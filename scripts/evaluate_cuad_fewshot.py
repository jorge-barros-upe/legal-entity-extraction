#!/usr/bin/env python3
"""
Evaluate CUAD extraction with Few-Shot optimized prompts.

This script uses real examples from the CUAD training set to improve
extraction accuracy for all 41 clause types.
"""

import os
import sys
import json
import re
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'src', 'approaches', 'llm'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import pandas as pd
from tqdm import tqdm

# LLM clients
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Import few-shot prompts
from cuad_fewshot_prompts import (
    CUADFewShotGenerator,
    CUAD_CLAUSE_DESCRIPTIONS,
    normalize_clause_type,
    create_cuad_fewshot_prompt,
    create_grouped_fewshot_prompt,
)


# =============================================================================
# CUAD DATA LOADING
# =============================================================================

def load_cuad_data(data_path: str) -> List[Dict]:
    """Load CUAD test data in QA format and convert to extraction format."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    documents = []

    for entry in data.get('data', []):
        title = entry.get('title', '')

        for para in entry.get('paragraphs', []):
            context = para.get('context', '')
            entities = []

            for qa in para.get('qas', []):
                question = qa.get('question', '')
                answers = qa.get('answers', [])
                is_impossible = qa.get('is_impossible', False)

                # Extract clause type from question
                if '"' in question and not is_impossible:
                    start = question.find('"') + 1
                    end = question.find('"', start)
                    if end > start:
                        clause_type = question[start:end]
                        normalized_type = normalize_clause_type(clause_type)

                        for ans in answers:
                            if ans.get('text'):
                                entities.append({
                                    'text': ans['text'],
                                    'type': normalized_type,
                                    'start': ans.get('answer_start', 0)
                                })

            if entities:
                documents.append({
                    'id': title,
                    'text': context,
                    'entities': entities
                })

    return documents


# =============================================================================
# LLM EXTRACTOR
# =============================================================================

class FewShotExtractor:
    """Extract entities using few-shot prompts."""

    def __init__(self, model_name: str = "gpt-4o", use_groups: bool = False):
        self.model_name = model_name
        self.use_groups = use_groups
        self.generator = CUADFewShotGenerator()

        # Initialize client
        if "gemini" in model_name.lower():
            self._init_gemini()
        else:
            self._init_openai()

    def _init_openai(self):
        """Initialize Azure OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed")

        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

        model_map = {
            "gpt-4o": os.getenv("AZURE_OPENAI_MODEL", "gpt-4o"),
            "gpt-4-turbo": "gpt-4-turbo",
        }
        self.deployment = model_map.get(self.model_name, self.model_name)
        self.provider = "openai"

    def _init_gemini(self):
        """Initialize Gemini client."""
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(self.model_name)
        self.provider = "gemini"

    def extract(self, text: str) -> Tuple[List[Dict], float]:
        """Extract entities from text."""
        start_time = time.time()

        if self.use_groups:
            entities = self._extract_with_groups(text)
        else:
            entities = self._extract_single_pass(text)

        latency = time.time() - start_time
        return entities, latency

    def _extract_single_pass(self, text: str) -> List[Dict]:
        """Extract all entities in a single pass."""
        prompt = create_cuad_fewshot_prompt(text)
        response = self._call_llm(prompt)
        return self._parse_response(response)

    def _extract_with_groups(self, text: str) -> List[Dict]:
        """Extract entities in multiple passes by group."""
        all_entities = []
        groups = ["basic", "termination", "restrictions", "ip", "financial", "legal"]

        for group in groups:
            prompt = create_grouped_fewshot_prompt(text, group, self.generator)
            response = self._call_llm(prompt)
            entities = self._parse_response(response)
            all_entities.extend(entities)

        # Deduplicate
        seen = set()
        unique_entities = []
        for ent in all_entities:
            key = (ent.get('text', ''), ent.get('type', ''))
            if key not in seen:
                seen.add(key)
                unique_entities.append(ent)

        return unique_entities

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4096,
                )
                return response.choices[0].message.content
            else:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=4096,
                    )
                )
                return response.text
        except Exception as e:
            print(f"LLM error: {e}")
            return "{}"

    def _parse_response(self, response: str) -> List[Dict]:
        """Parse LLM response to extract entities."""
        if not response:
            return []

        # Try to extract JSON
        try:
            # Find JSON block
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                entities = data.get('entities', [])
                return [e for e in entities if e.get('text') and e.get('type')]
        except json.JSONDecodeError:
            pass

        return []


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_token_overlap(pred_text: str, gold_text: str) -> float:
    """Calculate token-level overlap between prediction and gold."""
    pred_tokens = set(pred_text.lower().split())
    gold_tokens = set(gold_text.lower().split())

    if not gold_tokens:
        return 0.0

    intersection = pred_tokens & gold_tokens
    return len(intersection) / len(gold_tokens)


def evaluate_extraction(
    predictions: List[Dict],
    ground_truth: List[Dict],
    overlap_threshold: float = 0.5
) -> Dict[str, Any]:
    """Evaluate extraction results against ground truth."""

    # Group by type
    pred_by_type = defaultdict(list)
    gold_by_type = defaultdict(list)

    for p in predictions:
        pred_by_type[p.get('type', 'UNKNOWN')].append(p)
    for g in ground_truth:
        gold_by_type[g.get('type', 'UNKNOWN')].append(g)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    type_metrics = {}

    all_types = set(pred_by_type.keys()) | set(gold_by_type.keys())

    for entity_type in all_types:
        preds = pred_by_type.get(entity_type, [])
        golds = gold_by_type.get(entity_type, [])

        matched_golds = set()
        tp = 0

        for pred in preds:
            pred_text = pred.get('text', '')
            best_overlap = 0
            best_gold_idx = -1

            for idx, gold in enumerate(golds):
                if idx in matched_golds:
                    continue
                gold_text = gold.get('text', '')
                overlap = calculate_token_overlap(pred_text, gold_text)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_gold_idx = idx

            if best_overlap >= overlap_threshold and best_gold_idx >= 0:
                tp += 1
                matched_golds.add(best_gold_idx)

        fp = len(preds) - tp
        fn = len(golds) - len(matched_golds)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        type_metrics[entity_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'support': len(golds)
        }

    # Overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'support': sum(len(golds) for golds in gold_by_type.values()),
        'by_type': type_metrics
    }


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate CUAD with few-shot prompts")
    parser.add_argument("--data", type=str, required=True, help="Path to CUAD test.json")
    parser.add_argument("--output", type=str, default="experiments/results", help="Output directory")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Model to use")
    parser.add_argument("--use-groups", action="store_true", help="Use grouped extraction")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to evaluate")
    args = parser.parse_args()

    print(f"Loading CUAD data from {args.data}...")
    documents = load_cuad_data(args.data)
    print(f"Loaded {len(documents)} documents")

    if args.max_docs:
        documents = documents[:args.max_docs]
        print(f"Limited to {len(documents)} documents")

    # Count ground truth
    total_gt = sum(len(doc['entities']) for doc in documents)
    print(f"Total ground truth annotations: {total_gt}")

    # Initialize extractor
    extractor = FewShotExtractor(args.model, use_groups=args.use_groups)

    # Run evaluation
    all_predictions = []
    all_ground_truth = []
    latencies = []
    errors = 0

    strategy = "grouped" if args.use_groups else "single"
    desc = f"{args.model} ({strategy})"

    for doc in tqdm(documents, desc=desc):
        try:
            predictions, latency = extractor.extract(doc['text'])
            all_predictions.extend(predictions)
            all_ground_truth.extend(doc['entities'])
            latencies.append(latency)
        except Exception as e:
            print(f"Error processing {doc['id']}: {e}")
            errors += 1
            all_ground_truth.extend(doc['entities'])

    # Calculate metrics
    metrics = evaluate_extraction(all_predictions, all_ground_truth)

    print(f"\n{'='*60}")
    print(f"Results: {args.model} ({strategy})")
    print(f"{'='*60}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Support:   {metrics['support']}")
    print(f"TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
    print(f"Latency:   {sum(latencies)/len(latencies):.2f}s ± {(sum((l-sum(latencies)/len(latencies))**2 for l in latencies)/len(latencies))**0.5:.2f}s")
    print(f"Errors:    {errors}")

    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_clean = args.model.replace("-", "_").replace(".", "_")

    # Summary CSV
    summary_path = os.path.join(args.output, f"cuad_fewshot_{model_clean}_{strategy}_{timestamp}.csv")
    summary_df = pd.DataFrame([{
        'model': args.model,
        'strategy': strategy,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'support': metrics['support'],
        'tp': metrics['tp'],
        'fp': metrics['fp'],
        'fn': metrics['fn'],
        'latency_avg': sum(latencies)/len(latencies) if latencies else 0,
        'num_docs': len(documents),
        'errors': errors
    }])
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Per-type results
    type_results = []
    for entity_type, type_metrics in metrics['by_type'].items():
        type_results.append({
            'type': entity_type,
            'precision': type_metrics['precision'],
            'recall': type_metrics['recall'],
            'f1': type_metrics['f1'],
            'support': type_metrics['support'],
            'tp': type_metrics['tp'],
            'fp': type_metrics['fp'],
            'fn': type_metrics['fn']
        })

    type_df = pd.DataFrame(type_results)
    type_df = type_df.sort_values('support', ascending=False)
    type_path = os.path.join(args.output, f"cuad_fewshot_{model_clean}_{strategy}_{timestamp}_per_type.csv")
    type_df.to_csv(type_path, index=False)
    print(f"Per-type results saved to: {type_path}")

    # Print top 10 types by support
    print(f"\n{'='*60}")
    print("Top 10 Entity Types by Support")
    print(f"{'='*60}")
    print(type_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
