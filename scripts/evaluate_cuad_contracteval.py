#!/usr/bin/env python3
"""
CUAD Evaluation - ContractEval Methodology

Implements the exact methodology from ContractEval (2025) which achieves F1=64.4% with GPT-4.1 mini.

Key differences from our previous approach:
1. Zero-shot prompting (no few-shot examples)
2. Full document context (no chunking) - requires 128K+ context
3. Q&A format: one question per clause type
4. Exact span extraction: "respond with exact sentences from the Context"
5. Span-based matching: True Positive = prediction fully covers labeled span
6. "No related clause" for negative cases
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
from functools import wraps

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'src', 'utils'))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

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


# =============================================================================
# CUAD 41 CLAUSE TYPES - Exact definitions from ContractEval
# =============================================================================

CUAD_CLAUSE_QUESTIONS = {
    "DOCUMENT_NAME": "What is the name or title of this document/agreement?",
    "PARTIES": "Who are the parties to this agreement?",
    "AGREEMENT_DATE": "What is the date of the agreement or effective date?",
    "EFFECTIVE_DATE": "When does this agreement become effective?",
    "EXPIRATION_DATE": "When does this agreement expire or terminate?",
    "RENEWAL_TERM": "What are the renewal terms of this agreement?",
    "NOTICE_PERIOD_TO_TERMINATE_RENEWAL": "What notice period is required to terminate renewal?",
    "GOVERNING_LAW": "What law governs this agreement?",
    "MOST_FAVORED_NATION": "Are there any most favored nation clauses?",
    "NON_COMPETE": "Are there any non-compete restrictions?",
    "EXCLUSIVITY": "Are there any exclusivity provisions?",
    "NO_SOLICIT_OF_CUSTOMERS": "Are there any non-solicitation of customers clauses?",
    "NO_SOLICIT_OF_EMPLOYEES": "Are there any non-solicitation of employees clauses?",
    "NON_DISPARAGEMENT": "Are there any non-disparagement clauses?",
    "TERMINATION_FOR_CONVENIENCE": "Can either party terminate for convenience?",
    "ROFR_ROFO_ROFN": "Are there any right of first refusal, offer, or negotiation clauses?",
    "CHANGE_OF_CONTROL": "Are there any change of control provisions?",
    "ANTI_ASSIGNMENT": "Are there any anti-assignment clauses?",
    "REVENUE_PROFIT_SHARING": "Are there any revenue or profit sharing provisions?",
    "PRICE_RESTRICTIONS": "Are there any price restrictions?",
    "MINIMUM_COMMITMENT": "Are there any minimum commitment requirements?",
    "VOLUME_RESTRICTION": "Are there any volume restrictions?",
    "IP_OWNERSHIP_ASSIGNMENT": "Are there any IP ownership or assignment clauses?",
    "JOINT_IP_OWNERSHIP": "Are there any joint IP ownership provisions?",
    "LICENSE_GRANT": "What licenses are granted in this agreement?",
    "NON_TRANSFERABLE_LICENSE": "Is the license non-transferable?",
    "AFFILIATE_LICENSE_LICENSEE": "Can the licensee grant licenses to affiliates?",
    "AFFILIATE_LICENSE_LICENSOR": "Can the licensor grant licenses to affiliates?",
    "UNLIMITED_ALL_YOU_CAN_EAT_LICENSE": "Is there an unlimited or all-you-can-eat license?",
    "IRREVOCABLE_OR_PERPETUAL_LICENSE": "Is the license irrevocable or perpetual?",
    "SOURCE_CODE_ESCROW": "Are there any source code escrow provisions?",
    "POST_TERMINATION_SERVICES": "Are there any post-termination service obligations?",
    "AUDIT_RIGHTS": "Are there any audit rights?",
    "UNCAPPED_LIABILITY": "Is there uncapped liability?",
    "CAP_ON_LIABILITY": "Is there a cap on liability?",
    "LIQUIDATED_DAMAGES": "Are there any liquidated damages provisions?",
    "WARRANTY_DURATION": "What is the warranty duration?",
    "INSURANCE": "Are there any insurance requirements?",
    "COVENANT_NOT_TO_SUE": "Are there any covenants not to sue?",
    "THIRD_PARTY_BENEFICIARY": "Are there any third-party beneficiary provisions?",
    "COMPETITIVE_RESTRICTION_EXCEPTION": "Are there any exceptions to competitive restrictions?",
}

CUAD_ALL_TYPES = set(CUAD_CLAUSE_QUESTIONS.keys())


# =============================================================================
# CONTRACTEVAL STYLE PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an assistant with strong legal knowledge, specialized in commercial contract review.

Your task is to identify and extract exact text spans from contracts that relate to specific legal clauses.

CRITICAL INSTRUCTIONS:
1. Respond ONLY with exact sentences or phrases directly copied from the contract
2. Do NOT rephrase, summarize, or paraphrase in any way
3. If no relevant clause exists, respond with exactly: "No related clause"
4. Extract ALL relevant spans, not just the first occurrence
5. Separate multiple spans with "|||" delimiter"""


def create_contracteval_prompt(contract_text: str, clause_type: str, question: str) -> str:
    """Create a ContractEval-style prompt for single clause extraction."""
    return f"""Context:
{contract_text}

Question: {question}

Instructions:
- Extract the exact text from the contract that answers this question
- Copy the text verbatim - do not rephrase or summarize
- If multiple relevant clauses exist, separate them with "|||"
- If no relevant clause exists, respond with exactly: "No related clause"

Answer:"""


def create_batch_extraction_prompt(contract_text: str, clause_types: List[str]) -> str:
    """Create a batch extraction prompt for multiple clause types."""
    questions = []
    for i, ct in enumerate(clause_types, 1):
        q = CUAD_CLAUSE_QUESTIONS.get(ct, f"Extract {ct} clauses")
        questions.append(f"{i}. {ct}: {q}")

    questions_text = "\n".join(questions)

    return f"""Context:
{contract_text}

Extract exact text spans from the contract for each clause type below.
- Copy text VERBATIM from the contract - do not rephrase
- Use "No related clause" if nothing relevant exists
- Separate multiple spans with "|||"

Questions:
{questions_text}

Respond in this exact JSON format:
{{
  "CLAUSE_TYPE_1": "exact text from contract" or "No related clause",
  "CLAUSE_TYPE_2": "span1 ||| span2" or "No related clause",
  ...
}}

JSON Response:"""


# =============================================================================
# MATCHING FUNCTIONS - ContractEval Style
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def compute_span_coverage(pred: str, gold: str) -> float:
    """
    Compute how much of the gold span is covered by the prediction.
    ContractEval: True Positive = prediction fully covers labeled span
    """
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)

    if not gold_norm:
        return 0.0

    # Check if prediction contains the gold span
    if gold_norm in pred_norm:
        return 1.0

    # Check token-level overlap (Jaccard-style)
    pred_tokens = set(pred_norm.split())
    gold_tokens = set(gold_norm.split())

    if not gold_tokens:
        return 0.0

    intersection = pred_tokens & gold_tokens
    coverage = len(intersection) / len(gold_tokens)

    return coverage


def compute_jaccard_similarity(pred: str, gold: str) -> float:
    """Compute Jaccard similarity between prediction and gold."""
    pred_tokens = set(normalize_text(pred).split())
    gold_tokens = set(normalize_text(gold).split())

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    intersection = pred_tokens & gold_tokens
    union = pred_tokens | gold_tokens

    return len(intersection) / len(union)


def match_prediction_to_gold(pred: str, golds: List[str], threshold: float = 0.7) -> Tuple[bool, Optional[str], float]:
    """
    Match a prediction against gold spans using ContractEval criteria.
    Returns: (is_match, matched_gold, coverage_score)
    """
    if not golds:
        return False, None, 0.0

    best_match = None
    best_score = 0.0

    for gold in golds:
        coverage = compute_span_coverage(pred, gold)
        if coverage > best_score:
            best_score = coverage
            best_match = gold

    is_match = best_score >= threshold
    return is_match, best_match, best_score


# =============================================================================
# LLM CLIENTS
# =============================================================================

class ContractEvalExtractor:
    """Extractor following ContractEval methodology."""

    def __init__(self, model: str = "gpt-4o", batch_size: int = 10, request_delay: float = 0.0):
        self.model = model
        self.batch_size = batch_size  # Number of clause types per request
        self.request_delay = request_delay  # Delay between requests to avoid rate limits
        self._init_client()

    def _init_client(self):
        """Initialize the appropriate LLM client."""
        if "gemini" in self.model.lower():
            if not GEMINI_AVAILABLE:
                raise RuntimeError("Google Generative AI not installed")
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            # Use stable model name
            model_name = "gemini-2.0-flash" if "2.0" in self.model else "gemini-1.5-flash"
            self.client = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT
            )
            self.provider = "gemini"
        else:
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI not installed")
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            self.provider = "azure"

    def _call_llm(self, prompt: str, max_retries: int = 5) -> str:
        """Call LLM with retry logic."""
        last_error = None

        for attempt in range(max_retries):
            try:
                if self.provider == "gemini":
                    response = self.client.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=0.0,
                            max_output_tokens=8000,
                        )
                    )
                    return response.text
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0,
                        max_tokens=4096,
                    )
                    return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Longer delays for rate limits
                    if "429" in str(e) or "rate" in str(e).lower():
                        delay = 30 * (attempt + 1)  # 30s, 60s, 90s...
                    else:
                        delay = 5 * (attempt + 1)
                    print(f"  Retry {attempt + 1}: {str(e)[:60]}... waiting {delay}s", flush=True)
                    time.sleep(delay)

        raise last_error

    def extract_all_clauses(self, contract_text: str) -> Dict[str, List[str]]:
        """Extract all clause types from a contract using batch processing."""
        results = {}
        clause_types = list(CUAD_ALL_TYPES)

        # Process in batches
        for i in range(0, len(clause_types), self.batch_size):
            batch = clause_types[i:i + self.batch_size]
            prompt = create_batch_extraction_prompt(contract_text, batch)

            try:
                response = self._call_llm(prompt)
                batch_results = self._parse_batch_response(response, batch)
                results.update(batch_results)

                # Add delay between requests to avoid rate limits
                if self.request_delay > 0 and i + self.batch_size < len(clause_types):
                    time.sleep(self.request_delay)

            except Exception as e:
                print(f"  Error in batch {i//self.batch_size + 1}: {e}")
                # Mark all as empty on error
                for ct in batch:
                    results[ct] = []

        return results

    def _parse_batch_response(self, response: str, clause_types: List[str]) -> Dict[str, List[str]]:
        """Parse batch extraction response."""
        results = {ct: [] for ct in clause_types}

        try:
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())

                for key, value in data.items():
                    # Normalize key
                    norm_key = key.upper().replace(" ", "_").replace("-", "_")

                    if norm_key in results:
                        if isinstance(value, str):
                            if value.lower().strip() != "no related clause":
                                # Split by delimiter
                                spans = [s.strip() for s in value.split("|||")]
                                results[norm_key] = [s for s in spans if s and s.lower() != "no related clause"]
                        elif isinstance(value, list):
                            results[norm_key] = [str(v).strip() for v in value if str(v).strip().lower() != "no related clause"]
        except json.JSONDecodeError:
            # Try to parse line by line
            for line in response.split('\n'):
                for ct in clause_types:
                    if ct in line.upper():
                        # Extract value after colon
                        if ':' in line:
                            value = line.split(':', 1)[1].strip()
                            value = value.strip('"\'')
                            if value.lower() != "no related clause":
                                results[ct] = [s.strip() for s in value.split("|||") if s.strip()]
        except Exception as e:
            print(f"  Parse error: {e}")

        return results


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cuad_test_data(data_path: str) -> List[Dict]:
    """Load CUAD test data."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    documents = []

    for doc in data['data']:
        title = doc['title']

        for para in doc['paragraphs']:
            context = para['context']
            ground_truth = defaultdict(list)

            for qa in para['qas']:
                question = qa['question']

                # Extract clause type from question
                match = re.search(r'related to "([^"]+)"', question)
                if match:
                    category = match.group(1)
                    norm_type = category.upper().replace(" ", "_").replace("-", "_").replace("/", "_")

                    if norm_type in CUAD_ALL_TYPES:
                        for answer in qa.get('answers', []):
                            text = answer.get('text', '').strip()
                            if text:
                                ground_truth[norm_type].append(text)

            if ground_truth:
                documents.append({
                    'id': title,
                    'text': context,
                    'ground_truth': dict(ground_truth)
                })

    return documents


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_document(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth using ContractEval metrics.
    """
    metrics = {
        'per_type': {},
        'overall': {'tp': 0, 'fp': 0, 'fn': 0}
    }

    all_types = set(predictions.keys()) | set(ground_truth.keys())

    for clause_type in all_types:
        preds = predictions.get(clause_type, [])
        golds = ground_truth.get(clause_type, [])

        tp = 0
        fp = 0
        matched_golds = set()

        # For each prediction, check if it matches a gold
        for pred in preds:
            is_match, matched_gold, score = match_prediction_to_gold(pred, golds, threshold)
            if is_match and matched_gold:
                # Find index of matched gold
                for i, g in enumerate(golds):
                    if g == matched_gold and i not in matched_golds:
                        tp += 1
                        matched_golds.add(i)
                        break
                else:
                    fp += 1  # Already matched this gold
            else:
                fp += 1

        fn = len(golds) - len(matched_golds)

        metrics['per_type'][clause_type] = {'tp': tp, 'fp': fp, 'fn': fn}
        metrics['overall']['tp'] += tp
        metrics['overall']['fp'] += fp
        metrics['overall']['fn'] += fn

    return metrics


def compute_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Compute precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# =============================================================================
# MAIN
# =============================================================================

def run_evaluation(
    data_path: str,
    model: str = "gpt-4o",
    batch_size: int = 10,
    match_threshold: float = 0.7,
    max_docs: Optional[int] = None,
    output_dir: str = "experiments/results"
):
    """Run full evaluation."""
    print("=" * 70)
    print("CUAD EVALUATION - ContractEval Methodology")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Batch size: {batch_size}")
    print(f"Match threshold: {match_threshold}")
    print()

    # Load data
    print(f"Loading data from: {data_path}")
    documents = load_cuad_test_data(data_path)
    print(f"Loaded {len(documents)} documents")

    if max_docs:
        documents = documents[:max_docs]
        print(f"Limited to {max_docs} documents")

    # Initialize extractor
    print(f"\nInitializing {model} extractor...")
    # Add delay to avoid rate limits (Gemini: 10 RPM, GPT-4o: variable)
    if "gemini" in model.lower():
        request_delay = 7.0
    elif "gpt" in model.lower():
        request_delay = 5.0  # Azure rate limits
    else:
        request_delay = 0.0
    extractor = ContractEvalExtractor(model=model, batch_size=batch_size, request_delay=request_delay)

    # Run extraction
    print("\nRunning extraction...")
    all_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    per_type_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    errors = 0

    start_time = time.time()

    for doc in tqdm(documents, desc=f"{model} (ContractEval)"):
        try:
            predictions = extractor.extract_all_clauses(doc['text'])
            doc_metrics = evaluate_document(predictions, doc['ground_truth'], match_threshold)

            all_metrics['tp'] += doc_metrics['overall']['tp']
            all_metrics['fp'] += doc_metrics['overall']['fp']
            all_metrics['fn'] += doc_metrics['overall']['fn']

            for ct, m in doc_metrics['per_type'].items():
                per_type_metrics[ct]['tp'] += m['tp']
                per_type_metrics[ct]['fp'] += m['fp']
                per_type_metrics[ct]['fn'] += m['fn']

        except Exception as e:
            print(f"\n  Error processing {doc['id']}: {e}")
            errors += 1

    total_time = time.time() - start_time

    # Compute overall metrics
    precision, recall, f1 = compute_f1(all_metrics['tp'], all_metrics['fp'], all_metrics['fn'])

    # Compute per-type metrics
    per_type_results = {}
    for ct in CUAD_ALL_TYPES:
        m = per_type_metrics[ct]
        p, r, f = compute_f1(m['tp'], m['fp'], m['fn'])
        support = m['tp'] + m['fn']
        per_type_results[ct] = {
            'precision': p, 'recall': r, 'f1': f,
            'tp': m['tp'], 'fp': m['fp'], 'fn': m['fn'],
            'support': support
        }

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS - ContractEval Methodology")
    print("=" * 70)
    print(f"Model:      {model}")
    print(f"Documents:  {len(documents)} ({errors} errors)")
    print(f"Time:       {total_time:.1f}s ({total_time/len(documents):.2f}s/doc)")
    print()
    print("OVERALL METRICS:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Support:   {all_metrics['tp'] + all_metrics['fn']}")
    print()

    # Top 10 types by F1
    print("TOP 10 TYPES BY F1:")
    print("-" * 70)
    sorted_types = sorted(per_type_results.items(), key=lambda x: x[1]['f1'], reverse=True)[:10]
    print(f"{'Clause Type':<40} {'P':>8} {'R':>8} {'F1':>8} {'Support':>8}")
    print("-" * 70)
    for ct, m in sorted_types:
        print(f"{ct:<40} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['support']:>8}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.replace("-", "_").replace(".", "_")

    results = {
        'model': model,
        'methodology': 'contracteval',
        'batch_size': batch_size,
        'match_threshold': match_threshold,
        'timestamp': timestamp,
        'num_documents': len(documents),
        'errors': errors,
        'total_time': total_time,
        'overall': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': all_metrics['tp'],
            'fp': all_metrics['fp'],
            'fn': all_metrics['fn'],
            'support': all_metrics['tp'] + all_metrics['fn']
        },
        'per_type': per_type_results
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"cuad_contracteval_{model_name}_{timestamp}.json")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="CUAD Evaluation - ContractEval Methodology")
    parser.add_argument("--data", default="data/cuad/test.json", help="CUAD test data path")
    parser.add_argument("--model", default="gpt-4o", help="Model to use")
    parser.add_argument("--batch-size", type=int, default=10, help="Clause types per request")
    parser.add_argument("--threshold", type=float, default=0.7, help="Match threshold")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to process")
    parser.add_argument("--output", default="experiments/results", help="Output directory")
    args = parser.parse_args()

    run_evaluation(
        data_path=args.data,
        model=args.model,
        batch_size=args.batch_size,
        match_threshold=args.threshold,
        max_docs=args.max_docs,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
