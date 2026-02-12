"""
Few-Shot Optimized Prompts for CUAD Entity Extraction.

This module generates prompts with real examples from the CUAD training set
to improve extraction accuracy for all 41 clause types.
"""

import json
import os
import random
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Path to training data
CUAD_TRAIN_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..',
    'data', 'cuad', 'train_separate_questions.json'
)

# All 41 CUAD clause types with descriptions
CUAD_CLAUSE_DESCRIPTIONS = {
    "Document Name": "The name or title of the contract/agreement",
    "Parties": "The entities (companies, individuals) that are parties to the contract",
    "Agreement Date": "The date when the agreement was signed or executed",
    "Effective Date": "The date when the agreement becomes effective",
    "Expiration Date": "The date when the agreement expires or ends",
    "Renewal Term": "Terms describing automatic renewal or extension of the agreement",
    "Notice Period To Terminate Renewal": "Required notice period to prevent automatic renewal",
    "Governing Law": "The jurisdiction whose laws govern the agreement",
    "Most Favored Nation": "Clauses ensuring equal or better terms than given to others",
    "Non-Compete": "Restrictions on competing activities during or after the agreement",
    "Exclusivity": "Exclusive rights or restrictions granted under the agreement",
    "No-Solicit Of Customers": "Restrictions on soliciting the other party's customers",
    "No-Solicit Of Employees": "Restrictions on hiring or soliciting the other party's employees",
    "Non-Disparagement": "Restrictions on making negative statements about the other party",
    "Termination For Convenience": "Right to terminate the agreement without cause",
    "Rofr/Rofo/Rofn": "Right of first refusal, first offer, or first negotiation",
    "Change Of Control": "Provisions triggered by change in ownership or control",
    "Anti-Assignment": "Restrictions on assigning rights or obligations under the agreement",
    "Revenue/Profit Sharing": "Terms for sharing revenue or profits between parties",
    "Price Restrictions": "Limitations on pricing or price changes",
    "Minimum Commitment": "Minimum purchase, sales, or other commitment requirements",
    "Volume Restriction": "Limitations on volume or quantity",
    "Ip Ownership Assignment": "Transfer or assignment of intellectual property rights",
    "Joint Ip Ownership": "Shared ownership of intellectual property",
    "License Grant": "Grant of license rights (e.g., to use IP, software, etc.)",
    "Non-Transferable License": "Restrictions on transferring licensed rights",
    "Affiliate License-Licensor": "License rights extended to licensor's affiliates",
    "Affiliate License-Licensee": "License rights extended to licensee's affiliates",
    "Unlimited/All-You-Can-Eat-License": "Unlimited usage rights under the license",
    "Irrevocable Or Perpetual License": "License that cannot be revoked or lasts indefinitely",
    "Source Code Escrow": "Provisions for source code escrow arrangements",
    "Post-Termination Services": "Obligations or services required after termination",
    "Audit Rights": "Rights to audit the other party's records or compliance",
    "Uncapped Liability": "Provisions where liability is not capped or limited",
    "Cap On Liability": "Limitations on liability amounts",
    "Liquidated Damages": "Pre-determined damages for breach",
    "Warranty Duration": "Duration of warranties provided",
    "Insurance": "Requirements for insurance coverage",
    "Covenant Not To Sue": "Agreement not to bring legal action",
    "Third Party Beneficiary": "Rights granted to third parties under the agreement",
    "Competitive Restriction Exception": "Exceptions to competitive restrictions",
}

# Normalize clause type names to match evaluation format
def normalize_clause_type(clause_type: str) -> str:
    """Convert clause type to normalized format."""
    return clause_type.upper().replace(" ", "_").replace("/", "_").replace("-", "_")


class CUADFewShotGenerator:
    """Generate few-shot prompts with real CUAD examples."""

    def __init__(self, train_path: str = None, examples_per_type: int = 3):
        self.train_path = train_path or CUAD_TRAIN_PATH
        self.examples_per_type = examples_per_type
        self.examples_cache = None

    def load_examples(self) -> Dict[str, List[Dict]]:
        """Load and cache examples from training data."""
        if self.examples_cache is not None:
            return self.examples_cache

        clause_examples = defaultdict(list)

        try:
            with open(self.train_path, 'r') as f:
                train_data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Training file not found at {self.train_path}")
            return {}

        for doc in train_data.get('data', []):
            title = doc.get('title', '')
            for para in doc.get('paragraphs', []):
                context = para.get('context', '')
                for qa in para.get('qas', []):
                    question = qa.get('question', '')
                    answers = qa.get('answers', [])
                    is_impossible = qa.get('is_impossible', False)

                    # Extract clause type from question
                    if '"' in question:
                        start = question.find('"') + 1
                        end = question.find('"', start)
                        if end > start:
                            clause_type = question[start:end]

                            if answers and not is_impossible:
                                for ans in answers:
                                    text = ans.get('text', '').strip()
                                    if text and len(text) > 5:  # Filter very short answers
                                        # Get context around the answer
                                        ans_start = ans.get('answer_start', 0)
                                        context_before = context[max(0, ans_start-150):ans_start].strip()
                                        context_after = context[ans_start+len(text):ans_start+len(text)+150].strip()

                                        clause_examples[clause_type].append({
                                            'text': text[:500],  # Limit length
                                            'context_before': context_before[-100:] if context_before else "",
                                            'context_after': context_after[:100] if context_after else "",
                                            'doc_title': title
                                        })

        self.examples_cache = clause_examples
        return clause_examples

    def get_best_examples(self, clause_type: str, n: int = 3) -> List[Dict]:
        """Get the best examples for a clause type."""
        examples = self.load_examples()
        type_examples = examples.get(clause_type, [])

        if not type_examples:
            return []

        # Prefer diverse examples (different documents, different lengths)
        seen_docs = set()
        selected = []

        # Sort by text length (prefer medium-length examples)
        sorted_examples = sorted(type_examples, key=lambda x: abs(len(x['text']) - 100))

        for ex in sorted_examples:
            if ex['doc_title'] not in seen_docs and len(selected) < n:
                selected.append(ex)
                seen_docs.add(ex['doc_title'])

        # If we need more, add from same docs
        if len(selected) < n:
            for ex in sorted_examples:
                if ex not in selected and len(selected) < n:
                    selected.append(ex)

        return selected[:n]

    def format_example(self, example: Dict, clause_type: str) -> str:
        """Format a single example for the prompt."""
        text = example['text']

        # Truncate if too long
        if len(text) > 200:
            text = text[:200] + "..."

        return f'  - "{text}"'

    def create_fewshot_prompt(self, contract_text: str, max_text_len: int = 20000) -> str:
        """Create a prompt with few-shot examples for all clause types."""

        # Build examples section
        examples_section = "\n## EXAMPLES FROM CUAD TRAINING DATA\n\n"

        for clause_type, description in CUAD_CLAUSE_DESCRIPTIONS.items():
            examples = self.get_best_examples(clause_type, self.examples_per_type)

            if examples:
                normalized_type = normalize_clause_type(clause_type)
                examples_section += f"### {normalized_type}\n"
                examples_section += f"Description: {description}\n"
                examples_section += "Examples:\n"
                for ex in examples[:2]:  # Use 2 examples per type to save tokens
                    examples_section += self.format_example(ex, clause_type) + "\n"
                examples_section += "\n"

        # Truncate contract text if needed
        if len(contract_text) > max_text_len:
            contract_text = contract_text[:max_text_len] + "\n[...TEXT TRUNCATED...]"

        prompt = f"""You are an expert legal contract analyst. Extract all clause types from the contract below.

## TASK
Extract entities for ALL 41 CUAD clause types. Return ONLY valid JSON.

## ENTITY TYPES (41 total)
{', '.join([normalize_clause_type(ct) for ct in CUAD_CLAUSE_DESCRIPTIONS.keys()])}

{examples_section}

## CRITICAL INSTRUCTIONS
1. Extract the EXACT text from the document - do not paraphrase
2. Include ALL instances of each clause type found
3. For each extraction, provide:
   - "text": exact text from document
   - "type": one of the 41 types listed above
   - "confidence": 0.85-0.99
4. If a clause type is not present, do not include it
5. Pay special attention to: PARTIES, DOCUMENT_NAME, AGREEMENT_DATE, EFFECTIVE_DATE

## CONTRACT TO ANALYZE
```
{contract_text}
```

## RESPONSE FORMAT
Respond with valid JSON only:
```json
{{
  "entities": [
    {{"text": "exact text from document", "type": "CLAUSE_TYPE", "confidence": 0.95}}
  ]
}}
```
"""
        return prompt


def create_grouped_fewshot_prompt(contract_text: str, group: str, generator: CUADFewShotGenerator = None) -> str:
    """Create a prompt for a specific group of clause types.

    Groups:
    - "basic": Document Name, Parties, Agreement Date, Effective Date, Expiration Date
    - "termination": Renewal Term, Notice Period, Termination For Convenience, Post-Termination
    - "restrictions": Non-Compete, Exclusivity, No-Solicit, Non-Disparagement, Anti-Assignment
    - "ip": IP Ownership, License Grant, Joint IP, Source Code Escrow
    - "financial": Revenue Sharing, Price Restrictions, Minimum Commitment, Cap On Liability
    - "legal": Governing Law, Audit Rights, Insurance, Warranty, Covenant Not To Sue
    """

    groups = {
        "basic": [
            "Document Name", "Parties", "Agreement Date",
            "Effective Date", "Expiration Date"
        ],
        "termination": [
            "Renewal Term", "Notice Period To Terminate Renewal",
            "Termination For Convenience", "Post-Termination Services",
            "Change Of Control"
        ],
        "restrictions": [
            "Non-Compete", "Exclusivity", "No-Solicit Of Customers",
            "No-Solicit Of Employees", "Non-Disparagement",
            "Anti-Assignment", "Competitive Restriction Exception"
        ],
        "ip": [
            "Ip Ownership Assignment", "Joint Ip Ownership", "License Grant",
            "Non-Transferable License", "Affiliate License-Licensor",
            "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat-License",
            "Irrevocable Or Perpetual License", "Source Code Escrow"
        ],
        "financial": [
            "Revenue/Profit Sharing", "Price Restrictions",
            "Minimum Commitment", "Volume Restriction",
            "Cap On Liability", "Uncapped Liability",
            "Liquidated Damages", "Most Favored Nation"
        ],
        "legal": [
            "Governing Law", "Audit Rights", "Insurance",
            "Warranty Duration", "Covenant Not To Sue",
            "Third Party Beneficiary", "Rofr/Rofo/Rofn"
        ]
    }

    if group not in groups:
        raise ValueError(f"Unknown group: {group}. Available: {list(groups.keys())}")

    if generator is None:
        generator = CUADFewShotGenerator()

    clause_types = groups[group]

    # Build examples for this group only
    examples_section = f"\n## EXAMPLES FOR {group.upper()} CLAUSES\n\n"

    for clause_type in clause_types:
        examples = generator.get_best_examples(clause_type, 2)
        description = CUAD_CLAUSE_DESCRIPTIONS.get(clause_type, "")

        if examples:
            normalized_type = normalize_clause_type(clause_type)
            examples_section += f"### {normalized_type}\n"
            examples_section += f"{description}\n"
            examples_section += "Examples:\n"
            for ex in examples:
                examples_section += generator.format_example(ex, clause_type) + "\n"
            examples_section += "\n"

    types_list = [normalize_clause_type(ct) for ct in clause_types]

    prompt = f"""You are an expert legal contract analyst. Extract {group.upper()} clauses from the contract.

## TASK
Extract ONLY these clause types: {', '.join(types_list)}

{examples_section}

## CRITICAL INSTRUCTIONS
1. Extract the EXACT text from the document
2. Include ALL instances found for each type
3. Only extract types from the list above
4. If a type is not present, do not include it

## CONTRACT
```
{contract_text[:15000]}
```

## RESPONSE (JSON only)
```json
{{"entities": [{{"text": "exact text", "type": "TYPE", "confidence": 0.95}}]}}
```
"""
    return prompt


# Pre-built generator instance
_generator = None

def get_generator() -> CUADFewShotGenerator:
    """Get or create the global generator instance."""
    global _generator
    if _generator is None:
        _generator = CUADFewShotGenerator()
    return _generator


def create_cuad_fewshot_prompt(text: str) -> str:
    """Create few-shot prompt for CUAD extraction."""
    return get_generator().create_fewshot_prompt(text)


__all__ = [
    'CUADFewShotGenerator',
    'CUAD_CLAUSE_DESCRIPTIONS',
    'normalize_clause_type',
    'create_cuad_fewshot_prompt',
    'create_grouped_fewshot_prompt',
    'get_generator',
]
