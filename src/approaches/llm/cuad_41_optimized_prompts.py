"""
Optimized Prompts for CUAD 41-Type Entity Extraction.

Based on ContractEval (2025) methodology that achieves F1=64.1% with GPT-4.1.

Key improvements over original prompts:
1. Question-Answering format (one question per clause type)
2. Chain-of-Thought prompting
3. Structured output with JSON schema
4. Semantic grouping for efficient batch extraction
"""

import json
from typing import List, Dict, Any, Optional


# =============================================================================
# CUAD CLAUSE TYPES WITH ENHANCED DESCRIPTIONS (41 TYPES)
# =============================================================================

CUAD_CLAUSE_TYPES_ENHANCED = {
    "DOCUMENT_NAME": {
        "question": "What is the name or title of this contract/agreement?",
        "description": "The formal name of the contract document",
        "keywords": ["agreement", "contract", "license", "amendment"],
        "typical_location": "First page, title section",
        "examples": ["LICENSE AGREEMENT", "ASSET PURCHASE AGREEMENT", "MASTER SERVICE AGREEMENT"]
    },
    "PARTIES": {
        "question": "Who are the parties (companies or individuals) that signed this contract?",
        "description": "The entities entering into the contractual relationship",
        "keywords": ["party", "parties", "between", "by and between", "hereinafter"],
        "typical_location": "Preamble, first paragraph",
        "examples": ["ABC Corporation, a Delaware corporation", "John Smith, an individual"]
    },
    "AGREEMENT_DATE": {
        "question": "What is the date when this agreement was signed or executed?",
        "description": "The date of contract execution/signing",
        "keywords": ["dated", "as of", "entered into", "made and entered"],
        "typical_location": "First paragraph, after parties",
        "examples": ["January 15, 2020", "as of March 1, 2019"]
    },
    "EFFECTIVE_DATE": {
        "question": "What is the date when this agreement becomes effective?",
        "description": "The date when contractual obligations begin",
        "keywords": ["effective date", "effective as of", "shall become effective", "commencement date"],
        "typical_location": "Preamble or definitions section",
        "examples": ["effective as of January 1, 2020", "shall become effective on the Closing Date"]
    },
    "EXPIRATION_DATE": {
        "question": "What is the date when this contract's initial term expires?",
        "description": "End date of the initial contract term",
        "keywords": ["expire", "expiration", "termination date", "end date", "initial term"],
        "typical_location": "Term section",
        "examples": ["December 31, 2025", "three years from the Effective Date"]
    },
    "RENEWAL_TERM": {
        "question": "What are the renewal terms after the initial term expires?",
        "description": "Automatic extensions or renewal provisions",
        "keywords": ["renew", "renewal", "automatically extend", "successive", "additional term"],
        "typical_location": "Term section",
        "examples": ["automatically renew for successive one-year terms", "may be extended for additional 12-month periods"]
    },
    "NOTICE_PERIOD_TO_TERMINATE_RENEWAL": {
        "question": "What notice period is required to terminate or prevent automatic renewal?",
        "description": "Required advance notice to stop renewal",
        "keywords": ["notice", "days prior", "written notice", "terminate renewal"],
        "typical_location": "Term section",
        "examples": ["30 days prior written notice", "90 days before expiration"]
    },
    "GOVERNING_LAW": {
        "question": "Which state or country's law governs this contract?",
        "description": "Jurisdiction whose laws apply to the contract",
        "keywords": ["governed by", "governing law", "laws of", "jurisdiction"],
        "typical_location": "General provisions, near end",
        "examples": ["governed by the laws of the State of Delaware", "New York law shall govern"]
    },
    "MOST_FAVORED_NATION": {
        "question": "Is there a most favored nation clause ensuring equal or better terms?",
        "description": "Guarantee of most favorable terms",
        "keywords": ["most favored", "no less favorable", "most favorable terms"],
        "typical_location": "Pricing or commercial terms",
        "examples": ["most favored customer pricing", "shall receive terms no less favorable than"]
    },
    "NON_COMPETE": {
        "question": "Are there restrictions on competing activities?",
        "description": "Restrictions on competition",
        "keywords": ["non-compete", "not compete", "competitive activities", "competing business"],
        "typical_location": "Restrictive covenants section",
        "examples": ["shall not compete in the Territory", "agrees not to engage in any competing business"]
    },
    "EXCLUSIVITY": {
        "question": "Are there exclusive dealing commitments or exclusivity provisions?",
        "description": "Exclusive rights or restrictions",
        "keywords": ["exclusive", "sole", "exclusivity", "only supplier", "only customer"],
        "typical_location": "License grant or commercial terms",
        "examples": ["exclusive license", "sole and exclusive right", "shall not license to any third party"]
    },
    "NO_SOLICIT_OF_CUSTOMERS": {
        "question": "Are there restrictions on soliciting customers?",
        "description": "Prohibition on customer solicitation",
        "keywords": ["solicit", "customers", "clients", "accounts"],
        "typical_location": "Restrictive covenants",
        "examples": ["shall not solicit any customer", "agrees not to contact clients"]
    },
    "COMPETITIVE_RESTRICTION_EXCEPTION": {
        "question": "Are there exceptions to non-compete, exclusivity, or no-solicit provisions?",
        "description": "Carveouts to competitive restrictions",
        "keywords": ["except", "exception", "excluding", "carveout", "shall not apply"],
        "typical_location": "After restrictive covenants",
        "examples": ["except for passive investments", "shall not apply to products developed independently"]
    },
    "NO_SOLICIT_OF_EMPLOYEES": {
        "question": "Are there restrictions on soliciting or hiring employees?",
        "description": "Prohibition on employee solicitation/hiring",
        "keywords": ["solicit", "hire", "employ", "recruit", "personnel"],
        "typical_location": "Restrictive covenants",
        "examples": ["shall not hire any employee", "agrees not to solicit personnel"]
    },
    "NON_DISPARAGEMENT": {
        "question": "Are there requirements not to disparage the other party?",
        "description": "Prohibition on negative statements",
        "keywords": ["disparage", "disparagement", "negative statements", "criticize"],
        "typical_location": "Restrictive covenants",
        "examples": ["shall not make any disparaging statements", "agrees not to publicly criticize"]
    },
    "TERMINATION_FOR_CONVENIENCE": {
        "question": "Can a party terminate this contract without cause?",
        "description": "Right to terminate without reason",
        "keywords": ["terminate for convenience", "terminate without cause", "any reason", "no reason"],
        "typical_location": "Termination section",
        "examples": ["may terminate for any reason upon 30 days notice", "either party may terminate without cause"]
    },
    "ROFR_ROFO_ROFN": {
        "question": "Is there a right of first refusal, first offer, or first negotiation?",
        "description": "Preferential rights to purchase/license",
        "keywords": ["first refusal", "first offer", "first negotiation", "right of first"],
        "typical_location": "Special rights section",
        "examples": ["right of first refusal", "right of first negotiation", "first opportunity to purchase"]
    },
    "CHANGE_OF_CONTROL": {
        "question": "Are there provisions triggered by a change of ownership or control?",
        "description": "Rights upon ownership change",
        "keywords": ["change of control", "merger", "acquisition", "voting stock", "substantially all assets"],
        "typical_location": "Assignment or termination section",
        "examples": ["upon a change of control", "in the event of a merger or acquisition"]
    },
    "ANTI_ASSIGNMENT": {
        "question": "Are there restrictions on assigning this contract to a third party?",
        "description": "Prohibition/consent requirement for assignment",
        "keywords": ["assign", "assignment", "transfer", "consent required"],
        "typical_location": "General provisions",
        "examples": ["shall not assign without prior written consent", "may not transfer this Agreement"]
    },
    "REVENUE_PROFIT_SHARING": {
        "question": "Is there revenue or profit sharing between parties?",
        "description": "Obligations to share revenue/profits",
        "keywords": ["revenue sharing", "profit sharing", "royalty", "percentage of sales"],
        "typical_location": "Compensation section",
        "examples": ["shall pay 10% of net revenues", "profit sharing arrangement", "royalty of 5%"]
    },
    "PRICE_RESTRICTIONS": {
        "question": "Are there restrictions on price changes?",
        "description": "Limitations on pricing",
        "keywords": ["price", "pricing", "increase", "adjustment", "cap"],
        "typical_location": "Pricing section",
        "examples": ["prices shall not increase by more than 3% annually", "shall maintain current pricing"]
    },
    "MINIMUM_COMMITMENT": {
        "question": "Is there a minimum purchase, order, or commitment requirement?",
        "description": "Minimum quantity/value obligations",
        "keywords": ["minimum", "at least", "not less than", "commitment"],
        "typical_location": "Commercial terms",
        "examples": ["minimum purchase of $100,000 per year", "shall order at least 1,000 units"]
    },
    "VOLUME_RESTRICTION": {
        "question": "Are there fees or restrictions if usage exceeds certain thresholds?",
        "description": "Volume-based restrictions or fees",
        "keywords": ["volume", "threshold", "exceed", "additional fees", "overage"],
        "typical_location": "Pricing or license terms",
        "examples": ["if usage exceeds 10,000 users", "additional fees for volume above threshold"]
    },
    "IP_OWNERSHIP_ASSIGNMENT": {
        "question": "Is intellectual property assigned or transferred between parties?",
        "description": "IP ownership transfer",
        "keywords": ["assign", "transfer", "ownership", "intellectual property", "all rights"],
        "typical_location": "IP section",
        "examples": ["all IP shall be owned by Company", "assigns all rights, title and interest"]
    },
    "JOINT_IP_OWNERSHIP": {
        "question": "Is there joint ownership of intellectual property?",
        "description": "Shared IP ownership",
        "keywords": ["joint", "jointly owned", "shared ownership", "co-owned"],
        "typical_location": "IP section",
        "examples": ["jointly owned intellectual property", "shared ownership of developments"]
    },
    "LICENSE_GRANT": {
        "question": "Is a license granted from one party to another?",
        "description": "License rights granted",
        "keywords": ["grant", "license", "right to use", "permission"],
        "typical_location": "License grant section",
        "examples": ["hereby grants a license", "non-exclusive license to use"]
    },
    "NON_TRANSFERABLE_LICENSE": {
        "question": "Is the license non-transferable or restricted from sublicensing?",
        "description": "Restrictions on license transfer",
        "keywords": ["non-transferable", "may not sublicense", "personal", "non-assignable"],
        "typical_location": "License restrictions",
        "examples": ["non-transferable license", "may not sublicense"]
    },
    "AFFILIATE_LICENSE_LICENSOR": {
        "question": "Does the license include IP from licensor's affiliates?",
        "description": "Licensor affiliate rights",
        "keywords": ["licensor", "affiliates", "parent", "subsidiary"],
        "typical_location": "License grant",
        "examples": ["Licensor and its Affiliates grant", "including IP owned by Affiliates"]
    },
    "AFFILIATE_LICENSE_LICENSEE": {
        "question": "Does the license extend to licensee's affiliates?",
        "description": "Licensee affiliate rights",
        "keywords": ["licensee", "affiliates", "subsidiaries"],
        "typical_location": "License grant",
        "examples": ["Licensee and its Affiliates may use", "extends to Affiliates of Licensee"]
    },
    "UNLIMITED_ALL_YOU_CAN_EAT_LICENSE": {
        "question": "Is there an unlimited or all-you-can-eat license?",
        "description": "Unlimited usage rights",
        "keywords": ["unlimited", "enterprise", "all you can eat", "no restrictions"],
        "typical_location": "License scope",
        "examples": ["unlimited license", "enterprise-wide license", "no restrictions on usage"]
    },
    "IRREVOCABLE_OR_PERPETUAL_LICENSE": {
        "question": "Is the license irrevocable or perpetual?",
        "description": "Permanent license rights",
        "keywords": ["irrevocable", "perpetual", "survive termination", "permanent"],
        "typical_location": "License term",
        "examples": ["irrevocable license", "perpetual license", "license shall survive termination"]
    },
    "SOURCE_CODE_ESCROW": {
        "question": "Is there a source code escrow arrangement?",
        "description": "Code escrow provisions",
        "keywords": ["escrow", "source code", "escrow agent", "release conditions"],
        "typical_location": "Special provisions",
        "examples": ["source code escrow agreement", "shall deposit source code with escrow agent"]
    },
    "POST_TERMINATION_SERVICES": {
        "question": "Are there obligations after contract termination?",
        "description": "Post-termination requirements",
        "keywords": ["post-termination", "after termination", "wind-down", "transition"],
        "typical_location": "Termination effects",
        "examples": ["transition services for 90 days", "wind-down period"]
    },
    "AUDIT_RIGHTS": {
        "question": "Are there rights to audit records or facilities?",
        "description": "Audit provisions",
        "keywords": ["audit", "inspect", "examine", "books and records"],
        "typical_location": "Compliance section",
        "examples": ["right to audit books and records", "may inspect facilities"]
    },
    "UNCAPPED_LIABILITY": {
        "question": "Is liability uncapped for any type of breach?",
        "description": "Unlimited liability provisions",
        "keywords": ["uncapped", "unlimited", "not limited", "no cap"],
        "typical_location": "Limitation of liability",
        "examples": ["liability shall not be limited for", "unlimited liability for breach of confidentiality"]
    },
    "CAP_ON_LIABILITY": {
        "question": "Is there a cap or limit on liability?",
        "description": "Liability limitations",
        "keywords": ["cap", "limit", "not exceed", "maximum"],
        "typical_location": "Limitation of liability",
        "examples": ["liability shall not exceed $1,000,000", "limited to fees paid in prior 12 months"]
    },
    "LIQUIDATED_DAMAGES": {
        "question": "Are there liquidated damages or termination fees?",
        "description": "Pre-determined damages",
        "keywords": ["liquidated damages", "termination fee", "penalty", "predetermined"],
        "typical_location": "Remedies section",
        "examples": ["liquidated damages of $10,000", "termination fee"]
    },
    "WARRANTY_DURATION": {
        "question": "What is the duration of any warranty?",
        "description": "Warranty period",
        "keywords": ["warranty", "warranted", "warranty period", "defect"],
        "typical_location": "Warranty section",
        "examples": ["12-month warranty period", "warranty for one year from delivery"]
    },
    "INSURANCE": {
        "question": "Are there insurance requirements?",
        "description": "Required insurance coverage",
        "keywords": ["insurance", "coverage", "policy", "insure"],
        "typical_location": "Insurance section",
        "examples": ["shall maintain insurance coverage of at least $1,000,000", "proof of insurance required"]
    },
    "COVENANT_NOT_TO_SUE": {
        "question": "Is there a covenant not to sue or challenge IP validity?",
        "description": "Litigation restriction",
        "keywords": ["covenant not to sue", "shall not challenge", "not bring claim"],
        "typical_location": "IP or general provisions",
        "examples": ["covenants not to sue", "shall not challenge the validity of patents"]
    },
    "THIRD_PARTY_BENEFICIARY": {
        "question": "Are there third party beneficiary rights?",
        "description": "Rights for non-contracting parties",
        "keywords": ["third party", "beneficiary", "enforce"],
        "typical_location": "General provisions",
        "examples": ["third party beneficiary rights", "Affiliates shall be third party beneficiaries"]
    },
}

# List of all clause types
CUAD_ALL_TYPES = list(CUAD_CLAUSE_TYPES_ENHANCED.keys())


# =============================================================================
# SEMANTIC GROUPS FOR EFFICIENT BATCH EXTRACTION
# =============================================================================

CLAUSE_GROUPS = {
    "basic_info": ["DOCUMENT_NAME", "PARTIES", "AGREEMENT_DATE", "EFFECTIVE_DATE"],
    "term_termination": [
        "EXPIRATION_DATE", "RENEWAL_TERM", "NOTICE_PERIOD_TO_TERMINATE_RENEWAL",
        "TERMINATION_FOR_CONVENIENCE", "POST_TERMINATION_SERVICES"
    ],
    "governance": ["GOVERNING_LAW", "ANTI_ASSIGNMENT", "CHANGE_OF_CONTROL"],
    "competition": [
        "NON_COMPETE", "EXCLUSIVITY", "NO_SOLICIT_OF_CUSTOMERS",
        "COMPETITIVE_RESTRICTION_EXCEPTION", "NO_SOLICIT_OF_EMPLOYEES", "NON_DISPARAGEMENT"
    ],
    "commercial": [
        "MOST_FAVORED_NATION", "ROFR_ROFO_ROFN", "REVENUE_PROFIT_SHARING",
        "PRICE_RESTRICTIONS", "MINIMUM_COMMITMENT", "VOLUME_RESTRICTION"
    ],
    "ip": [
        "IP_OWNERSHIP_ASSIGNMENT", "JOINT_IP_OWNERSHIP", "LICENSE_GRANT",
        "NON_TRANSFERABLE_LICENSE", "AFFILIATE_LICENSE_LICENSOR", "AFFILIATE_LICENSE_LICENSEE",
        "UNLIMITED_ALL_YOU_CAN_EAT_LICENSE", "IRREVOCABLE_OR_PERPETUAL_LICENSE", "SOURCE_CODE_ESCROW"
    ],
    "liability": [
        "UNCAPPED_LIABILITY", "CAP_ON_LIABILITY", "LIQUIDATED_DAMAGES",
        "WARRANTY_DURATION", "INSURANCE", "COVENANT_NOT_TO_SUE", "AUDIT_RIGHTS"
    ],
    "other": ["THIRD_PARTY_BENEFICIARY"]
}


# =============================================================================
# Q&A FORMAT PROMPT (ContractEval style - one type at a time)
# =============================================================================

def create_qa_extraction_prompt(
    text: str,
    clause_type: str,
    include_cot: bool = True,
    max_text_len: int = 100000
) -> str:
    """
    Create a Question-Answering style prompt for a single clause type.
    This is the format used by ContractEval that achieves F1=64.1%.

    Args:
        text: Contract text
        clause_type: The clause type to extract
        include_cot: Whether to include Chain-of-Thought reasoning
        max_text_len: Maximum text length

    Returns:
        Prompt string
    """
    if len(text) > max_text_len:
        text = text[:max_text_len] + "\n[...TEXT TRUNCATED...]"

    clause_info = CUAD_CLAUSE_TYPES_ENHANCED.get(clause_type, {})
    question = clause_info.get("question", f"Are there clauses related to {clause_type}?")
    description = clause_info.get("description", "")
    keywords = clause_info.get("keywords", [])
    examples = clause_info.get("examples", [])
    typical_location = clause_info.get("typical_location", "")

    examples_str = "\n".join([f'  - "{ex}"' for ex in examples[:3]])

    if include_cot:
        cot_section = f"""
## REASONING STEPS
Before extracting, consider:
1. What section of the contract would typically contain {clause_type}? ({typical_location})
2. What keywords indicate this clause type? ({", ".join(keywords[:5])})
3. Is the text you found actually about {description.lower()}?
4. Is it the complete clause or just a fragment?
"""
    else:
        cot_section = ""

    prompt = f"""You are a legal assistant specialized in contract analysis.

## QUESTION
{question}

## CLAUSE TYPE: {clause_type}
{description}
{cot_section}
## EXAMPLES
{examples_str}

## INSTRUCTIONS
1. Read the contract carefully
2. Find ALL sentences that relate to "{clause_type}"
3. Extract the EXACT text - do not paraphrase
4. If multiple clauses exist, extract all of them
5. If no relevant clause exists, return empty list

## CONTRACT
```
{text}
```

## RESPONSE (JSON only)
```json
{{"clauses": ["exact text 1", "exact text 2"]}}
```
"""
    return prompt


# =============================================================================
# GROUPED EXTRACTION PROMPT (more efficient than one-at-a-time)
# =============================================================================

def create_grouped_extraction_prompt(
    text: str,
    group: str,
    include_cot: bool = True,
    max_text_len: int = 80000
) -> str:
    """
    Create a prompt for extracting a group of related clause types.

    Args:
        text: Contract text
        group: Group name from CLAUSE_GROUPS
        include_cot: Whether to include Chain-of-Thought
        max_text_len: Maximum text length

    Returns:
        Prompt string
    """
    if group not in CLAUSE_GROUPS:
        raise ValueError(f"Unknown group: {group}. Available: {list(CLAUSE_GROUPS.keys())}")

    if len(text) > max_text_len:
        text = text[:max_text_len] + "\n[...TEXT TRUNCATED...]"

    clause_types = CLAUSE_GROUPS[group]

    # Build clause descriptions
    clause_section = ""
    for ct in clause_types:
        info = CUAD_CLAUSE_TYPES_ENHANCED.get(ct, {})
        question = info.get("question", "")
        description = info.get("description", "")
        examples = info.get("examples", [])[:2]
        examples_str = ", ".join([f'"{e}"' for e in examples])

        clause_section += f"""
### {ct}
Question: {question}
Description: {description}
Examples: {examples_str}
"""

    if include_cot:
        cot_section = f"""
## REASONING APPROACH
For the {group.upper()} clause types:
1. Identify relevant sections of the contract
2. Look for keywords and patterns
3. Extract complete clause text
4. Verify extraction matches description
"""
    else:
        cot_section = ""

    prompt = f"""You are a legal assistant. Extract the following clause types.

## CLAUSE TYPES ({group.upper()} GROUP)
{clause_section}
{cot_section}
## INSTRUCTIONS
1. For each type, find ALL relevant text
2. Extract EXACT text - do not paraphrase
3. If type not present, use empty list
4. Include ALL occurrences

## CONTRACT
```
{text}
```

## RESPONSE (JSON only)
```json
{{
  "entities": [
    {{"text": "exact clause text", "type": "CLAUSE_TYPE", "confidence": 0.95}}
  ]
}}
```
"""
    return prompt


# =============================================================================
# FULL EXTRACTION WITH CoT (Single Pass for all 41 types)
# =============================================================================

def create_cot_full_extraction_prompt(text: str, max_text_len: int = 60000) -> str:
    """
    Create a Chain-of-Thought prompt for extracting all 41 clause types in one pass.

    Args:
        text: Contract text
        max_text_len: Maximum text length

    Returns:
        Prompt string
    """
    if len(text) > max_text_len:
        text = text[:max_text_len] + "\n[...TEXT TRUNCATED...]"

    # Build condensed type descriptions
    type_list = []
    for ct, info in CUAD_CLAUSE_TYPES_ENHANCED.items():
        desc = info.get("description", "")
        type_list.append(f"- {ct}: {desc}")

    types_str = "\n".join(type_list)

    prompt = f"""You are an expert legal analyst. Extract all clause types from this contract.

## CLAUSE TYPES (41 total)
{types_str}

## CHAIN-OF-THOUGHT APPROACH

### Step 1: Document Overview
Identify: Contract type, main parties, key dates

### Step 2: Section-by-Section Analysis
- Preamble: PARTIES, AGREEMENT_DATE, EFFECTIVE_DATE, DOCUMENT_NAME
- Definitions: Key terms
- Main Terms: LICENSE_GRANT, EXCLUSIVITY, MINIMUM_COMMITMENT
- Term/Termination: EXPIRATION_DATE, RENEWAL_TERM, TERMINATION_FOR_CONVENIENCE
- IP Rights: IP_OWNERSHIP_ASSIGNMENT, LICENSE_GRANT, SOURCE_CODE_ESCROW
- Liability: CAP_ON_LIABILITY, UNCAPPED_LIABILITY, LIQUIDATED_DAMAGES
- General: GOVERNING_LAW, ANTI_ASSIGNMENT, AUDIT_RIGHTS

### Step 3: Extraction
For each clause found:
1. Copy EXACT text
2. Verify it matches definition
3. Assign confidence (0.8-0.99)

## CONTRACT
```
{text}
```

## RESPONSE (JSON only)
```json
{{
  "entities": [
    {{"text": "exact text", "type": "CLAUSE_TYPE", "confidence": 0.95}}
  ]
}}
```

Important:
- Extract EXACT text, not summaries
- Include ALL occurrences
- Only confident extractions (>0.8)
"""
    return prompt


# =============================================================================
# JSON SCHEMA FOR STRUCTURED OUTPUT (OpenAI)
# =============================================================================

EXTRACTION_JSON_SCHEMA = {
    "name": "clause_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Exact text from document"},
                        "type": {"type": "string", "enum": CUAD_ALL_TYPES},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["text", "type", "confidence"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["entities"],
        "additionalProperties": False
    }
}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CUAD_CLAUSE_TYPES_ENHANCED',
    'CUAD_ALL_TYPES',
    'CLAUSE_GROUPS',
    'EXTRACTION_JSON_SCHEMA',
    'create_qa_extraction_prompt',
    'create_grouped_extraction_prompt',
    'create_cot_full_extraction_prompt',
]
