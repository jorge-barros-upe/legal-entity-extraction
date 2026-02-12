"""
Optimized Prompts for Entity Extraction in CUAD (English Legal Contracts).

This module contains carefully crafted prompts designed to improve F1-Score
for CUAD dataset extraction, particularly for PARTY entities which have
the lowest baseline performance (F1=0.457).

Target: Match or exceed baseline F1=0.62 with improved PARTY extraction.
"""

from typing import List, Dict, Any


# =============================================================================
# CUAD ENTITY TYPES
# =============================================================================

CUAD_ENTITY_TYPES = [
    "PARTY",      # Company names, person names, organizations party to contract
    "DOC_NAME",   # Document/agreement title (e.g., "LICENSE AGREEMENT")
    "AGMT_DATE",  # Agreement dates, effective dates
]


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT_EXPERT = """You are an expert legal analyst specializing in commercial contract review with 20 years of experience in:
- Mergers and acquisitions agreements
- License and distribution agreements
- Service and consulting contracts
- Joint venture and partnership agreements

FUNDAMENTAL PRINCIPLES:
1. ACCURACY > QUANTITY: Better to extract fewer entities with high confidence than many with low confidence
2. FIDELITY TO TEXT: Extract EXACTLY as it appears in the document, preserving original formatting
3. CONTEXT VALIDATION: Verify each entity makes sense in the legal context
4. CAUTION: If uncertain, DO NOT extract - mark as not found"""


# =============================================================================
# ENTITY DESCRIPTIONS - Detailed for better extraction
# =============================================================================

CUAD_ENTITY_DESCRIPTIONS = """
ENTITIES TO EXTRACT (exactly these 3 types):

1. PARTY: Companies, persons, or organizations that are parties to the contract
   - Include: Contracting parties mentioned in the preamble/recitals
   - Include: Parties defined as "Company", "Licensee", "Licensor", "Vendor", "Client", etc.
   - Include: Parent companies, subsidiaries, affiliates when they are contracting parties
   - Include: Individuals who are signatories or principals
   - EXCLUDE: Attorneys, witnesses, notaries (unless they are parties)
   - EXCLUDE: Third-party references that are not contracting parties
   - EXCLUDE: Generic terms like "the parties" without specific names

   CRITICAL - EXTRACT ALL NAME VARIATIONS:
   - If a party has BOTH a full name AND an abbreviation, extract BOTH separately
   - Example: "I-ESCROW, INC." AND "i-Escrow" → extract both as separate PARTY entities
   - Example: "ADAMS GOLF, LTD." AND "ADAMS GOLF" → extract both as separate entities
   - Extract defined terms when they refer to a party (e.g., "CONSULTANT", "Licensor")
   - Each variation counts as a separate entity

   Examples:
   - "ABC Corporation" (company - full name)
   - "ABC" (short form - ALSO extract separately)
   - "Licensor" (defined term referring to party - ALSO extract)
   - "John Smith" (individual party)
   - "XYZ Holdings, LLC" (company with entity type)

2. DOC_NAME: The title or name of the document/agreement
   - Look for: Text in ALL CAPS at the beginning
   - Look for: "This [TYPE] AGREEMENT", "AGREEMENT FOR [PURPOSE]"
   - Look for: Headers identifying the contract type
   - Common types: LICENSE AGREEMENT, CONSULTING AGREEMENT, PURCHASE AGREEMENT,
     SERVICE AGREEMENT, DISTRIBUTION AGREEMENT, EMPLOYMENT AGREEMENT

   Examples:
   - "LICENSE AGREEMENT"
   - "ASSET PURCHASE AGREEMENT"
   - "SOFTWARE LICENSE AND SERVICES AGREEMENT"

3. AGMT_DATE: Agreement date, effective date, or execution date
   - Look for: "dated as of [DATE]", "effective [DATE]", "as of [DATE]"
   - Look for: Date in the preamble/introduction
   - Formats: "January 1, 2020", "1/1/2020", "01-01-2020", "1st day of January, 2020"
   - EXCLUDE: Amendment dates (unless it's the date of the amendment itself)
   - EXCLUDE: Termination dates, deadline dates

   Examples:
   - "January 15, 2019"
   - "November 9, 2005"
   - "the 21st day of June, 1999"
"""


# =============================================================================
# FEW-SHOT EXAMPLES - Curated for CUAD patterns
# =============================================================================

FEW_SHOT_EXAMPLES = [
    # Example 1: Standard agreement with multiple parties AND their abbreviations
    {
        "input": """LICENSE AGREEMENT

This License Agreement (this "Agreement") is made and entered into as of November 15, 2018 by and between ABC Software Inc., a Delaware corporation ("Licensor"), and XYZ Enterprises, LLC, a California limited liability company ("Licensee").

RECITALS

WHEREAS, Licensor owns certain proprietary software; and
WHEREAS, Licensee desires to obtain a license to use such software...""",

        "output": {
            "entities": [
                {"text": "LICENSE AGREEMENT", "type": "DOC_NAME", "confidence": 0.99},
                {"text": "License Agreement", "type": "DOC_NAME", "confidence": 0.98},
                {"text": "November 15, 2018", "type": "AGMT_DATE", "confidence": 0.98},
                {"text": "ABC Software Inc.", "type": "PARTY", "confidence": 0.98},
                {"text": "Licensor", "type": "PARTY", "confidence": 0.95},
                {"text": "XYZ Enterprises, LLC", "type": "PARTY", "confidence": 0.98},
                {"text": "Licensee", "type": "PARTY", "confidence": 0.95}
            ],
            "reasoning": "Extract BOTH the full company name AND the defined term (Licensor/Licensee). Also extract both DOC_NAME variations."
        }
    },

    # Example 2: Agreement showing all name variations
    {
        "input": """CONSULTING AGREEMENT

This Consulting Agreement ("Agreement") is effective as of March 1, 2020 (the "Effective Date") by and between Global Tech Solutions, Inc. ("Company") and Dr. Sarah Johnson ("Consultant").

The Company desires to retain Consultant to provide advisory services...""",

        "output": {
            "entities": [
                {"text": "CONSULTING AGREEMENT", "type": "DOC_NAME", "confidence": 0.99},
                {"text": "Consulting Agreement", "type": "DOC_NAME", "confidence": 0.98},
                {"text": "March 1, 2020", "type": "AGMT_DATE", "confidence": 0.98},
                {"text": "Global Tech Solutions, Inc.", "type": "PARTY", "confidence": 0.98},
                {"text": "Company", "type": "PARTY", "confidence": 0.95},
                {"text": "Dr. Sarah Johnson", "type": "PARTY", "confidence": 0.98},
                {"text": "Consultant", "type": "PARTY", "confidence": 0.95}
            ],
            "reasoning": "Extract full names AND defined terms. 'Company' and 'Consultant' are defined as referring to parties."
        }
    },

    # Example 3: Complex with multiple variations
    {
        "input": """MASTER SERVICES AGREEMENT

Dated: June 21, 1999

Between:
(1) ACME Holdings PLC, a company incorporated in England ("Parent")
(2) ACME Services Limited, a wholly-owned subsidiary of Parent ("Service Provider")
(3) BigCorp International, Inc., a Nevada corporation ("Client")

This Master Services Agreement sets forth the terms...""",

        "output": {
            "entities": [
                {"text": "MASTER SERVICES AGREEMENT", "type": "DOC_NAME", "confidence": 0.99},
                {"text": "Master Services Agreement", "type": "DOC_NAME", "confidence": 0.98},
                {"text": "June 21, 1999", "type": "AGMT_DATE", "confidence": 0.98},
                {"text": "ACME Holdings PLC", "type": "PARTY", "confidence": 0.98},
                {"text": "Parent", "type": "PARTY", "confidence": 0.95},
                {"text": "ACME Services Limited", "type": "PARTY", "confidence": 0.98},
                {"text": "Service Provider", "type": "PARTY", "confidence": 0.95},
                {"text": "BigCorp International, Inc.", "type": "PARTY", "confidence": 0.98},
                {"text": "Client", "type": "PARTY", "confidence": 0.95}
            ],
            "reasoning": "Extract BOTH full company names AND their defined terms (Parent, Service Provider, Client)."
        }
    }
]


# =============================================================================
# VALIDATION RULES
# =============================================================================

VALIDATION_RULES = """
VALIDATION RULES (check BEFORE including each entity):

1. PARTY VALIDATION:
   - Must be a specific named entity (person or organization)
   - Must be a party TO THE CONTRACT (not just mentioned)
   - Check if defined in the preamble with a role (Licensor, Licensee, Company, etc.)
   - Include full legal name (e.g., "Inc.", "LLC", "Ltd.", "Corp.")
   - EXCLUDE: Law firms, witnesses, unless they are actual parties

2. DOC_NAME VALIDATION:
   - Usually appears at the very top of the document
   - Often in ALL CAPS or title case
   - Should describe the type of agreement
   - EXCLUDE: Section headings, exhibit titles

3. AGMT_DATE VALIDATION:
   - Must be the main agreement date or effective date
   - Usually in the first paragraph or preamble
   - EXCLUDE: Amendment dates, termination dates, notice periods
   - EXCLUDE: Birth dates, incorporation dates (unless relevant)
"""


# =============================================================================
# ANTI-HALLUCINATION INSTRUCTIONS
# =============================================================================

ANTI_HALLUCINATION = """
CRITICAL RULES TO AVOID ERRORS:

1. NEVER INVENT DATA
   - If a party's full name isn't clear, extract what IS clear
   - If no date is visible in the preamble, don't guess
   - Don't add "Inc." or "LLC" if not in the text

2. EXTRACT ALL VARIATIONS (important for PARTY):
   - If a party appears with full name AND abbreviation, extract BOTH
   - "ABC Corp." and "ABC" are DIFFERENT entities - extract both
   - Defined terms like "Licensor", "Company", "Consultant" ARE valid PARTY entities
   - DOC_NAME: if document title appears in different formats (ALL CAPS and Title Case), extract both

3. CONTEXT IS KEY
   - A company mentioned as "competitor" or "third party" is NOT a contract party
   - Only extract parties from the "between/among" clause or definitions
   - Verify the entity is actually signing/bound by this contract

4. WHEN IN DOUBT ABOUT FORMAT, EXTRACT
   - Better to extract more variations than miss them
   - If the same party has multiple mentions, extract each unique text string

5. USE EXACT TYPES
   - Use exactly: "PARTY", "DOC_NAME", "AGMT_DATE"
   - Don't create new types or variations
"""


# =============================================================================
# MAIN EXTRACTION PROMPT
# =============================================================================

def create_cuad_extraction_prompt(text: str) -> str:
    """
    Create optimized prompt for CUAD entity extraction.

    Args:
        text: Contract text to extract from

    Returns:
        Complete prompt string
    """
    # Build examples section (use 2 examples for balance)
    examples_section = "\n\n## EXTRACTION EXAMPLES\n"
    for i, ex in enumerate(FEW_SHOT_EXAMPLES[:2]):
        examples_section += f"""
### Example {i+1}:
**Input:**
```
{ex['input'][:400]}...
```

**Correct Output:**
```json
{{"entities": {ex['output']['entities']}}}
```

**Reasoning:** {ex['output']['reasoning']}
"""

    return f"""{SYSTEM_PROMPT_EXPERT}

## TASK
Extract entities from the commercial contract below. Return ONLY valid JSON.

{CUAD_ENTITY_DESCRIPTIONS}

{VALIDATION_RULES}

{ANTI_HALLUCINATION}
{examples_section}

## CONTRACT TO ANALYZE
```
{text}
```

## RESPONSE FORMAT
Respond ONLY with valid JSON, no additional text:
```json
{{
  "entities": [
    {{"text": "exact_value_from_document", "type": "PARTY|DOC_NAME|AGMT_DATE", "confidence": 0.95}}
  ]
}}
```

CRITICAL INSTRUCTIONS:
- Extract ALL variations of each party (full name AND abbreviation/defined term)
- If "ABC Corp." is defined as "Company", extract BOTH "ABC Corp." AND "Company" as separate PARTY entities
- Document names may appear in ALL CAPS and Title Case - extract both variations
- The JSON may have 10+ entities - this is expected for contracts with multiple parties
- Confidence should be between 0.85 and 0.99
"""


# =============================================================================
# SELF-CONSISTENCY PROMPTS
# =============================================================================

def create_cuad_self_consistency_prompt(text: str, entity_types: List[str], variation: int) -> str:
    """
    Create prompt variation for self-consistency voting on CUAD.

    Args:
        text: Contract text
        entity_types: Entity types to extract
        variation: Variation number (0, 1, 2)

    Returns:
        Prompt string
    """
    validation_section = """
VALIDATION:
- PARTY: Named companies/persons in the "between" clause or defined as parties
- DOC_NAME: Agreement title, usually in ALL CAPS at document start
- AGMT_DATE: Main agreement date in preamble ("as of", "dated", "effective")

CRITICAL - EXTRACT ALL VARIATIONS:
- If a company has a full name AND an abbreviation (e.g., "ABC Corp." and "Company"), extract BOTH
- Defined terms like "Licensor", "Licensee", "Consultant", "Company" ARE valid PARTY entities
- Document names may appear twice (ALL CAPS and Title Case) - extract both
- Use exactly these types: PARTY, DOC_NAME, AGMT_DATE
- Extract text exactly as it appears in the document"""

    variations = [
        # Variation 0: Focus on precision
        f"""You are a meticulous legal auditor. Extract ONLY entities with 100% certainty.

RULE: When in doubt, DON'T extract. Precision matters more than recall.

{CUAD_ENTITY_DESCRIPTIONS}
{validation_section}

CONTRACT:
{text}

Respond in valid JSON:
{{"entities": [{{"text": "exact_value", "type": "PARTY|DOC_NAME|AGMT_DATE", "confidence": 0.95}}]}}""",

        # Variation 1: Focus on recall (especially for PARTY)
        f"""You are a thorough contract analyst. Identify ALL possible parties, document names, and dates.

SPECIAL ATTENTION TO PARTIES:
- Check the preamble for "by and between" or "among"
- Look for defined terms like "Company", "Client", "Vendor", "Licensor", "Licensee"
- Include parent companies, subsidiaries, and affiliates if they are parties
- Include individuals (founders, executives) if they are contracting parties

{CUAD_ENTITY_DESCRIPTIONS}
{validation_section}

CONTRACT:
{text}

Respond in valid JSON:
{{"entities": [{{"text": "exact_value", "type": "PARTY|DOC_NAME|AGMT_DATE", "confidence": 0.95}}]}}""",

        # Variation 2: Structured step-by-step
        f"""You are an expert in commercial contract analysis.

EXTRACTION PROCESS:
1. Find the DOCUMENT NAME (usually ALL CAPS at top)
2. Find the AGREEMENT DATE (in preamble, after "as of" or "dated")
3. Find ALL PARTIES (in "between/among" clause)
   - Look for company names with Inc., LLC, Corp., Ltd., PLC
   - Look for individual names with titles
   - Check for numbered party lists: (1), (2), (3)
4. Verify each extraction against the text

{CUAD_ENTITY_DESCRIPTIONS}
{validation_section}

CONTRACT:
{text}

Respond in valid JSON:
{{"entities": [{{"text": "exact_value", "type": "PARTY|DOC_NAME|AGMT_DATE", "confidence": 0.95}}]}}"""
    ]

    return variations[variation % len(variations)]


# =============================================================================
# COMPACT PROMPT (for rate limiting / token reduction)
# =============================================================================

def create_cuad_compact_prompt(text: str, max_text_len: int = 15000) -> str:
    """
    Create a compact prompt for CUAD extraction.

    Args:
        text: Contract text (will be truncated if too long)
        max_text_len: Maximum text length

    Returns:
        Compact prompt string
    """
    if len(text) > max_text_len:
        text = text[:max_text_len] + "\n[...TEXT TRUNCATED...]"

    return f"""Extract entities from this commercial contract. Types: PARTY (companies/persons in contract), DOC_NAME (agreement title), AGMT_DATE (agreement date)

RULES:
1. PARTY: Only from "between/among" clause or defined parties
2. DOC_NAME: Title at document start (often ALL CAPS)
3. AGMT_DATE: Date in preamble ("as of", "dated", "effective")
4. Extract text EXACTLY as written
5. Valid JSON only

CONTRACT:
{text}

JSON:
{{"entities":[{{"text":"value","type":"TYPE","confidence":0.95}}]}}"""


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'CUAD_ENTITY_TYPES',
    'SYSTEM_PROMPT_EXPERT',
    'CUAD_ENTITY_DESCRIPTIONS',
    'FEW_SHOT_EXAMPLES',
    'VALIDATION_RULES',
    'ANTI_HALLUCINATION',
    'create_cuad_extraction_prompt',
    'create_cuad_self_consistency_prompt',
    'create_cuad_compact_prompt',
]
