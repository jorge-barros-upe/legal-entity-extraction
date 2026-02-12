"""
Full CUAD Prompts for Entity Extraction (All 41 Clause Types).

This module contains prompts for extracting all 41 clause types from the
original CUAD benchmark, enabling valid comparison with published baselines.

Based on: Hendrycks et al., 2021 - CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review
"""

from typing import List, Dict, Any


# =============================================================================
# ALL 41 CUAD CLAUSE TYPES (from original benchmark)
# =============================================================================

CUAD_CLAUSE_TYPES = {
    "DOCUMENT_NAME": {
        "original_name": "Document Name",
        "description": "The name of the contract",
        "examples": ["LICENSE AGREEMENT", "ASSET PURCHASE AGREEMENT", "SERVICE AGREEMENT"],
    },
    "PARTIES": {
        "original_name": "Parties",
        "description": "The two or more parties who signed the contract",
        "examples": ["ABC Corporation", "John Smith", "XYZ Holdings, LLC"],
    },
    "AGREEMENT_DATE": {
        "original_name": "Agreement Date",
        "description": "The date of the contract",
        "examples": ["January 15, 2020", "as of March 1, 2019"],
    },
    "EFFECTIVE_DATE": {
        "original_name": "Effective Date",
        "description": "The date when the contract is effective",
        "examples": ["effective as of January 1, 2020", "shall become effective on the Closing Date"],
    },
    "EXPIRATION_DATE": {
        "original_name": "Expiration Date",
        "description": "On what date will the contract's initial term expire?",
        "examples": ["December 31, 2025", "three years from the Effective Date"],
    },
    "RENEWAL_TERM": {
        "original_name": "Renewal Term",
        "description": "What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice.",
        "examples": ["automatically renew for successive one-year terms", "may be extended for additional 12-month periods"],
    },
    "NOTICE_PERIOD_TO_TERMINATE_RENEWAL": {
        "original_name": "Notice Period To Terminate Renewal",
        "description": "What is the notice period required to terminate renewal?",
        "examples": ["30 days prior written notice", "90 days before expiration"],
    },
    "GOVERNING_LAW": {
        "original_name": "Governing Law",
        "description": "Which state/country's law governs the interpretation of the contract?",
        "examples": ["governed by the laws of the State of Delaware", "New York law shall govern"],
    },
    "MOST_FAVORED_NATION": {
        "original_name": "Most Favored Nation",
        "description": "Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?",
        "examples": ["most favored customer pricing", "shall receive terms no less favorable"],
    },
    "NON_COMPETE": {
        "original_name": "Non-Compete",
        "description": "Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector?",
        "examples": ["shall not compete in the Territory", "agrees not to engage in any competing business"],
    },
    "EXCLUSIVITY": {
        "original_name": "Exclusivity",
        "description": "Is there an exclusive dealing commitment with the counterparty? This includes a commitment to procure all requirements from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on collaborating or working with other parties, whether during the contract or after the contract ends (or both).",
        "examples": ["exclusive license", "sole and exclusive right", "shall not license to any third party"],
    },
    "NO_SOLICIT_OF_CUSTOMERS": {
        "original_name": "No-Solicit Of Customers",
        "description": "Is a party restricted from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both)?",
        "examples": ["shall not solicit any customer", "agrees not to contact clients of the Company"],
    },
    "COMPETITIVE_RESTRICTION_EXCEPTION": {
        "original_name": "Competitive Restriction Exception",
        "description": "This category includes the exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers above.",
        "examples": ["except for passive investments", "shall not apply to products developed independently"],
    },
    "NO_SOLICIT_OF_EMPLOYEES": {
        "original_name": "No-Solicit Of Employees",
        "description": "Is there a restriction on a party's soliciting or hiring employees and/or contractors from the counterparty, whether during the contract or after the contract ends (or both)?",
        "examples": ["shall not hire any employee", "agrees not to solicit personnel"],
    },
    "NON_DISPARAGEMENT": {
        "original_name": "Non-Disparagement",
        "description": "Is there a requirement on a party not to disparage the counterparty?",
        "examples": ["shall not make any disparaging statements", "agrees not to publicly criticize"],
    },
    "TERMINATION_FOR_CONVENIENCE": {
        "original_name": "Termination For Convenience",
        "description": "Can a party terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire)?",
        "examples": ["may terminate for any reason upon 30 days notice", "either party may terminate without cause"],
    },
    "ROFR_ROFO_ROFN": {
        "original_name": "Rofr/Rofo/Rofn",
        "description": "Is there a clause granting one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services?",
        "examples": ["right of first refusal", "right of first negotiation", "first opportunity to purchase"],
    },
    "CHANGE_OF_CONTROL": {
        "original_name": "Change Of Control",
        "description": "Does one party have the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?",
        "examples": ["upon a change of control", "in the event of a merger or acquisition", "if more than 50% of voting stock is transferred"],
    },
    "ANTI_ASSIGNMENT": {
        "original_name": "Anti-Assignment",
        "description": "Is consent or notice required of a party if the contract is assigned to a third party?",
        "examples": ["shall not assign without prior written consent", "may not transfer this Agreement"],
    },
    "REVENUE_PROFIT_SHARING": {
        "original_name": "Revenue/Profit Sharing",
        "description": "Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?",
        "examples": ["shall pay 10% of net revenues", "profit sharing arrangement", "royalty of 5%"],
    },
    "PRICE_RESTRICTIONS": {
        "original_name": "Price Restrictions",
        "description": "Is there a restriction on the ability of a party to raise or reduce prices of technology, goods, or services provided?",
        "examples": ["prices shall not increase by more than 3% annually", "shall maintain current pricing"],
    },
    "MINIMUM_COMMITMENT": {
        "original_name": "Minimum Commitment",
        "description": "Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?",
        "examples": ["minimum purchase of $100,000 per year", "shall order at least 1,000 units"],
    },
    "VOLUME_RESTRICTION": {
        "original_name": "Volume Restriction",
        "description": "Is there a fee increase or consent requirement, etc. if one party's use of the product/services exceeds certain threshold?",
        "examples": ["if usage exceeds 10,000 users", "additional fees for volume above threshold"],
    },
    "IP_OWNERSHIP_ASSIGNMENT": {
        "original_name": "Ip Ownership Assignment",
        "description": "Does intellectual property created by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?",
        "examples": ["all IP shall be owned by Company", "assigns all rights, title and interest"],
    },
    "JOINT_IP_OWNERSHIP": {
        "original_name": "Joint Ip Ownership",
        "description": "Is there any clause providing for joint or shared ownership of intellectual property between the parties to the contract?",
        "examples": ["jointly owned intellectual property", "shared ownership of developments"],
    },
    "LICENSE_GRANT": {
        "original_name": "License Grant",
        "description": "Does the contract contain a license granted by one party to its counterparty?",
        "examples": ["hereby grants a license", "non-exclusive license to use", "license to reproduce and distribute"],
    },
    "NON_TRANSFERABLE_LICENSE": {
        "original_name": "Non-Transferable License",
        "description": "Does the contract limit the ability of a party to transfer the license being granted to a third party?",
        "examples": ["non-transferable license", "may not sublicense", "personal and non-assignable"],
    },
    "AFFILIATE_LICENSE_LICENSOR": {
        "original_name": "Affiliate License-Licensor",
        "description": "Does the contract contain a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor?",
        "examples": ["Licensor and its Affiliates grant", "including IP owned by Affiliates"],
    },
    "AFFILIATE_LICENSE_LICENSEE": {
        "original_name": "Affiliate License-Licensee",
        "description": "Does the contract contain a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor?",
        "examples": ["Licensee and its Affiliates may use", "extends to Affiliates of Licensee"],
    },
    "UNLIMITED_ALL_YOU_CAN_EAT_LICENSE": {
        "original_name": "Unlimited/All-You-Can-Eat-License",
        "description": "Is there a clause granting one party an 'enterprise,' 'all you can eat' or unlimited usage license?",
        "examples": ["unlimited license", "enterprise-wide license", "no restrictions on usage"],
    },
    "IRREVOCABLE_OR_PERPETUAL_LICENSE": {
        "original_name": "Irrevocable Or Perpetual License",
        "description": "Does the contract contain a license grant that is irrevocable or perpetual?",
        "examples": ["irrevocable license", "perpetual license", "license shall survive termination"],
    },
    "SOURCE_CODE_ESCROW": {
        "original_name": "Source Code Escrow",
        "description": "Is one party required to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy, insolvency, etc.)?",
        "examples": ["source code escrow agreement", "shall deposit source code with escrow agent"],
    },
    "POST_TERMINATION_SERVICES": {
        "original_name": "Post-Termination Services",
        "description": "Is a party subject to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments?",
        "examples": ["transition services for 90 days", "wind-down period", "shall continue to provide support"],
    },
    "AUDIT_RIGHTS": {
        "original_name": "Audit Rights",
        "description": "Does a party have the right to audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?",
        "examples": ["right to audit books and records", "may inspect facilities", "audit rights upon reasonable notice"],
    },
    "UNCAPPED_LIABILITY": {
        "original_name": "Uncapped Liability",
        "description": "Is a party's liability uncapped upon the breach of its obligation in the contract? This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.",
        "examples": ["liability shall not be limited for", "unlimited liability for breach of confidentiality"],
    },
    "CAP_ON_LIABILITY": {
        "original_name": "Cap On Liability",
        "description": "Does the contract include a cap on liability upon the breach of a party's obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery.",
        "examples": ["liability shall not exceed $1,000,000", "limited to fees paid in prior 12 months"],
    },
    "LIQUIDATED_DAMAGES": {
        "original_name": "Liquidated Damages",
        "description": "Does the contract contain a clause that would award either party liquidated damages for breach or a fee upon the termination of a contract (termination fee)?",
        "examples": ["liquidated damages of $10,000", "termination fee", "penalty for early termination"],
    },
    "WARRANTY_DURATION": {
        "original_name": "Warranty Duration",
        "description": "What is the duration of any warranty against defects or errors in technology, products, or services provided under the contract?",
        "examples": ["12-month warranty period", "warranty for one year from delivery"],
    },
    "INSURANCE": {
        "original_name": "Insurance",
        "description": "Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty?",
        "examples": ["shall maintain insurance coverage of at least $1,000,000", "proof of insurance required"],
    },
    "COVENANT_NOT_TO_SUE": {
        "original_name": "Covenant Not To Sue",
        "description": "Is a party restricted from contesting the validity of the counterparty's ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract?",
        "examples": ["covenants not to sue", "shall not challenge the validity of patents"],
    },
    "THIRD_PARTY_BENEFICIARY": {
        "original_name": "Third Party Beneficiary",
        "description": "Is there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party?",
        "examples": ["third party beneficiary rights", "Affiliates shall be third party beneficiaries"],
    },
}

# List of all clause type keys for easy iteration
CUAD_ALL_TYPES = list(CUAD_CLAUSE_TYPES.keys())


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT_FULL = """You are an expert legal analyst specializing in commercial contract review with extensive experience in:
- Mergers and acquisitions agreements
- License and distribution agreements
- Service and consulting contracts
- Joint venture and partnership agreements
- Technology licensing and IP agreements

Your task is to extract specific contractual clauses from legal documents.

FUNDAMENTAL PRINCIPLES:
1. ACCURACY > QUANTITY: Better to extract fewer clauses with high confidence than many with low confidence
2. FIDELITY TO TEXT: Extract EXACTLY as it appears in the document, preserving original formatting
3. CONTEXT VALIDATION: Verify each extraction makes sense in the legal context
4. PRECISION: Extract the specific text span that contains the relevant clause, not the entire paragraph"""


# =============================================================================
# ENTITY DESCRIPTIONS (Grouped by category for better prompting)
# =============================================================================

def get_clause_descriptions() -> str:
    """Generate formatted clause descriptions for all 41 types."""

    # Group clauses by category for better organization
    groups = {
        "Basic Contract Info": ["DOCUMENT_NAME", "PARTIES", "AGREEMENT_DATE", "EFFECTIVE_DATE"],
        "Term and Termination": ["EXPIRATION_DATE", "RENEWAL_TERM", "NOTICE_PERIOD_TO_TERMINATE_RENEWAL", "TERMINATION_FOR_CONVENIENCE", "POST_TERMINATION_SERVICES"],
        "Governance": ["GOVERNING_LAW", "ANTI_ASSIGNMENT", "CHANGE_OF_CONTROL"],
        "Competition and Exclusivity": ["NON_COMPETE", "EXCLUSIVITY", "NO_SOLICIT_OF_CUSTOMERS", "COMPETITIVE_RESTRICTION_EXCEPTION", "NO_SOLICIT_OF_EMPLOYEES", "NON_DISPARAGEMENT"],
        "Commercial Terms": ["MOST_FAVORED_NATION", "ROFR_ROFO_ROFN", "REVENUE_PROFIT_SHARING", "PRICE_RESTRICTIONS", "MINIMUM_COMMITMENT", "VOLUME_RESTRICTION"],
        "Intellectual Property": ["IP_OWNERSHIP_ASSIGNMENT", "JOINT_IP_OWNERSHIP", "LICENSE_GRANT", "NON_TRANSFERABLE_LICENSE", "AFFILIATE_LICENSE_LICENSOR", "AFFILIATE_LICENSE_LICENSEE", "UNLIMITED_ALL_YOU_CAN_EAT_LICENSE", "IRREVOCABLE_OR_PERPETUAL_LICENSE", "SOURCE_CODE_ESCROW"],
        "Liability and Risk": ["UNCAPPED_LIABILITY", "CAP_ON_LIABILITY", "LIQUIDATED_DAMAGES", "WARRANTY_DURATION", "INSURANCE", "COVENANT_NOT_TO_SUE", "AUDIT_RIGHTS"],
        "Other": ["THIRD_PARTY_BENEFICIARY"],
    }

    desc = "CLAUSE TYPES TO EXTRACT (41 types):\n\n"

    for group_name, clause_keys in groups.items():
        desc += f"### {group_name}\n"
        for key in clause_keys:
            if key in CUAD_CLAUSE_TYPES:
                info = CUAD_CLAUSE_TYPES[key]
                examples = ", ".join([f'"{e}"' for e in info.get("examples", [])[:2]])
                desc += f"- **{key}**: {info['description']}"
                if examples:
                    desc += f" Examples: {examples}"
                desc += "\n"
        desc += "\n"

    return desc


# =============================================================================
# MAIN EXTRACTION PROMPT (Full 41 types)
# =============================================================================

def create_cuad_full_extraction_prompt(text: str, max_text_len: int = 50000) -> str:
    """
    Create prompt for extracting all 41 CUAD clause types.

    Args:
        text: Contract text to extract from
        max_text_len: Maximum text length (truncate if longer)

    Returns:
        Complete prompt string
    """
    if len(text) > max_text_len:
        text = text[:max_text_len] + "\n[...TEXT TRUNCATED...]"

    clause_desc = get_clause_descriptions()

    return f"""{SYSTEM_PROMPT_FULL}

## TASK
Extract all relevant clauses from the commercial contract below. Return ONLY valid JSON.

{clause_desc}

## EXTRACTION RULES

1. **Extract text spans**: For each clause found, extract the specific text that contains the clause
2. **One entry per occurrence**: If a clause type appears multiple times, create separate entries
3. **Exact text**: Copy the text exactly as it appears in the document
4. **High confidence only**: Only extract clauses you are confident about (confidence > 0.8)
5. **Type must match**: Use exactly the clause type names shown above (e.g., "PARTIES", "GOVERNING_LAW")

## CONTRACT TO ANALYZE
```
{text}
```

## RESPONSE FORMAT
Respond ONLY with valid JSON:
```json
{{
  "entities": [
    {{"text": "exact_text_from_document", "type": "CLAUSE_TYPE", "confidence": 0.95}},
    {{"text": "another_text_span", "type": "ANOTHER_TYPE", "confidence": 0.90}}
  ]
}}
```

Important:
- Use the exact clause type names (uppercase with underscores)
- Extract the specific text span, not summaries
- Include all occurrences of each clause type found
- Confidence should be between 0.80 and 0.99
- If no clauses of a type are found, simply don't include that type
"""


# =============================================================================
# FOCUSED EXTRACTION PROMPT (for specific clause types)
# =============================================================================

def create_cuad_focused_prompt(text: str, clause_types: List[str], max_text_len: int = 50000) -> str:
    """
    Create prompt focused on specific clause types.

    Args:
        text: Contract text
        clause_types: List of clause types to extract (e.g., ["PARTIES", "GOVERNING_LAW"])
        max_text_len: Maximum text length

    Returns:
        Prompt string
    """
    if len(text) > max_text_len:
        text = text[:max_text_len] + "\n[...TEXT TRUNCATED...]"

    # Build focused descriptions
    desc = "CLAUSE TYPES TO EXTRACT:\n\n"
    for ct in clause_types:
        if ct in CUAD_CLAUSE_TYPES:
            info = CUAD_CLAUSE_TYPES[ct]
            examples = ", ".join([f'"{e}"' for e in info.get("examples", [])])
            desc += f"**{ct}**: {info['description']}\n"
            if examples:
                desc += f"   Examples: {examples}\n"
            desc += "\n"

    return f"""{SYSTEM_PROMPT_FULL}

## TASK
Extract the following specific clause types from the contract. Return ONLY valid JSON.

{desc}

## CONTRACT
```
{text}
```

## RESPONSE
```json
{{
  "entities": [
    {{"text": "exact_text", "type": "CLAUSE_TYPE", "confidence": 0.95}}
  ]
}}
```
"""


# =============================================================================
# BATCH PROCESSING (for very long documents)
# =============================================================================

def create_cuad_batch_prompts(text: str, batch_size: int = 10) -> List[Dict[str, Any]]:
    """
    Create multiple prompts for batch processing of clause types.

    Useful for very long documents where extracting all 41 types at once
    may exceed context limits or reduce accuracy.

    Args:
        text: Contract text
        batch_size: Number of clause types per batch

    Returns:
        List of dicts with 'clause_types' and 'prompt'
    """
    batches = []
    all_types = list(CUAD_CLAUSE_TYPES.keys())

    for i in range(0, len(all_types), batch_size):
        batch_types = all_types[i:i+batch_size]
        prompt = create_cuad_focused_prompt(text, batch_types)
        batches.append({
            "clause_types": batch_types,
            "prompt": prompt,
            "batch_index": i // batch_size,
        })

    return batches


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'CUAD_CLAUSE_TYPES',
    'CUAD_ALL_TYPES',
    'SYSTEM_PROMPT_FULL',
    'get_clause_descriptions',
    'create_cuad_full_extraction_prompt',
    'create_cuad_focused_prompt',
    'create_cuad_batch_prompts',
]
