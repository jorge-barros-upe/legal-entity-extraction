"""
Optimized Prompts for Entity Extraction in Brazilian Legal Contracts.

This module contains carefully crafted prompts designed to improve F1-Score
from ~0.25 to target 0.60+ through:
1. Few-shot examples with edge cases
2. Chain-of-thought reasoning
3. Validation rules embedded in prompts
4. Anti-hallucination instructions
"""

from typing import List, Dict, Any


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT_EXPERT = """Você é um analista jurídico especializado em contratos societários brasileiros com 20 anos de experiência em:
- Constituição e alteração de sociedades limitadas
- Análise de contratos sociais
- Identificação de partes contratuais e seus dados

PRINCÍPIOS FUNDAMENTAIS:
1. PRECISÃO > QUANTIDADE: É melhor extrair menos entidades com alta confiança do que muitas com baixa confiança
2. FIDELIDADE AO TEXTO: Extraia EXATAMENTE como aparece no documento, sem normalização
3. VALIDAÇÃO: Verifique se cada entidade faz sentido no contexto jurídico
4. CAUTELA: Se não tiver certeza, NÃO extraia - marque como não encontrado"""


# =============================================================================
# FEW-SHOT EXAMPLES - Carefully curated for edge cases
# =============================================================================

FEW_SHOT_EXAMPLES = [
    # Example 1: Complete extraction with multiple sócios
    {
        "input": """ALTERAÇÃO CONTRATUAL
EMPRESA TECH SOLUTIONS LTDA
CNPJ: 12.345.678/0001-90
NIRE: 35.208.123.456

Os sócios abaixo qualificados:
1. JOÃO SILVA DOS SANTOS, brasileiro, casado, empresário, portador do RG nº 12.345.678-9 SSP/SP e CPF nº 123.456.789-00, residente na Rua das Flores, 100, apto 501, Jardim Paulista, CEP 01234-567, São Paulo/SP;

2. MARIA OLIVEIRA COSTA, brasileira, solteira, administradora, portadora do RG nº 98.765.432-1 SSP/SP e CPF nº 987.654.321-00, residente na Av. Paulista, 2000, sala 1501, Bela Vista, CEP 01310-100, São Paulo/SP;

Únicos sócios da sociedade empresária limitada denominada EMPRESA TECH SOLUTIONS LTDA, com sede na Rua Augusta, 500, 10º andar, Consolação, CEP 01304-000, São Paulo/SP, registrada na JUCESP sob NIRE 35.208.123.456 em 15/03/2020, RESOLVEM alterar o contrato social.""",

        "output": {
            "entities": [
                {"text": "12.345.678/0001-90", "type": "cnpj->sociedade", "confidence": 0.98},
                {"text": "JOÃO SILVA DOS SANTOS", "type": "nome->socio", "confidence": 0.98},
                {"text": "12.345.678-9", "type": "rg->socio", "confidence": 0.95},
                {"text": "123.456.789-00", "type": "cpf->socio", "confidence": 0.98},
                {"text": "01234-567", "type": "cep->socio", "confidence": 0.95},
                {"text": "São Paulo", "type": "municipio_(ou_cidade)->socio", "confidence": 0.95},
                {"text": "MARIA OLIVEIRA COSTA", "type": "nome->socio", "confidence": 0.98},
                {"text": "98.765.432-1", "type": "rg->socio", "confidence": 0.95},
                {"text": "987.654.321-00", "type": "cpf->socio", "confidence": 0.98},
                {"text": "01310-100", "type": "cep->socio", "confidence": 0.95},
                {"text": "São Paulo", "type": "municipio_(ou_cidade)->socio", "confidence": 0.95},
                {"text": "01304-000", "type": "cep->sociedade", "confidence": 0.95},
                {"text": "São Paulo", "type": "municipio_(ou_cidade)->sociedade", "confidence": 0.95},
                {"text": "15/03/2020", "type": "data_de_registro_do_contrato", "confidence": 0.90}
            ],
            "reasoning": "Identifiquei 2 sócios com seus dados completos e a sociedade com sede em SP. O NIRE não é tipo de entidade solicitado."
        },
        "notes": "Observe: Não extraí 'EMPRESA TECH SOLUTIONS LTDA' como nome->sociedade pois o tipo correto é 'nome_(ou_razao_social)->sociedade' que não foi solicitado. NIRE também não foi solicitado."
    },

    # Example 2: Incomplete data - demonstrating what NOT to extract
    {
        "input": """CONTRATO DE CONSTITUIÇÃO DE SOCIEDADE

PEDRO HENRIQUE FERREIRA, brasileiro, maior, empresário, CPF 111.222.333-44, e
LUCAS ANDRADE (dados pessoais constam em instrumento apartado), resolvem constituir
sociedade empresária com sede em Campinas/SP.""",

        "output": {
            "entities": [
                {"text": "PEDRO HENRIQUE FERREIRA", "type": "nome->socio", "confidence": 0.98},
                {"text": "111.222.333-44", "type": "cpf->socio", "confidence": 0.98},
                {"text": "LUCAS ANDRADE", "type": "nome->socio", "confidence": 0.95},
                {"text": "Campinas", "type": "municipio_(ou_cidade)->sociedade", "confidence": 0.90}
            ],
            "reasoning": "LUCAS ANDRADE tem dados em documento apartado - NÃO invento RG/CPF. O CEP da sociedade não consta no texto."
        },
        "notes": "IMPORTANTE: Quando dados estão ausentes ou em outro documento, NÃO os invente. Extraia apenas o que está explícito."
    },

    # Example 3: Similar entity types - disambiguation
    {
        "input": """A empresa XYZ COMÉRCIO LTDA, CNPJ 55.666.777/0001-88, com sede na
Rua do Comércio, 50, Centro, CEP 20040-020, Rio de Janeiro/RJ, representada por
seu administrador CARLOS EDUARDO LIMA, CPF 444.555.666-77, residente na
Av. Copacabana, 1000, apto 202, CEP 22070-000, Rio de Janeiro/RJ.""",

        "output": {
            "entities": [
                {"text": "55.666.777/0001-88", "type": "cnpj->sociedade", "confidence": 0.98},
                {"text": "20040-020", "type": "cep->sociedade", "confidence": 0.95},
                {"text": "Rio de Janeiro", "type": "municipio_(ou_cidade)->sociedade", "confidence": 0.95},
                {"text": "CARLOS EDUARDO LIMA", "type": "nome->socio", "confidence": 0.90},
                {"text": "444.555.666-77", "type": "cpf->socio", "confidence": 0.98},
                {"text": "22070-000", "type": "cep->socio", "confidence": 0.95},
                {"text": "Rio de Janeiro", "type": "municipio_(ou_cidade)->socio", "confidence": 0.95}
            ],
            "reasoning": "CARLOS é administrador/representante, considero como sócio para fins de extração. Diferenciei CEP da sociedade (sede) do CEP do sócio (residência)."
        },
        "notes": "A cidade 'Rio de Janeiro' aparece 2x - uma para sociedade (sede) e outra para sócio (residência). São entidades DISTINTAS."
    }
]


# =============================================================================
# VALIDATION RULES (embedded in prompt)
# =============================================================================

VALIDATION_RULES = """
REGRAS DE VALIDAÇÃO (verifique ANTES de incluir cada entidade):

1. CPF: Deve ter exatamente 11 dígitos (formato XXX.XXX.XXX-XX ou XXXXXXXXXXX)
   - Se tiver menos ou mais dígitos, NÃO extraia
   - Se for claramente um número de outro documento, NÃO extraia como CPF

2. CNPJ: Deve ter exatamente 14 dígitos (formato XX.XXX.XXX/XXXX-XX)
   - Filiais têm /0002-, /0003-, etc. - extraia corretamente

3. CEP: Deve ter 8 dígitos (formato XXXXX-XXX ou XXXXXXXX)
   - CEPs brasileiros começam de 01000 a 99999

4. RG: Varia por estado, geralmente 7-9 dígitos + letra opcional
   - Pode incluir órgão expedidor (SSP, DETRAN, etc.)
   - NÃO confunda com número de registro profissional (CREA, OAB, CRM)

5. NOMES DE SÓCIOS:
   - Deve ser nome completo de pessoa física (não apelidos ou abreviações)
   - NÃO extraia nomes de testemunhas, contadores ou advogados
   - Administradores/representantes que também são sócios DEVEM ser extraídos

6. DATAS:
   - Apenas data de registro/arquivamento do contrato
   - NÃO extraia datas de nascimento, validade de documentos, etc.

7. MUNICÍPIOS:
   - Diferencie município da SEDE (->sociedade) do município de RESIDÊNCIA (->socio)
   - Não confunda com naturalidade ou foro de eleição"""


# =============================================================================
# ANTI-HALLUCINATION INSTRUCTIONS
# =============================================================================

ANTI_HALLUCINATION = """
⚠️ REGRAS CRÍTICAS PARA EVITAR ERROS:

1. NUNCA INVENTE DADOS
   - Se um sócio não tem CPF no texto, NÃO crie um CPF
   - Se a sociedade não tem CEP explícito, NÃO extraia CEP

2. EVITE DUPLICATAS
   - Se o mesmo CPF/CNPJ aparece múltiplas vezes, extraia APENAS UMA VEZ
   - Exceção: Se são claramente entidades diferentes (ex: 2 sócios com CEPs diferentes)

3. CONTEXTO É FUNDAMENTAL
   - Um número pode parecer CPF mas ser RG, cartão CNPJ, etc.
   - Leia o contexto ao redor para confirmar o tipo

4. QUANDO EM DÚVIDA, NÃO EXTRAIA
   - É melhor ter recall menor que precision baixa
   - Se confiança < 80%, considere não incluir

5. TIPOS DE ENTIDADE EXATOS
   - Use EXATAMENTE os tipos solicitados (ex: "nome->socio", não "nome_socio")
   - Não crie novos tipos que não foram pedidos"""


# =============================================================================
# CHAIN-OF-THOUGHT TEMPLATE
# =============================================================================

COT_TEMPLATE = """
PROCESSO DE EXTRAÇÃO (siga mentalmente antes de responder):

PASSO 1 - IDENTIFICAR ESTRUTURA
□ Que tipo de documento é? (constituição, alteração, dissolução)
□ Quantos sócios são mencionados?
□ A sociedade tem sede definida?

PASSO 2 - LOCALIZAR CADA SÓCIO
Para cada sócio identificado:
□ Nome completo está presente?
□ CPF está presente e é válido (11 dígitos)?
□ RG está presente?
□ Endereço/CEP de residência?
□ Município de residência?

PASSO 3 - LOCALIZAR DADOS DA SOCIEDADE
□ CNPJ está presente e é válido (14 dígitos)?
□ Endereço da sede?
□ CEP da sede?
□ Município da sede?

PASSO 4 - DATAS E OUTROS
□ Data de registro/arquivamento?
□ Verificar se não confundi com outras datas

PASSO 5 - REVISÃO FINAL
□ Cada entidade tem o tipo correto?
□ Não há duplicatas?
□ Todos os CPFs/CNPJs têm formato válido?
□ Confiança é alta o suficiente?"""


# =============================================================================
# MAIN EXTRACTION PROMPT - FEW-SHOT OPTIMIZED
# =============================================================================

def create_optimized_prompt(
    text: str,
    entity_types: List[str],
    use_cot: bool = True,
    use_few_shot: bool = True,
    num_examples: int = 2
) -> str:
    """
    Create an optimized extraction prompt with few-shot examples.

    Args:
        text: Contract text to extract from
        entity_types: List of entity types to extract
        use_cot: Include chain-of-thought guidance
        use_few_shot: Include few-shot examples
        num_examples: Number of examples to include (1-3)

    Returns:
        Complete prompt string
    """
    # Build entity types section
    types_formatted = "\n".join([f"  • {t}" for t in entity_types])

    # Build examples section
    examples_section = ""
    if use_few_shot:
        examples_section = "\n\n## EXEMPLOS DE EXTRAÇÃO CORRETA\n"
        for i, ex in enumerate(FEW_SHOT_EXAMPLES[:num_examples]):
            examples_section += f"""
### Exemplo {i+1}:
**Entrada:**
```
{ex['input'][:500]}...
```

**Saída Correta:**
```json
{{"entities": {ex['output']['entities'][:5]}}}
```

**Raciocínio:** {ex['output']['reasoning']}
**Nota:** {ex['notes']}
"""

    # Build CoT section
    cot_section = ""
    if use_cot:
        cot_section = f"\n\n## PROCESSO DE ANÁLISE\n{COT_TEMPLATE}"

    # Main prompt
    prompt = f"""{SYSTEM_PROMPT_EXPERT}

## TAREFA
Extraia as seguintes entidades do contrato brasileiro abaixo. Retorne APENAS JSON válido.

## TIPOS DE ENTIDADES A EXTRAIR
{types_formatted}

{VALIDATION_RULES}

{ANTI_HALLUCINATION}
{examples_section}
{cot_section}

## CONTRATO PARA ANÁLISE
```
{text}
```

## FORMATO DE RESPOSTA
Responda APENAS com JSON válido, sem texto adicional:
```json
{{
  "entities": [
    {{"text": "valor_exato_do_documento", "type": "tipo_da_entidade", "confidence": 0.95}}
  ]
}}
```

IMPORTANTE:
- Se uma entidade não existir no documento, NÃO a inclua no JSON
- Mantenha o array "entities" vazio se nenhuma entidade for encontrada
- Cada entidade deve ter confidence entre 0.80 e 0.99
"""

    return prompt


# =============================================================================
# TWO-PHASE EXTRACTION PROMPTS
# =============================================================================

PHASE1_SECTION_IDENTIFICATION = """Você é um analista de documentos jurídicos. Sua tarefa é identificar as seções relevantes de um contrato brasileiro.

TAREFA: Analise o contrato abaixo e identifique:
1. Onde estão os DADOS DOS SÓCIOS (nome, CPF, RG, endereço)
2. Onde estão os DADOS DA SOCIEDADE (CNPJ, sede, razão social)
3. Onde está a DATA DE REGISTRO/ARQUIVAMENTO

Para cada seção encontrada, extraia o TRECHO EXATO do texto (máximo 500 caracteres por trecho).

CONTRATO:
{text}

RESPONDA EM JSON:
{{
  "sections": {{
    "socios": ["trecho 1 com dados do sócio 1...", "trecho 2 com dados do sócio 2..."],
    "sociedade": ["trecho com dados da empresa..."],
    "registro": ["trecho com data de registro..."]
  }},
  "num_socios_identificados": 2,
  "tem_dados_sociedade": true,
  "tem_data_registro": true
}}"""


PHASE2_EXTRACT_FROM_SECTION = """Extraia as entidades do trecho abaixo. Este trecho contém dados de {section_type}.

TIPOS ESPERADOS NESTA SEÇÃO:
{expected_types}

TRECHO:
{section_text}

RESPONDA APENAS EM JSON:
{{
  "entities": [
    {{"text": "valor", "type": "tipo", "confidence": 0.95}}
  ]
}}"""


# =============================================================================
# SELF-CONSISTENCY PROMPT (for voting)
# =============================================================================

def create_self_consistency_prompt(text: str, entity_types: List[str], variation: int) -> str:
    """
    Create prompt variation for self-consistency voting.

    Args:
        text: Contract text
        entity_types: Entity types to extract
        variation: Variation number (0, 1, 2) for different prompt styles

    Returns:
        Prompt string
    """
    # Check if using core entity types
    is_core = len(entity_types) <= 12 and all(t in CORE_ENTITY_TYPES for t in entity_types)

    if is_core:
        # Use detailed core descriptions for better accuracy
        types_section = CORE_ENTITY_DESCRIPTIONS
    else:
        types_section = "TIPOS A EXTRAIR:\n" + "\n".join([f"  • {t}" for t in entity_types])

    # Common validation rules for all variations
    validation_section = """
VALIDAÇÃO OBRIGATÓRIA:
- CPF: exatamente 11 dígitos (formato XXX.XXX.XXX-XX)
- CNPJ: exatamente 14 dígitos (formato XX.XXX.XXX/XXXX-XX)
- CEP: exatamente 8 dígitos (formato XXXXX-XXX)
- Nome: deve ter pelo menos 2 palavras

ATENÇÃO - MÚLTIPLOS SÓCIOS:
Este contrato pode ter MUITOS sócios. Extraia TODOS, não apenas os primeiros.

TIPOS DE ENTIDADE:
Use EXATAMENTE os tipos listados acima. Não invente tipos diferentes."""

    variations = [
        # Variation 0: Focus on precision
        f"""Você é um auditor jurídico meticuloso. Extraia APENAS entidades com 100% de certeza.

REGRA: Na dúvida, NÃO extraia. Precisão é mais importante que recall.

{types_section}
{validation_section}

CONTRATO:
{text}

RESPONDA em JSON válido:
{{"entities": [{{"text": "valor_exato", "type": "tipo_da_lista_acima", "confidence": 0.95}}]}}""",

        # Variation 1: Focus on recall
        f"""Você é um assistente de análise contratual. Identifique TODAS as possíveis entidades.

TAREFA: Extraia todas as entidades do contrato, especialmente TODOS os sócios.

{types_section}
{validation_section}

CONTRATO:
{text}

RESPONDA em JSON válido:
{{"entities": [{{"text": "valor_exato", "type": "tipo_da_lista_acima", "confidence": 0.95}}]}}""",

        # Variation 2: Balanced with step-by-step
        f"""Você é um especialista em contratos societários brasileiros.

PROCESSO:
1. Leia TODO o documento
2. Identifique TODOS os sócios (pode haver 10+)
3. Extraia os dados de cada sócio
4. Extraia os dados da sociedade
5. Valide cada entidade antes de incluir

{types_section}
{validation_section}

CONTRATO:
{text}

RESPONDA em JSON válido:
{{"entities": [{{"text": "valor_exato", "type": "tipo_da_lista_acima", "confidence": 0.95}}]}}"""
    ]

    return variations[variation % len(variations)]


# =============================================================================
# COMPACT PROMPT (for Gemini to avoid truncation)
# =============================================================================

def create_compact_prompt(text: str, entity_types: List[str], max_types: int = 10) -> str:
    """
    Create a compact prompt optimized for Gemini to avoid JSON truncation.

    Args:
        text: Contract text (will be truncated if too long)
        entity_types: Entity types (will limit to most important)
        max_types: Maximum number of entity types

    Returns:
        Compact prompt string
    """
    # Prioritize most common/important entity types
    priority_types = [
        "nome->socio", "cpf->socio", "rg->socio", "cep->socio",
        "municipio_(ou_cidade)->socio", "cnpj->sociedade",
        "cep->sociedade", "municipio_(ou_cidade)->sociedade",
        "data_de_registro_do_contrato"
    ]

    # Filter and limit types
    filtered_types = [t for t in priority_types if t in entity_types]
    other_types = [t for t in entity_types if t not in priority_types]
    final_types = (filtered_types + other_types)[:max_types]

    types_str = ", ".join(final_types)

    # Truncate text if needed
    max_text_len = 15000  # Keep text shorter for Gemini
    if len(text) > max_text_len:
        text = text[:max_text_len] + "\n[...TEXTO TRUNCADO...]"

    return f"""Extraia entidades do contrato. Tipos: {types_str}

REGRAS:
1. CPF=11 dígitos, CNPJ=14 dígitos, CEP=8 dígitos
2. Não invente dados ausentes
3. JSON curto e válido

CONTRATO:
{text}

JSON:
{{"entities":[{{"text":"valor","type":"tipo","confidence":0.95}}]}}"""


# =============================================================================
# CORE ENTITY TYPES - Most representative for contract analysis
# =============================================================================

CORE_ENTITY_TYPES = [
    # Identificação de Sócios
    "nome->socio",
    "cpf->socio",
    "rg->socio",

    # Identificação da Sociedade
    "cnpj->sociedade",
    "nire->sociedade",

    # Localização - Sócio
    "cep->socio",
    "municipio_(ou_cidade)->socio",

    # Localização - Sociedade
    "cep->sociedade",
    "municipio_(ou_cidade)->sociedade",

    # Datas
    "data_de_registro_do_contrato",
]


CORE_ENTITY_DESCRIPTIONS = """
ENTIDADES CORE A EXTRAIR (apenas estas 10):

1. nome->socio: Nome completo de pessoa física sócia
   Exemplo: "JOÃO SILVA DOS SANTOS", "MARIA OLIVEIRA COSTA"

2. cpf->socio: CPF do sócio (11 dígitos)
   Formato: XXX.XXX.XXX-XX ou XXXXXXXXXXX
   Exemplo: "123.456.789-00"

3. rg->socio: RG do sócio (número da identidade)
   Exemplo: "12.345.678-9", "MG-12.345.678"

4. cnpj->sociedade: CNPJ da empresa (14 dígitos)
   Formato: XX.XXX.XXX/XXXX-XX
   Exemplo: "12.345.678/0001-90"

5. nire->sociedade: NIRE (registro na Junta Comercial)
   Exemplo: "35.208.123.456"

6. cep->socio: CEP de residência do sócio (8 dígitos)
   Formato: XXXXX-XXX
   Exemplo: "01234-567"

7. municipio_(ou_cidade)->socio: Cidade de residência do sócio
   Exemplo: "São Paulo", "Rio de Janeiro"

8. cep->sociedade: CEP da sede da empresa
   Exemplo: "04567-890"

9. municipio_(ou_cidade)->sociedade: Cidade da sede
   Exemplo: "São Paulo", "Belo Horizonte"

10. data_de_registro_do_contrato: Data de registro/arquivamento
    Exemplo: "15/03/2020", "20 de março de 2020"
"""


def create_core_extraction_prompt(text: str) -> str:
    """
    Create optimized prompt specifically for CORE entity types.
    Focused, precise, with clear examples and validation rules.
    """
    return f"""{SYSTEM_PROMPT_EXPERT}

{CORE_ENTITY_DESCRIPTIONS}

## REGRAS DE VALIDAÇÃO
- CPF: exatamente 11 dígitos
- CNPJ: exatamente 14 dígitos
- CEP: exatamente 8 dígitos
- Nome: mínimo 2 palavras, nome completo
- Não extraia dados que não existem no texto

## ⚠️ ATENÇÃO ESPECIAL - MÚLTIPLOS SÓCIOS
Este contrato pode conter MUITOS sócios (10, 20 ou mais).
Você DEVE extrair TODOS os sócios encontrados, não apenas os primeiros.
Percorra TODO o documento até o final.

Para cada sócio identificado, extraia:
- nome->socio (nome completo)
- cpf->socio (se disponível)
- rg->socio (se disponível)
- cep->socio (se disponível)
- municipio_(ou_cidade)->socio (se disponível)

## ⚠️ DATAS - FORMATOS VARIADOS
A data de registro pode aparecer em vários formatos:
- "15/03/2020"
- "15.03.2020"
- "15 de março de 2020"
- "2 0 MAR. 2020" (com espaços)
- "em sessão de 20/03/2020"
Extraia a data exatamente como aparece no documento.

## CONTRATO PARA ANÁLISE
```
{text}
```

## RESPOSTA (JSON válido apenas):
```json
{{
  "entities": [
    {{"text": "valor_exato", "type": "tipo_da_entidade", "confidence": 0.95}}
  ]
}}
```

LEMBRETE FINAL:
- Extraia TODOS os sócios, não pare nos primeiros
- Percorra o documento COMPLETO
- O JSON pode ter muitas entidades (50+), isso é esperado"""


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'SYSTEM_PROMPT_EXPERT',
    'FEW_SHOT_EXAMPLES',
    'VALIDATION_RULES',
    'ANTI_HALLUCINATION',
    'COT_TEMPLATE',
    'create_optimized_prompt',
    'PHASE1_SECTION_IDENTIFICATION',
    'PHASE2_EXTRACT_FROM_SECTION',
    'create_self_consistency_prompt',
    'create_compact_prompt'
]
