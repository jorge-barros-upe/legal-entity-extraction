"""
Optimized Prompts for DI2WIN Dataset - Brazilian Social Contracts (Contratos Sociais).

This module contains carefully crafted prompts designed to improve F1-Score
on the di2win dataset which contains 143 entity types from Brazilian social contracts.

Key improvements:
1. Entity type hierarchy matching di2win's annotation schema
2. Specific examples for common Brazilian document patterns
3. Handling of multiple shareholders (sócios), administrators, and representatives
4. Support for company branches (filiais) and contract amendments
"""

from typing import List, Dict, Any, Set


# =============================================================================
# DI2WIN ENTITY TYPE HIERARCHY
# =============================================================================

# Top 30 most frequent entity types (covers ~90% of annotations)
HIGH_FREQUENCY_TYPES = {
    # Sociedade (Company) - 305+ occurrences each
    "nome_(ou_razao_social)->sociedade",  # 305
    "nire->sociedade",  # 182
    "cnpj->sociedade",  # 115

    # Sócio (Shareholder) - Core identification
    "nome->socio",  # 266
    "cpf->socio",  # 101
    "rg->socio",  # 89
    "nacionalidade->socio",  # 106
    "estado_civil->socio",  # 104
    "trabalho->socio",  # 97
    "data_nascimento->socio",  # 79

    # Sócio (Shareholder) - Address
    "nome_da_rua->socio",  # 113
    "numero_da_rua->socio",  # 104
    "complemento->socio",  # 76
    "bairro->socio",  # 103
    "municipio_(ou_cidade)->socio",  # 108
    "uf->socio",  # 79
    "cep->socio",  # 106

    # Sócio (Shareholder) - Capital/Quotas
    "numero_de_quotas->socio",  # 97
    "valor_total_das_cotas->socio",  # 119
    "percentual_de_participacao->socio",  # 49

    # Contract/Registration
    "data_de_registro_do_contrato",  # 160
    "numero_do_registro_do_contrato",  # 154
    "numero_alteracao_contratual->contrato",  # 78

    # Administrator
    "nome->adm_1",  # 52
    "poderes->administrador",  # 158
    "vetos->administrador",  # 90

    # Sociedade - Address
    "nome_da_rua->sociedade",  # 62
    "numero_da_rua->sociedade",  # 55
    "bairro->sociedade",  # 56
    "municipio_(ou_cidade)->sociedade",  # 61
    "cep->sociedade",  # 58
    "uf->sociedade",  # 39

    # Sociedade - Capital
    "valor_total_em_reais->sociedade",  # 77
    "numero_de_quotas_total->sociedade",  # 60
    "valor_nominal_quota->sociedade",  # 56

    # Other
    "quem_assina",  # 132
    "data->assinatura_contrato",  # 42
    "municipio_(ou_cidade)->assinatura_contrato",  # 41
    "nome_foro_eleito->sociedade",  # 38
}

# Medium frequency types (10-50 occurrences)
MEDIUM_FREQUENCY_TYPES = {
    "orgao_expedidor_rg->socio",
    "uf_do_orgao_expedidor_rg->socio",
    "comunhao_de_bens->socio",
    "municipio_naturalidade->socio",
    "uf_naturalidade->socio",
    "pais->socio",
    "complemento->sociedade",
    "nome_(ou_razao_social)_NOVO->sociedade",
    "nome_da_rua_NOVO->sociedade",
    "municipio_(ou_cidade)_NOVO->sociedade",
    "cep_NOVO->sociedade",
    "bairro_NOVO->sociedade",
    "uf_NOVO->sociedade",
    "data_de_fundacao_sociedade",
    "uf->assinatura_contrato",
    "estado_foro_eleito->sociedade",
    "nome->adm_2",
    "nome_fantasia->sociedade",
    "cnh->socio",
    "orgao_expedidor_cnh->socio",
    "CIC->socio",
    "assinatura_ou_administracao_isolada_ou_conjunta->administrador",
}


# =============================================================================
# SYSTEM PROMPT - Specialized for DI2WIN
# =============================================================================

SYSTEM_PROMPT_DI2WIN = """Você é um especialista em análise de contratos sociais brasileiros com experiência em:
- Constituição de sociedades limitadas (LTDA)
- Alterações contratuais e consolidações
- Identificação de sócios, administradores e suas qualificações completas

CONTEXTO DO DOCUMENTO:
Os documentos são contratos sociais e alterações contratuais de empresas brasileiras.
Eles contêm informações detalhadas sobre:
1. A sociedade (razão social, CNPJ, NIRE, sede, capital, quotas)
2. Os sócios (nome, CPF, RG, endereço, participação societária, estado civil, regime de bens)
3. Os administradores (nome, poderes, vedações/vetos, forma de administração)
4. Capital social (total, quotas, valores nominais, distribuição entre sócios)
5. Datas importantes (registro, fundação, assinatura)
6. Assinatura (quem assina, local e data da assinatura)

ATENÇÃO ESPECIAL:
- PODERES DO ADMINISTRADOR: Textos longos que descrevem o que o administrador PODE fazer
- VETOS DO ADMINISTRADOR: Textos que começam com "vedado", "proibido", "não poderá", descrevem RESTRIÇÕES
- QUEM ASSINA: Identifique TODAS as pessoas que assinam o contrato (geralmente no final)
- REGIME DE BENS (comunhao_de_bens->socio): "comunhão parcial", "separação total", "comunhão universal"
- NÚMERO TOTAL DE QUOTAS: Extraia o total de quotas da sociedade, não confunda com quotas individuais

PRINCÍPIOS FUNDAMENTAIS:
1. EXTRAIA EXATAMENTE como aparece no texto original
2. Use os tipos de entidade EXATOS da lista fornecida
3. Identifique TODOS os sócios, mesmo que sejam muitos (10+)
4. Diferencie dados da SOCIEDADE vs dados dos SÓCIOS vs dados de FILIAIS"""


# =============================================================================
# ENTITY TYPE DESCRIPTIONS - Detailed for LLM understanding
# =============================================================================

ENTITY_DESCRIPTIONS = """
## TIPOS DE ENTIDADES DO DI2WIN

### 1. DADOS DA SOCIEDADE (prefixo ->sociedade)
| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| nome_(ou_razao_social)->sociedade | Razão social completa | "EMPRESA TECH SOLUTIONS LTDA" |
| cnpj->sociedade | CNPJ da matriz (14 dígitos) | "12.345.678/0001-90" |
| nire->sociedade | Número de registro na Junta Comercial | "35.208.123.456" |
| nome_da_rua->sociedade | Nome da rua da sede | "Rua Augusta" |
| numero_da_rua->sociedade | Número do endereço da sede | "500" |
| complemento->sociedade | Complemento do endereço | "10º andar, sala 1001" |
| bairro->sociedade | Bairro da sede | "Consolação" |
| municipio_(ou_cidade)->sociedade | Cidade da sede | "São Paulo" |
| uf->sociedade | Estado da sede | "SP" |
| cep->sociedade | CEP da sede (8 dígitos) | "01304-000" |
| valor_total_em_reais->sociedade | Capital social total | "R$ 100.000,00" |
| numero_de_quotas_total->sociedade | Total de quotas | "100.000" |
| valor_nominal_quota->sociedade | Valor por quota | "R$ 1,00" |
| nome_foro_eleito->sociedade | Foro para disputas | "São Paulo" |
| nome_fantasia->sociedade | Nome fantasia | "Tech Solutions" |

### 2. DADOS DOS SÓCIOS (prefixo ->socio)
| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| nome->socio | Nome completo do sócio | "JOÃO SILVA DOS SANTOS" |
| cpf->socio | CPF do sócio (11 dígitos) | "123.456.789-00" |
| rg->socio | RG do sócio | "12.345.678-9" |
| orgao_expedidor_rg->socio | Órgão que emitiu o RG | "SSP/SP" |
| nacionalidade->socio | Nacionalidade | "brasileiro" |
| estado_civil->socio | Estado civil | "casado", "solteiro" |
| comunhao_de_bens->socio | Regime de bens | "comunhão parcial de bens" |
| trabalho->socio | Profissão | "empresário", "administrador" |
| data_nascimento->socio | Data de nascimento | "15/03/1980" |
| nome_da_rua->socio | Rua de residência | "Rua das Flores" |
| numero_da_rua->socio | Número da residência | "100" |
| complemento->socio | Complemento | "apto 501" |
| bairro->socio | Bairro de residência | "Jardim Paulista" |
| municipio_(ou_cidade)->socio | Cidade de residência | "São Paulo" |
| uf->socio | Estado de residência | "SP" |
| cep->socio | CEP de residência | "01234-567" |
| numero_de_quotas->socio | Quotas que possui | "50.000" |
| valor_total_das_cotas->socio | Valor das quotas | "R$ 50.000,00" |
| percentual_de_participacao->socio | % de participação | "50%" |

### 3. DADOS DE REGISTRO E DATAS
| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| data_de_registro_do_contrato | Data de registro na Junta | "15/03/2020" |
| numero_do_registro_do_contrato | Número do registro | "123456" |
| numero_alteracao_contratual->contrato | Número da alteração | "1ª", "2ª" |
| data_de_fundacao_sociedade | Data de fundação | "01/01/2010" |
| data->assinatura_contrato | Data da assinatura | "10/03/2020" |
| municipio_(ou_cidade)->assinatura_contrato | Local da assinatura | "São Paulo" |

### 4. DADOS DE ADMINISTRAÇÃO E ASSINATURA
| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| nome->adm_1 | Nome do 1º administrador | "JOÃO SILVA" |
| nome->adm_2 | Nome do 2º administrador | "MARIA COSTA" |
| poderes->administrador | O que o administrador PODE fazer (texto longo!) | "praticar todos os atos necessários à administração da sociedade, representá-la perante terceiros" |
| vetos->administrador | O que é VEDADO/PROIBIDO ao administrador | "vedada a alienação ou oneração de bens imóveis do ativo permanente" |
| quem_assina | Nome de CADA pessoa que assina (aparecem no final do contrato) | "JOÃO SILVA" |
| assinatura_ou_administracao_isolada_ou_conjunta->administrador | Se o administrador atua sozinho ou em conjunto | "isoladamente", "conjuntamente" |

### 5. ALTERAÇÕES (prefixo _NOVO)
Quando há mudança de endereço/dados, usar sufixo _NOVO:
| Tipo | Descrição |
|------|-----------|
| nome_(ou_razao_social)_NOVO->sociedade | Nova razão social |
| nome_da_rua_NOVO->sociedade | Novo endereço |
| cep_NOVO->sociedade | Novo CEP |
"""


# =============================================================================
# VALIDATION RULES - Specific to DI2WIN
# =============================================================================

VALIDATION_RULES_DI2WIN = """
## REGRAS DE VALIDAÇÃO

### Documentos de Identificação
- **CPF**: 11 dígitos (XXX.XXX.XXX-XX)
- **CNPJ**: 14 dígitos (XX.XXX.XXX/XXXX-XX)
  - Matriz: /0001-XX
  - Filiais: /0002-XX, /0003-XX, etc.
- **CEP**: 8 dígitos (XXXXX-XXX)
- **NIRE**: Geralmente 11-15 dígitos, formato varia por estado
- **RG**: Varia por estado (7-12 dígitos + letra opcional)

### Nomes
- **nome->socio**: Nome COMPLETO de pessoa física
- **nome_(ou_razao_social)->sociedade**: Razão social COMPLETA incluindo tipo (LTDA, S/A, etc.)
- **nome_fantasia->sociedade**: Nome comercial (diferente da razão social)

### Endereços
- Diferenciar endereço da SEDE (->sociedade) do endereço de RESIDÊNCIA (->socio)
- A mesma cidade pode aparecer múltiplas vezes para entidades diferentes

### Datas
- **data_de_registro_do_contrato**: Data de registro/arquivamento NA JUNTA COMERCIAL
- **data->assinatura_contrato**: Data em que o contrato foi ASSINADO
- **data_de_fundacao_sociedade**: Data de constituição ORIGINAL da empresa
- NÃO confundir com datas de nascimento ou validade de documentos

### Capital e Quotas
- **valor_total_em_reais->sociedade**: Capital social TOTAL
- **valor_total_das_cotas->socio**: Valor das quotas DE CADA SÓCIO
- **percentual_de_participacao->socio**: Porcentagem de cada sócio
"""


# =============================================================================
# FEW-SHOT EXAMPLES - DI2WIN Specific
# =============================================================================

FEW_SHOT_EXAMPLES_DI2WIN = [
    {
        "input": """ALTERAÇÃO CONTRATUAL
DIGICROM ANALITICA LTDA
CNPJ: 12.345.678/0001-90
NIRE: 35.208.123.456

RENATO BARBOSA PRANDINI, brasileiro, divorciado, comerciante, CRC-SP 123456, portador da cédula de identidade RG nº 12.345.678-9 SSP/SP, inscrito no CPF/MF sob nº 123.456.789-00, DRT/SP, residente e domiciliado na Rua Professor Atílio Innocenti, nº 165, apartamento 141, Brooklin, CEP 04538-000, São Paulo/SP;
MARIA LUIZA PRANDINI INNOCENTI, brasileira, viúva, empresária, portadora do RG nº 98.765.432-1 SSP/SP, inscrita no CPF/MF sob nº 987.654.321-00, residente na Rua Professor Atílio Innocenti, nº 165, apartamento 141, Brooklin, CEP 04538-000, São Paulo/SP;

Únicos sócios da sociedade DIGICROM ANALITICA LTDA, com sede na Rua Tabapuã, 500, 5º andar, Itaim Bibi, CEP 04533-001, São Paulo/SP, inscrita na JUCESP sob NIRE 35.208.123.456 em sessão de 2 0 MAR. 2020.""",

        "output": [
            {"text": "DIGICROM ANALITICA LTDA", "type": "nome_(ou_razao_social)->sociedade"},
            {"text": "12.345.678/0001-90", "type": "cnpj->sociedade"},
            {"text": "35.208.123.456", "type": "nire->sociedade"},
            {"text": "RENATO BARBOSA PRANDINI", "type": "nome->socio"},
            {"text": "brasileiro", "type": "nacionalidade->socio"},
            {"text": "divorciado", "type": "estado_civil->socio"},
            {"text": "comerciante", "type": "trabalho->socio"},
            {"text": "12.345.678-9", "type": "rg->socio"},
            {"text": "SSP/SP", "type": "orgao_expedidor_rg->socio"},
            {"text": "123.456.789-00", "type": "cpf->socio"},
            {"text": "Rua Professor Atílio Innocenti", "type": "nome_da_rua->socio"},
            {"text": "165", "type": "numero_da_rua->socio"},
            {"text": "apartamento 141", "type": "complemento->socio"},
            {"text": "Brooklin", "type": "bairro->socio"},
            {"text": "04538-000", "type": "cep->socio"},
            {"text": "São Paulo", "type": "municipio_(ou_cidade)->socio"},
            {"text": "SP", "type": "uf->socio"},
            {"text": "MARIA LUIZA PRANDINI INNOCENTI", "type": "nome->socio"},
            {"text": "brasileira", "type": "nacionalidade->socio"},
            {"text": "viúva", "type": "estado_civil->socio"},
            {"text": "empresária", "type": "trabalho->socio"},
            {"text": "98.765.432-1", "type": "rg->socio"},
            {"text": "SSP/SP", "type": "orgao_expedidor_rg->socio"},
            {"text": "987.654.321-00", "type": "cpf->socio"},
            {"text": "Rua Professor Atílio Innocenti", "type": "nome_da_rua->socio"},
            {"text": "165", "type": "numero_da_rua->socio"},
            {"text": "apartamento 141", "type": "complemento->socio"},
            {"text": "Brooklin", "type": "bairro->socio"},
            {"text": "04538-000", "type": "cep->socio"},
            {"text": "São Paulo", "type": "municipio_(ou_cidade)->socio"},
            {"text": "SP", "type": "uf->socio"},
            {"text": "Rua Tabapuã", "type": "nome_da_rua->sociedade"},
            {"text": "500", "type": "numero_da_rua->sociedade"},
            {"text": "5º andar", "type": "complemento->sociedade"},
            {"text": "Itaim Bibi", "type": "bairro->sociedade"},
            {"text": "04533-001", "type": "cep->sociedade"},
            {"text": "São Paulo", "type": "municipio_(ou_cidade)->sociedade"},
            {"text": "SP", "type": "uf->sociedade"},
            {"text": "2 0 MAR. 2020", "type": "data_de_registro_do_contrato"}
        ],
        "notes": "Note: Each shareholder has SEPARATE entries for their personal data. CRC-SP is a professional registration (accountant), not RG."
    },
    {
        "input": """O capital social é de R$ 100.000,00 (cem mil reais), dividido em 100.000 (cem mil) quotas no valor nominal de R$ 1,00 (um real) cada uma, totalmente integralizado, assim distribuído entre os sócios:

JOÃO SILVA - 60.000 quotas, no valor total de R$ 60.000,00, correspondente a 60% do capital social;
MARIA COSTA - 40.000 quotas, no valor total de R$ 40.000,00, correspondente a 40% do capital social;

CLÁUSULA TERCEIRA - DA ADMINISTRAÇÃO
A sociedade será administrada pelo sócio JOÃO SILVA, que exercerá seus poderes isoladamente, podendo praticar todos os atos necessários à representação da sociedade, vedada a alienação de bens imóveis.""",

        "output": [
            {"text": "R$ 100.000,00", "type": "valor_total_em_reais->sociedade"},
            {"text": "100.000", "type": "numero_de_quotas_total->sociedade"},
            {"text": "R$ 1,00", "type": "valor_nominal_quota->sociedade"},
            {"text": "JOÃO SILVA", "type": "nome->socio"},
            {"text": "60.000", "type": "numero_de_quotas->socio"},
            {"text": "R$ 60.000,00", "type": "valor_total_das_cotas->socio"},
            {"text": "60%", "type": "percentual_de_participacao->socio"},
            {"text": "MARIA COSTA", "type": "nome->socio"},
            {"text": "40.000", "type": "numero_de_quotas->socio"},
            {"text": "R$ 40.000,00", "type": "valor_total_das_cotas->socio"},
            {"text": "40%", "type": "percentual_de_participacao->socio"},
            {"text": "JOÃO SILVA", "type": "nome->adm_1"},
            {"text": "praticar todos os atos necessários à representação da sociedade", "type": "poderes->administrador"},
            {"text": "vedada a alienação de bens imóveis", "type": "vetos->administrador"},
            {"text": "isoladamente", "type": "assinatura_ou_administracao_isolada_ou_conjunta->administrador"}
        ],
        "notes": "Capital distribution and administration clauses require careful extraction of quotas, values, and powers."
    },
    {
        "input": """PEDRO HENRIQUE SOUZA, brasileiro, casado sob o regime da comunhão parcial de bens, empresário, nascido em 15/08/1975, portador do RG nº 25.678.901-2 SSP/SP, CPF 456.789.012-34, residente na Rua das Palmeiras, 789, apto 302, Centro, CEP 01310-100, São Paulo/SP;

E assim, por estarem justos e acordados, os sócios assinam o presente instrumento em 03 (três) vias de igual teor.

São Paulo, 20 de março de 2023.

_____________________
PEDRO HENRIQUE SOUZA

_____________________
MARIA CLARA OLIVEIRA""",

        "output": [
            {"text": "PEDRO HENRIQUE SOUZA", "type": "nome->socio"},
            {"text": "brasileiro", "type": "nacionalidade->socio"},
            {"text": "casado", "type": "estado_civil->socio"},
            {"text": "comunhão parcial de bens", "type": "comunhao_de_bens->socio"},
            {"text": "empresário", "type": "trabalho->socio"},
            {"text": "15/08/1975", "type": "data_nascimento->socio"},
            {"text": "25.678.901-2", "type": "rg->socio"},
            {"text": "SSP/SP", "type": "orgao_expedidor_rg->socio"},
            {"text": "456.789.012-34", "type": "cpf->socio"},
            {"text": "Rua das Palmeiras", "type": "nome_da_rua->socio"},
            {"text": "789", "type": "numero_da_rua->socio"},
            {"text": "apto 302", "type": "complemento->socio"},
            {"text": "Centro", "type": "bairro->socio"},
            {"text": "01310-100", "type": "cep->socio"},
            {"text": "São Paulo", "type": "municipio_(ou_cidade)->socio"},
            {"text": "SP", "type": "uf->socio"},
            {"text": "São Paulo", "type": "municipio_(ou_cidade)->assinatura_contrato"},
            {"text": "20 de março de 2023", "type": "data->assinatura_contrato"},
            {"text": "PEDRO HENRIQUE SOUZA", "type": "quem_assina"},
            {"text": "MARIA CLARA OLIVEIRA", "type": "quem_assina"}
        ],
        "notes": "IMPORTANT: 'quem_assina' extracts EACH person who signs (usually at the end). 'comunhao_de_bens->socio' is the marital property regime."
    }
]


# =============================================================================
# MAIN EXTRACTION PROMPT
# =============================================================================

def create_di2win_prompt(
    text: str,
    entity_types: List[str],
    include_examples: bool = True,
    max_examples: int = 2
) -> str:
    """
    Create optimized extraction prompt for DI2WIN dataset.

    Args:
        text: Contract text to extract from
        entity_types: List of entity types to extract
        include_examples: Whether to include few-shot examples
        max_examples: Maximum number of examples to include

    Returns:
        Complete prompt string
    """
    # Group entity types by category for clarity
    types_formatted = "\n".join([f"- {t}" for t in sorted(entity_types)])

    # Build examples section
    examples_section = ""
    if include_examples and max_examples > 0:
        examples_section = "\n## EXEMPLOS\n"
        for i, ex in enumerate(FEW_SHOT_EXAMPLES_DI2WIN[:max_examples]):
            # Truncate example output for prompt length
            output_preview = ex['output'][:10]
            examples_section += f"""
### Exemplo {i+1}:
Entrada (trecho):
```
{ex['input'][:800]}...
```

Saída (primeiras 10 entidades):
```json
{{"entities": {output_preview}}}
```
Nota: {ex['notes']}
"""

    prompt = f"""{SYSTEM_PROMPT_DI2WIN}

## TAREFA
Extraia TODAS as entidades dos tipos listados abaixo do contrato social brasileiro.

## TIPOS DE ENTIDADES A EXTRAIR
{types_formatted}

{VALIDATION_RULES_DI2WIN}
{examples_section}

## CONTRATO PARA ANÁLISE
```
{text}
```

## INSTRUÇÕES FINAIS
1. Extraia TODAS as entidades encontradas, não apenas as primeiras
2. Use os tipos de entidade EXATAMENTE como listados acima
3. Se houver múltiplos sócios, extraia os dados de CADA UM
4. Extraia poderes->administrador (o que PODE fazer) e vetos->administrador (o que é VEDADO)
5. Extraia quem_assina - CADA nome que aparece nas assinaturas
6. Extraia comunhao_de_bens->socio para casados (regime de bens)
7. Se um dado não existir no texto, NÃO invente
8. Retorne APENAS JSON válido

## FORMATO DE RESPOSTA
```json
{{
  "entities": [
    {{"text": "valor_exato_do_documento", "type": "tipo_exato_da_lista", "confidence": 0.95}}
  ]
}}
```
"""
    return prompt


def create_di2win_simple_prompt(text: str, entity_types: List[str]) -> str:
    """
    Create a simpler, more direct prompt for DI2WIN extraction.
    Better for models that struggle with long prompts.
    """
    # Get top 20 most important types
    important_types = [t for t in entity_types if t in HIGH_FREQUENCY_TYPES]
    if len(important_types) < 10:
        important_types = entity_types[:30]

    types_str = ", ".join(important_types[:20])

    return f"""Extraia entidades do contrato social brasileiro.

TIPOS: {types_str}

REGRAS:
- CPF: 11 dígitos (XXX.XXX.XXX-XX)
- CNPJ: 14 dígitos (XX.XXX.XXX/XXXX-XX)
- CEP: 8 dígitos (XXXXX-XXX)
- Extraia TODOS os sócios
- Use os tipos EXATAMENTE como listados

CONTRATO:
{text}

RESPOSTA (JSON):
{{"entities": [{{"text": "valor", "type": "tipo", "confidence": 0.95}}]}}"""


def create_di2win_self_consistency_prompt(
    text: str,
    entity_types: List[str],
    variation: int
) -> str:
    """
    Create prompt variation for self-consistency voting on DI2WIN.

    Args:
        text: Contract text
        entity_types: Entity types to extract
        variation: Variation number (0, 1, 2)

    Returns:
        Prompt string
    """
    # Use HIGH_FREQUENCY_TYPES for better focus
    filtered_types = [t for t in entity_types if t in HIGH_FREQUENCY_TYPES]
    if len(filtered_types) < 15:
        filtered_types = entity_types[:30]
    types_formatted = "\n".join([f"- {t}" for t in filtered_types[:25]])

    json_format = '''```json
{"entities": [{"text": "valor_exato", "type": "tipo_da_lista"}]}
```'''

    variations = [
        # Variation 0: Precision focus
        f"""Extraia entidades do contrato social brasileiro abaixo.

TIPOS DE ENTIDADES:
{types_formatted}

REGRAS:
- Extraia EXATAMENTE como aparece no texto
- CPF: 11 dígitos, CNPJ: 14 dígitos, CEP: 8 dígitos
- Use APENAS tipos da lista acima

CONTRATO:
{text}

Responda APENAS com JSON válido no formato:
{json_format}""",

        # Variation 1: Recall focus
        f"""Você é um extrator de entidades de contratos sociais brasileiros.

TIPOS ACEITOS:
{types_formatted}

IMPORTANTE:
- Extraia TODAS as ocorrências de cada tipo
- Pode haver múltiplos sócios - extraia TODOS
- Use os tipos EXATAMENTE como listados

CONTRATO:
{text}

Responda APENAS com JSON válido:
{json_format}""",

        # Variation 2: Structured approach
        f"""Analise o contrato social e extraia as entidades.

CATEGORIAS:
1. SOCIEDADE: razão social, CNPJ, NIRE, endereço
2. SÓCIOS: nome, CPF, RG, endereço, quotas
3. ADMINISTRAÇÃO: administrador, poderes
4. DATAS: registro, assinatura

TIPOS VÁLIDOS:
{types_formatted}

CONTRATO:
{text}

Responda APENAS com JSON válido:
{json_format}"""
    ]

    return variations[variation % len(variations)]


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'SYSTEM_PROMPT_DI2WIN',
    'ENTITY_DESCRIPTIONS',
    'VALIDATION_RULES_DI2WIN',
    'HIGH_FREQUENCY_TYPES',
    'MEDIUM_FREQUENCY_TYPES',
    'FEW_SHOT_EXAMPLES_DI2WIN',
    'create_di2win_prompt',
    'create_di2win_simple_prompt',
    'create_di2win_self_consistency_prompt',
]
