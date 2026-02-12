#!/usr/bin/env python3
"""
Comprehensive Multi-Model Evaluation for di2win Dataset (Portuguese Legal Contracts).

Evaluates multiple model types:
1. LLMs: GPT-4o, GPT-4 Turbo, Gemini 2.0 Flash, Gemini 1.5 Pro, Claude
2. RAG: Dense retrieval + LLM generation with various embeddings
3. SLM: BERTimbau, Legal-BERTimbau, mBERT (fine-tuned NER models)

Outputs:
- di2win_model_comparison.csv: Overall metrics per model
- di2win_entity_performance_{model}.csv: Per-entity metrics for each model
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import csv

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ModelResult:
    """Results for a single model evaluation."""
    model_name: str
    model_type: str  # llm, rag, slm
    provider: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0
    latency_avg: float = 0.0
    latency_std: float = 0.0
    num_documents: int = 0
    errors: int = 0
    per_entity_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvaluationMetrics:
    """Metrics for evaluation."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    per_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    latencies: List[float] = field(default_factory=list)


# =============================================================================
# DI2WIN ENTITY TYPES - High frequency types from dataset
# =============================================================================

CORE_ENTITY_TYPES = [
    # Sociedade (Company)
    "nome_(ou_razao_social)->sociedade",
    "cnpj->sociedade",
    "nire->sociedade",
    "nome_da_rua->sociedade",
    "numero_da_rua->sociedade",
    "bairro->sociedade",
    "municipio_(ou_cidade)->sociedade",
    "cep->sociedade",
    "uf->sociedade",
    "valor_total_em_reais->sociedade",
    "numero_de_quotas_total->sociedade",

    # Socio (Shareholder)
    "nome->socio",
    "cpf->socio",
    "rg->socio",
    "nacionalidade->socio",
    "estado_civil->socio",
    "trabalho->socio",
    "nome_da_rua->socio",
    "numero_da_rua->socio",
    "bairro->socio",
    "municipio_(ou_cidade)->socio",
    "cep->socio",
    "uf->socio",
    "numero_de_quotas->socio",
    "valor_total_das_cotas->socio",
    "comunhao_de_bens->socio",

    # Administracao
    "nome->adm_1",
    "poderes->administrador",
    "vetos->administrador",
    "quem_assina",

    # Datas e Registro
    "data_de_registro_do_contrato",
    "numero_do_registro_do_contrato",
    "data->assinatura_contrato",
    "municipio_(ou_cidade)->assinatura_contrato",
]


# =============================================================================
# DI2WIN DATA LOADER
# =============================================================================

def load_di2win_data(data_path: str) -> List[Dict]:
    """Load di2win dataset (supports JSON and JSONL formats)."""
    documents = []

    # Try JSONL format first
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            f.seek(0)

            # Check if JSONL (each line is a JSON object)
            if first_line.startswith('{'):
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    doc = {
                        "id": str(item.get("id", len(documents))),
                        "text": item.get("text", ""),
                        "annotations": []
                    }

                    # Handle label format: [[start, end, label], ...]
                    if "label" in item:
                        for label in item["label"]:
                            if isinstance(label, list) and len(label) >= 3:
                                start, end, label_type = label[0], label[1], label[2]
                                if start < len(item["text"]):
                                    doc["annotations"].append({
                                        "type": label_type,
                                        "value": item["text"][start:end],
                                        "start": start,
                                        "end": end
                                    })

                    # Handle labels format (same as label)
                    elif "labels" in item:
                        for label in item["labels"]:
                            if isinstance(label, list) and len(label) >= 3:
                                start, end, label_type = label[0], label[1], label[2]
                                if start < len(item["text"]):
                                    doc["annotations"].append({
                                        "type": label_type,
                                        "value": item["text"][start:end],
                                        "start": start,
                                        "end": end
                                    })

                    # Handle entities format
                    elif "entities" in item:
                        for ent in item["entities"]:
                            if isinstance(ent, list) and len(ent) >= 3:
                                start, end, label_type = ent[0], ent[1], ent[2]
                                if start < len(item["text"]):
                                    doc["annotations"].append({
                                        "type": label_type,
                                        "value": item["text"][start:end],
                                        "start": start,
                                        "end": end
                                    })

                    if doc["text"] and doc["annotations"]:
                        documents.append(doc)

                return documents
    except json.JSONDecodeError:
        pass

    # Try regular JSON format
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            doc = {
                "id": str(item.get("id", len(documents))),
                "text": item.get("text", ""),
                "annotations": []
            }

            # Handle different annotation formats
            if "annotations" in item:
                for ann in item["annotations"]:
                    doc["annotations"].append({
                        "type": ann.get("type", ann.get("label", "")),
                        "value": ann.get("value", ann.get("text", "")),
                        "start": ann.get("start", 0),
                        "end": ann.get("end", 0)
                    })
            elif "label" in item:
                for label in item["label"]:
                    if isinstance(label, list) and len(label) >= 3:
                        start, end, label_type = label[0], label[1], label[2]
                        if start < len(item["text"]):
                            doc["annotations"].append({
                                "type": label_type,
                                "value": item["text"][start:end],
                                "start": start,
                                "end": end
                            })

            if doc["text"] and doc["annotations"]:
                documents.append(doc)

    return documents


# =============================================================================
# BRAZILIAN DOCUMENT VALIDATOR
# =============================================================================

class BrazilianDocumentValidator:
    """Validator for Brazilian document formats."""

    @staticmethod
    def validate_cpf(cpf: str) -> bool:
        """Validate CPF format (11 digits)."""
        import re
        digits = re.sub(r'\D', '', cpf)
        return len(digits) == 11

    @staticmethod
    def validate_cnpj(cnpj: str) -> bool:
        """Validate CNPJ format (14 digits)."""
        import re
        digits = re.sub(r'\D', '', cnpj)
        return len(digits) == 14

    @staticmethod
    def validate_cep(cep: str) -> bool:
        """Validate CEP format (8 digits)."""
        import re
        digits = re.sub(r'\D', '', cep)
        return len(digits) == 8

    @staticmethod
    def validate_rg(rg: str) -> bool:
        """Validate RG format (at least 5 digits)."""
        import re
        digits = re.sub(r'\D', '', rg)
        return len(digits) >= 5

    @staticmethod
    def validate_nire(nire: str) -> bool:
        """Validate NIRE format (11 digits typically)."""
        import re
        digits = re.sub(r'\D', '', nire)
        return len(digits) >= 8

    @classmethod
    def validate_entity(cls, entity: Dict) -> bool:
        """Validate entity based on type."""
        entity_type = entity.get("type", "").lower()
        value = entity.get("text", entity.get("value", ""))

        if "cpf" in entity_type:
            return cls.validate_cpf(value)
        elif "cnpj" in entity_type:
            return cls.validate_cnpj(value)
        elif "cep" in entity_type:
            return cls.validate_cep(value)
        elif "rg" in entity_type:
            return cls.validate_rg(value)
        elif "nire" in entity_type:
            return cls.validate_nire(value)

        return True  # No specific validation for other types


# =============================================================================
# LLM EXTRACTORS
# =============================================================================

class BaseLLMExtractor:
    """Base class for LLM extractors."""

    def __init__(self, model_name: str, provider: str):
        self.model_name = model_name
        self.provider = provider
        self.client = None

    def extract(self, text: str, entity_types: List[str]) -> List[Dict]:
        """Extract entities from text."""
        raise NotImplementedError

    def _create_prompt(self, text: str, entity_types: List[str]) -> str:
        """Create extraction prompt for di2win."""
        from src.approaches.llm.di2win_optimized_prompts import create_di2win_prompt
        return create_di2win_prompt(text[:25000], entity_types, include_examples=True, max_examples=2)


class AzureGPTExtractor(BaseLLMExtractor):
    """Azure OpenAI GPT extractor."""

    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__(model_name, "azure")
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", model_name)

    def extract(self, text: str, entity_types: List[str]) -> Tuple[List[Dict], float]:
        """Extract entities and return with latency."""
        prompt = self._create_prompt(text, entity_types)

        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )
        latency = time.time() - start_time

        content = response.choices[0].message.content
        entities = self._parse_response(content, entity_types)

        return entities, latency

    def _parse_response(self, content: str, entity_types: List[str]) -> List[Dict]:
        """Parse JSON response and filter by entity types."""
        try:
            data = json.loads(content)
            entities = data.get("entities", [])
            # Filter by requested entity types
            filtered = [e for e in entities if e.get("type") in entity_types]
            return filtered
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response")
            return []


class GeminiExtractor(BaseLLMExtractor):
    """Google Gemini extractor."""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__(model_name, "gemini")
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 3000,
                "response_mime_type": "application/json"
            }
        )

    def extract(self, text: str, entity_types: List[str]) -> Tuple[List[Dict], float]:
        """Extract entities and return with latency."""
        prompt = self._create_prompt(text, entity_types)

        start_time = time.time()
        response = self.model.generate_content(prompt)
        latency = time.time() - start_time

        entities = self._parse_response(response.text, entity_types)

        return entities, latency

    def _parse_response(self, content: str, entity_types: List[str]) -> List[Dict]:
        """Parse JSON response."""
        import re
        try:
            # Clean response - remove markdown code blocks
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Try direct parse
            try:
                data = json.loads(cleaned)
                entities = data.get("entities", [])
                filtered = [e for e in entities if e.get("type") in entity_types]
                return filtered
            except json.JSONDecodeError:
                pass

            # Try to find JSON in response with regex
            json_match = re.search(r'\{[\s\S]*"entities"[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                entities = data.get("entities", [])
                filtered = [e for e in entities if e.get("type") in entity_types]
                return filtered

            return []
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Failed to parse Gemini response: {e}")
            return []


class AnthropicExtractor(BaseLLMExtractor):
    """Anthropic Claude extractor."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        super().__init__(model_name, "anthropic")
        from anthropic import Anthropic

        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def extract(self, text: str, entity_types: List[str]) -> Tuple[List[Dict], float]:
        """Extract entities and return with latency."""
        prompt = self._create_prompt(text, entity_types)

        start_time = time.time()
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = time.time() - start_time

        content = response.content[0].text
        entities = self._parse_response(content, entity_types)

        return entities, latency

    def _parse_response(self, content: str, entity_types: List[str]) -> List[Dict]:
        """Parse JSON response."""
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                entities = data.get("entities", [])
                filtered = [e for e in entities if e.get("type") in entity_types]
                return filtered
            return []
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Claude response")
            return []


# =============================================================================
# RAG EXTRACTOR
# =============================================================================

class RAGExtractor:
    """RAG-based entity extractor for Portuguese contracts."""

    def __init__(self, embedding_model: str = "text-embedding-3-small", llm_model: str = "gpt-4o"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.provider = "rag"
        self.model_name = f"RAG-{embedding_model.split('/')[-1]}"

        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    def extract(self, text: str, entity_types: List[str]) -> Tuple[List[Dict], float]:
        """Extract entities using RAG approach."""
        start_time = time.time()

        # Chunk the document
        chunks = self._chunk_document(text)

        # For simplicity, use first chunks (production would use retrieval)
        relevant_chunks = chunks[:5]
        context = "\n\n".join(relevant_chunks)

        # Create extraction prompt with context
        from src.approaches.llm.optimized_prompts import CORE_ENTITY_DESCRIPTIONS

        prompt = f"""Extraia entidades deste contrato societário brasileiro.

{CORE_ENTITY_DESCRIPTIONS}

CONTEXTO:
{context[:12000]}

VALIDAÇÃO:
- CPF: exatamente 11 dígitos
- CNPJ: exatamente 14 dígitos
- CEP: exatamente 8 dígitos
- Nome: pelo menos 2 palavras

Responda em JSON válido:
{{"entities": [{{"text": "valor_exato", "type": "tipo_da_lista", "confidence": 0.95}}]}}"""

        response = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )

        latency = time.time() - start_time

        content = response.choices[0].message.content
        try:
            data = json.loads(content)
            entities = data.get("entities", [])
            # Filter by entity types
            entities = [e for e in entities if e.get("type") in entity_types]
        except json.JSONDecodeError:
            entities = []

        return entities, latency

    def _chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split document into chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                last_period = text[start:end].rfind('.')
                if last_period > chunk_size // 2:
                    end = start + last_period + 1
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else end
        return chunks


# =============================================================================
# SLM EXTRACTOR (Placeholder - requires fine-tuned model)
# =============================================================================

class SLMExtractor:
    """Fine-tuned SLM extractor (BERTimbau, Legal-BERTimbau, etc.)."""

    def __init__(self, model_name: str = "neuralmind/bert-base-portuguese-cased", checkpoint_path: Optional[str] = None):
        self.model_name = model_name
        self.provider = "slm"
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_model()

    def _load_model(self):
        """Load fine-tuned model."""
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
            self.model = AutoModelForTokenClassification.from_pretrained(self.checkpoint_path)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded SLM model from {self.checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to load SLM model: {e}")
            self.model = None

    def extract(self, text: str, entity_types: List[str]) -> Tuple[List[Dict], float]:
        """Extract entities using fine-tuned NER model."""
        if self.model is None:
            return [], 0.0

        import torch
        start_time = time.time()

        inputs = self.tokenizer(
            text[:4096],
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
            return_offsets_mapping=True
        )

        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
            probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        entities = self._predictions_to_entities(predictions, probabilities, offset_mapping, text, entity_types)

        latency = time.time() - start_time
        return entities, latency

    def _predictions_to_entities(self, predictions, probabilities, offset_mapping, text, entity_types) -> List[Dict]:
        """Convert BIO predictions to entity list."""
        entities = []
        current_entity = None

        id2label = getattr(self.model.config, 'id2label', {})

        for idx, pred_id in enumerate(predictions):
            if idx >= len(offset_mapping):
                break

            start, end = offset_mapping[idx]
            if start == end:
                continue

            label = id2label.get(str(pred_id), id2label.get(int(pred_id), "O"))
            confidence = float(probabilities[idx][pred_id])

            if label == "O":
                if current_entity:
                    if current_entity["type"] in entity_types:
                        entities.append(current_entity)
                    current_entity = None
            elif label.startswith("B-"):
                if current_entity and current_entity["type"] in entity_types:
                    entities.append(current_entity)
                entity_type = label[2:]
                current_entity = {
                    "text": text[start:end],
                    "type": entity_type,
                    "confidence": confidence,
                    "start": start,
                    "end": end
                }
            elif label.startswith("I-") and current_entity:
                if label[2:] == current_entity["type"]:
                    current_entity["text"] = text[current_entity["start"]:end]
                    current_entity["end"] = end
                    current_entity["confidence"] = min(current_entity["confidence"], confidence)

        if current_entity and current_entity["type"] in entity_types:
            entities.append(current_entity)

        return entities


# =============================================================================
# EVALUATION
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    import re
    # For CPF/CNPJ/CEP, normalize to digits only
    if re.match(r'^[\d\.\-/]+$', text.replace(' ', '')):
        return re.sub(r'\D', '', text)
    return " ".join(text.lower().split())


def compute_metrics(predictions: List[Dict], ground_truth: List[Dict], entity_types: List[str]) -> EvaluationMetrics:
    """Compute evaluation metrics."""
    metrics = EvaluationMetrics()

    # Filter by entity types
    predictions = [p for p in predictions if p.get("type") in entity_types]
    ground_truth = [g for g in ground_truth if g.get("type") in entity_types]

    # Create sets for exact match
    pred_set = {(p.get("type", ""), normalize_text(p.get("text", p.get("value", "")))) for p in predictions}
    true_set = {(g.get("type", ""), normalize_text(g.get("value", g.get("text", "")))) for g in ground_truth}

    metrics.tp = len(pred_set & true_set)
    metrics.fp = len(pred_set - true_set)
    metrics.fn = len(true_set - pred_set)

    metrics.precision = metrics.tp / (metrics.tp + metrics.fp) if (metrics.tp + metrics.fp) > 0 else 0.0
    metrics.recall = metrics.tp / (metrics.tp + metrics.fn) if (metrics.tp + metrics.fn) > 0 else 0.0
    metrics.f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall) if (metrics.precision + metrics.recall) > 0 else 0.0

    # Per-type metrics
    for entity_type in entity_types:
        type_preds = {normalize_text(p.get("text", p.get("value", ""))) for p in predictions if p.get("type") == entity_type}
        type_true = {normalize_text(g.get("value", g.get("text", ""))) for g in ground_truth if g.get("type") == entity_type}

        tp = len(type_preds & type_true)
        fp = len(type_preds - type_true)
        fn = len(type_true - type_preds)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.per_type[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": len(type_true),
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

    return metrics


def aggregate_metrics(all_metrics: List[EvaluationMetrics]) -> EvaluationMetrics:
    """Aggregate metrics from multiple documents."""
    total = EvaluationMetrics()

    for m in all_metrics:
        total.tp += m.tp
        total.fp += m.fp
        total.fn += m.fn
        total.latencies.extend(m.latencies)

        for entity_type, type_metrics in m.per_type.items():
            if entity_type not in total.per_type:
                total.per_type[entity_type] = {"tp": 0, "fp": 0, "fn": 0, "support": 0}
            total.per_type[entity_type]["tp"] += type_metrics.get("tp", 0)
            total.per_type[entity_type]["fp"] += type_metrics.get("fp", 0)
            total.per_type[entity_type]["fn"] += type_metrics.get("fn", 0)
            total.per_type[entity_type]["support"] += type_metrics.get("support", 0)

    # Compute overall metrics
    total.precision = total.tp / (total.tp + total.fp) if (total.tp + total.fp) > 0 else 0.0
    total.recall = total.tp / (total.tp + total.fn) if (total.tp + total.fn) > 0 else 0.0
    total.f1 = 2 * total.precision * total.recall / (total.precision + total.recall) if (total.precision + total.recall) > 0 else 0.0

    # Compute per-type metrics
    for entity_type in total.per_type:
        tp = total.per_type[entity_type]["tp"]
        fp = total.per_type[entity_type]["fp"]
        fn = total.per_type[entity_type]["fn"]

        total.per_type[entity_type]["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        total.per_type[entity_type]["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        p, r = total.per_type[entity_type]["precision"], total.per_type[entity_type]["recall"]
        total.per_type[entity_type]["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return total


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

def get_model_configs() -> List[Dict]:
    """Get configurations for all models to evaluate."""
    return [
        # LLMs
        {"name": "GPT-4o", "type": "llm", "provider": "azure", "class": AzureGPTExtractor, "args": {"model_name": "gpt-4o"}},
        {"name": "GPT-4-Turbo", "type": "llm", "provider": "azure", "class": AzureGPTExtractor, "args": {"model_name": "gpt-4-turbo"}},
        {"name": "Gemini-2.0-Flash", "type": "llm", "provider": "gemini", "class": GeminiExtractor, "args": {"model_name": "gemini-2.0-flash"}},
        # {"name": "Gemini-1.5-Pro", "type": "llm", "provider": "gemini", "class": GeminiExtractor, "args": {"model_name": "gemini-1.5-pro"}},
        # {"name": "Claude-3.5-Sonnet", "type": "llm", "provider": "anthropic", "class": AnthropicExtractor, "args": {"model_name": "claude-3-5-sonnet-20241022"}},

        # RAG
        {"name": "RAG-GPT4o", "type": "rag", "provider": "rag", "class": RAGExtractor, "args": {"embedding_model": "text-embedding-3-small", "llm_model": "gpt-4o"}},

        # SLM (requires fine-tuned checkpoints)
        # {"name": "BERTimbau", "type": "slm", "provider": "slm", "class": SLMExtractor, "args": {"model_name": "neuralmind/bert-base-portuguese-cased", "checkpoint_path": "checkpoints/bertimbau-di2win"}},
        # {"name": "Legal-BERTimbau", "type": "slm", "provider": "slm", "class": SLMExtractor, "args": {"model_name": "rufimelo/Legal-BERTimbau-base", "checkpoint_path": "checkpoints/legal-bertimbau-di2win"}},
    ]


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_model(
    extractor,
    documents: List[Dict],
    entity_types: List[str],
    max_samples: int = 50,
    delay: float = 2.0
) -> Tuple[EvaluationMetrics, int]:
    """Evaluate a single model on documents."""
    all_metrics = []
    errors = 0

    for i, doc in enumerate(documents[:max_samples]):
        try:
            # Extract entities
            predictions, latency = extractor.extract(doc["text"], entity_types)

            # Compute metrics
            metrics = compute_metrics(predictions, doc["annotations"], entity_types)
            metrics.latencies = [latency]
            all_metrics.append(metrics)

            logger.info(f"  Doc {i+1}/{min(len(documents), max_samples)}: TP={metrics.tp}, FP={metrics.fp}, FN={metrics.fn}")

            time.sleep(delay)

        except Exception as e:
            logger.error(f"  Error on doc {i+1}: {e}")
            errors += 1

    # Aggregate metrics
    total_metrics = aggregate_metrics(all_metrics)

    return total_metrics, errors


def save_results_csv(results: List[ModelResult], output_dir: str, timestamp: str):
    """Save model comparison to CSV."""
    # Overall comparison CSV
    comparison_file = os.path.join(output_dir, f"di2win_model_comparison_{timestamp}.csv")

    with open(comparison_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Type", "Provider", "Precision", "Recall", "F1",
            "Support", "Latency_Avg", "Latency_Std", "Num_Documents", "Errors"
        ])

        for r in results:
            writer.writerow([
                r.model_name, r.model_type, r.provider,
                f"{r.precision:.4f}", f"{r.recall:.4f}", f"{r.f1:.4f}",
                r.support, f"{r.latency_avg:.2f}", f"{r.latency_std:.2f}",
                r.num_documents, r.errors
            ])

    logger.info(f"Saved comparison to {comparison_file}")

    # Per-entity CSV for each model
    for r in results:
        if r.per_entity_metrics:
            entity_file = os.path.join(output_dir, f"di2win_entity_performance_{r.model_name.replace(' ', '_')}_{timestamp}.csv")

            with open(entity_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Model", "Entity_Type", "Precision", "Recall", "F1", "Support", "TP", "FP", "FN"
                ])

                for entity_type, metrics in r.per_entity_metrics.items():
                    writer.writerow([
                        r.model_name, entity_type,
                        f"{metrics.get('precision', 0):.4f}",
                        f"{metrics.get('recall', 0):.4f}",
                        f"{metrics.get('f1', 0):.4f}",
                        metrics.get('support', 0),
                        metrics.get('tp', 0),
                        metrics.get('fp', 0),
                        metrics.get('fn', 0)
                    ])

            logger.info(f"Saved entity metrics to {entity_file}")


def main():
    """Main evaluation function."""
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Evaluate multiple models on di2win dataset")
    parser.add_argument("--data-path", default="./data/di2win/contratros_test_files_extractor.jsonl", help="Path to di2win data")
    parser.add_argument("--output-dir", default="./experiments/results", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=50, help="Max documents to evaluate")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between API calls")
    parser.add_argument("--entity-types", nargs="+", default=CORE_ENTITY_TYPES, help="Entity types to evaluate")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("DI2WIN MULTI-MODEL EVALUATION (Portuguese Legal Contracts)")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    documents = load_di2win_data(args.data_path)
    print(f"Loaded {len(documents)} documents")

    # Filter documents with annotations for the entity types
    documents = [d for d in documents if any(a.get("type") in args.entity_types for a in d.get("annotations", []))]
    print(f"Documents with target annotations: {len(documents)}")

    # Get model configs
    model_configs = get_model_configs()
    print(f"\nModels to evaluate: {len(model_configs)}")
    for cfg in model_configs:
        print(f"  - {cfg['name']} ({cfg['type']})")

    print(f"\nEntity types: {len(args.entity_types)}")
    for et in args.entity_types[:5]:
        print(f"  - {et}")
    if len(args.entity_types) > 5:
        print(f"  ... and {len(args.entity_types) - 5} more")

    # Evaluate each model
    results = []

    for cfg in model_configs:
        print(f"\n{'=' * 80}")
        print(f"MODEL: {cfg['name']} ({cfg['type']})")
        print("=" * 80)

        try:
            # Initialize extractor
            extractor = cfg["class"](**cfg.get("args", {}))

            # Evaluate
            metrics, errors = evaluate_model(
                extractor, documents, args.entity_types,
                max_samples=args.max_samples, delay=args.delay
            )

            # Calculate latency stats
            latency_avg = np.mean(metrics.latencies) if metrics.latencies else 0.0
            latency_std = np.std(metrics.latencies) if metrics.latencies else 0.0

            result = ModelResult(
                model_name=cfg["name"],
                model_type=cfg["type"],
                provider=cfg["provider"],
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
                support=metrics.tp + metrics.fn,
                latency_avg=latency_avg,
                latency_std=latency_std,
                num_documents=min(len(documents), args.max_samples) - errors,
                errors=errors,
                per_entity_metrics=metrics.per_type
            )
            results.append(result)

            print(f"\n  Results for {cfg['name']}:")
            print(f"    Precision: {result.precision:.4f}")
            print(f"    Recall:    {result.recall:.4f}")
            print(f"    F1-Score:  {result.f1:.4f}")
            print(f"    Latency:   {result.latency_avg:.2f}s ± {result.latency_std:.2f}s")

            print(f"\n  Per-Entity Metrics (top 5 by support):")
            sorted_types = sorted(metrics.per_type.items(), key=lambda x: x[1].get('support', 0), reverse=True)
            for entity_type, type_metrics in sorted_types[:5]:
                print(f"    {entity_type}: P={type_metrics['precision']:.4f}, R={type_metrics['recall']:.4f}, F1={type_metrics['f1']:.4f}, Support={type_metrics['support']}")

        except Exception as e:
            logger.error(f"Failed to evaluate {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_results_csv(results, args.output_dir, timestamp)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<25} {'Type':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Latency':>12}")
    print("-" * 80)

    # Sort by F1
    results.sort(key=lambda x: x.f1, reverse=True)
    for r in results:
        print(f"{r.model_name:<25} {r.model_type:<8} {r.precision:>10.4f} {r.recall:>10.4f} {r.f1:>10.4f} {r.latency_avg:>10.2f}s")

    # Save full results as JSON
    json_file = os.path.join(args.output_dir, f"di2win_full_results_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to:")
    print(f"  - {args.output_dir}/di2win_model_comparison_{timestamp}.csv")
    print(f"  - {args.output_dir}/di2win_entity_performance_*_{timestamp}.csv")
    print(f"  - {json_file}")


if __name__ == "__main__":
    main()
