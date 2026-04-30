"""
Módulo de Named Entity Recognition (NER) para quejas CFPB.
"""
from typing import Dict, List

import spacy

from src.config import FINANCIAL_ENTITIES

# Cargar modelo spaCy con NER habilitado
_nlp_ner = spacy.load("en_core_web_sm")

# Añadir ruler de entidades financieras específicas
_ruler = _nlp_ner.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})

_patterns = []
for entity in FINANCIAL_ENTITIES:
    _patterns.append({"label": "ORG", "pattern": entity})

# Patrones legales
_legal_patterns = [
    {"label": "LAW", "pattern": "FCRA"},
    {"label": "LAW", "pattern": "FDCPA"},
    {"label": "LAW", "pattern": "FCBA"},
    {"label": "LAW", "pattern": "ECOA"},
    {"label": "LAW", "pattern": "TILA"},
    {"label": "LAW", "pattern": [{"LOWER": "section"}, {"LIKE_NUM": True}]},
    {"label": "LAW", "pattern": [{"LOWER": "fair"}, {"LOWER": "credit"}, {"LOWER": "reporting"}, {"LOWER": "act"}]},
    {"label": "LAW", "pattern": [{"LOWER": "fair"}, {"LOWER": "debt"}, {"LOWER": "collection"}]},
]
_ruler.add_patterns(_patterns + _legal_patterns)


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extrae entidades de un texto y las agrupa por tipo.
    Retorna dict: {LABEL: [entidad1, entidad2, ...]}
    """
    if not isinstance(text, str) or not text.strip():
        return {}

    doc = _nlp_ner(text)
    entities: Dict[str, List[str]] = {}
    for ent in doc.ents:
        # Filtrar entidades con máscaras
        if "[MASK]" in ent.text:
            continue
        label = ent.label_
        if label not in entities:
            entities[label] = []
        entities[label].append(ent.text)

    # Deduplicar y ordenar
    for label in entities:
        entities[label] = sorted(list(set(entities[label])))

    return entities


def extract_entities_batch(texts: List[str]) -> List[Dict[str, List[str]]]:
    """Extrae entidades de una lista de textos usando spaCy pipe."""
    results = []
    for doc in _nlp_ner.pipe(texts, batch_size=200):
        entities: Dict[str, List[str]] = {}
        for ent in doc.ents:
            if "[MASK]" in ent.text:
                continue
            label = ent.label_
            if label not in entities:
                entities[label] = []
            entities[label].append(ent.text)
        for label in entities:
            entities[label] = sorted(list(set(entities[label])))
        results.append(entities)
    return results


def count_entity_types(entities: Dict[str, List[str]]) -> Dict[str, int]:
    """Cuenta cuántas entidades hay por tipo."""
    return {k: len(v) for k, v in entities.items()}
