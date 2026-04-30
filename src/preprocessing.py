"""
Módulo de preprocesamiento de texto para quejas CFPB.
"""
import re
import unicodedata
from typing import List

import spacy
from nltk.corpus import stopwords

from src.config import ADDITIONAL_STOPWORDS

# Cargar modelo spaCy una sola vez (deshabilitar parser/ner para velocidad en preprocesamiento)
_nlp_preprocess = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Stopwords combinadas
_STOPWORDS = set(stopwords.words("english")) | ADDITIONAL_STOPWORDS


def clean_text(text: str) -> str:
    """
    Pipeline de limpieza completo:
    - Normalización Unicode NFKC
    - Reemplazo de XXXX -> [MASK]
    - Remoción de URLs y emails
    - Minúsculas
    - Remoción de caracteres no alfabéticos (excepto [MASK])
    - Normalización de espacios
    """
    if not isinstance(text, str):
        return ""

    # Unicode NFKC
    text = unicodedata.normalize("NFKC", text)

    # Reemplazar XXXX por token genérico
    text = re.sub(r"\b[Xx]{2,}\b", "[MASK]", text)

    # Remover URLs y emails
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)

    # Minúsculas
    text = text.lower()

    # Mantener solo letras, espacios y el token [mask]
    text = re.sub(r"[^a-z\s\[\]]", " ", text)

    # Normalizar espacios
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_lemmatize(text: str) -> List[str]:
    """
    Tokeniza, lematiza y filtra stopwords/puntuación con spaCy.
    Retorna lista de tokens limpios.
    """
    doc = _nlp_preprocess(text)
    tokens = []
    for token in doc:
        if (
            token.is_space
            or token.is_punct
            or token.is_digit
            or token.like_url
            or token.lemma_ in _STOPWORDS
        ):
            continue
        lemma = token.lemma_.lower().strip()
        if lemma and len(lemma) > 1:
            tokens.append(lemma)
    return tokens


def preprocess_pipeline(texts: List[str]) -> List[List[str]]:
    """
    Aplica limpieza + tokenización/lemmatización a una lista de textos.
    Usa spaCy pipe para eficiencia.
    """
    cleaned = [clean_text(t) for t in texts]
    processed = []
    for doc in _nlp_preprocess.pipe(cleaned, batch_size=500):
        tokens = []
        for token in doc:
            if (
                token.is_space
                or token.is_punct
                or token.is_digit
                or token.like_url
                or token.lemma_.lower() in _STOPWORDS
            ):
                continue
            lemma = token.lemma_.lower().strip()
            if lemma and len(lemma) > 1:
                tokens.append(lemma)
        processed.append(tokens)
    return processed


def preprocess_single(text: str) -> str:
    """
    Versión simplificada que retorna string de tokens separados por espacio.
    Útil para columnas de texto procesado en DataFrames.
    """
    tokens = tokenize_and_lemmatize(clean_text(text))
    return " ".join(tokens)
