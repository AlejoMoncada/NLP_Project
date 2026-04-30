"""
Utilidades generales del proyecto.
"""
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Optional

import pandas as pd


def normalize_unicode(text: str) -> str:
    """Aplica NFKC y elimina caracteres de control."""
    text = unicodedata.normalize("NFKC", text)
    # Eliminar caracteres de control excepto tab, newline, etc.
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\t\n ")
    return text


def clean_narrative_basic(text: str) -> str:
    """Limpieza básica: XXXX -> [MASK], URLs, emails, múltiples espacios."""
    if not isinstance(text, str):
        return ""
    # Reemplazar XXXX (máscara de PII del CFPB) por token genérico
    text = re.sub(r"\b[Xx]{2,}\b", "[MASK]", text)
    # Remover URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remover emails
    text = re.sub(r"\S+@\S+", "", text)
    # Normalizar espacios
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def save_interim(df: pd.DataFrame, path: Path) -> None:
    """Guarda DataFrame intermedio en CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[Guardado] {path.name}: {len(df)} filas, {len(df.columns)} columnas")


def load_interim(path: Path) -> Optional[pd.DataFrame]:
    """Carga DataFrame intermedio si existe."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def narrative_length_stats(series: pd.Series) -> dict:
    """Retorna estadísticas básicas de longitud de narrativas."""
    lengths = series.dropna().astype(str).str.len()
    return {
        "mean": lengths.mean(),
        "median": lengths.median(),
        "std": lengths.std(),
        "min": lengths.min(),
        "max": lengths.max(),
        "q25": lengths.quantile(0.25),
        "q75": lengths.quantile(0.75),
    }


def sample_by_group(df: pd.DataFrame, group_col: str, n: int = 3, seed: int = 42) -> pd.DataFrame:
    """Muestra estratificada por grupo."""
    return df.groupby(group_col, group_keys=False).apply(
        lambda x: x.sample(min(n, len(x)), random_state=seed)
    )
