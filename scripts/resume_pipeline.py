"""
Retoma el pipeline desde el paso 4 (NER + Sentimiento + Features).
Lee data/interim/02_preprocesado.csv (completo, ~73k filas) y genera
03_ner_sentimiento.csv, 04_features.csv y el parquet final.
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
from textblob import TextBlob
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA_INTERIM = Path("data/interim")
DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Cargar datos ya preprocesados
print("[1/4] Cargando 02_preprocesado.csv...")
df = pd.read_csv(DATA_INTERIM / "02_preprocesado.csv")
print(f"  Filas: {len(df):,}")

# ========== 2. NER ==========
print("\n[2/4] Extrayendo entidades (NER)...")
nlp_ner = spacy.load("en_core_web_sm")
ruler = nlp_ner.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})

financial = [
    "Equifax", "Experian", "TransUnion", "FICO", "VantageScore",
    "CFPB", "Consumer Financial Protection Bureau", "IRS", "FTC",
    "PennyMac", "Navient", "Nelnet", "Great Lakes", "Sallie Mae",
    "Citibank", "Chase", "Bank of America", "Wells Fargo",
    "Capital One", "Discover", "American Express", "Synchrony",
]
patterns = [{"label": "ORG", "pattern": ent} for ent in financial]
patterns += [
    {"label": "LAW", "pattern": "FCRA"},
    {"label": "LAW", "pattern": "FDCPA"},
    {"label": "LAW", "pattern": "FCBA"},
    {"label": "LAW", "pattern": "ECOA"},
    {"label": "LAW", "pattern": "TILA"},
    {"label": "LAW", "pattern": [{"LOWER": "section"}, {"LIKE_NUM": True}]},
    {"label": "LAW", "pattern": [{"LOWER": "fair"}, {"LOWER": "credit"}, {"LOWER": "reporting"}, {"LOWER": "act"}]},
]
ruler.add_patterns(patterns)


def extract_entities(text):
    if not isinstance(text, str) or not text.strip():
        return {}
    doc = nlp_ner(text)
    entities = {}
    for ent in doc.ents:
        if "[MASK]" in ent.text:
            continue
        entities.setdefault(ent.label_, []).append(ent.text)
    return {k: sorted(list(set(v))) for k, v in entities.items()}


# Procesar en batches para poder guardar progreso
BATCH_SIZE = 5000
total = len(df)
all_entities = []
for start in range(0, total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total)
    batch_texts = df["Consumer complaint narrative"].iloc[start:end].astype(str).tolist()
    batch_ents = [extract_entities(t) for t in tqdm(batch_texts, desc=f"NER {start}-{end}")]
    all_entities.extend(batch_ents)
    print(f"  Batch {start}-{end} completado")

df["entities"] = all_entities
df["entity_count"] = df["entities"].apply(lambda x: sum(len(v) for v in x.values()) if isinstance(x, dict) else 0)
df["org_count"] = df["entities"].apply(lambda x: len(x.get("ORG", [])) if isinstance(x, dict) else 0)
df["law_count"] = df["entities"].apply(lambda x: len(x.get("LAW", [])) if isinstance(x, dict) else 0)
df["money_count"] = df["entities"].apply(lambda x: len(x.get("MONEY", [])) if isinstance(x, dict) else 0)
df["gpe_count"] = df["entities"].apply(lambda x: len(x.get("GPE", [])) if isinstance(x, dict) else 0)

# Guardar checkpoint NER
df.to_csv(DATA_INTERIM / "03_ner_sentimiento.csv", index=False)
print("  NER guardado en 03_ner_sentimiento.csv")

# ========== 3. SENTIMIENTO ==========
print("\n[3/4] Analizando sentimiento...")
vader = SentimentIntensityAnalyzer()


def vader_analysis(text):
    s = vader.polarity_scores(text)
    c = s["compound"]
    label = "Muy Negativo" if c <= -0.5 else "Negativo" if c < -0.05 else "Neutral" if c <= 0.05 else "Positivo" if c < 0.5 else "Muy Positivo"
    return s["compound"], s["neg"], s["neu"], s["pos"], label


def textblob_analysis(text):
    b = TextBlob(text)
    p = b.sentiment.polarity
    label = "Muy Negativo" if p <= -0.3 else "Negativo" if p < -0.05 else "Neutral" if p <= 0.05 else "Positivo" if p < 0.3 else "Muy Positivo"
    return p, b.sentiment.subjectivity, label


# VADER en batches
v_res = []
for start in range(0, total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total)
    batch_texts = df["Consumer complaint narrative"].iloc[start:end].astype(str).tolist()
    batch_res = [vader_analysis(t) for t in tqdm(batch_texts, desc=f"VADER {start}-{end}")]
    v_res.extend(batch_res)
    print(f"  VADER batch {start}-{end} completado")

# TextBlob en batches
tb_res = []
for start in range(0, total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total)
    batch_texts = df["Consumer complaint narrative"].iloc[start:end].astype(str).tolist()
    batch_res = [textblob_analysis(t) for t in tqdm(batch_texts, desc=f"TextBlob {start}-{end}")]
    tb_res.extend(batch_res)
    print(f"  TextBlob batch {start}-{end} completado")

df["vader_compound"] = [r[0] for r in v_res]
df["vader_neg"] = [r[1] for r in v_res]
df["vader_neu"] = [r[2] for r in v_res]
df["vader_pos"] = [r[3] for r in v_res]
df["vader_label"] = [r[4] for r in v_res]

df["textblob_polarity"] = [r[0] for r in tb_res]
df["textblob_subjectivity"] = [r[1] for r in tb_res]
df["textblob_label"] = [r[2] for r in tb_res]

# Guardar checkpoint
df.to_csv(DATA_INTERIM / "03_ner_sentimiento.csv", index=False)
print("  Sentimiento guardado")

# ========== 4. FEATURES ==========
print("\n[4/4] Feature engineering...")
df["word_count"] = df["processed_text"].astype(str).apply(lambda x: len(x.split()))
df["unique_words"] = df["processed_text"].astype(str).apply(lambda x: len(set(x.split())))
df["lexical_diversity"] = df["unique_words"] / (df["word_count"] + 1)

legal_terms = [
    "FCRA", "FDCPA", "FCBA", "ECOA", "TILA", "violation", "violated",
    "compliance", "dispute", "validation", "verification", "investigate",
    "lawsuit", "litigation", "attorney", "fraud", "fraudulent", "identity theft",
]
df["legal_term_count"] = df["Consumer complaint narrative"].astype(str).apply(
    lambda x: sum(1 for term in legal_terms if term.lower() in x.lower())
)
df["uppercase_ratio"] = df["Consumer complaint narrative"].astype(str).apply(
    lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
)
df["mask_count"] = df["Consumer complaint narrative"].astype(str).str.count(r"\[MASK\]")

df.to_csv(DATA_INTERIM / "04_features.csv", index=False)
df.to_parquet(DATA_PROCESSED / "muestra_nlp_procesada.parquet", index=False)

print(f"\n{'='*60}")
print("PIPELINE COMPLETADO - DATASET COMPLETO")
print(f"{'='*60}")
print(f"Dataset final: {len(df):,} filas, {len(df.columns)} columnas")
print(f"Archivo: {DATA_PROCESSED / 'muestra_nlp_procesada.parquet'}")
print(f"{'='*60}")
