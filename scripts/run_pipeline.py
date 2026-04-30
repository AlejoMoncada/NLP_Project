"""
Ejecuta el pipeline completo de NLP sobre el dataset CFPB.
Genera todos los archivos intermedios y el dataset final procesado.
"""
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
from nltk.corpus import stopwords
from textblob import TextBlob
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Rutas
DATA_RAW = Path("data/raw/muestra_nlp_limpia.csv")
DATA_INTERIM = Path("data/interim")
DATA_PROCESSED = Path("data/processed")
DATA_INTERIM.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = None  # None = procesar todo el dataset

def main():
    print("=" * 60)
    print("PIPELINE NLP - CFPB Complaints")
    print("=" * 60)

    # ========== 1. CARGA ==========
    print("\n[1/6] Cargando datos...")
    df = pd.read_csv(DATA_RAW, engine="python", on_bad_lines="skip")
    print(f"  Filas cargadas: {len(df):,}")
    df.to_csv(DATA_INTERIM / "00_raw_fixed.csv", index=False)

    # ========== 2. LIMPIEZA ==========
    print("\n[2/6] Limpiando datos...")
    min_length = 20
    short_mask = df["Consumer complaint narrative"].astype(str).str.len() < min_length
    print(f"  Eliminando {short_mask.sum()} narrativas cortas")
    df_clean = df[~short_mask].copy()
    df_clean["narrative_length"] = df_clean["Consumer complaint narrative"].astype(str).str.len()
    df_clean["Date received"] = pd.to_datetime(df_clean["Date received"], errors="coerce")
    df_clean["year"] = df_clean["Date received"].dt.year
    df_clean["month"] = df_clean["Date received"].dt.month
    df_clean.to_csv(DATA_INTERIM / "01_limpio.csv", index=False)
    print(f"  Dataset limpio: {len(df_clean):,} filas")

    # Muestra para procesamiento rápido
    if SAMPLE_SIZE:
        sample_df = df_clean.head(SAMPLE_SIZE).copy()
        print(f"  Usando muestra de {SAMPLE_SIZE:,} filas para procesamiento")
    else:
        sample_df = df_clean.copy()

    # ========== 3. PREPROCESAMIENTO ==========
    print("\n[3/6] Preprocesando texto...")
    nlp_preprocess = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    additional_stops = {
        "xxxx", "xx", "xxxxx", "xx/xx/xxxx", "xxx",
        "company", "consumer", "complaint", "report",
        "account", "information", "requested", "please",
        "also", "would", "could", "should", "said",
        "told", "called", "spoke", "stated", "mentioned",
        "however", "therefore", "furthermore", "accordingly",
    }
    stop_words = set(stopwords.words("english")) | additional_stops

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\b[Xx]{2,}\b", "[MASK]", text)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = text.lower()
        text = re.sub(r"[^a-z\s\[\]]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    texts = sample_df["Consumer complaint narrative"].astype(str).tolist()
    cleaned = [clean_text(t) for t in texts]
    processed = []
    for doc in tqdm(nlp_preprocess.pipe(cleaned, batch_size=500), total=len(cleaned), desc="Lematizando"):
        tokens = []
        for token in doc:
            if token.is_space or token.is_punct or token.is_digit or token.like_url:
                continue
            lemma = token.lemma_.lower().strip()
            if lemma and len(lemma) > 1 and lemma not in stop_words:
                tokens.append(lemma)
        processed.append(tokens)

    sample_df["tokens"] = processed
    sample_df["processed_text"] = sample_df["tokens"].apply(lambda x: " ".join(x))
    sample_df.to_csv(DATA_INTERIM / "02_preprocesado.csv", index=False)
    print("  Preprocesamiento completado")

    # ========== 4. NER ==========
    print("\n[4/6] Extrayendo entidades (NER)...")
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

    sample_df["entities"] = [extract_entities(t) for t in tqdm(sample_df["Consumer complaint narrative"].astype(str), desc="NER")]
    sample_df["entity_count"] = sample_df["entities"].apply(lambda x: sum(len(v) for v in x.values()) if isinstance(x, dict) else 0)
    sample_df["org_count"] = sample_df["entities"].apply(lambda x: len(x.get("ORG", [])) if isinstance(x, dict) else 0)
    sample_df["law_count"] = sample_df["entities"].apply(lambda x: len(x.get("LAW", [])) if isinstance(x, dict) else 0)
    sample_df["money_count"] = sample_df["entities"].apply(lambda x: len(x.get("MONEY", [])) if isinstance(x, dict) else 0)
    sample_df["gpe_count"] = sample_df["entities"].apply(lambda x: len(x.get("GPE", [])) if isinstance(x, dict) else 0)
    sample_df.to_csv(DATA_INTERIM / "03_ner_sentimiento.csv", index=False)
    print("  NER completado")

    # ========== 5. SENTIMIENTO ==========
    print("\n[5/6] Analizando sentimiento...")
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

    v_res = [vader_analysis(t) for t in tqdm(sample_df["Consumer complaint narrative"].astype(str), desc="VADER")]
    tb_res = [textblob_analysis(t) for t in tqdm(sample_df["Consumer complaint narrative"].astype(str), desc="TextBlob")]

    sample_df["vader_compound"] = [r[0] for r in v_res]
    sample_df["vader_neg"] = [r[1] for r in v_res]
    sample_df["vader_neu"] = [r[2] for r in v_res]
    sample_df["vader_pos"] = [r[3] for r in v_res]
    sample_df["vader_label"] = [r[4] for r in v_res]

    sample_df["textblob_polarity"] = [r[0] for r in tb_res]
    sample_df["textblob_subjectivity"] = [r[1] for r in tb_res]
    sample_df["textblob_label"] = [r[2] for r in tb_res]
    sample_df.to_csv(DATA_INTERIM / "03_ner_sentimiento.csv", index=False)
    print("  Sentimiento completado")

    # ========== 6. FEATURES ==========
    print("\n[6/6] Feature engineering...")
    sample_df["word_count"] = sample_df["processed_text"].astype(str).apply(lambda x: len(x.split()))
    sample_df["unique_words"] = sample_df["processed_text"].astype(str).apply(lambda x: len(set(x.split())))
    sample_df["lexical_diversity"] = sample_df["unique_words"] / (sample_df["word_count"] + 1)

    legal_terms = [
        "FCRA", "FDCPA", "FCBA", "ECOA", "TILA", "violation", "violated",
        "compliance", "dispute", "validation", "verification", "investigate",
        "lawsuit", "litigation", "attorney", "fraud", "fraudulent", "identity theft",
    ]
    sample_df["legal_term_count"] = sample_df["Consumer complaint narrative"].astype(str).apply(
        lambda x: sum(1 for term in legal_terms if term.lower() in x.lower())
    )
    sample_df["uppercase_ratio"] = sample_df["Consumer complaint narrative"].astype(str).apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )
    sample_df["mask_count"] = sample_df["Consumer complaint narrative"].astype(str).str.count(r"\[MASK\]")

    sample_df.to_csv(DATA_INTERIM / "04_features.csv", index=False)
    sample_df.to_parquet(DATA_PROCESSED / "muestra_nlp_procesada.parquet", index=False)
    print(f"  Dataset final: {len(sample_df):,} filas, {len(sample_df.columns)} columnas")
    print(f"  Guardado en: {DATA_PROCESSED / 'muestra_nlp_procesada.parquet'}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
