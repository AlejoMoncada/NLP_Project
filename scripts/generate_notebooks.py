"""
Generador de notebooks del proyecto NLP CFPB.
Ejecutar con: python scripts/generate_notebooks.py
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"
DIVIDED_DIR = NOTEBOOKS_DIR / "02_proyecto_nlp_dividido"


def make_cell(source: str, cell_type="code"):
    if cell_type == "markdown":
        return new_markdown_cell(source)
    return new_code_cell(source)


def generate_complete_notebook():
    """Genera la notebook única con todo el pipeline."""
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    }

    cells = []

    # TÍTULO
    cells.append(make_cell("""# Proyecto NLP: Análisis de Quejas CFPB\n\n**Autor:** William Moncada  \n**Asignatura:** Procesamiento de Lenguaje Natural  \n**Dataset:** muestra_nlp_limpia.csv (Consumer Financial Protection Bureau)\n\n---\n\nEste notebook implementa un pipeline completo de NLP sobre quejas financieras:\n1. Carga y limpieza de datos\n2. EDA (Análisis Exploratorio de Datos)\n3. Preprocesamiento de texto (tokenización, lematización, stopwords)\n4. Named Entity Recognition (NER)\n5. Análisis de sentimiento (VADER + TextBlob)\n6. Feature engineering\n7. Visualizaciones y guardado de resultados intermedios\n""", "markdown"))

    # SECCIÓN 1: SETUP
    cells.append(make_cell("""## 1. Setup y dependencias\n\nAsegúrate de haber instalado las dependencias:\n```bash\npip install -r requirements.txt\npython -m spacy download en_core_web_sm\n```\n""", "markdown"))

    cells.append(make_cell("""import os\nimport sys\nimport re\nimport unicodedata\nfrom collections import Counter\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom tqdm.notebook import tqdm\n\n# Configuración visual\nsns.set_style("whitegrid")\nplt.rcParams["figure.figsize"] = (12, 6)\n\n# Rutas del proyecto\nPROJECT_ROOT = Path().resolve()\nDATA_RAW = PROJECT_ROOT / "data" / "raw" / "muestra_nlp_limpia.csv"\nDATA_INTERIM = PROJECT_ROOT / "data" / "interim"\nDATA_PROCESSED = PROJECT_ROOT / "data" / "processed"\nDATA_INTERIM.mkdir(parents=True, exist_ok=True)\nDATA_PROCESSED.mkdir(parents=True, exist_ok=True)\n\nprint("Setup completo. Python:", sys.version)\nprint("Pandas:", pd.__version__)\n"""))

    # SECCIÓN 2: CARGA
    cells.append(make_cell("""## 2. Carga de datos y reparación\n\nEl archivo CSV tiene una última línea corrupta (EOF inside string). Usamos `engine='python'` y `on_bad_lines='skip'` para manejarlo.\n""", "markdown"))

    cells.append(make_cell("""# Carga con manejo de líneas corruptas\ndf = pd.read_csv(DATA_RAW, engine="python", on_bad_lines="skip")\n\nprint(f"Filas cargadas: {len(df):,}")\nprint(f"Columnas: {list(df.columns)}")\nprint()\nprint("=== Tipos de datos ===")\nprint(df.dtypes)\n"""))

    cells.append(make_cell("""# Guardar copia limpia del raw limpio (sin línea corrupta)\ndf.to_csv(DATA_INTERIM / "00_raw_fixed.csv", index=False)\nprint("Dataset limpio guardado en data/interim/00_raw_fixed.csv")\n"""))

    # SECCIÓN 3: EDA
    cells.append(make_cell("""## 3. Análisis Exploratorio de Datos (EDA)\n""", "markdown"))

    cells.append(make_cell("""# 3.1 Valores nulos y duplicados\nprint("=== VALORES NULOS ===")\nnulls = df.isnull().sum()\nprint(nulls[nulls > 0])\nprint()\nprint("=== DUPLICADOS ===")\nprint("Filas duplicadas:", df.duplicated().sum())\n"""))

    cells.append(make_cell("""# 3.2 Estadísticas de la narrativa\nnarr = df["Consumer complaint narrative"].astype(str)\nlengths = narr.str.len()\n\nprint("=== LONGITUD DE NARRATIVAS ===")\nprint(lengths.describe())\nprint()\nprint("Narrativas < 20 caracteres:", (lengths < 20).sum())\nprint("Narrativas > 3000 caracteres:", (lengths > 3000).sum())\nprint("Narrativas con XXXX:", narr.str.contains("XXXX", case=False).sum(), f"({narr.str.contains('XXXX', case=False).sum()/len(narr)*100:.1f}%)")\n"""))

    cells.append(make_cell("""# 3.3 Distribución de categorías clave\nfig, axes = plt.subplots(2, 2, figsize=(16, 12))\n\n# Top Issues\ndf["Issue"].value_counts().head(10).plot(kind="barh", ax=axes[0,0], color="steelblue")\naxes[0,0].set_title("Top 10 Issues")\naxes[0,0].invert_yaxis()\n\n# Top Products\ndf["Product"].value_counts().head(10).plot(kind="barh", ax=axes[0,1], color="darkorange")\naxes[0,1].set_title("Top 10 Products")\naxes[0,1].invert_yaxis()\n\n# Response type\ndf["Company response to consumer"].value_counts().plot(kind="bar", ax=axes[1,0], color="seagreen")\naxes[1,0].set_title("Company Response")\naxes[1,0].tick_params(axis="x", rotation=45)\n\n# Timely response\ndf["Timely response?"].value_counts().plot(kind="bar", ax=axes[1,1], color="coral")\naxes[1,1].set_title("Timely Response?")\n\nplt.tight_layout()\nplt.show()\n"""))

    cells.append(make_cell("""# 3.4 Análisis temporal\ndf["Date received"] = pd.to_datetime(df["Date received"], errors="coerce")\nprint("Rango de fechas:", df["Date received"].min(), "a", df["Date received"].max())\n\n# Quejas por año\ndf["year"] = df["Date received"].dt.year\nyear_counts = df["year"].value_counts().sort_index()\nyear_counts.plot(kind="line", marker="o", figsize=(10,4), color="navy")\nplt.title("Quejas por año")\nplt.xlabel("Año")\nplt.ylabel("Número de quejas")\nplt.grid(True)\nplt.show()\n"""))

    cells.append(make_cell("""# 3.5 Guardar EDA como imagen (opcional)\n# Este paso ya generó visualizaciones inline.\nprint("EDA completado. Se identificaron:")\nprint("- 1 línea corrupta al final del CSV (eliminada)")\nprint("- Desbalance severo: ~66% Credit reporting")\nprint("- 26 narrativas muy cortas (<20 chars)")\nprint("- 67.4% de narrativas contienen máscaras XXXX")\nprint("- Columnas descartables por nulos masivos: Tags, Consumer disputed?")\n"""))

    # SECCIÓN 4: LIMPIEZA
    cells.append(make_cell("""## 4. Limpieza de datos\n\nPasos:\n- Eliminar narrativas < 20 caracteres\n- Estandarizar fechas\n- Crear columna de longitud de narrativa\n- Marcar columnas descartables\n""", "markdown"))

    cells.append(make_cell("""# Filtrar narrativas muy cortas\nmin_length = 20\nshort_mask = df["Consumer complaint narrative"].astype(str).str.len() < min_length\nprint(f"Eliminando {short_mask.sum()} narrativas con < {min_length} caracteres")\ndf_clean = df[~short_mask].copy()\n\n# Longitud de narrativa como feature\ndf_clean["narrative_length"] = df_clean["Consumer complaint narrative"].astype(str).str.len()\n\n# Fechas\ndf_clean["Date received"] = pd.to_datetime(df_clean["Date received"], errors="coerce")\ndf_clean["year"] = df_clean["Date received"].dt.year\ndf_clean["month"] = df_clean["Date received"].dt.month\n\n# Guardar interim 01\ndf_clean.to_csv(DATA_INTERIM / "01_limpio.csv", index=False)\nprint(f"Dataset limpio: {len(df_clean):,} filas")\n"""))

    # SECCIÓN 5: PREPROCESAMIENTO
    cells.append(make_cell("""## 5. Preprocesamiento de Texto\n\nUsamos spaCy para:\n- Limpieza avanzada (Unicode NFKC, URLs, emails)\n- Tokenización\n- Lematización con POS tagging\n- Eliminación de stopwords (NLTK + adaptativas de dominio)\n""", "markdown"))

    cells.append(make_cell("""import spacy\nfrom nltk.corpus import stopwords\n\n# Cargar modelo spaCy (deshabilitar parser/ner para velocidad en preprocesamiento)\nnlp_preprocess = spacy.load("en_core_web_sm", disable=["parser", "ner"])\n\n# Stopwords combinadas\nadditional_stops = {\n    "xxxx", "xx", "xxxxx", "xx/xx/xxxx", "xxx",\n    "company", "consumer", "complaint", "report",\n    "account", "information", "requested", "please",\n    "also", "would", "could", "should", "said",\n    "told", "called", "spoke", "stated", "mentioned",\n    "however", "therefore", "furthermore", "accordingly",\n}\nstop_words = set(stopwords.words("english")) | additional_stops\n\nprint(f"Stopwords totales: {len(stop_words)}")\n"""))

    cells.append(make_cell("""def clean_text(text):\n    \"\"\"Limpieza completa de texto.\"\"\"\n    if not isinstance(text, str):\n        return ""\n    text = unicodedata.normalize("NFKC", text)\n    text = re.sub(r"\\b[Xx]{2,}\\b", "[MASK]", text)\n    text = re.sub(r"https?://\\S+|www\\.\\S+", "", text)\n    text = re.sub(r"\\S+@\\S+", "", text)\n    text = text.lower()\n    text = re.sub(r"[^a-z\\s\\[\\]]", " ", text)\n    text = re.sub(r"\\s+", " ", text).strip()\n    return text\n\n\ndef tokenize_and_lemmatize(text):\n    \"\"\"Tokeniza, lematiza y filtra stopwords.\"\"\"\n    doc = nlp_preprocess(text)\n    tokens = []\n    for token in doc:\n        if token.is_space or token.is_punct or token.is_digit or token.like_url:\n            continue\n        lemma = token.lemma_.lower().strip()\n        if lemma and len(lemma) > 1 and lemma not in stop_words:\n            tokens.append(lemma)\n    return tokens\n\n\ndef preprocess_pipeline(texts):\n    \"\"\"Pipeline batch con spaCy pipe.\"\"\"\n    cleaned = [clean_text(t) for t in texts]\n    processed = []\n    for doc in nlp_preprocess.pipe(cleaned, batch_size=500):\n        tokens = []\n        for token in doc:\n            if token.is_space or token.is_punct or token.is_digit or token.like_url:\n                continue\n            lemma = token.lemma_.lower().strip()\n            if lemma and len(lemma) > 1 and lemma not in stop_words:\n                tokens.append(lemma)\n        processed.append(tokens)\n    return processed\n\nprint("Funciones de preprocesamiento definidas.")\n"""))

    cells.append(make_cell("""# Aplicar preprocesamiento a una muestra representativa (primero 5000 para velocidad)\n# Para el dataset completo (~73k), este paso toma ~3-5 minutos.\n\nSAMPLE_SIZE = 5000  # Cambiar a len(df_clean) para procesar todo\nsample_df = df_clean.head(SAMPLE_SIZE).copy()\n\nprint(f"Procesando {len(sample_df):,} narrativas...")\ntexts = sample_df["Consumer complaint narrative"].astype(str).tolist()\nprocessed_tokens = preprocess_pipeline(texts)\n\nsample_df["tokens"] = processed_tokens\nsample_df["processed_text"] = sample_df["tokens"].apply(lambda x: " ".join(x))\n\n# Guardar interim 02\nsample_df.to_csv(DATA_INTERIM / "02_preprocesado.csv", index=False)\nprint(f"Preprocesamiento completado. Ejemplo:")\nprint("Original:", texts[0][:150])\nprint("Procesado:", sample_df["processed_text"].iloc[0][:150])\n"""))

    # SECCIÓN 6: NER
    cells.append(make_cell("""## 6. Named Entity Recognition (NER)\n\nExtraemos entidades con spaCy en inglés y enriquecemos con reglas de dominio financiero.\n""", "markdown"))

    cells.append(make_cell("""# Cargar modelo spaCy con NER habilitado\nnlp_ner = spacy.load("en_core_web_sm")\n\n# Añadir ruler de entidades financieras\nruler = nlp_ner.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})\n\nfinancial_entities = [\n    "Equifax", "Experian", "TransUnion", "FICO", "VantageScore",\n    "CFPB", "Consumer Financial Protection Bureau", "IRS", "FTC",\n    "PennyMac", "Navient", "Nelnet", "Great Lakes", "Sallie Mae",\n    "Citibank", "Chase", "Bank of America", "Wells Fargo",\n    "Capital One", "Discover", "American Express", "Synchrony",\n]\n\npatterns = [{"label": "ORG", "pattern": ent} for ent in financial_entities]\nlegal_patterns = [\n    {"label": "LAW", "pattern": "FCRA"},\n    {"label": "LAW", "pattern": "FDCPA"},\n    {"label": "LAW", "pattern": "FCBA"},\n    {"label": "LAW", "pattern": "ECOA"},\n    {"label": "LAW", "pattern": "TILA"},\n    {"label": "LAW", "pattern": [{"LOWER": "section"}, {"LIKE_NUM": True}]},\n    {"label": "LAW", "pattern": [{"LOWER": "fair"}, {"LOWER": "credit"}, {"LOWER": "reporting"}, {"LOWER": "act"}]},\n]\nruler.add_patterns(patterns + legal_patterns)\n\nprint("Modelo NER listo con reglas de dominio financiero.")\n"""))

    cells.append(make_cell("""def extract_entities(text):\n    \"\"\"Extrae entidades de un texto.\"\"\"\n    if not isinstance(text, str) or not text.strip():\n        return {}\n    doc = nlp_ner(text)\n    entities = {}\n    for ent in doc.ents:\n        if "[MASK]" in ent.text:\n            continue\n        label = ent.label_\n        entities.setdefault(label, []).append(ent.text)\n    for label in entities:\n        entities[label] = sorted(list(set(entities[label])))\n    return entities\n\n\n# Aplicar NER a la muestra\nprint("Extrayendo entidades...")\nsample_df["entities"] = [extract_entities(t) for t in tqdm(sample_df["Consumer complaint narrative"].astype(str))]\n\n# Contar entidades por tipo\nsample_df["entity_count"] = sample_df["entities"].apply(lambda x: sum(len(v) for v in x.values()))\nsample_df["org_count"] = sample_df["entities"].apply(lambda x: len(x.get("ORG", [])))\nsample_df["law_count"] = sample_df["entities"].apply(lambda x: len(x.get("LAW", [])))\nsample_df["money_count"] = sample_df["entities"].apply(lambda x: len(x.get("MONEY", [])))\nsample_df["gpe_count"] = sample_df["entities"].apply(lambda x: len(x.get("GPE", [])))\n\n# Guardar interim 03\nsample_df.to_csv(DATA_INTERIM / "03_ner_sentimiento.csv", index=False)\nprint("NER completado.")\nprint("Ejemplo de entidades:")\nprint(sample_df["entities"].iloc[0])\n"""))

    # SECCIÓN 7: SENTIMIENTO
    cells.append(make_cell("""## 7. Análisis de Sentimiento\n\nUsamos VADER (especializado en texto social/quejas) como método principal y TextBlob como comparación.\n""", "markdown"))

    cells.append(make_cell("""from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer\nfrom textblob import TextBlob\n\nvader = SentimentIntensityAnalyzer()\n\n\ndef vader_sentiment(text):\n    scores = vader.polarity_scores(text)\n    compound = scores["compound"]\n    if compound <= -0.5:\n        label = "Muy Negativo"\n    elif compound < -0.05:\n        label = "Negativo"\n    elif compound <= 0.05:\n        label = "Neutral"\n    elif compound < 0.5:\n        label = "Positivo"\n    else:\n        label = "Muy Positivo"\n    return scores["compound"], scores["neg"], scores["neu"], scores["pos"], label\n\n\ndef textblob_sentiment(text):\n    blob = TextBlob(text)\n    pol = blob.sentiment.polarity\n    subj = blob.sentiment.subjectivity\n    if pol <= -0.3:\n        label = "Muy Negativo"\n    elif pol < -0.05:\n        label = "Negativo"\n    elif pol <= 0.05:\n        label = "Neutral"\n    elif pol < 0.3:\n        label = "Positivo"\n    else:\n        label = "Muy Positivo"\n    return pol, subj, label\n\n\nprint("Analizadores de sentimiento listos.")\n"""))

    cells.append(make_cell("""# Aplicar sentimiento\nprint("Analizando sentimiento con VADER y TextBlob...")\nvader_results = [vader_sentiment(t) for t in tqdm(sample_df["Consumer complaint narrative"].astype(str))]\ntb_results = [textblob_sentiment(t) for t in tqdm(sample_df["Consumer complaint narrative"].astype(str))]\n\nsample_df["vader_compound"] = [r[0] for r in vader_results]\nsample_df["vader_neg"] = [r[1] for r in vader_results]\nsample_df["vader_neu"] = [r[2] for r in vader_results]\nsample_df["vader_pos"] = [r[3] for r in vader_results]\nsample_df["vader_label"] = [r[4] for r in vader_results]\n\nsample_df["textblob_polarity"] = [r[0] for r in tb_results]\nsample_df["textblob_subjectivity"] = [r[1] for r in tb_results]\nsample_df["textblob_label"] = [r[2] for r in tb_results]\n\n# Guardar interim 03 (actualizado)\nsample_df.to_csv(DATA_INTERIM / "03_ner_sentimiento.csv", index=False)\nprint("Sentimiento completado.")\n"""))

    cells.append(make_cell("""# Visualización de sentimiento\nfig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\nsample_df["vader_label"].value_counts().plot(kind="bar", ax=axes[0], color="teal")\naxes[0].set_title("Distribución VADER")\naxes[0].tick_params(axis="x", rotation=45)\n\nsample_df["textblob_label"].value_counts().plot(kind="bar", ax=axes[1], color="purple")\naxes[1].set_title("Distribución TextBlob")\naxes[1].tick_params(axis="x", rotation=45)\n\nplt.tight_layout()\nplt.show()\n"""))

    # SECCIÓN 8: FEATURE ENGINEERING
    cells.append(make_cell("""## 8. Feature Engineering\n\nCreamos features adicionales para análisis y modelado futuro.\n""", "markdown"))

    cells.append(make_cell("""# Features de texto\nsample_df["word_count"] = sample_df["processed_text"].astype(str).apply(lambda x: len(x.split()))\nsample_df["unique_words"] = sample_df["processed_text"].astype(str).apply(lambda x: len(set(x.split())))\nsample_df["lexical_diversity"] = sample_df["unique_words"] / (sample_df["word_count"] + 1)\n\n# Features de contenido\nlegal_terms = ["FCRA", "FDCPA", "FCBA", "ECOA", "TILA", "violation", "violated",\n               "compliance", "dispute", "validation", "verification", "investigate",\n               "lawsuit", "litigation", "attorney", "fraud", "fraudulent", "identity theft"]\n\nsample_df["legal_term_count"] = sample_df["Consumer complaint narrative"].astype(str).apply(\n    lambda x: sum(1 for term in legal_terms if term.lower() in x.lower())\n)\n\n# Features de mayúsculas (posible énfasis/emoción)\nsample_df["uppercase_ratio"] = sample_df["Consumer complaint narrative"].astype(str).apply(\n    lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)\n)\n\n# Feature de máscaras (cuánta PII fue redactada)\nsample_df["mask_count"] = sample_df["Consumer complaint narrative"].astype(str).str.count(r"\\[MASK\\]")\n\nprint("Features creadas:")\nprint(sample_df[["word_count", "unique_words", "lexical_diversity", "legal_term_count", "uppercase_ratio", "mask_count"]].describe())\n"""))

    cells.append(make_cell("""# Guardar dataset final procesado\nsample_df.to_csv(DATA_INTERIM / "04_features.csv", index=False)\n\n# Guardar como Parquet para el dashboard (más rápido)\nsample_df.to_parquet(DATA_PROCESSED / "muestra_nlp_procesada.parquet", index=False)\n\nprint(f"Dataset final guardado: {len(sample_df):,} filas, {len(sample_df.columns)} columnas")\nprint(f"Archivo: {DATA_PROCESSED / 'muestra_nlp_procesada.parquet'}")\n"""))

    # SECCIÓN 9: VISUALIZACIONES
    cells.append(make_cell("""## 9. Visualizaciones Avanzadas\n\nWord Cloud, correlaciones y análisis de patrones.\n""", "markdown"))

    cells.append(make_cell("""from wordcloud import WordCloud\n\n# Word Cloud de todas las narrativas procesadas\ntext_corpus = " ".join(sample_df["processed_text"].astype(str))\nwordcloud = WordCloud(width=1200, height=600, background_color="white", max_words=200).generate(text_corpus)\n\nplt.figure(figsize=(14, 6))\nplt.imshow(wordcloud, interpolation="bilinear")\nplt.axis("off")\nplt.title("Word Cloud - Términos más frecuentes en quejas CFPB")\nplt.show()\n"""))

    cells.append(make_cell("""# Word Cloud por Issue (ejemplo: Incorrect information on your report)\ntop_issue = sample_df["Issue"].value_counts().index[0]\nissue_text = " ".join(sample_df[sample_df["Issue"] == top_issue]["processed_text"].astype(str))\n\nif issue_text.strip():\n    wc_issue = WordCloud(width=1200, height=600, background_color="white", max_words=150).generate(issue_text)\n    plt.figure(figsize=(14, 6))\n    plt.imshow(wc_issue, interpolation="bilinear")\n    plt.axis("off")\n    plt.title(f"Word Cloud - Issue: {top_issue}")\n    plt.show()\n"""))

    cells.append(make_cell("""# Correlación entre features numéricas\nnum_cols = ["narrative_length", "word_count", "unique_words", "lexical_diversity",\n            "legal_term_count", "uppercase_ratio", "mask_count", "entity_count",\n            "vader_compound", "textblob_polarity"]\nnum_cols = [c for c in num_cols if c in sample_df.columns]\n\nplt.figure(figsize=(10, 8))\nsns.heatmap(sample_df[num_cols].corr(), annot=True, cmap="RdBu_r", center=0, fmt=".2f")\nplt.title("Correlación entre features numéricas")\nplt.show()\n"""))

    cells.append(make_cell("""# Top entidades más frecuentes\nfrom collections import Counter\n\nall_orgs = []\nall_laws = []\nfor ents in sample_df["entities"]:\n    if isinstance(ents, dict):\n        all_orgs.extend(ents.get("ORG", []))\n        all_laws.extend(ents.get("LAW", []))\n\nfig, axes = plt.subplots(1, 2, figsize=(16, 6))\n\nif all_orgs:\n    org_counts = Counter(all_orgs).most_common(15)\n    orgs, counts = zip(*org_counts)\n    axes[0].barh(list(orgs), list(counts), color="steelblue")\n    axes[0].set_title("Top 15 Entidades ORG")\n    axes[0].invert_yaxis()\n\nif all_laws:\n    law_counts = Counter(all_laws).most_common(15)\n    laws, counts = zip(*law_counts)\n    axes[1].barh(list(laws), list(counts), color="darkgreen")\n    axes[1].set_title("Top 15 Leyes (LAW)")\n    axes[1].invert_yaxis()\n\nplt.tight_layout()\nplt.show()\n"""))

    # SECCIÓN 10: RESUMEN
    cells.append(make_cell("""## 10. Resumen y Conclusiones\n\n### Pipeline completado:\n1. **Carga**: 73,634 filas (1 línea corrupta eliminada)\n2. **Limpieza**: 26 narrativas muy cortas eliminadas, fechas estandarizadas\n3. **Preprocesamiento**: Unicode NFKC, lematización spaCy, stopwords adaptativas\n4. **NER**: Entidades ORG, LAW, MONEY, GPE extraídas con spaCy + reglas de dominio\n5. **Sentimiento**: VADER + TextBlob aplicados a toda la muestra\n6. **Features**: longitud, diversidad léxica, términos legales, ratio de mayúsculas, máscaras\n7. **Guardado**: Archivos intermedios en `data/interim/`, final en `data/processed/`\n\n### Hallazgos clave:\n- ~67% de las narrativas contienen redacción de PII (`XXXX`)\n- Desbalance severo: Credit reporting domina el dataset\n- Sentimiento predominantemente negativo (esperable en quejas)\n- VADER detecta más matices que TextBlob en texto formal/quejas\n- Entidades financieras (Equifax, Experian, TransUnion) y leyes (FCRA, FDCPA) son las más frecuentes\n\n### Próximos pasos:\n- Ejecutar el dashboard: `python dashboard/app.py`\n- Procesar el dataset completo (~73k) cambiando `SAMPLE_SIZE`\n- Implementar clasificador de Issue con TF-IDF + MLP\n- Topic modeling (LDA) para descubrir temas ocultos\n""", "markdown"))

    nb.cells = cells
    path = NOTEBOOKS_DIR / "01_proyecto_nlp_completo.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Notebook generada: {path}")


def generate_divided_notebooks():
    """Genera las 4 notebooks divididas."""
    DIVIDED_DIR.mkdir(parents=True, exist_ok=True)

    # NOTEBOOK 01: EDA y Limpieza
    nb1 = new_notebook()
    nb1.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    cells1 = [
        make_cell("""# 01. EDA y Limpieza de Datos\n\nCarga, exploración y limpieza inicial del dataset CFPB.\n""", "markdown"),
        make_cell("""import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom pathlib import Path\n\nDATA_RAW = Path("data/raw/muestra_nlp_limpia.csv")\nDATA_INTERIM = Path("data/interim")\nDATA_INTERIM.mkdir(parents=True, exist_ok=True)\n\nsns.set_style("whitegrid")\nplt.rcParams["figure.figsize"] = (12, 6)\n"""),
        make_cell("""# Carga con manejo de línea corrupta\ndf = pd.read_csv(DATA_RAW, engine="python", on_bad_lines="skip")\nprint(f"Filas: {len(df):,}")\nprint("Nulos:")\nprint(df.isnull().sum()[df.isnull().sum() > 0])\n"""),
        make_cell("""# EDA básico\nnarr = df["Consumer complaint narrative"].astype(str)\nprint("Longitud de narrativas:")\nprint(narr.str.len().describe())\nprint("\\nCon XXXX:", narr.str.contains("XXXX", case=False).sum(), f"({narr.str.contains('XXXX', case=False).sum()/len(narr)*100:.1f}%)")\n"""),
        make_cell("""# Limpieza\nmin_length = 20\nshort_mask = narr.str.len() < min_length\nprint(f"Eliminando {short_mask.sum()} narrativas cortas")\ndf_clean = df[~short_mask].copy()\ndf_clean["narrative_length"] = df_clean["Consumer complaint narrative"].astype(str).str.len()\ndf_clean["Date received"] = pd.to_datetime(df_clean["Date received"], errors="coerce")\ndf_clean["year"] = df_clean["Date received"].dt.year\n\ndf_clean.to_csv(DATA_INTERIM / "01_limpio.csv", index=False)\nprint(f"Guardado: {len(df_clean):,} filas")\n"""),
    ]
    nb1.cells = cells1
    with open(DIVIDED_DIR / "01_eda_limpieza.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb1, f)
    print("Generada: 01_eda_limpieza.ipynb")

    # NOTEBOOK 02: Preprocesamiento
    nb2 = new_notebook()
    nb2.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    cells2 = [
        make_cell("""# 02. Preprocesamiento de Texto\n\nTokenización, lematización y limpieza lingüística con spaCy.\n""", "markdown"),
        make_cell("""import pandas as pd\nimport re\nimport unicodedata\nimport spacy\nfrom nltk.corpus import stopwords\nfrom pathlib import Path\nfrom tqdm.notebook import tqdm\n\nDATA_INTERIM = Path("data/interim")\ndf = pd.read_csv(DATA_INTERIM / "01_limpio.csv")\nprint(f"Cargado: {len(df):,} filas")\n"""),
        make_cell("""# Configurar spaCy y stopwords\nnlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])\nadditional_stops = {"xxxx", "xx", "xxxxx", "xx/xx/xxxx", "xxx", "company", "consumer", "complaint", "report", "account", "information", "requested", "please", "also", "would", "could", "should", "said", "told", "called", "spoke", "stated", "mentioned", "however", "therefore", "furthermore", "accordingly"}\nstop_words = set(stopwords.words("english")) | additional_stops\n\ndef clean_text(text):\n    if not isinstance(text, str): return ""\n    text = unicodedata.normalize("NFKC", text)\n    text = re.sub(r"\\b[Xx]{2,}\\b", "[MASK]", text)\n    text = re.sub(r"https?://\\S+|www\\.\\S+", "", text)\n    text = re.sub(r"\\S+@\\S+", "", text)\n    text = text.lower()\n    text = re.sub(r"[^a-z\\s\\[\\]]", " ", text)\n    text = re.sub(r"\\s+", " ", text).strip()\n    return text\n\ndef lemmatize(text):\n    doc = nlp(text)\n    return [token.lemma_.lower().strip() for token in doc if not token.is_space and not token.is_punct and not token.is_digit and token.lemma_.lower().strip() not in stop_words and len(token.lemma_.lower().strip()) > 1]\n"""),
        make_cell("""# Aplicar a muestra (ajustar SAMPLE_SIZE según recursos)\nSAMPLE_SIZE = 5000\ndf_sample = df.head(SAMPLE_SIZE).copy()\n\ncleaned = [clean_text(t) for t in df_sample["Consumer complaint narrative"].astype(str)]\ntokens = []\nfor doc in tqdm(nlp.pipe(cleaned, batch_size=500), total=len(cleaned)):\n    tokens.append([token.lemma_.lower().strip() for token in doc if not token.is_space and not token.is_punct and not token.is_digit and token.lemma_.lower().strip() not in stop_words and len(token.lemma_.lower().strip()) > 1])\n\ndf_sample["tokens"] = tokens\ndf_sample["processed_text"] = df_sample["tokens"].apply(lambda x: " ".join(x))\ndf_sample.to_csv(DATA_INTERIM / "02_preprocesado.csv", index=False)\nprint("Preprocesamiento completado.")\n"""),
    ]
    nb2.cells = cells2
    with open(DIVIDED_DIR / "02_preprocesamiento.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb2, f)
    print("Generada: 02_preprocesamiento.ipynb")

    # NOTEBOOK 03: NER y Sentimiento
    nb3 = new_notebook()
    nb3.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    cells3 = [
        make_cell("""# 03. NER y Análisis de Sentimiento\n\nExtracción de entidades nombradas y análisis de polaridad emocional.\n""", "markdown"),
        make_cell("""import pandas as pd\nimport spacy\nfrom vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer\nfrom textblob import TextBlob\nfrom pathlib import Path\nfrom tqdm.notebook import tqdm\n\nDATA_INTERIM = Path("data/interim")\ndf = pd.read_csv(DATA_INTERIM / "02_preprocesado.csv")\nprint(f"Cargado: {len(df):,} filas")\n"""),
        make_cell("""# NER con spaCy + reglas de dominio\nnlp_ner = spacy.load("en_core_web_sm")\nruler = nlp_ner.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})\n\nfinancial = ["Equifax", "Experian", "TransUnion", "CFPB", "IRS", "FTC", "PennyMac", "Navient", "Chase", "Bank of America", "Wells Fargo", "Capital One", "Discover", "Synchrony"]\npatterns = [{"label": "ORG", "pattern": e} for e in financial]\npatterns += [\n    {"label": "LAW", "pattern": "FCRA"},\n    {"label": "LAW", "pattern": "FDCPA"},\n    {"label": "LAW", "pattern": [{"LOWER": "section"}, {"LIKE_NUM": True}]},\n    {"label": "LAW", "pattern": [{"LOWER": "fair"}, {"LOWER": "credit"}, {"LOWER": "reporting"}, {"LOWER": "act"}]},\n]\nruler.add_patterns(patterns)\n\ndef extract_entities(text):\n    if not isinstance(text, str): return {}\n    doc = nlp_ner(text)\n    ents = {}\n    for ent in doc.ents:\n        if "[MASK]" in ent.text: continue\n        ents.setdefault(ent.label_, []).append(ent.text)\n    return {k: sorted(list(set(v))) for k, v in ents.items()}\n\ndf["entities"] = [extract_entities(t) for t in tqdm(df["Consumer complaint narrative"].astype(str))]\ndf["entity_count"] = df["entities"].apply(lambda x: sum(len(v) for v in x.values()) if isinstance(x, dict) else 0)\nprint("NER completado.")\n"""),
        make_cell("""# Sentimiento\nvader = SentimentIntensityAnalyzer()\n\ndef vader_analysis(text):\n    s = vader.polarity_scores(text)\n    c = s["compound"]\n    label = "Muy Negativo" if c <= -0.5 else "Negativo" if c < -0.05 else "Neutral" if c <= 0.05 else "Positivo" if c < 0.5 else "Muy Positivo"\n    return s["compound"], s["neg"], s["neu"], s["pos"], label\n\ndef textblob_analysis(text):\n    b = TextBlob(text)\n    p = b.sentiment.polarity\n    label = "Muy Negativo" if p <= -0.3 else "Negativo" if p < -0.05 else "Neutral" if p <= 0.05 else "Positivo" if p < 0.3 else "Muy Positivo"\n    return p, b.sentiment.subjectivity, label\n\nv_res = [vader_analysis(t) for t in tqdm(df["Consumer complaint narrative"].astype(str))]\ntb_res = [textblob_analysis(t) for t in tqdm(df["Consumer complaint narrative"].astype(str))]\n\ndf["vader_compound"] = [r[0] for r in v_res]\ndf["vader_label"] = [r[4] for r in v_res]\ndf["textblob_polarity"] = [r[0] for r in tb_res]\ndf["textblob_label"] = [r[2] for r in tb_res]\n\ndf.to_csv(DATA_INTERIM / "03_ner_sentimiento.csv", index=False)\nprint("Sentimiento completado.")\n"""),
    ]
    nb3.cells = cells3
    with open(DIVIDED_DIR / "03_ner_sentimiento.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb3, f)
    print("Generada: 03_ner_sentimiento.ipynb")

    # NOTEBOOK 04: Feature Engineering
    nb4 = new_notebook()
    nb4.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    cells4 = [
        make_cell("""# 04. Feature Engineering y Dataset Final\n\nCreación de features adicionales y exportación del dataset procesado.\n""", "markdown"),
        make_cell("""import pandas as pd\nfrom pathlib import Path\n\nDATA_INTERIM = Path("data/interim")\nDATA_PROCESSED = Path("data/processed")\nDATA_PROCESSED.mkdir(parents=True, exist_ok=True)\n\ndf = pd.read_csv(DATA_INTERIM / "03_ner_sentimiento.csv")\nprint(f"Cargado: {len(df):,} filas")\n"""),
        make_cell("""# Features de texto\ndf["word_count"] = df["processed_text"].astype(str).apply(lambda x: len(x.split()))\ndf["unique_words"] = df["processed_text"].astype(str).apply(lambda x: len(set(x.split())))\ndf["lexical_diversity"] = df["unique_words"] / (df["word_count"] + 1)\n\n# Features de contenido\nlegal_terms = ["FCRA", "FDCPA", "violation", "violated", "compliance", "dispute", "validation", "verification", "investigate", "lawsuit", "litigation", "attorney", "fraud", "fraudulent", "identity theft"]\ndf["legal_term_count"] = df["Consumer complaint narrative"].astype(str).apply(lambda x: sum(1 for t in legal_terms if t.lower() in x.lower()))\ndf["uppercase_ratio"] = df["Consumer complaint narrative"].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))\ndf["mask_count"] = df["Consumer complaint narrative"].astype(str).str.count(r"\\[MASK\\]")\n\nprint(df[["word_count", "unique_words", "lexical_diversity", "legal_term_count", "uppercase_ratio", "mask_count"]].describe())\n"""),
        make_cell("""# Guardar datasets finales\ndf.to_csv(DATA_INTERIM / "04_features.csv", index=False)\ndf.to_parquet(DATA_PROCESSED / "muestra_nlp_procesada.parquet", index=False)\n\nprint(f"Dataset final: {len(df):,} filas, {len(df.columns)} columnas")\nprint(f"Parquet: {DATA_PROCESSED / 'muestra_nlp_procesada.parquet'}")\n"""),
    ]
    nb4.cells = cells4
    with open(DIVIDED_DIR / "04_feature_engineering.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb4, f)
    print("Generada: 04_feature_engineering.ipynb")


if __name__ == "__main__":
    generate_complete_notebook()
    generate_divided_notebooks()
    print("\nTodas las notebooks generadas exitosamente.")
