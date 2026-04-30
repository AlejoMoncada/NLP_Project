"""
Configuración central del proyecto NLP de Quejas CFPB.
"""
import os
from pathlib import Path

# Rutas base
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Archivos
RAW_CSV = RAW_DIR / "muestra_nlp_limpia.csv"
INTERIM_01 = INTERIM_DIR / "01_limpio.csv"
INTERIM_02 = INTERIM_DIR / "02_preprocesado.csv"
INTERIM_03 = INTERIM_DIR / "03_ner_sentimiento.csv"
INTERIM_04 = INTERIM_DIR / "04_features.csv"
PROCESSED_PARQUET = PROCESSED_DIR / "muestra_nlp_procesada.parquet"

# Columnas relevantes
NARRATIVE_COL = "Consumer complaint narrative"
DATE_COL = "Date received"
PRODUCT_COL = "Product"
ISSUE_COL = "Issue"
SUB_ISSUE_COL = "Sub-issue"
COMPANY_COL = "Company"
STATE_COL = "State"
RESPONSE_COL = "Company response to consumer"
TIMELY_COL = "Timely response?"

# Stopwords adaptativas al dominio financiero/quejas
# Se excluyen modificadores de polaridad que NLTK incluye
ADDITIONAL_STOPWORDS = {
    "xxxx", "xx", "xxxxx", "xx/xx/xxxx", "xxx",
    "company", "consumer", "complaint", "report",
    "account", "information", "requested", "please",
    "also", "would", "could", "should", "said",
    "told", "called", "spoke", "stated", "mentioned",
    "however", "therefore", "furthermore", "accordingly",
}

# Términos legales relevantes para feature engineering
LEGAL_TERMS = [
    "FCRA", "FDCPA", "FCBA", "ECOA", "TILA",
    "Section 609", "Section 611", "Section 1681",
    "Fair Credit Reporting Act", "Fair Debt Collection Practices Act",
    "violation", "violated", "compliance", "non-compliance",
    "dispute", "validation", "verification", "investigate",
    "lawsuit", "litigation", "attorney", "legal action",
    "fraud", "fraudulent", "identity theft",
]

# Entidades financieras específicas para enriquecer NER
FINANCIAL_ENTITIES = [
    "Equifax", "Experian", "TransUnion",
    "FICO", "VantageScore",
    "CFPB", "Consumer Financial Protection Bureau",
    "IRS", "FTC", "Federal Trade Commission",
    "Department of Education",
    "PennyMac", "Navient", "Nelnet", "Great Lakes",
    "Sallie Mae", "Citibank", "Chase", "Bank of America",
    "Wells Fargo", "Capital One", "Discover", "American Express",
    "Synchrony", "Comenity", "Portfolio Recovery",
]

# Categorías de respuesta para análisis
RESPONSE_CATEGORIES = {
    "Closed with explanation": "explanation",
    "Closed with non-monetary relief": "relief_non_monetary",
    "Closed with monetary relief": "relief_monetary",
    "Untimely response": "untimely",
    "Closed": "closed",
    "In progress": "in_progress",
}
