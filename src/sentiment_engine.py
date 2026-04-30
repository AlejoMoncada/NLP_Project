"""
Módulo de análisis de sentimiento para quejas CFPB.
Usa VADER como principal y TextBlob como referencia comparativa.
"""
from typing import Dict

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Inicializar analizadores
_vader = SentimentIntensityAnalyzer()


def analyze_vader(text: str) -> Dict[str, float]:
    """
    Analiza sentimiento con VADER.
    Retorna dict con neg, neu, pos, compound.
    """
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    scores = _vader.polarity_scores(text)
    return scores


def vader_label(compound: float) -> str:
    """
    Clasifica el sentimiento basado en el score compound de VADER.
    """
    if compound <= -0.5:
        return "Muy Negativo"
    elif compound < -0.05:
        return "Negativo"
    elif compound <= 0.05:
        return "Neutral"
    elif compound < 0.5:
        return "Positivo"
    else:
        return "Muy Positivo"


def analyze_textblob(text: str) -> Dict[str, float]:
    """
    Analiza sentimiento con TextBlob.
    Retorna polaridad (-1 a 1) y subjetividad (0 a 1).
    """
    if not isinstance(text, str) or not text.strip():
        return {"polarity": 0.0, "subjectivity": 0.0}
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}


def textblob_label(polarity: float) -> str:
    """Clasifica sentimiento basado en polaridad de TextBlob."""
    if polarity <= -0.3:
        return "Muy Negativo"
    elif polarity < -0.05:
        return "Negativo"
    elif polarity <= 0.05:
        return "Neutral"
    elif polarity < 0.3:
        return "Positivo"
    else:
        return "Muy Positivo"


def full_sentiment_analysis(text: str) -> Dict:
    """
    Análisis completo combinando VADER + TextBlob.
    Retorna dict consolidado.
    """
    vader_scores = analyze_vader(text)
    tb_scores = analyze_textblob(text)

    return {
        "vader_compound": vader_scores["compound"],
        "vader_neg": vader_scores["neg"],
        "vader_neu": vader_scores["neu"],
        "vader_pos": vader_scores["pos"],
        "vader_label": vader_label(vader_scores["compound"]),
        "textblob_polarity": tb_scores["polarity"],
        "textblob_subjectivity": tb_scores["subjectivity"],
        "textblob_label": textblob_label(tb_scores["polarity"]),
    }
