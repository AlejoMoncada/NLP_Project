# Proyecto NLP: Análisis de Quejas CFPB

**Autor:** Alejandro Moncada - Noel Perez - Julian Mendez 
**Asignatura:** Procesamiento de Lenguaje Natural  
**Dataset:** `muestra_nlp_limpia.csv` (Consumer Financial Protection Bureau)

---

## Descripción

Este proyecto implementa un pipeline completo de Procesamiento de Lenguaje Natural (NLP) sobre un dataset de ~73,600 quejas financieras del CFPB. El objetivo es analizar patrones en las narrativas de los consumidores mediante técnicas de limpieza, preprocesamiento, Named Entity Recognition (NER), análisis de sentimiento y visualización interactiva.

## Estructura del proyecto

```
PLN/
├── README.md                              # Este archivo
├── requirements.txt                       # Dependencias
├── .gitignore
├── data/
│   ├── raw/muestra_nlp_limpia.csv         # Dataset original
│   ├── interim/                           # Archivos intermedios del pipeline
│   └── processed/muestra_nlp_procesada.parquet  # Dataset final
├── notebooks/
│   ├── 01_proyecto_nlp_completo.ipynb     # Notebook única con todo el pipeline
│   └── 02_proyecto_nlp_dividido/          # Notebooks modulares
│       ├── 01_eda_limpieza.ipynb
│       ├── 02_preprocesamiento.ipynb
│       ├── 03_ner_sentimiento.ipynb
│       └── 04_feature_engineering.ipynb
├── src/                                   # Módulos Python reutilizables
│   ├── config.py
│   ├── utils.py
│   ├── preprocessing.py
│   ├── ner_engine.py
│   └── sentiment_engine.py
├── dashboard/                             # Dashboard interactivo con Dash
│   └── app.py
├── docs/
│   └── informe_tecnico.md                 # Documentación técnica detallada
└── scripts/
    ├── generate_notebooks.py              # Generador de notebooks
    ├── run_pipeline.py                    # Ejecutor del pipeline NLP
    └── run_dashboard.py                   # Lanzador del dashboard
```

## Instalación

```bash
# 1. Crear entorno virtual (opcional pero recomendado)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar modelo spaCy en inglés
python -m spacy download en_core_web_sm

# 4. Descargar datos NLTC
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('wordnet')"
```

## Uso

### Opción 1: Ejecutar el pipeline completo

```bash
python scripts/run_pipeline.py
```

Esto genera todos los archivos intermedios y el dataset final procesado.

### Opción 2: Usar las notebooks

Abre `notebooks/01_proyecto_nlp_completo.ipynb` en Jupyter/JupyterLab para ejecutar el pipeline paso a paso con visualizaciones inline.

También puedes usar las notebooks divididas en `notebooks/02_proyecto_nlp_dividido/`.

### Opción 3: Dashboard interactivo

```bash
python scripts/run_dashboard.py
```

Luego abre tu navegador en: **http://127.0.0.1:8050**

El dashboard incluye:
- Filtros por Producto, Issue, Compañía, Estado, Sentimiento y Año
- **Visión General**: KPIs, evolución temporal, top issues, distribución de sentimiento, mapa de estados
- **Detalle y Patrones**: Word Cloud interactivo, top entidades, tabla de narrativas destacadas

## Pipeline de procesamiento

| Paso | Descripción | Output |
|:---|:---|:---|
| 1. Carga | Lectura CSV con manejo de línea corrupta | `00_raw_fixed.csv` |
| 2. Limpieza | Eliminar narrativas <20 chars, estandarizar fechas | `01_limpio.csv` |
| 3. Preprocesamiento | Unicode NFKC, lematización spaCy, stopwords | `02_preprocesado.csv` |
| 4. NER | Entidades ORG, LAW, MONEY, GPE con reglas de dominio | `03_ner_sentimiento.csv` |
| 5. Sentimiento | VADER + TextBlob con categorización | `03_ner_sentimiento.csv` |
| 6. Features | Longitud, diversidad léxica, términos legales, mayúsculas | `04_features.csv` + `.parquet` |

## Hallazgos principales

- **67.4%** de las narrativas contienen redacción de PII (`XXXX`)
- **Desbalance severo**: Credit reporting domina ~66% del dataset
- Sentimiento **predominantemente negativo** (esperable en quejas)
- **VADER** detecta más matices que TextBlob en texto formal/quejas
- Entidades más frecuentes: Equifax, Experian, TransUnion, FCRA, FDCPA

## Decisiones de diseño

- **Idioma**: Inglés (coherente con dataset CFPB de EEUU)
- **Lematización**: spaCy `en_core_web_sm` (precisión semántica > velocidad)
- **Sentimiento**: VADER como principal (especializado en texto social/quejas), TextBlob como referencia
- **Stopwords**: NLTK + adaptativas de dominio financiero (excluyendo modificadores de polaridad)
- **Dataset final**: Parquet para carga rápida en el dashboard

## Tecnologías

- **Python 3.13**
- **spaCy** (NLP, NER, lematización)
- **pandas, polars** (manejo de datos)
- **VADER, TextBlob** (análisis de sentimiento)
- **Dash + Plotly** (dashboard interactivo)
- **WordCloud** (visualización de términos)
- **scikit-learn** (feature engineering)

## Licencia

Proyecto académico para la asignatura de Procesamiento de Lenguaje Natural.
