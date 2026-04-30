# Informe Técnico — Proyecto NLP: Análisis de Quejas CFPB

**Autor:** William Moncada  
**Fecha:** Abril 2026  
**Asignatura:** Procesamiento de Lenguaje Natural

---

## 1. Introducción

Este documento describe las decisiones técnicas, el análisis de resultados, las limitaciones identificadas y las propuestas de mejora del proyecto de NLP sobre quejas del Consumer Financial Protection Bureau (CFPB).

## 2. Decisiones de diseño y justificación

### 2.1 Idioma del pipeline
El dataset es 100% en inglés (quejas de consumidores estadounidenses). Por coherencia técnica, todo el pipeline se implementó en inglés usando modelos spaCy `en_core_web_sm`. No se forzó soporte multilingüe ni se introdujeron entidades colombianas que no aparecen en la data.

### 2.2 Elección de spaCy sobre NLTK para lematización
- **spaCy** usa POS tagging contextual: "saw" como verbo → "see", "saw" como sustantivo → "saw".
- **NLTK WordNetLemmatizer** requiere especificar POS manualmente y es más lento en batch.
- Para 73k narrativas, spaCy `pipe()` con batch_size=500 ofrece throughput superior.

### 2.3 Sentimiento: VADER como principal
- **VADER** está calibrado para texto social y quejas. Maneja mayúsculas (énfasis), puntuación repetida ("!!!") y negaciones ("not good").
- **TextBlob** usa PatternAnalyzer basado en diccionarios. Es más rápido pero menos robusto para texto formal con términos legales.
- En las pruebas, VADER clasificó correctamente narrativas con lenguaje legal agresivo ("violation", "fraud", "lawsuit") como negativas, mientras que TextBlob las tendió a neutralizar.

### 2.4 Stopwords adaptativas
Se combinó la lista de NLTK con términos de dominio que no aportan poder discriminativo: "company", "consumer", "complaint", "report", "account". Crucialmente, se **excluyeron** de la lista los modificadores de polaridad: "no", "not", "never", "very", "more".

### 2.5 Representación de entidades
Las entidades se almacenan como diccionarios JSON por fila: `{"ORG": ["Equifax"], "LAW": ["FCRA"]}`. Esto permite análisis agregados sin perder la trazabilidad por queja.

## 3. Análisis de resultados

### 3.1 Calidad del dataset
- **Línea corrupta**: 1 fila al final del CSV (EOF inside string) eliminada.
- **Narrativas cortas**: 26 registros (<20 chars) sin valor semántico eliminados.
- **Valores nulos masivos**: `Tags` (90.6%), `Consumer disputed?` (95.6%). Descartados del análisis.
- **Desbalance de clases**: 66% Credit reporting. Impacta cualquier modelo de clasificación supervisada.

### 3.2 Distribución de sentimiento (muestra de 5,000)
| Categoría VADER | % |
|:---|:---|
| Muy Negativo | ~35% |
| Negativo | ~40% |
| Neutral | ~20% |
| Positivo | ~4% |
| Muy Positivo | ~1% |

Esto confirma la naturaleza adversarial del dataset. Las quejas raramente son neutrales o positivas.

### 3.3 Entidades más frecuentes
- **ORG**: Equifax, Experian, TransUnion, Capital One, Chase, Bank of America
- **LAW**: FCRA, FDCPA, Section 609, Fair Credit Reporting Act
- **GPE**: CA, TX, FL, NY (estados con más quejas)

### 3.4 Correlaciones observadas
- `vader_compound` vs `uppercase_ratio`: correlación negativa débil (-0.15). Mayúsculas indican énfasis/emoción negativa.
- `legal_term_count` vs `narrative_length`: correlación positiva moderada (0.42). Quejas más largas tienden a citar más leyes.
- `mask_count` vs `narrative_length`: correlación positiva (0.38). Narrativas largas contienen más PII redactada.

## 4. Limitaciones identificadas

1. **Muestra limitada**: El pipeline se ejecutó sobre 5,000 filas por restricciones de tiempo. El dataset completo (73k) tomaría ~15-20 minutos en procesar.
2. **Masking masivo**: 67% de narrativas contienen `XXXX`. Esto elimina información semántica valiosa (nombres, fechas, montos).
3. **Desbalance de clases**: Credit reporting domina. Un clasificador naive siempre predeciría esa clase con 66% accuracy.
4. **NER limitado**: spaCy `en_core_web_sm` no detecta todas las variantes de nombres de compañías. El entity ruler mitiga esto parcialmente.
5. **Sin modelado de tópicos**: No se implementó LDA o NMF para descubrir temas ocultos.
6. **Memory usage**: spaCy mantiene el modelo en memoria. Para escalar a millones de registros, se requeriría procesamiento por chunks o GPU.

## 5. Propuestas de mejora y futuras direcciones

### Corto plazo
- **Procesar dataset completo**: Cambiar `SAMPLE_SIZE = None` en `scripts/run_pipeline.py`.
- **Clasificador de Issue**: Entrenar TF-IDF + MLP/Random Forest para predecir la categoría de Issue desde la narrativa.
- **Topic Modeling**: Implementar LDA o BERTopic para descubrir subtemas dentro de cada Issue.

### Mediano plazo
- **Embeddings contextuales**: Migrar de TF-IDF a sentence-transformers (`all-MiniLM-L6-v2`) para capturar semántica más profunda.
- **Dashboard avanzado**: Añadir mapa coroplético de EEUU, serie temporal con forecast, y análisis de bigramas/trigramas.
- **Pipeline de producción**: Dockerizar el proyecto y exponer la API con FastAPI para inferencia en tiempo real.

### Largo plazo
- **Fine-tuning de LLM**: Adaptar un modelo tipo BERT o GPT con fine-tuning sobre este dataset para tareas de clasificación, resumen y extracción de entidades.
- **Detección de anomalías**: Identificar quejas atípicas o patrones de fraude sistemático usando clustering sobre embeddings.
- **Multilingüe real**: Si en el futuro se obtiene data en español, entrenar un pipeline paralelo con `es_core_news_md`.

## 6. Referencias

- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
- Honnibal, M. & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.
- Consumer Financial Protection Bureau. (2024). Consumer Complaint Database.
- Jurafsky, D. & Martin, J.H. (2023). Speech and Language Processing (3rd ed.).
