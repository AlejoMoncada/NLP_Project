# Elevator Pitch — Dashboard NLP: Quejas Financieras CFPB

---

## ¿Cuál es el problema?

Cada año, millones de consumidores presentan quejas ante la Oficina de Protección Financiera del Consumidor (CFPB) de Estados Unidos. Estas quejas contienen narrativas escritas en lenguaje natural: relatos de fraudes, errores crediticios, prácticas abusivas y violaciones de derechos.

> **El problema:** esas 73 600 narrativas son texto no estructurado. Sin procesar, son datos muertos.

---

## ¿Qué construimos?

Un **pipeline completo de Procesamiento de Lenguaje Natural (PLN)** que transforma esas narrativas en inteligencia accionable, visualizada en un **dashboard interactivo** con enfoque en storytelling.

---

## El Pipeline en 5 etapas

```
Texto crudo  →  Limpieza  →  Preprocesamiento  →  NER + Sentimiento  →  Features
  73 635       73 609         Tokens / Lemas       Entidades / VADER     43 variables
```

| Etapa | Técnica | Resultado |
|-------|---------|-----------|
| Limpieza | Regex, NFKC, enmascaramiento PII | Texto normalizado |
| Tokenización & Lematización | spaCy `en_core_web_sm` | Tokens semánticos |
| NER | spaCy + Entity Ruler | ORG, LAW, GPE, MONEY |
| Sentimiento | VADER + TextBlob | Score −1 a +1 por narrativa |
| Feature Engineering | Lexical, domain, entidades | 43 features por registro |

---

## Lo que el dashboard revela

### 📊 Panorama
- **75% de las quejas** expresan sentimiento negativo o muy negativo
- El sector de **reportes crediticios** concentra el 66% de los casos
- **Equifax, Experian y TransUnion** son las entidades más mencionadas

### ⚖️ Hallazgos legales
- La **FCRA** (Fair Credit Reporting Act) aparece en 1 de cada 6 narrativas
- La **FDCPA** es la segunda ley más citada: prácticas de cobro abusivas
- Mayor longitud de narrativa correlaciona con mayor número de términos legales (r = 0.42)

### 😟 Sentimiento
- VADER supera a TextBlob en precisión sobre texto de quejas formales con lenguaje legal
- Los consumidores con menciones a regulaciones escriben narrativas **más largas y más negativas**

---

## Propuesta de valor

| Para… | El dashboard permite… |
|-------|-----------------------|
| Reguladores | Identificar productos y compañías con mayor volumen de insatisfacción |
| Analistas financieros | Detectar patrones legales emergentes en tiempo real |
| Instituciones | Priorizar respuesta según severidad del sentimiento y entidades citadas |
| Investigadores NLP | Explorar un pipeline end-to-end con datos reales y complejos |

---

## Stack tecnológico

```
Python 3.13  ·  spaCy  ·  VADER  ·  TextBlob  ·  pandas  ·  Dash / Plotly
WordCloud  ·  scikit-learn  ·  pyarrow (Parquet)  ·  NLTK
```

---

## En 30 segundos

> Tomamos **73 600 quejas financieras reales**, las procesamos con un pipeline de PLN de 5 etapas —limpieza, lematización, NER y análisis de sentimiento— y construimos un dashboard interactivo que permite a cualquier analista explorar **quién se queja, de qué, con qué emociones y citando qué leyes**, todo con filtros en tiempo real.
>
> El resultado: datos que antes eran texto inerte ahora son inteligencia accionable.

---

*Proyecto académico — Maestría en Inteligencia Artificial · PLN · 2026*
