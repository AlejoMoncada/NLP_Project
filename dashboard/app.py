"""
Dashboard interactivo NLP — Análisis de Quejas Financieras CFPB
Storytelling: Panorama → Pipeline NLP → Sentimiento → Entidades & Leyes
Ejecutar: python scripts/run_dashboard.py
"""
import json
from collections import Counter
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from wordcloud import WordCloud

# ══════════════════════════════════════════════════════════════════
#  CARGA Y NORMALIZACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════
DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "processed" / "muestra_nlp_procesada.parquet"
)

try:
    df = pd.read_parquet(DATA_PATH)
    print(f"[OK] Dataset cargado: {len(df):,} filas × {df.shape[1]} columnas")
except Exception as e:
    print(f"[ERROR] {e}")
    df = pd.DataFrame()


def _ent_to_list(v):
    """Convierte numpy.ndarray / None / list a lista Python limpia."""
    if v is None:
        return []
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, list):
        return v
    return []


def _normalize_entity_dict(d):
    """Normaliza un dict de entidades: valores → listas Python."""
    if not isinstance(d, dict):
        return {}
    return {k: _ent_to_list(v) for k, v in d.items()}


# Normalizar entidades al cargar (una sola vez)
if "entities" in df.columns:
    df["entities"] = df["entities"].apply(_normalize_entity_dict)

# ── Limpiar processed_text: eliminar token [MASK] y nulls ─────────
WORD_STOPWORDS = {
    "mask", "xxxx", "x", "xx", "xxx", "xxxxxxxxxxx",
}

# ── Limpiar entidades ORG (ruido / falsos positivos) ──────────────
ORG_NOISE = {
    "XXXX", "U.S.C", "U.S.C 1681", "USC", "CFPB",
    "Consumer Financial Protection Bureau",
    "the Federal Trade Commission",
    "NEVER", "Sections 609", "Social Security",
    "Federal", "United States",
}

# ── Normalizar LAW (fusionar variantes del mismo concepto) ─────────
LAW_NORMALIZE = {
    "Fair Credit Reporting Act":          "FCRA",
    "Fair Credit Reporting act":          "FCRA",
    "fair credit reporting act":          "FCRA",
    "the Fair Debt Collection Practices Act": "FDCPA",
    "Fair Debt Collection Practices Act": "FDCPA",
    "Section 2":                          "Sec. 2",
    "section 2":                          "Sec. 2",
    "section 602":                        "Sec. 602",
    "Section 602":                        "Sec. 602",
    "section 604":                        "Sec. 604",
    "Section 604":                        "Sec. 604",
    "section 1637":                       "Sec. 1637",
}


def _clean_law(name):
    return LAW_NORMALIZE.get(name, name)


# ══════════════════════════════════════════════════════════════════
#  CONSTANTES DE DISEÑO
# ══════════════════════════════════════════════════════════════════
PALETTE = {
    "Muy Negativo": "#922b21",
    "Negativo":     "#e74c3c",
    "Neutral":      "#f4d03f",
    "Positivo":     "#58d68d",
    "Muy Positivo": "#1e8449",
}
SENT_ORDER = ["Muy Negativo", "Negativo", "Neutral", "Positivo", "Muy Positivo"]

PRIMARY    = "#1a3a5c"
ACCENT     = "#2e86c1"
BG_LIGHT   = "#f0f4f8"
BG_CARD    = "#ffffff"
TEXT_MUTED = "#5d6d7e"
FONT       = "Inter, Segoe UI, Arial, sans-serif"
TMPL       = "plotly_white"

# ── Opciones de filtros ────────────────────────────────────────────
def _opts(col):
    if col not in df.columns:
        return []
    vals = sorted(df[col].dropna().unique())
    return [{"label": v, "value": v} for v in vals]


products  = _opts("Product")
issues    = _opts("Issue")
states    = _opts("State")
companies = _opts("Company")
sent_opts = [{"label": s, "value": s} for s in SENT_ORDER
             if "vader_label" in df.columns]
years     = sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else []

FILTER_IDS = ["f-product", "f-issue", "f-company", "f-state",
              "f-sentiment", "f-year-from", "f-year-to"]


# ══════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "NLP · Quejas CFPB"

# ── Helpers de layout ──────────────────────────────────────────────
def _card(value, label, color=ACCENT, icon=""):
    return html.Div([
        html.Div(icon, style={"fontSize": "1.5rem", "marginBottom": "4px"}),
        html.Div(str(value), style={"fontSize": "1.6rem", "fontWeight": "800", "color": color}),
        html.Div(label, style={"fontSize": ".75rem", "color": TEXT_MUTED, "marginTop": "2px"}),
    ], style={
        "background": BG_CARD, "borderRadius": "12px", "padding": "18px 16px",
        "boxShadow": "0 2px 10px rgba(0,0,0,.07)", "textAlign": "center",
        "flex": "1", "minWidth": "130px", "borderTop": f"4px solid {color}",
    })


def _row(*children, gap="16px"):
    return html.Div(list(children),
                    style={"display": "flex", "flexWrap": "wrap",
                           "gap": gap, "marginBottom": "16px"})


def _box(*children, flex="1", min_width="300px"):
    return html.Div(list(children), style={
        "background": BG_CARD, "borderRadius": "12px", "padding": "16px",
        "boxShadow": "0 2px 8px rgba(0,0,0,.06)", "flex": flex, "minWidth": min_width,
    })


# ── Header ────────────────────────────────────────────────────────
HEADER = html.Div([
    html.Div([
        html.H1("Análisis NLP de Quejas Financieras",
                style={"margin": "0", "fontSize": "1.7rem", "fontWeight": "700", "color": "#fff"}),
        html.P("Consumer Financial Protection Bureau · 73 608 narrativas · Pipeline PLN completo",
               style={"margin": "4px 0 0", "color": "rgba(255,255,255,.75)", "fontSize": ".88rem"}),
    ], style={"flex": "1"}),
    html.Div([
        html.Span(t, style={"background": "rgba(255,255,255,.18)", "padding": "4px 12px",
                             "borderRadius": "20px", "fontSize": ".73rem", "color": "#fff"})
        for t in ["VADER + TextBlob", "spaCy NER", "Lematización", "73 K narrativas"]
    ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "alignItems": "center"}),
], style={
    "background": f"linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 100%)",
    "padding": "20px 32px", "display": "flex",
    "alignItems": "center", "justifyContent": "space-between", "flexWrap": "wrap", "gap": "12px",
})

# ── Panel de filtros ──────────────────────────────────────────────
def _filter_block(label, id_, options, placeholder="Todos"):
    return html.Div([
        html.Label(label, style={"fontSize": ".73rem", "color": TEXT_MUTED,
                                  "fontWeight": "600", "marginBottom": "3px", "display": "block"}),
        dcc.Dropdown(id=id_, options=options, multi=True,
                     placeholder=placeholder, style={"fontSize": ".83rem"}),
    ], style={"flex": "1", "minWidth": "155px"})


FILTERS = html.Div([
    _filter_block("Producto",   "f-product",   products),
    _filter_block("Issue",      "f-issue",      issues),
    _filter_block("Compañía",   "f-company",    companies),
    _filter_block("Estado",     "f-state",      states),
    _filter_block("Sentimiento","f-sentiment",  sent_opts),
    html.Div([
        html.Label("Año desde / hasta",
                   style={"fontSize": ".73rem", "color": TEXT_MUTED,
                          "fontWeight": "600", "marginBottom": "3px", "display": "block"}),
        html.Div([
            dcc.Dropdown(id="f-year-from",
                         options=[{"label": str(y), "value": y} for y in years],
                         placeholder="Desde",
                         style={"fontSize": ".83rem", "width": "92px", "display": "inline-block"}),
            dcc.Dropdown(id="f-year-to",
                         options=[{"label": str(y), "value": y} for y in years],
                         placeholder="Hasta",
                         style={"fontSize": ".83rem", "width": "92px",
                                "display": "inline-block", "marginLeft": "6px"}),
        ]),
    ], style={"flex": "1", "minWidth": "200px"}),
], style={
    "display": "flex", "flexWrap": "wrap", "gap": "14px",
    "padding": "14px 28px", "background": "#fff",
    "borderBottom": f"2px solid {BG_LIGHT}", "alignItems": "flex-end",
})

# ── Tabs ──────────────────────────────────────────────────────────
tab_style    = {"fontWeight": "600", "fontSize": ".88rem"}
tab_selected = {"fontWeight": "700", "fontSize": ".88rem",
                "borderTop": f"3px solid {ACCENT}", "color": PRIMARY}

TABS = dcc.Tabs(id="tabs", value="tab-overview", children=[
    dcc.Tab(label="📊 Panorama",           value="tab-overview",   style=tab_style, selected_style=tab_selected),
    dcc.Tab(label="🔬 Pipeline NLP",       value="tab-nlp",        style=tab_style, selected_style=tab_selected),
    dcc.Tab(label="😟 Sentimiento",        value="tab-sentiment",  style=tab_style, selected_style=tab_selected),
    dcc.Tab(label="🏛️ Entidades & Leyes",  value="tab-entities",   style=tab_style, selected_style=tab_selected),
], style={"fontFamily": FONT})

app.layout = html.Div([
    HEADER,
    FILTERS,
    TABS,
    html.Div(id="tab-content"),
], style={"fontFamily": FONT, "background": BG_LIGHT, "minHeight": "100vh"})


# ══════════════════════════════════════════════════════════════════
#  HELPER: filtrar dataframe
# ══════════════════════════════════════════════════════════════════
def _filter(prod, iss, comp, st, sent, y_from, y_to):
    dff = df
    if prod:   dff = dff[dff["Product"].isin(prod)]
    if iss:    dff = dff[dff["Issue"].isin(iss)]
    if comp:   dff = dff[dff["Company"].isin(comp)]
    if st:     dff = dff[dff["State"].isin(st)]
    if sent and "vader_label" in dff.columns:
        dff = dff[dff["vader_label"].isin(sent)]
    if y_from and "year" in dff.columns: dff = dff[dff["year"] >= y_from]
    if y_to   and "year" in dff.columns: dff = dff[dff["year"] <= y_to]
    return dff


# ══════════════════════════════════════════════════════════════════
#  RENDER DE TABS
# ══════════════════════════════════════════════════════════════════
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    P = {"padding": "24px 28px"}

    if tab == "tab-overview":
        return html.Div([
            html.Div(id="kpis",
                     style={"display": "flex", "flexWrap": "wrap",
                            "gap": "14px", "marginBottom": "20px"}),
            _row(
                _box(dcc.Graph(id="g-timeline"),  flex="2", min_width="420px"),
                _box(dcc.Graph(id="g-products"),  flex="1", min_width="300px"),
            ),
            _row(
                _box(dcc.Graph(id="g-issues"),    flex="1", min_width="300px"),
                _box(dcc.Graph(id="g-states"),    flex="1", min_width="300px"),
                _box(dcc.Graph(id="g-response"),  flex="1", min_width="280px"),
            ),
        ], style=P)

    if tab == "tab-nlp":
        return html.Div([
            html.Div(id="nlp-funnel", style={"marginBottom": "20px"}),
            _row(
                _box(dcc.Graph(id="g-wordcloud"),  flex="2", min_width="420px"),
                _box(dcc.Graph(id="g-freq-bar"),   flex="1", min_width="300px"),
            ),
            _row(
                _box(dcc.Graph(id="g-len-hist"),   flex="1", min_width="280px"),
                _box(dcc.Graph(id="g-lex-div"),    flex="1", min_width="280px"),
                _box(dcc.Graph(id="g-wc-box"),     flex="1", min_width="280px"),
            ),
        ], style=P)

    if tab == "tab-sentiment":
        return html.Div([
            _row(
                _box(dcc.Graph(id="g-sent-pie"),      flex="1", min_width="280px"),
                _box(dcc.Graph(id="g-sent-product"),  flex="2", min_width="420px"),
            ),
            _row(
                _box(dcc.Graph(id="g-sent-trend"),    flex="2", min_width="420px"),
                _box(dcc.Graph(id="g-vader-dist"),    flex="1", min_width="280px"),
            ),
            _row(
                _box(dcc.Graph(id="g-tb-scatter"),    flex="1", min_width="340px"),
                _box(dcc.Graph(id="g-subjectivity"),  flex="1", min_width="340px"),
            ),
        ], style=P)

    if tab == "tab-entities":
        return html.Div([
            _row(
                _box(dcc.Graph(id="g-orgs"),         flex="1", min_width="300px"),
                _box(dcc.Graph(id="g-laws"),         flex="1", min_width="300px"),
                _box(dcc.Graph(id="g-gpe"),          flex="1", min_width="260px"),
            ),
            _row(
                _box(dcc.Graph(id="g-legal-terms"),  flex="1", min_width="320px"),
                _box(dcc.Graph(id="g-entity-sent"),  flex="1.5", min_width="380px"),
            ),
            _box(html.Div([
                html.H4("Narrativas de muestra",
                        style={"margin": "0 0 12px", "color": PRIMARY, "fontSize": ".95rem"}),
                html.Div(id="narratives-table"),
            ]), flex="none", min_width="100%"),
        ], style=P)

    return html.Div("Selecciona una pestaña.")


# ══════════════════════════════════════════════════════════════════
#  TAB 1: PANORAMA
# ══════════════════════════════════════════════════════════════════
@app.callback(Output("kpis", "children"), [Input(i, "value") for i in FILTER_IDS])
def update_kpis(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    n = len(dff)
    if n == 0:
        return [_card("—", lbl) for lbl in
                ["Quejas","% Negativo","Issue frecuente","Top Compañía","Tokens mediana","Citan ley"]]

    neg_pct   = dff["vader_label"].isin(["Muy Negativo","Negativo"]).sum() / n * 100
    top_issue = dff["Issue"].value_counts().index[0] if "Issue" in dff else "—"
    top_co    = dff["Company"].value_counts().index[0] if "Company" in dff else "—"
    med_tok   = int(dff["word_count"].median()) if "word_count" in dff else 0
    legal_pct = dff["legal_term_count"].gt(0).sum() / n * 100 if "legal_term_count" in dff else 0

    def _short(s, n=22):
        s = str(s)
        return s[:n] + "…" if len(s) > n else s

    return [
        _card(f"{n:,}",              "Quejas",             color=ACCENT,    icon="📁"),
        _card(f"{neg_pct:.1f}%",     "Sentimiento negativo",color="#c0392b", icon="😠"),
        _card(_short(top_issue),     "Issue más frecuente", color="#8e44ad", icon="❗"),
        _card(_short(top_co),        "Compañía más citada", color="#16a085", icon="🏦"),
        _card(med_tok,               "Tokens por narrativa",color="#d4ac0d", icon="📝"),
        _card(f"{legal_pct:.1f}%",   "Citan una ley",      color="#e67e22", icon="⚖️"),
    ]


@app.callback(Output("g-timeline", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_timeline(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "year" not in dff.columns or dff.empty:
        return go.Figure()
    ts = dff["year"].value_counts().sort_index().reset_index()
    ts.columns = ["Año", "Quejas"]
    fig = px.area(ts, x="Año", y="Quejas", markers=True,
                  title="📈 Evolución temporal de quejas",
                  color_discrete_sequence=[ACCENT])
    fig.update_traces(fill="tozeroy", line_width=2.5, marker_size=5)
    fig.update_layout(template=TMPL, margin=dict(t=46, b=20), hovermode="x unified")
    return fig


@app.callback(Output("g-products", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_products(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "Product" not in dff.columns or dff.empty:
        return go.Figure()
    top = dff["Product"].value_counts().head(8).reset_index()
    top.columns = ["Product", "Quejas"]
    # Etiquetas cortas para el treemap
    top["Label"] = top["Product"].str.replace(
        r"Credit reporting.*", "Credit reporting", regex=True
    ).str[:35]
    fig = px.treemap(top, path=["Label"], values="Quejas",
                     title="🗂️ Distribución por Producto",
                     color="Quejas", color_continuous_scale="Blues")
    fig.update_traces(textfont_size=13)
    fig.update_layout(template=TMPL, margin=dict(t=46, b=10, l=10, r=10))
    fig.update_coloraxes(showscale=False)
    return fig


@app.callback(Output("g-issues", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_issues(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "Issue" not in dff.columns or dff.empty:
        return go.Figure()
    top = dff["Issue"].value_counts().head(10)
    short_ix = [s[:40] + "…" if len(s) > 42 else s for s in top.index]
    fig = px.bar(y=short_ix, x=top.values, orientation="h",
                 labels={"x": "Quejas", "y": ""},
                 title="🔝 Top 10 Issues",
                 color=top.values, color_continuous_scale="Blues")
    fig.update_layout(template=TMPL, yaxis={"categoryorder": "total ascending"},
                      margin=dict(t=46, b=20))
    fig.update_coloraxes(showscale=False)
    return fig


@app.callback(Output("g-states", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_states(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "State" not in dff.columns or dff.empty:
        return go.Figure()
    top = dff["State"].dropna().value_counts().head(15)
    fig = px.bar(y=top.index, x=top.values, orientation="h",
                 labels={"x": "Quejas", "y": "Estado"},
                 title="🗺️ Top 15 Estados",
                 color=top.values, color_continuous_scale="Teal")
    fig.update_layout(template=TMPL, yaxis={"categoryorder": "total ascending"},
                      margin=dict(t=46, b=20))
    fig.update_coloraxes(showscale=False)
    return fig


@app.callback(Output("g-response", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_response(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    col = "Company response to consumer"
    if col not in dff.columns or dff.empty:
        return go.Figure()
    counts = dff[col].dropna().value_counts().head(6)
    labels = [s[:30] + "…" if len(s) > 32 else s for s in counts.index]
    fig = px.bar(x=labels, y=counts.values,
                 labels={"x": "", "y": "Quejas"},
                 title="✉️ Respuesta de la Compañía",
                 color=counts.values, color_continuous_scale="Greens")
    fig.update_layout(template=TMPL, margin=dict(t=46, b=80), xaxis_tickangle=-35)
    fig.update_coloraxes(showscale=False)
    return fig


# ══════════════════════════════════════════════════════════════════
#  TAB 2: PIPELINE NLP
# ══════════════════════════════════════════════════════════════════
@app.callback(Output("nlp-funnel", "children"), [Input(i, "value") for i in FILTER_IDS])
def update_funnel(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    n = len(dff)
    steps = [
        ("1 · Raw Data",           73635, "#2e86c1", "Narrativas CFPB originales"),
        ("2 · Limpieza",           73609, "#2874a6", "Eliminadas <20 chars, parse fechas"),
        ("3 · Preprocesado",       n,     "#1f618d", "Tokenización · Lematización · Stopwords"),
        ("4 · NER + Sentimiento",  n,     "#154360", "spaCy · VADER · TextBlob"),
        ("5 · Feature Engineering",n,     "#0e2f44", "43 variables estructuradas"),
    ]
    items = []
    for i, (lbl, cnt, color, desc) in enumerate(steps):
        items.append(html.Div([
            html.Div(f"{cnt:,}",
                     style={"fontSize": "1.25rem", "fontWeight": "800", "color": color}),
            html.Div(lbl,
                     style={"fontSize": ".8rem", "fontWeight": "700",
                            "color": PRIMARY, "margin": "4px 0 2px"}),
            html.Div(desc,
                     style={"fontSize": ".71rem", "color": TEXT_MUTED}),
        ], style={
            "background": BG_CARD, "borderRadius": "10px", "padding": "14px 16px",
            "boxShadow": "0 2px 8px rgba(0,0,0,.07)", "flex": "1", "minWidth": "155px",
            "borderLeft": f"4px solid {color}",
        }))
        if i < len(steps) - 1:
            items.append(html.Div("→",
                                  style={"fontSize": "1.3rem", "color": TEXT_MUTED,
                                         "display": "flex", "alignItems": "center",
                                         "padding": "0 4px"}))
    return html.Div(items, style={"display": "flex", "flexWrap": "wrap",
                                   "gap": "8px", "marginBottom": "4px"})


@app.callback(Output("g-wordcloud", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_wordcloud(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "processed_text" not in dff.columns or dff.empty:
        return go.Figure()

    sample_n = min(6000, len(dff))
    texts = dff["processed_text"].dropna().sample(sample_n, random_state=42)
    # Filtrar token "mask" (PII) del texto
    cleaned = texts.astype(str).str.replace(r"\bmask\b", "", regex=True)
    text = " ".join(cleaned)
    if not text.strip():
        return go.Figure()

    wc = WordCloud(
        width=900, height=400,
        background_color="white",
        colormap="Blues",
        max_words=120,
        stopwords=WORD_STOPWORDS,
        collocations=False,
        min_font_size=9,
        prefer_horizontal=0.8,
    ).generate(text)

    fig = px.imshow(wc,
                    title="☁️ Word Cloud — Términos más frecuentes (sin tokens de enmascaramiento PII)")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(template=TMPL, margin=dict(t=50, b=10, l=10, r=10))
    return fig


@app.callback(Output("g-freq-bar", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_freq_bar(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "processed_text" not in dff.columns or dff.empty:
        return go.Figure()
    sample_n = min(8000, len(dff))
    texts = dff["processed_text"].dropna().sample(sample_n, random_state=42)
    all_tokens = " ".join(texts.astype(str)).split()
    # Excluir stopwords del Word Cloud
    filtered = [t for t in all_tokens if t not in WORD_STOPWORDS and len(t) > 2]
    top = Counter(filtered).most_common(20)
    if not top:
        return go.Figure()
    words, freqs = zip(*top)
    fig = px.bar(y=list(words), x=list(freqs), orientation="h",
                 labels={"x": "Frecuencia", "y": "Término"},
                 title="📊 Top 20 Términos (tokens procesados)",
                 color=list(freqs), color_continuous_scale="Blues")
    fig.update_layout(template=TMPL, yaxis={"categoryorder": "total ascending"},
                      margin=dict(t=46, b=20))
    fig.update_coloraxes(showscale=False)
    return fig


@app.callback(Output("g-len-hist", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_len_hist(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "narrative_length" not in dff.columns or dff.empty:
        return go.Figure()
    col = dff["narrative_length"].clip(upper=5000)
    fig = px.histogram(col, nbins=50,
                       labels={"value": "Caracteres (recortado en 5 000)", "count": "Quejas"},
                       title="📏 Longitud de narrativa",
                       color_discrete_sequence=[ACCENT])
    median_val = dff["narrative_length"].median()
    fig.add_vline(x=median_val, line_dash="dash", line_color="#e74c3c",
                  annotation_text=f"Mediana: {median_val:.0f}",
                  annotation_position="top right")
    fig.update_layout(template=TMPL, showlegend=False, margin=dict(t=46, b=20))
    return fig


@app.callback(Output("g-lex-div", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_lex_div(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "lexical_diversity" not in dff.columns or dff.empty:
        return go.Figure()
    fig = px.histogram(dff["lexical_diversity"].clip(0, 1), nbins=40,
                       labels={"value": "Diversidad (unique/total)", "count": "Quejas"},
                       title="🔤 Diversidad Léxica",
                       color_discrete_sequence=["#8e44ad"])
    fig.update_layout(template=TMPL, showlegend=False, margin=dict(t=46, b=20))
    return fig


@app.callback(Output("g-wc-box", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_wc_box(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "word_count" not in dff.columns or "vader_label" not in dff.columns or dff.empty:
        return go.Figure()
    # Recortar outliers extremos
    clipped = dff[dff["word_count"] <= dff["word_count"].quantile(0.98)].copy()
    fig = px.box(clipped, x="vader_label", y="word_count",
                 color="vader_label",
                 category_orders={"vader_label": SENT_ORDER},
                 labels={"word_count": "Tokens", "vader_label": "Sentimiento"},
                 title="📦 Tokens por Sentimiento",
                 color_discrete_map=PALETTE)
    fig.update_layout(template=TMPL, showlegend=False, margin=dict(t=46, b=20))
    return fig


# ══════════════════════════════════════════════════════════════════
#  TAB 3: SENTIMIENTO
# ══════════════════════════════════════════════════════════════════
@app.callback(Output("g-sent-pie", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_sent_pie(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "vader_label" not in dff.columns or dff.empty:
        return go.Figure()
    counts = dff["vader_label"].value_counts().reindex(SENT_ORDER).dropna()
    fig = px.pie(values=counts.values, names=counts.index,
                 title="😟 Distribución Sentimiento (VADER)",
                 color=counts.index, color_discrete_map=PALETTE,
                 hole=0.42)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(template=TMPL, margin=dict(t=46, b=20),
                      showlegend=False)
    return fig


@app.callback(Output("g-sent-product", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_sent_product(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "vader_label" not in dff.columns or "Product" not in dff.columns or dff.empty:
        return go.Figure()
    top_prods = dff["Product"].value_counts().head(8).index
    sub = dff[dff["Product"].isin(top_prods)].copy()
    # Etiquetas cortas
    sub["ProdLabel"] = sub["Product"].str.replace(
        r"Credit reporting.*", "Credit reporting", regex=True
    ).str[:35]
    pivot = sub.groupby(["ProdLabel", "vader_label"]).size().reset_index(name="n")
    fig = px.bar(pivot, x="ProdLabel", y="n", color="vader_label",
                 barmode="stack",
                 labels={"ProdLabel": "Producto", "n": "Quejas", "vader_label": "Sentimiento"},
                 title="📦 Sentimiento por Producto",
                 color_discrete_map=PALETTE,
                 category_orders={"vader_label": SENT_ORDER})
    fig.update_layout(template=TMPL, margin=dict(t=46, b=90),
                      xaxis_tickangle=-35,
                      legend=dict(orientation="h", y=-0.3, title=""))
    return fig


@app.callback(Output("g-sent-trend", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_sent_trend(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "vader_label" not in dff.columns or "year" not in dff.columns or dff.empty:
        return go.Figure()
    pivot = (dff.groupby(["year", "vader_label"])
               .size().reset_index(name="n"))
    fig = px.line(pivot, x="year", y="n", color="vader_label",
                  markers=True,
                  labels={"year": "Año", "n": "Quejas", "vader_label": "Sentimiento"},
                  title="📈 Tendencia de Sentimiento por Año",
                  color_discrete_map=PALETTE,
                  category_orders={"vader_label": SENT_ORDER})
    fig.update_layout(template=TMPL, margin=dict(t=46, b=20),
                      hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2, title=""))
    return fig


@app.callback(Output("g-vader-dist", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_vader_dist(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "vader_compound" not in dff.columns or dff.empty:
        return go.Figure()
    fig = px.histogram(dff["vader_compound"], nbins=60,
                       labels={"vader_compound": "Compound Score (−1 a +1)"},
                       title="🎯 Distribución Score VADER",
                       color_discrete_sequence=[ACCENT])
    fig.add_vline(x=0,    line_dash="dash", line_color="gray",
                  annotation_text="Neutro")
    fig.add_vline(x=-0.5, line_dash="dot",  line_color="#e74c3c",
                  annotation_text="Muy Negativo")
    fig.add_vline(x=0.5,  line_dash="dot",  line_color="#1e8449",
                  annotation_text="Muy Positivo")
    fig.update_layout(template=TMPL, showlegend=False, margin=dict(t=46, b=20))
    return fig


@app.callback(Output("g-tb-scatter", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_tb_scatter(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "vader_compound" not in dff.columns or "textblob_polarity" not in dff.columns or dff.empty:
        return go.Figure()
    sample = dff.sample(min(4000, len(dff)), random_state=42)
    fig = px.scatter(sample, x="vader_compound", y="textblob_polarity",
                     color="vader_label", opacity=0.4,
                     labels={"vader_compound": "VADER compound",
                             "textblob_polarity": "TextBlob polarity",
                             "vader_label": "Sentimiento"},
                     title="🔀 VADER vs TextBlob",
                     color_discrete_map=PALETTE,
                     category_orders={"vader_label": SENT_ORDER})
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(template=TMPL, margin=dict(t=46, b=20),
                      legend=dict(orientation="h", y=-0.25, title=""))
    return fig


@app.callback(Output("g-subjectivity", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_subjectivity(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "textblob_subjectivity" not in dff.columns or dff.empty:
        return go.Figure()
    # Solo Muy Negativo y Muy Positivo para claridad
    sub = dff[dff["vader_label"].isin(["Muy Negativo", "Muy Positivo", "Neutral"])]
    if sub.empty:
        sub = dff
    fig = px.histogram(sub, x="textblob_subjectivity",
                       color="vader_label",
                       nbins=40, barmode="overlay", opacity=0.65,
                       labels={"textblob_subjectivity": "Subjetividad (0=objetivo, 1=subjetivo)",
                               "vader_label": "Sentimiento"},
                       title="🧠 Subjetividad por Sentimiento (TextBlob)",
                       color_discrete_map=PALETTE,
                       category_orders={"vader_label": SENT_ORDER})
    fig.update_layout(template=TMPL, margin=dict(t=46, b=20),
                      legend=dict(orientation="h", y=-0.2, title=""))
    return fig


# ══════════════════════════════════════════════════════════════════
#  TAB 4: ENTIDADES & LEYES
# ══════════════════════════════════════════════════════════════════
def _top_entities(dff, key, n=15, blacklist=None, normalize_fn=None):
    """Extrae las n entidades más frecuentes de una columna entities."""
    if "entities" not in dff.columns or dff.empty:
        return [], []
    items = []
    for ents in dff["entities"]:
        if not isinstance(ents, dict):
            continue
        vals = ents.get(key, [])
        if not vals:
            continue
        for v in vals:
            v = str(v).strip()
            if not v:
                continue
            if blacklist and v in blacklist:
                continue
            if normalize_fn:
                v = normalize_fn(v)
            items.append(v)
    if not items:
        return [], []
    top = Counter(items).most_common(n)
    names, counts = zip(*top)
    return list(names), list(counts)


@app.callback(Output("g-orgs", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_orgs(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    names, counts = _top_entities(dff, "ORG", n=15, blacklist=ORG_NOISE)
    if not names:
        return go.Figure()
    fig = px.bar(y=names, x=counts, orientation="h",
                 labels={"x": "Menciones", "y": ""},
                 title="🏦 Top Entidades ORG",
                 color=counts, color_continuous_scale="Blues")
    fig.update_layout(template=TMPL, yaxis={"categoryorder": "total ascending"},
                      margin=dict(t=46, b=20))
    fig.update_coloraxes(showscale=False)
    return fig


@app.callback(Output("g-laws", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_laws(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    names, counts = _top_entities(dff, "LAW", n=15, normalize_fn=_clean_law)
    if not names:
        return go.Figure()
    fig = px.bar(y=names, x=counts, orientation="h",
                 labels={"x": "Menciones", "y": ""},
                 title="⚖️ Regulaciones Citadas (LAW)",
                 color=counts, color_continuous_scale="Reds")
    fig.update_layout(template=TMPL, yaxis={"categoryorder": "total ascending"},
                      margin=dict(t=46, b=20))
    fig.update_coloraxes(showscale=False)
    return fig


@app.callback(Output("g-gpe", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_gpe(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    names, counts = _top_entities(dff, "GPE", n=15)
    if not names:
        return go.Figure()
    fig = px.bar(y=names, x=counts, orientation="h",
                 labels={"x": "Menciones", "y": ""},
                 title="📍 Entidades Geográficas (GPE)",
                 color=counts, color_continuous_scale="Greens")
    fig.update_layout(template=TMPL, yaxis={"categoryorder": "total ascending"},
                      margin=dict(t=46, b=20))
    fig.update_coloraxes(showscale=False)
    return fig


@app.callback(Output("g-legal-terms", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_legal_terms(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "processed_text" not in dff.columns or dff.empty:
        return go.Figure()
    LEGAL = ["fcra", "fdcpa", "fcba", "ecoa", "tila",
             "fraud", "violation", "lawsuit", "dispute",
             "investigation", "inaccurate", "unauthorized",
             "breach", "discrimination", "harassment",
             "garnishment", "repossession", "foreclosure"]
    sample_n = min(10000, len(dff))
    texts = " ".join(dff["processed_text"].dropna().sample(sample_n, random_state=42).astype(str)).lower().split()
    freqs = {t: texts.count(t) for t in LEGAL if t in texts}
    if not freqs:
        return go.Figure()
    srt = dict(sorted(freqs.items(), key=lambda x: x[1]))
    fig = px.bar(y=list(srt.keys()), x=list(srt.values()), orientation="h",
                 labels={"x": "Ocurrencias", "y": "Término legal"},
                 title="📜 Frecuencia de Términos Legales",
                 color=list(srt.values()), color_continuous_scale="Oranges")
    fig.update_layout(template=TMPL, margin=dict(t=46, b=20))
    fig.update_coloraxes(showscale=False)
    return fig


@app.callback(Output("g-entity-sent", "figure"), [Input(i, "value") for i in FILTER_IDS])
def update_entity_sent(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if "entities" not in dff.columns or "vader_label" not in dff.columns or dff.empty:
        return go.Figure()
    rows = []
    for ents, label in zip(dff["entities"], dff["vader_label"]):
        if not isinstance(ents, dict):
            continue
        for org in (ents.get("ORG") or []):
            org = str(org).strip()
            if org and org not in ORG_NOISE:
                rows.append({"org": org, "sentiment": label})
    if not rows:
        return go.Figure()
    edf = pd.DataFrame(rows)
    top_orgs = edf["org"].value_counts().head(8).index
    edf = edf[edf["org"].isin(top_orgs)]
    pivot = edf.groupby(["org", "sentiment"]).size().reset_index(name="n")
    fig = px.bar(pivot, x="org", y="n", color="sentiment",
                 barmode="stack",
                 labels={"org": "Entidad", "n": "Quejas", "sentiment": "Sentimiento"},
                 title="🏦 Sentimiento por Entidad ORG",
                 color_discrete_map=PALETTE,
                 category_orders={"sentiment": SENT_ORDER})
    fig.update_layout(template=TMPL, margin=dict(t=46, b=80),
                      xaxis_tickangle=-30,
                      legend=dict(orientation="h", y=-0.3, title=""))
    return fig


@app.callback(Output("narratives-table", "children"), [Input(i, "value") for i in FILTER_IDS])
def update_table(prod, iss, comp, st, sent, y_from, y_to):
    dff = _filter(prod, iss, comp, st, sent, y_from, y_to)
    if dff.empty:
        return html.P("Sin datos para mostrar.", style={"color": TEXT_MUTED})

    sample = dff.sample(min(8, len(dff)), random_state=42)
    cols_show = ["Date received", "Product", "Issue", "Company",
                 "vader_label", "Consumer complaint narrative"]
    cols_show = [c for c in cols_show if c in sample.columns]

    sent_bg = {
        "Muy Negativo": "#fde8e8", "Negativo": "#fce4e4",
        "Neutral": "#fefce8", "Positivo": "#e8faf0", "Muy Positivo": "#d5f5e3",
    }

    def _cell(val, col):
        txt = str(val)
        if len(txt) > 250:
            txt = txt[:250] + "…"
        style = {
            "padding": "8px 10px", "fontSize": "11.5px",
            "borderBottom": "1px solid #eef0f3",
            "verticalAlign": "top", "maxWidth": "280px",
            "wordBreak": "break-word", "lineHeight": "1.4",
        }
        if col == "vader_label":
            style["background"] = sent_bg.get(val, "#fff")
            style["fontWeight"] = "700"
            style["textAlign"] = "center"
        return html.Td(txt, style=style)

    th_s = {
        "padding": "10px 10px", "background": PRIMARY,
        "color": "#fff", "fontSize": "11.5px",
        "fontWeight": "700", "textAlign": "left",
    }
    return html.Table([
        html.Thead(html.Tr([html.Th(c, style=th_s) for c in cols_show])),
        html.Tbody([
            html.Tr([_cell(row[c], c) for c in cols_show],
                    style={"background": "#fff" if i % 2 == 0 else "#f8fafc"})
            for i, (_, row) in enumerate(sample.iterrows())
        ]),
    ], style={
        "width": "100%", "borderCollapse": "collapse",
        "borderRadius": "8px", "overflow": "hidden",
        "boxShadow": "0 1px 4px rgba(0,0,0,.06)",
    })


if __name__ == "__main__":
    app.run(debug=True, port=8050)
