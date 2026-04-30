"""Grammar correction pipeline

Usage: python scripts/grammar_correction.py

This script fetches a Socrata dataset, attempts to detect a text column,
runs language-tool-python corrections, and writes results to CSV.
"""

import argparse
import sys
import time
from typing import Tuple, List

import pandas as pd
import requests
from tqdm import tqdm

try:
    import language_tool_python
except Exception:
    language_tool_python = None


def fetch_dataset(dataset_id: str, limit: int = 50, timeout: int = 30) -> pd.DataFrame:
    url = f"https://www.datos.gov.co/resource/{dataset_id}.json"
    params = {"$limit": limit}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return pd.DataFrame(data)


def detect_text_column(df: pd.DataFrame) -> str:
    # Prefer common names
    candidates = ["descripcion", "descripcion_hecho", "observaciones", "observacion", "descripcion_hechos", "detalle", "texto", "description"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: pick the column with the largest average string length
    str_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    if not str_cols:
        raise ValueError("No string-like columns found in dataset.")
    avg_len = {c: df[c].dropna().astype(str).map(len).mean() if not df[c].dropna().empty else 0 for c in str_cols}
    best = max(avg_len, key=avg_len.get)
    return best


def analizar_y_corregir_tool(tool, texto: str) -> Tuple[str, int, List[str]]:
    if not isinstance(texto, str) or not texto.strip():
        return texto, 0, []
    matches = tool.check(texto)
    texto_corregido = tool.correct(texto)
    errores = [getattr(m, "ruleIssueType", str(m.ruleId) if hasattr(m, "ruleId") else "unknown") for m in matches]
    return texto_corregido, len(matches), errores


def main(argv=None):
    parser = argparse.ArgumentParser(description="Corrige gramática en un dataset Socrata usando LanguageTool")
    parser.add_argument("--dataset", "-d", default="h8wr-bahk", help="ID del dataset en datos.gov.co (Socrata)")
    parser.add_argument("--limit", "-n", type=int, default=50, help="Número de registros a descargar")
    parser.add_argument("--out", "-o", default="out_corrected.csv", help="Archivo CSV de salida")
    args = parser.parse_args(argv)

    if language_tool_python is None:
        print("El paquete language-tool-python no está instalado. Instálalo con: pip install language-tool-python")
        sys.exit(1)

    print("Descargando datos...")
    df = fetch_dataset(args.dataset, limit=args.limit)
    print(f"Dataset cargado con éxito. Dimensiones: {df.shape}")

    try:
        columna_texto = detect_text_column(df)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    print(f"Analizando la columna: '{columna_texto}'")
    df = df.dropna(subset=[columna_texto]).reset_index(drop=True)

    print("Inicializando LanguageTool (español)...")
    tool = language_tool_python.LanguageTool('es')

    resultados = []
    print("Analizando y corrigiendo textos...")
    for t in tqdm(df[columna_texto].astype(str), total=len(df)):
        try:
            corr, cnt, errs = analizar_y_corregir_tool(tool, t)
        except Exception as e:
            corr, cnt, errs = t, 0, [f"error:{e}"]
        resultados.append((corr, cnt, errs))
        # polite pause to avoid local tool rate issues
        time.sleep(0.01)

    df['texto_corregido'] = [r[0] for r in resultados]
    df['cantidad_errores'] = [r[1] for r in resultados]
    df['tipos_errores'] = [r[2] for r in resultados]

    df.to_csv(args.out, index=False)
    print(f"Resultados guardados en {args.out}")

    tool.close()


if __name__ == '__main__':
    main()
