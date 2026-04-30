"""
Script para lanzar el dashboard fácilmente.
"""
from pathlib import Path
import sys

# Asegurar que el directorio raíz está en el path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.app import app

if __name__ == "__main__":
    print("=" * 60)
    print("Dashboard NLP - Quejas CFPB")
    print("=" * 60)
    print("Abre tu navegador en: http://127.0.0.1:8050")
    print("Presiona Ctrl+C para detener")
    print("=" * 60)
    app.run(debug=True, port=8050)
