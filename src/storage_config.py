
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(
    os.getenv("DOCUMIND_DATA_DIR", str(PROJECT_ROOT))
).resolve()

CHROMA_PATH = DATA_DIR / "chroma_db"
DATABASE_PATH = DATA_DIR / "documind.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PATH.mkdir(parents=True, exist_ok=True)
