import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "evaluation_results"

BASELINE_FILE = RESULTS_DIR / "rag_baseline.json"


def save_baseline(results):
    """
    Save the current RAG evaluation results
    as the regression baseline.
    """

    RESULTS_DIR.mkdir(
        exist_ok=True
    )

    baseline = {
        "created_at": datetime.now().isoformat(),
        "metrics": results
    }

    with open(
        BASELINE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            baseline,
            file,
            indent=4
        )

    print()
    print(
        f"Baseline saved to: {BASELINE_FILE}"
    )


def load_baseline():

    if not BASELINE_FILE.exists():
        return None

    with open(
        BASELINE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)