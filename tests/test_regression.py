import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_PATH = (
    PROJECT_ROOT
    / "evaluation_results"
    / "rag_baseline.json"
)


def load_baseline():
    with open(BASELINE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def test_regression_baseline_exists():
    assert BASELINE_PATH.exists(), (
        "RAG regression baseline does not exist."
    )


def test_regression_baseline_is_valid():
    baseline = load_baseline()

    assert isinstance(baseline, dict)
    assert baseline


def test_regression_baseline_contains_metrics():
    baseline = load_baseline()

    assert "metrics" in baseline
    assert isinstance(baseline["metrics"], dict)
    assert len(baseline["metrics"]) > 0


def test_regression_baseline_contains_expected_metrics():
    baseline = load_baseline()

    metrics = baseline["metrics"]

    assert "citation_completeness" in metrics
    assert "citation_correctness" in metrics
    assert "faithfulness" in metrics
    assert "semantic_correctness" in metrics


def test_regression_metrics_are_valid_scores():
    baseline = load_baseline()

    for metric_name, value in baseline["metrics"].items():
        assert isinstance(value, (int, float))
        assert 0.0 <= value <= 1.0, (
            f"{metric_name} must be between 0 and 1"
        )


def assert_metric_not_regressed(
    baseline_value,
    current_value,
    tolerance=0.05
):
    minimum_allowed = baseline_value - tolerance

    assert current_value >= minimum_allowed, (
        f"Metric regressed: "
        f"baseline={baseline_value}, "
        f"current={current_value}, "
        f"allowed_minimum={minimum_allowed}"
    )