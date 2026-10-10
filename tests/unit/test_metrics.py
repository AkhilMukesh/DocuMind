
import pytest

from metrics import MetricsRegistry


def test_counter_increments():
    registry = MetricsRegistry()

    registry.increment("ingestion.success")
    registry.increment("ingestion.success")
    registry.increment("ingestion.failure")

    snapshot = registry.snapshot()

    assert snapshot["counters"]["ingestion.success"] == 2
    assert snapshot["counters"]["ingestion.failure"] == 1


def test_duration_statistics():
    registry = MetricsRegistry()

    registry.observe_duration("llm.request", 1.0)
    registry.observe_duration("llm.request", 3.0)

    duration = registry.snapshot()["durations"]["llm.request"]

    assert duration["count"] == 2
    assert duration["total_seconds"] == 4.0
    assert duration["average_seconds"] == 2.0
    assert duration["max_seconds"] == 3.0


def test_track_duration_records_success_and_failure():
    registry = MetricsRegistry()

    with registry.track_duration("retrieval"):
        pass

    with pytest.raises(ValueError):
        with registry.track_duration("retrieval"):
            raise ValueError("Simulated failure")

    duration = registry.snapshot()["durations"]["retrieval"]

    assert duration["count"] == 2
    assert duration["total_seconds"] >= 0


def test_negative_metric_values_are_rejected():
    registry = MetricsRegistry()

    with pytest.raises(ValueError):
        registry.increment("test.counter", -1)

    with pytest.raises(ValueError):
        registry.observe_duration("test.duration", -0.1)


def test_reset_clears_metrics():
    registry = MetricsRegistry()
    registry.increment("test.counter")
    registry.observe_duration("test.duration", 0.5)

    registry.reset()
    snapshot = registry.snapshot()

    assert snapshot["counters"] == {}
    assert snapshot["durations"] == {}
