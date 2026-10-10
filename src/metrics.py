
import threading
import time
from collections import defaultdict
from contextlib import contextmanager


class MetricsRegistry:
    """Thread-safe, in-process application metrics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters = defaultdict(int)
        self._durations = defaultdict(list)

    def increment(self, name, amount=1):
        if amount < 0:
            raise ValueError("Counter increments cannot be negative")

        with self._lock:
            self._counters[name] += amount

    def observe_duration(self, name, seconds):
        if seconds < 0:
            raise ValueError("Duration cannot be negative")

        with self._lock:
            self._durations[name].append(seconds)

    @contextmanager
    def track_duration(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.observe_duration(name, elapsed)

    def snapshot(self):
        with self._lock:
            counters = dict(self._counters)
            durations = {
                name: {
                    "count": len(values),
                    "total_seconds": sum(values),
                    "average_seconds": (
                        sum(values) / len(values) if values else 0.0
                    ),
                    "max_seconds": max(values) if values else 0.0,
                }
                for name, values in self._durations.items()
            }

        return {
            "counters": counters,
            "durations": durations,
        }

    def reset(self):
        """Reset metrics, primarily for isolated tests."""
        with self._lock:
            self._counters.clear()
            self._durations.clear()


# Shared registry for this Python process.
metrics = MetricsRegistry()

