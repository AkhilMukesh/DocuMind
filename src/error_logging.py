
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def log_operation_errors(operation, **context):
    """
    Log the duration and traceback of a failed operation.

    Exceptions are re-raised so existing error handling
    continues to work.
    """
    start_time = time.perf_counter()

    try:
        yield
    except Exception:
        duration_seconds = time.perf_counter() - start_time

        # Only pass approved, non-sensitive context here.
        safe_context = " ".join(
            "{}={!r}".format(key, value)
            for key, value in context.items()
        )

        logger.exception(
            "Operation failed | operation=%s | %s "
            "| duration_seconds=%.3f",
            operation,
            safe_context,
            duration_seconds
        )

        raise
