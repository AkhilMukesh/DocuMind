
import logging
import sys

from request_context import get_request_id


class StructuredFormatter(logging.Formatter):

    def format(self, record):
        timestamp = self.formatTime(
            record,
            "%Y-%m-%dT%H:%M:%S"
        )

        request_id = get_request_id() or "-"

        return (
            f"timestamp={timestamp} "
            f"level={record.levelname} "
            f"logger={record.name} "
            f"request_id={request_id} "
            f"message={record.getMessage()}"
        )


def configure_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        force=True
    )
