import logging
import sys


class StructuredFormatter(logging.Formatter):

    def format(self, record):
        timestamp = self.formatTime(
            record,
            "%Y-%m-%dT%H:%M:%S"
        )

        return (
            f"timestamp={timestamp} "
            f"level={record.levelname} "
            f"logger={record.name} "
            f"message={record.getMessage()}"
        )


def configure_logging():
    handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(
        StructuredFormatter()
    )

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        force=True
    )