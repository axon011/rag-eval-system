import os
import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger


LOG_FORMAT = os.getenv("LOG_FORMAT", "json")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)

        log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"
        log_record["level"] = record.levelname
        log_record["logger"] = record.name

        if hasattr(record, "latency_ms"):
            log_record["latency_ms"] = record.latency_ms

        if hasattr(record, "user_id"):
            log_record["user_id"] = record.user_id

        if hasattr(record, "event"):
            log_record["event"] = record.event


def setup_logging(name: str = "rag") -> logging.Logger:
    """Setup structured logging."""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    if logger.handlers:
        return logger

    if LOG_FORMAT == "json":
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(event)s %(message)s"
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        file_handler = RotatingFileHandler(
            "/app/logs/rag.log", maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_event(logger: logging.Logger, event: str, **kwargs):
    """Log structured event."""
    extra = {"event": event}
    extra.update(kwargs)
    logger.info(event, extra=extra)
