"""Logging Module.

This module provides a structured logging wrapper using `structlog`. It is designed 
to handle high-throughput AI inference and training workloads by prioritizing 
machine-readable outputs, contextual metadata, and operational safety.

The module supports:
    - Context-local variable binding (traceability across function calls).
    - Automatic PII masking.
    - Environment-based formatting (JSON for Prod, Pretty-Print for Dev).
    - Standardized ISO timestamps.

Typical usage example:
    from logger import configure_logger, get_logger
    
    configure_logger(enable_json=True)
    log = get_logger()
    log.info("model_loaded", model_version="1.0.4", device="cuda:0")
"""

from typing import Any

import structlog
from structlog.types import EventDict, Processor


def mask_sensitive_data(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Masks sensitive information in the log event dictionary.

    Iterates through a predefined list of sensitive keys and replaces their 
    values with a mask string to prevent accidental leakage of credentials 
    or PII in logs.

    Args:
        _: The logger instance (unused).
        __: The log level name (unused).
        event_dict: The dictionary containing log metadata and the message.

    Returns:
        The event dictionary with sensitive values redacted.
    """
    sensitive_keys = {"password", "token", "api_key", "secret", "access_key"}
    for key in sensitive_keys:
        if key in event_dict:
            event_dict[key] = "********"
    return event_dict


def configure_logger(enable_json: bool = False, log_level: str = "INFO") -> None:
    """Configures the global structlog pipeline.

    Sets up the processors for structured logging, including timestamping, 
    level labeling, and exception handling. It toggles between human-readable 
    console output and production-ready JSON based on the `enable_json` flag.

    Args:
        enable_json: If True, outputs logs in JSON format for ingestion by 
            log aggregators (e.g., ELK, Datadog). If False, uses colorful 
            pretty-printing for local development. Defaults to False.
        log_level: The minimum logging level to emit. Must be one of 
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL". Defaults to "INFO".

    Note:
        This function should be called once at the application entry point 
        (e.g., in `main.py` or `__init__.py`).
    """
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        mask_sensitive_data,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if enable_json:
        processors = shared_processors + [structlog.processors.JSONRenderer()]
    else:
        processors = shared_processors + [structlog.dev.ConsoleRenderer()]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger() -> structlog.BoundLogger:
    """Retrieves a pre-configured bound logger instance.

    Returns:
        An instance of a structlog BoundLogger which supports contextual 
        binding and structured logging methods.
    """
    return structlog.get_logger()


import logging

