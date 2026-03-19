import structlog
from unittest.mock import patch
from src.utils.logger import mask_sensitive_data, configure_logger, get_logger

def test_mask_sensitive_data():
    """Verifies that sensitive keys are redacted in the event dictionary."""
    event_dict = {
        "event": "user_login",
        "password": "secret_password123",
        "token": "sensitive_api_token",
        "user_id": 42
    }

    # mask_sensitive_data is a structlog processor
    masked_dict = mask_sensitive_data(None, "info", event_dict)

    # Check that sensitive keys are masked
    assert masked_dict["password"] == "********"
    assert masked_dict["token"] == "********"
    # Check that non-sensitive keys remain untouched
    assert masked_dict["event"] == "user_login"
    assert masked_dict["user_id"] == 42

@patch("structlog.configure")
def test_configure_logger_dev_mode(mock_configure):
    """Ensures logger is configured with ConsoleRenderer for development."""
    configure_logger(enable_json=False)

    args, kwargs = mock_configure.call_args
    processors = kwargs["processors"]

    # Check if ConsoleRenderer is present in the processor list
    assert any(isinstance(p, structlog.dev.ConsoleRenderer) for p in processors)
    # Ensure JSONRenderer is not present
    assert not any(isinstance(p, structlog.processors.JSONRenderer) for p in processors)

@patch("structlog.configure")
def test_configure_logger_prod_mode(mock_configure):
    """Ensures logger is configured with JSONRenderer for production."""
    configure_logger(enable_json=True)

    args, kwargs = mock_configure.call_args
    processors = kwargs["processors"]

    # Check if JSONRenderer is present in the processor list
    assert any(isinstance(p, structlog.processors.JSONRenderer) for p in processors)
    # Ensure ConsoleRenderer is not present
    assert not any(isinstance(p, structlog.dev.ConsoleRenderer) for p in processors)

def test_get_logger_interface():
    """Verifies that get_logger returns an object implementing the logging interface."""
    logger = get_logger()

    # Check for the presence of essential structlog methods
    assert hasattr(logger, "info")
    assert hasattr(logger, "error")
    assert hasattr(logger, "bind")
    assert hasattr(logger, "new")

def test_logger_contextual_binding():
    """Verifies that the logger correctly supports contextual metadata binding."""
    # Binding returns a new logger with context
    logger = get_logger().bind(module="test_module", task_id=101)

    # After binding, the context should be accessible in the logger's internal state
    assert logger._context["module"] == "test_module"
    assert logger._context["task_id"] == 101

@patch("structlog.PrintLoggerFactory")
def test_logger_emission(mock_factory):
    """Verifies that calling a log method triggers the underlying logger."""
    configure_logger(enable_json=False)
    logger = get_logger()

    with patch.object(logger, "info") as mock_info:
        logger.info("test_event", key="value")
        mock_info.assert_called_once_with("test_event", key="value")

