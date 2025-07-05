"""
Centralized logging configuration using pure Loguru for the RAG application.
"""
import sys
from pathlib import Path
from typing import Optional
from loguru import logger

from config import get_settings


def configure_logging():
    """
    Configure Loguru logging for the RAG application.

    This function sets up:
    - Console logging with color support
    - File logging with rotation and retention
    - Different log levels for different sinks
    - Structured logging format
    - Pure Loguru implementation (no Python logging module)
    """
    settings = get_settings()

    # Remove default handler to prevent duplicate logs
    logger.remove()

    # Console handler with colors and appropriate level
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=console_format,
        level="INFO",
        colorize=True,
        enqueue=True  # Thread-safe logging
    )

    # File handler for all logs
    log_dir = Path("logs")
    try:
        log_dir.mkdir(exist_ok=True)
    except PermissionError:
        # Fallback to current directory if we can't create logs directory
        log_dir = Path(".")
        logger.warning("Could not create logs directory, using current directory for log files")

    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    # Add file handlers with error handling
    try:
        logger.add(
            log_dir / "rag_app.log",
            format=file_format,
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            enqueue=True,
            serialize=False  # Human-readable logs
        )

        # Error-only file handler
        logger.add(
            log_dir / "errors.log",
            format=file_format,
            level="ERROR",
            rotation="5 MB",
            retention="30 days",
            compression="gz",
            enqueue=True
        )

        # JSON structured logging for potential log aggregation
        logger.add(
            log_dir / "structured.json",
            format=file_format,
            level="INFO",
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            enqueue=True,
            serialize=True  # JSON format for structured logging
        )
    except Exception as e:
        logger.warning(f"Could not set up file logging: {e}. Continuing with console logging only.")

    logger.info("Pure Loguru logging configured successfully")


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance with optional module-specific context.

    Args:
        name: Optional name to bind to the logger for context

    Returns:
        Configured Loguru logger instance
    """
    if name:
        return logger.bind(module=name)
    return logger


# Context managers for special logging scenarios
def log_execution_time(operation_name: str):
    """
    Context manager to log execution time of operations.

    Usage:
        with log_execution_time("document_processing"):
            # Your code here
            pass
    """
    import time
    from contextlib import contextmanager

    @contextmanager
    def timer():
        start_time = time.time()
        logger.info(f"Starting operation: {operation_name}")
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger.info(f"Completed operation: {operation_name} in {duration:.3f}s")

    return timer()


def log_errors_with_context(func):
    """
    Decorator to automatically log function errors with context.

    Usage:
        @log_errors_with_context
        def my_function():
            # Your code here
            pass
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.opt(exception=True).error(
                f"Error in {func.__name__}: {str(e)}"
            )
            raise

    return wrapper
