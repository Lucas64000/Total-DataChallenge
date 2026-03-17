from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path


class LogLevel(Enum):
    """Supported log levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogCategory(Enum):
    """Logical categories used in logger names."""

    PREPROCESSING = "preprocessing"
    DATA_ANALYSIS = "data_analysis"
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    API = "api"
    GENERAL = "general"


class PhotoTrapLogger:
    """Wrapper around Python logging."""

    _ROOT_NAME = "PhotoTrap"
    _LOG_FORMAT = "%(asctime)s | %(name)-45s | %(levelname)-8s | %(message)s"
    _DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        base_log_dir: str | Path = "logs",
        console_output: bool = True,
        file_output: bool = True,
        log_level: LogLevel = LogLevel.INFO,
    ) -> None:
        self._level = log_level.value
        self._root = logging.getLogger(self._ROOT_NAME)
        self._configure_handlers(base_log_dir, console_output, file_output)

    def _configure_handlers(
        self,
        base_log_dir: str | Path,
        console_output: bool,
        file_output: bool,
    ) -> None:
        self._root.setLevel(self._level)
        self._root.propagate = False

        for handler in self._root.handlers[:]:
            handler.close()
            self._root.removeHandler(handler)

        formatter = logging.Formatter(self._LOG_FORMAT, datefmt=self._DATE_FORMAT)

        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self._level)
            console_handler.setFormatter(formatter)
            self._root.addHandler(console_handler)

        if file_output:
            log_dir = Path(base_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "phototrap.log", encoding="utf-8")
            file_handler.setLevel(self._level)
            file_handler.setFormatter(formatter)
            self._root.addHandler(file_handler)

    def get_logger(
        self,
        category: LogCategory = LogCategory.GENERAL,
        module_name: str | None = None,
    ) -> logging.Logger:
        """Return a logger scoped by category and optional module name."""
        name = f"{self._ROOT_NAME}.{category.value}"
        if module_name:
            name = f"{name}.{module_name}"

        logger = logging.getLogger(name)
        logger.setLevel(self._level)
        logger.propagate = True
        return logger

    def set_level(self, level: LogLevel) -> None:
        """Update the root logger level and handlers."""
        self._level = level.value
        self._root.setLevel(self._level)
        for handler in self._root.handlers:
            handler.setLevel(self._level)


_global_logger_instance: PhotoTrapLogger | None = None


def get_phototrap_logger() -> PhotoTrapLogger:
    """Get the global logger instance."""
    global _global_logger_instance
    if _global_logger_instance is None:
        _global_logger_instance = PhotoTrapLogger()
    return _global_logger_instance


def init_logging(
    log_dir: str | Path = "logs",
    console_output: bool = True,
    file_output: bool = True,
    log_level: LogLevel = LogLevel.INFO,
) -> PhotoTrapLogger:
    """Reinitialize the global logger instance."""
    global _global_logger_instance
    _global_logger_instance = PhotoTrapLogger(
        base_log_dir=log_dir,
        console_output=console_output,
        file_output=file_output,
        log_level=log_level,
    )
    return _global_logger_instance


def _log(level: str, category: LogCategory, message: str, module_name: str | None = None) -> None:
    logger = get_phototrap_logger().get_logger(category, module_name)
    getattr(logger, level)(message)


def log_info(category: LogCategory, message: str, module_name: str | None = None) -> None:
    _log("info", category, message, module_name)


def log_error(category: LogCategory, message: str, module_name: str | None = None) -> None:
    _log("error", category, message, module_name)


def log_warning(category: LogCategory, message: str, module_name: str | None = None) -> None:
    _log("warning", category, message, module_name)


def log_debug(category: LogCategory, message: str, module_name: str | None = None) -> None:
    _log("debug", category, message, module_name)
