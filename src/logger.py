import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

# Set up default paths relative to project root
DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "app.log"
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Keep max ~5MB of logs locally (1MB per file x 5 backups)
MAX_BYTES = 1 * 1024 * 1024
BACKUP_COUNT = 5


def setup_logger(
    log_file: Path | str = DEFAULT_LOG_FILE,
    level: str = DEFAULT_LOG_LEVEL,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
) -> logging.Logger:
    """Configures the root logger with rotating file handler."""
    root_logger = logging.getLogger()

    # Avoid duplicate handlers if setup is called multiple times
    if root_logger.hasHandlers():
        return root_logger

    root_logger.setLevel(level)

    # Make sure logs/ directory exists before creating file
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Format: [2026-07-21 19:27:31.347] [INFO] [module:function:line] - message
    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Use UTF-8 and automatic file rotation
    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Convenience getter that ensures setup runs on first call."""
    # Auto-initialize logger if imported before setup_logger() was called
    if not logging.getLogger().hasHandlers():
        setup_logger()

    return logging.getLogger(name)
