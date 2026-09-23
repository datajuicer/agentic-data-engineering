from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def get_logs_dir(project_root_dir: str | Path, category: str) -> Path:
    path = Path(project_root_dir) / "logs" / category
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(name: str, log_dir: str | Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = RotatingFileHandler(Path(log_dir) / f"{name}.log.jsonl", maxBytes=10 * 1024 * 1024, backupCount=5)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
