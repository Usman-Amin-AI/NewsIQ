import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.config import AppConfig


def configure_logger(config: AppConfig) -> logging.Logger:
    Path(os.path.dirname(config.log_path)).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("newsbot")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(config.log_path, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def record_event(config: AppConfig, event: str, data: Dict[str, Any]) -> None:
    logger = configure_logger(config)
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event": event,
        **data,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))
