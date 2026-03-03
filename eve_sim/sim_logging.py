from __future__ import annotations

import logging
from pathlib import Path

from .config import EngineConfig


def get_sim_logger(config: EngineConfig) -> logging.Logger:
    logger = logging.getLogger("eve_sim.sim")
    logger.propagate = False

    if not config.detailed_logging:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.flush()
            except Exception:
                pass
            try:
                handler.close()
            except Exception:
                pass
        logger.setLevel(logging.CRITICAL + 1)
        logger.disabled = True
        return logger

    logger.disabled = False
    logger.setLevel(logging.DEBUG)

    log_path = Path(config.detail_log_file)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    target = str(log_path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve().as_posix() == Path(target).resolve().as_posix():
            return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.FileHandler(target, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
