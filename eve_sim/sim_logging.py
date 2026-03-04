from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .config import EngineConfig


MAX_LOG_FILE_BYTES = 100 * 1024 * 1024
TRIMMED_LOG_BYTES = 95 * 1024 * 1024


def _trim_log_head_if_needed(path: Path, max_bytes: int = MAX_LOG_FILE_BYTES, trim_to_bytes: int = TRIMMED_LOG_BYTES) -> None:
    try:
        if not path.exists():
            return
        size = path.stat().st_size
        if size <= max_bytes:
            return
        keep = max(1, min(trim_to_bytes, max_bytes, size))
        with path.open("rb+") as fp:
            if keep < size:
                fp.seek(-keep, os.SEEK_END)
            else:
                fp.seek(0)
            tail = fp.read()
            line_break = tail.find(b"\n")
            if line_break >= 0 and (line_break + 1) < len(tail):
                tail = tail[line_break + 1 :]
            fp.seek(0)
            fp.write(tail)
            fp.truncate()
    except Exception:
        return


class _BoundedFileHandler(logging.FileHandler):
    def __init__(
        self,
        filename: str,
        mode: str = "a",
        encoding: str | None = None,
        delay: bool = False,
        max_bytes: int = MAX_LOG_FILE_BYTES,
        trim_to_bytes: int = TRIMMED_LOG_BYTES,
    ) -> None:
        self.max_bytes = max_bytes
        self.trim_to_bytes = trim_to_bytes
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self._trim_if_needed()

    def _trim_if_needed(self) -> None:
        stream = self.stream
        if stream is None:
            return
        try:
            stream.flush()
        except Exception:
            pass
        try:
            size = stream.tell()
        except Exception:
            try:
                size = os.path.getsize(self.baseFilename)
            except Exception:
                return
        if size <= self.max_bytes:
            return
        try:
            stream.close()
        except Exception:
            pass
        _trim_log_head_if_needed(Path(self.baseFilename), self.max_bytes, self.trim_to_bytes)
        self.stream = self._open()


def _stringify_field(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text if text else "0"
    if isinstance(value, (list, tuple, set)):
        return ",".join(_stringify_field(v) for v in value)
    text = str(value)
    return text.replace("\n", " ").replace("\r", " ").strip()


def format_sim_event(event: str, **fields: Any) -> str:
    parts: list[str] = [f"event={event}"]
    for key in sorted(fields.keys()):
        value = fields[key]
        if value is None:
            continue
        raw = _stringify_field(value)
        if not raw:
            continue
        safe = raw.replace('"', "'")
        if any(ch.isspace() for ch in safe) or "|" in safe:
            safe = f'"{safe}"'
        parts.append(f"{key}={safe}")
    return " ".join(parts)


def log_sim_event(logger: logging.Logger | None, event: str, **fields: Any) -> None:
    if logger is None or logger.disabled:
        return
    logger.info(format_sim_event(event, **fields))


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
    logger.setLevel(logging.INFO)

    log_path = Path(config.detail_log_file)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _trim_log_head_if_needed(log_path)

    target = str(log_path.resolve())
    normalized_target = Path(target).resolve().as_posix()
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler_path = Path(handler.baseFilename).resolve().as_posix()
            if handler_path != normalized_target:
                logger.removeHandler(handler)
                try:
                    handler.flush()
                except Exception:
                    pass
                try:
                    handler.close()
                except Exception:
                    pass

    for handler in logger.handlers:
        if isinstance(handler, _BoundedFileHandler) and Path(handler.baseFilename).resolve().as_posix() == normalized_target:
            handler.setLevel(logging.INFO)
            handler._trim_if_needed()
            return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = _BoundedFileHandler(target, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
