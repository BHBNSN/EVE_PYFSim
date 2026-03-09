from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence

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


class _SimEventFilter(logging.Filter):
    def __init__(self, *, event: str, include: bool) -> None:
        super().__init__()
        self.event = event
        self.include = include

    def filter(self, record: logging.LogRecord) -> bool:
        record_event = getattr(record, "sim_event", None)
        is_match = record_event == self.event
        return is_match if self.include else not is_match


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
    logger.info(format_sim_event(event, **fields), extra={"sim_event": event})


def _resolve_log_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    _trim_log_head_if_needed(path)
    return path.resolve()


def _close_handler(handler: logging.Handler) -> None:
    try:
        handler.flush()
    except Exception:
        pass
    try:
        handler.close()
    except Exception:
        pass


def _filter_signature(filters: Sequence[object]) -> tuple[tuple[str, bool], ...]:
    signature: list[tuple[str, bool]] = []
    for filt in filters:
        if isinstance(filt, _SimEventFilter):
            signature.append((filt.event, filt.include))
    return tuple(signature)


def _ensure_file_handler(
    logger: logging.Logger,
    *,
    path: Path,
    filters: list[logging.Filter],
) -> None:
    normalized_target = path.as_posix()
    target_signature = _filter_signature(filters)
    for handler in logger.handlers:
        if not isinstance(handler, _BoundedFileHandler):
            continue
        handler_path = Path(handler.baseFilename).resolve().as_posix()
        handler_signature = _filter_signature(list(handler.filters))
        if handler_path == normalized_target and handler_signature == target_signature:
            handler.setLevel(logging.INFO)
            handler._trim_if_needed()
            return

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = _BoundedFileHandler(str(path), encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    for filt in filters:
        handler.addFilter(filt)
    logger.addHandler(handler)


def get_sim_logger(config: EngineConfig) -> logging.Logger:
    logger = logging.getLogger("eve_sim.sim")
    logger.propagate = False

    if not (config.detailed_logging or getattr(config, "hotspot_logging", False)):
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            _close_handler(handler)
        logger.setLevel(logging.CRITICAL + 1)
        logger.disabled = True
        return logger

    logger.disabled = False
    logger.setLevel(logging.INFO)

    desired_handlers: set[tuple[str, tuple[tuple[str, bool], ...]]] = set()
    if config.detailed_logging:
        detail_path = _resolve_log_path(config.detail_log_file)
        detail_filters: list[logging.Filter] = [_SimEventFilter(event="hotspot", include=False)]
        desired_handlers.add((detail_path.as_posix(), _filter_signature(detail_filters)))
        _ensure_file_handler(logger, path=detail_path, filters=detail_filters)

    if getattr(config, "hotspot_logging", False):
        hotspot_path = _resolve_log_path(getattr(config, "hotspot_log_file", "logs/sim_hotspot.log"))
        hotspot_filters: list[logging.Filter] = [_SimEventFilter(event="hotspot", include=True)]
        desired_handlers.add((hotspot_path.as_posix(), _filter_signature(hotspot_filters)))
        _ensure_file_handler(logger, path=hotspot_path, filters=hotspot_filters)

    for handler in list(logger.handlers):
        if not isinstance(handler, _BoundedFileHandler):
            logger.removeHandler(handler)
            _close_handler(handler)
            continue
        handler_key = (
            Path(handler.baseFilename).resolve().as_posix(),
            _filter_signature(list(handler.filters)),
        )
        if handler_key not in desired_handlers:
            logger.removeHandler(handler)
            _close_handler(handler)

    return logger
