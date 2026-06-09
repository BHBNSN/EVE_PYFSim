from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json

from .config import resolve_pyfa_source_dir


DEFAULT_MANIFEST_PATH = Path(__file__).with_name("version_manifest.json")


@dataclass(frozen=True, slots=True)
class DataVersionManifest:
    sde_build: str | None = None
    pyfa_version: str | None = None
    eve_db_sha256: str | None = None
    eve_db_size: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataVersionManifest":
        return cls(
            sde_build=None if data.get("sde_build") is None else str(data.get("sde_build")),
            pyfa_version=None if data.get("pyfa_version") is None else str(data.get("pyfa_version")),
            eve_db_sha256=None if data.get("eve_db_sha256") is None else str(data.get("eve_db_sha256")),
            eve_db_size=None if data.get("eve_db_size") is None else int(data.get("eve_db_size")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sde_build": self.sde_build,
            "pyfa_version": self.pyfa_version,
            "eve_db_sha256": self.eve_db_sha256,
            "eve_db_size": self.eve_db_size,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pyfa_version(root: Path) -> str | None:
    version_file = root / "version.yml"
    if not version_file.exists():
        return None
    for line in version_file.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("version:"):
            continue
        return line.split(":", 1)[1].strip() or None
    return None


def build_current_manifest(pyfa_root: str | Path | None = None) -> DataVersionManifest:
    root = Path(pyfa_root) if pyfa_root is not None else resolve_pyfa_source_dir()
    eve_db = root / "eve.db"
    pyfa_version = _read_pyfa_version(root)
    if not eve_db.exists():
        return DataVersionManifest(pyfa_version=pyfa_version)
    return DataVersionManifest(
        pyfa_version=pyfa_version,
        eve_db_sha256=_sha256(eve_db),
        eve_db_size=eve_db.stat().st_size,
    )


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> DataVersionManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DataVersionManifest.from_dict(data)


def validate_manifest(
    expected: DataVersionManifest,
    current: DataVersionManifest | None = None,
) -> list[str]:
    actual = current or build_current_manifest()
    mismatches: list[str] = []
    for field in ("sde_build", "pyfa_version", "eve_db_sha256", "eve_db_size"):
        wanted = getattr(expected, field)
        if wanted is None:
            continue
        got = getattr(actual, field)
        if got != wanted:
            mismatches.append(f"{field}: expected {wanted!r}, got {got!r}")
    return mismatches


def validate_default_manifest() -> list[str]:
    return validate_manifest(load_manifest(DEFAULT_MANIFEST_PATH))


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DataVersionManifest",
    "build_current_manifest",
    "load_manifest",
    "validate_default_manifest",
    "validate_manifest",
]
