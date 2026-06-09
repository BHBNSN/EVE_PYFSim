from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_under_repo(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} resolved outside repo: {resolved}") from exc
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def write_text_if_changed(path: Path, content: str) -> bool:
    if path.exists() and normalized_text(path.read_text(encoding="utf-8")) == normalized_text(content):
        return False
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    return True


def copy_file_if_changed(source: Path, target: Path) -> bool:
    if target.exists() and sha256(source) == sha256(target):
        return False
    shutil.copyfile(source, target)
    return True


def read_pyfa_version(version_path: Path) -> str:
    text = version_path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*version:\s*(\S+)\s*$", text)
    if match is None:
        raise RuntimeError(f"Could not parse Pyfa version from {version_path}")
    version = match.group(1)
    if not version.startswith("v"):
        raise RuntimeError(f"Pyfa version should look like a git tag such as v2.66.4, got: {version}")
    return version


def read_zip_db_signature(zip_path: Path) -> tuple[int, str] | None:
    if not zip_path.exists():
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            info = archive.getinfo("eve.db")
            digest = hashlib.sha256()
            size = 0
            with archive.open(info, "r") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    size += len(chunk)
                    digest.update(chunk)
            return size, digest.hexdigest()
    except (KeyError, zipfile.BadZipFile, OSError):
        return None


def write_zip(db_path: Path, zip_path: Path, tmp_root: Path, *, db_size: int, db_hash: str) -> tuple[int, str, bool]:
    existing_signature = read_zip_db_signature(zip_path)
    if existing_signature == (db_size, db_hash):
        return zip_path.stat().st_size, db_hash, True

    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_zip = tmp_root / "eve_db.zip"
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        info = zipfile.ZipInfo("eve.db", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, db_path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    restore_dir = tmp_root / "restore"
    restore_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp_zip, "r") as archive:
        archive.extractall(restore_dir)
    restored_db = restore_dir / "eve.db"
    if not restored_db.exists():
        raise RuntimeError("Compressed archive did not restore eve.db")
    restored_size = restored_db.stat().st_size
    restored_hash = sha256(restored_db)
    if restored_size != db_path.stat().st_size or restored_hash != sha256(db_path):
        raise RuntimeError("Compressed eve.db failed round-trip validation")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    copy_file_if_changed(tmp_zip, zip_path)
    return zip_path.stat().st_size, restored_hash, False


def update_manifest(path: Path, *, pyfa_version: str, db_hash: str, db_size: int) -> None:
    sde_build = None
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        sde_build = data.get("sde_build")
    payload = {
        "sde_build": sde_build,
        "pyfa_version": pyfa_version,
        "eve_db_sha256": db_hash,
        "eve_db_size": db_size,
    }
    write_text_if_changed(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def update_readme(path: Path, *, pyfa_version: str, db_hash: str, db_size: int, zip_size: int) -> None:
    content = f"""# Pyfa test database

This directory intentionally stores only the compressed Pyfa `eve.db` used by
tests and CI.

- Pyfa source is fetched in CI from `pyfa-org/Pyfa` at tag `{pyfa_version}`.
- `eve_db.zip` contains `eve.db` copied from a working Pyfa install.
- Compressed `eve_db.zip` size: `{zip_size}` bytes.
- Uncompressed `eve.db` SHA256:
  `{db_hash}`
- Uncompressed `eve.db` size: `{db_size}` bytes.

The uncompressed database is not kept in the repository root because runtime
code expects it under `Pyfa-master/eve.db`, and `Pyfa-master/` remains ignored.

Regenerate these files after updating `Pyfa-master`:

```powershell
python scripts/update_pyfa_vendor.py
```
"""
    write_text_if_changed(path, content)


def update_ci_ref(path: Path, pyfa_version: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"(?ms)(repository:\s*pyfa-org/Pyfa\s+ref:\s*)\S+",
        rf"\g<1>{pyfa_version}",
        text,
        count=1,
    )
    if count != 1 and not re.search(rf"(?m)^\s*ref:\s*{re.escape(pyfa_version)}\s*$", text):
        raise RuntimeError(f"Could not update Pyfa ref in {path}")
    write_text_if_changed(path, updated)


def validate_manifest(root: Path) -> None:
    command = [
        sys.executable,
        "-c",
        (
            "from eve_sim.sde_manifest import validate_default_manifest; "
            "errors = validate_default_manifest(); print(errors); "
            "raise SystemExit(1 if errors else 0)"
        ),
    ]
    subprocess.run(command, cwd=root, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh vendored Pyfa test DB artifacts after manually updating "
            "Pyfa-master/version.yml and Pyfa-master/eve.db."
        )
    )
    parser.add_argument(
        "--pyfa-root",
        default=None,
        help="Path to Pyfa source root. Defaults to ./Pyfa-master.",
    )
    parser.add_argument(
        "--skip-manifest-validation",
        action="store_true",
        help="Skip validate_default_manifest() after writing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    pyfa_root = Path(args.pyfa_root).resolve() if args.pyfa_root else root / "Pyfa-master"
    manifest_path = ensure_under_repo(root / "eve_sim" / "version_manifest.json", root, "Manifest path")
    ci_path = ensure_under_repo(root / ".github" / "workflows" / "ci.yml", root, "CI workflow path")
    third_party_dir = ensure_under_repo(root / "third_party" / "pyfa", root, "Third-party pyfa path")
    zip_path = third_party_dir / "eve_db.zip"
    readme_path = third_party_dir / "README.md"
    tmp_root = ensure_under_repo(root / ".tmp" / "pyfa-vendor-update", root, "Temp path")

    version_path = pyfa_root / "version.yml"
    db_path = pyfa_root / "eve.db"
    if not version_path.exists():
        raise RuntimeError(f"Missing Pyfa version.yml: {version_path}")
    if not db_path.exists():
        raise RuntimeError(f"Missing Pyfa eve.db: {db_path}")

    pyfa_version = read_pyfa_version(version_path)
    db_size = db_path.stat().st_size
    db_hash = sha256(db_path)
    zip_size, restored_hash, zip_reused = write_zip(db_path, zip_path, tmp_root, db_size=db_size, db_hash=db_hash)
    if restored_hash != db_hash:
        raise RuntimeError("Restored eve.db hash changed unexpectedly")

    update_manifest(manifest_path, pyfa_version=pyfa_version, db_hash=db_hash, db_size=db_size)
    update_readme(readme_path, pyfa_version=pyfa_version, db_hash=db_hash, db_size=db_size, zip_size=zip_size)
    update_ci_ref(ci_path, pyfa_version)

    if not args.skip_manifest_validation:
        validate_manifest(root)

    print("Updated Pyfa vendor assets")
    print(f"  Pyfa version: {pyfa_version}")
    print(f"  eve.db SHA256: {db_hash}")
    print(f"  eve.db size: {db_size}")
    print(f"  eve_db.zip size: {zip_size}")
    print(f"  {'Reused' if zip_reused else 'Updated'}: third_party/pyfa/eve_db.zip")
    print("  Updated: eve_sim/version_manifest.json")
    print("  Updated: third_party/pyfa/README.md")
    print("  Updated: .github/workflows/ci.yml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
