from __future__ import annotations

import os
from pathlib import Path
import tempfile


_TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp" / "test-temp"
_TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)

for _name in ("TMP", "TEMP", "TMPDIR"):
    os.environ[_name] = str(_TEST_TEMP_ROOT)

tempfile.tempdir = str(_TEST_TEMP_ROOT)
