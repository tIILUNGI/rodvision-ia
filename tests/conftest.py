from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
PM = ROOT / ".venv" / "Scripts" / "python.exe"
SMOKE_CONFIG = ROOT / "configs" / "smoke.yaml"
MILSTONE_SEED = 10452
