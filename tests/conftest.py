"""Keep rebase tests bound to this checkout when another starVLA is installed."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if sys.path[0] != repo_root:
    sys.path.insert(0, repo_root)
