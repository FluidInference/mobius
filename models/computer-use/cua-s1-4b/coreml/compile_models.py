"""Compile .mlpackage -> .mlmodelc next to each package (what the published repo ships)."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import coremltools as ct

for root in sys.argv[1:]:
    for package in sorted(Path(root).glob("*.mlpackage")):
        target = package.with_suffix(".mlmodelc")
        if target.exists():
            continue
        t0 = time.time()
        compiled = Path(ct.utils.compile_model(str(package)))
        shutil.move(str(compiled), target)
        print(f"{target} ({time.time() - t0:.0f}s)", flush=True)
