"""Convert the pinned Decision 1.0 Lex native release to Core ML."""

import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parents[2] / "decision-vela" / "coreml"
sys.path.insert(0, str(SHARED))

from run import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--toolkit", str(Path(__file__).resolve().parent)]
    main()
