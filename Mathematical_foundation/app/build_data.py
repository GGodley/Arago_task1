#!/usr/bin/env python3
"""Precompute the dashboard's speed x memory grid for the preset workloads (writes app/data.json).

    python Mathematical_foundation/app/build_data.py
then serve the folder:  python -m http.server -d Mathematical_foundation/app 8000
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from foundation.optimizer import grid  # noqa: E402

PRESETS = [
    ("128³ (small)", 128, 128, 128), ("256³ (small)", 256, 256, 256),
    ("512³ (medium)", 512, 512, 512), ("1024³ (medium)", 1024, 1024, 1024),
    ("2048³ (large)", 2048, 2048, 2048), ("4096³ (large)", 4096, 4096, 4096),
    ("LLM decode 64×4096×4096", 64, 4096, 4096), ("Wide, short K 4096×64×4096", 4096, 64, 4096),
    ("Long K 64×8192×64", 64, 8192, 64), ("Padded 100×200×300", 100, 200, 300),
]

if __name__ == "__main__":
    data = [grid(M, K, N, name) for name, M, K, N in PRESETS]
    (HERE / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    print("wrote", HERE / "data.json")
