"""
Run the full refresh from the repository root:

    python run_pipeline.py

Order: prices -> news (deduped) -> FinBERT scores -> correlation summary.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODULES = [
    "pipeline.ingest_prices",
    "pipeline.ingest_news",
    "pipeline.sentiment",
    "pipeline.correlate",
]


def main() -> None:
    for mod in MODULES:
        print(f"\n=== {mod} ===\n")
        subprocess.run([sys.executable, "-m", mod], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
