"""Download and unpack the BEIR SciFact benchmark into ``data/scifact/``.

Uses only the standard library. Safe to re-run: it skips the download if the
extracted files are already present.
"""

from __future__ import annotations

import shutil
import ssl
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
SCIFACT_DIR = DATA_ROOT / "scifact"
ZIP_PATH = DATA_ROOT / "scifact.zip"
REQUIRED = ["corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv"]


def already_present() -> bool:
    return all((SCIFACT_DIR / rel).exists() for rel in REQUIRED)


def _fetch_with_urllib(dest: Path) -> bool:
    try:
        req = urllib.request.Request(BEIR_URL, headers={"User-Agent": "ragsearch/0.0.1"})
        with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
            dest.write_bytes(resp.read())
        return True
    except (urllib.error.URLError, ssl.SSLError) as exc:
        print(f"  urllib download failed ({exc}); trying curl")
        return False


def _fetch_with_curl(dest: Path) -> bool:
    curl = shutil.which("curl")
    if not curl:
        return False
    result = subprocess.run(
        [curl, "-sSL", "--fail", "-o", str(dest), BEIR_URL],
        check=False,
    )
    return result.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def download_and_extract() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if not (ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 0):
        print(f"downloading {BEIR_URL}")
        if not _fetch_with_urllib(ZIP_PATH) and not _fetch_with_curl(ZIP_PATH):
            raise RuntimeError("could not download scifact.zip via urllib or curl")
    print(f"  {ZIP_PATH.stat().st_size / 1e6:.1f} MB, extracting")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        # archive lays files out under a top-level "scifact/" directory
        zf.extractall(DATA_ROOT)


def main() -> int:
    if already_present():
        print(f"scifact already present at {SCIFACT_DIR}")
        return 0
    download_and_extract()
    if not already_present():
        missing = [rel for rel in REQUIRED if not (SCIFACT_DIR / rel).exists()]
        print(f"extraction incomplete, missing: {missing}", file=sys.stderr)
        return 1
    print(f"ready at {SCIFACT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
