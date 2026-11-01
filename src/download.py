"""Download the challenge package through the Source Cooperative proxy (data.source.coop).

The proxy is ~85x faster than the raw us-west-2 S3 endpoint from outside AWS (measured
2.8 MB/s vs 33 KB/s). Resumable: files whose size already matches are skipped.

Usage:  python src/download.py [region ...]     (default: all five)  -> $BB_DATA
        python src/download.py --strata            the small strata tables only (a few MB), enough for quick_check.py
"""
import os
import sys
import urllib.request
from pathlib import Path

DATA = Path(os.environ.get("BB_DATA", Path(__file__).resolve().parents[2] / "data"))
BASE = "https://data.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge"
REGIONS = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
REF = ["overture-pois", "overture-roads", "overture-roads-unfiltered", "overture-infrastructure",
       "overture-rail", "census-acs-housing", "overture-buildings"]
STRATA = ["strata-tract-table", "census-tracts", "tribal-tract-table"]
UA = {"User-Agent": "Mozilla/5.0 (bias-bounty downloader)"}  # proxy rejects urllib's default UA


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch(rel):
    url, dst = f"{BASE}/{rel}", DATA / rel
    size = remote_size(url)
    if dst.exists() and dst.stat().st_size == size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, \
            open(tmp, "wb") as fh:
        while chunk := r.read(1 << 22):
            fh.write(chunk)
    assert tmp.stat().st_size == size, f"truncated download: {rel}"
    tmp.replace(dst)
    print(f"  {rel}  {size / 1e6:.1f} MB", flush=True)


def region(r, strata_only=False):
    fetch(f"reference/{r}/{r}-sample-submission.csv")
    for layer in [] if strata_only else REF:
        fetch(f"reference/{r}/{r}-{layer}.parquet")
    for layer in STRATA:
        fetch(f"strata/{r}/{r}-{layer}.parquet")


if __name__ == "__main__":
    only = "--strata" in sys.argv
    for r in ([a for a in sys.argv[1:] if a != "--strata"] or REGIONS):
        print(r, flush=True)
        region(r, strata_only=only)
