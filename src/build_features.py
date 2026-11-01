"""Every per-tract feature table, in dependency order, from the downloaded challenge package (~25 min).

    python src/download.py            # the package into ../data (no credentials)
    python src/build_features.py      # then this; each step is a standalone script and can be rerun on its own

`table.py` writes derived/all.parquet (the features joined to the strata in submission order), which later
builders read; boundary-following needs routes and concurrent routes; designations need boundary-following; the
mixed boundary and rail measures need both.
"""
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
ORDER = ["features", "table",
         "features_routes", "features_concurrent", "features_boundary", "features_quality", "features_places",
         "features_misfiled", "features_commercial", "features_lineage", "features_settle", "features_schoolcat",
         "features_cbpscope", "features_dispatch", "features_xy",
         "features_bndfollow", "features_desig", "features_bfmix", "features_bfrail", "features_unrouted",
         "features_centroid", "features_geom", "features_tiger", "features_juris"]

if __name__ == "__main__":
    t0 = time.time()
    for i, m in enumerate(ORDER, 1):
        print(f"[{i}/{len(ORDER)}] {m}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        subprocess.run([sys.executable, str(SRC / f"{m}.py")], check=True)
    print(f"all feature tables built in {(time.time() - t0) / 60:.1f} min")
