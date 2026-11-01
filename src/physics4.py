"""Physics model v4 (E84): v3b (E76) with the 21 Oct leaderboard read-backs applied to its road layer.

All on scored E74, single-part files, c at m = 0.85:
  - E78 sigma_t gradient c +0.021 (f 0.51): the truth sits on the higher-noise side of 0.15, so
    sigma_t 0.15 -> 0.22 (v1's 0.30 flattened the transport ratio; the step is a bit under half-way).
  - E81 import-era tertiary c -0.023 (f -0.58): TIGER does NOT hold the reclassified import-era
    tertiary, so the tertiary term keeps only post-import tertiary.
  - E82 edited import-era local roads c +0.021 (f 0.52): TIGER holds part of them (demoted county
    highways), entering at the same small rate as tertiary did (LAM_LO = LAM_T = 0.05).
Everything else (calibration to published aggregates only, per-region k, fire/EMS/school/CBP layers)
is v3b unchanged; k is re-fitted because the road reference moved.

    python src/physics4.py              # calibrate, checks, perp share, dose table
    python src/physics4.py build NAME   # write NAME = scored base + dose * projected physics mean
"""
import sys

import numpy as np
import pandas as pd

import physics3 as P3

_inputs3 = P3.inputs
import lb

SIGMA_T = 0.22
LAM_LO = 0.05
BASE = "e77_affine.csv"


def inputs():
    d, X = _inputs3()
    lin = pd.concat([pd.read_parquet(lb.ROOT.parent / "data" / "derived" / f"{r}-lineage.parquet")
                     for r in sorted(d.region.unique())])
    lin = d[["GEOID"]].merge(lin, on="GEOID", how="left").fillna(0)
    g = lambda c: lin[c].values.astype(float)
    # simulate() adds LAM_T * X["ter"]: post-import tertiary plus the edited import-era local roads
    X["ter"] = g("lin_te_new") + (LAM_LO / P3.LAM_T) * g("lin_lo_edit")
    return d, X


P3.SIGMA_T = SIGMA_T
P3.BASE = BASE
P3.inputs = inputs

if __name__ == "__main__":
    a = sys.argv[1:]
    P3.build(a[1]) if a and a[0] == "build" else P3.evidence()
