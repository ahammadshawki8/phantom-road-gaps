"""Boundary-following TIGER length with railway boundaries taken out (E120, part A), provided data only.

`features_bndfollow` counts a tract-boundary sample as TIGER's own highway line when an Overture core route is
the nearest ROAD within 60 m and runs parallel to the boundary. But tract boundaries also follow railways
(TIGER R1011), and a railway beside a highway is not in the roads layer, so the nearest-road test passes and the
highway is credited with a boundary it does not hold. Checked on the three largest "transport defined, no
Overture highway inside" tracts on 28 Oct: the boundary runs 30-55 m from the highway along a railway or a river,
where a highway-following boundary sits within the Overture/TIGER tracing displacement (median 1.5 m, p90 5-6 m).
This module re-measures the boundary terms with `rail=True` (`overture-rail`, a sample is rejected when a rail
line is closer than the core route). It removes 303 km of 27,758 (1.1%) and makes 17 tracts road-undefined.

    bf_out_r                          bf_out with railway boundaries out
    bf_x_multi_r / _local_r / _named_r / _desig_r   the boundary extras (features_bndfollow / features_desig), same

Usage:  python src/features_bfrail.py [region ...]
Output: $BB_DATA/derived/<region>-bfrail.parquet
"""
import sys

import features as F
import features_bndfollow as FB
import features_desig as FD

COLS = ["bf_out", "bf_x_multi", "bf_x_local", "bf_x_named", "bf_x_desig"]


def build(r):
    df = FB.build(r, extra={"desig": FD.DESIG}, write=False, rail=True)
    df = df[["GEOID"] + COLS].rename(columns={c: c + "_r" for c in COLS})
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-bfrail.parquet", index=False)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
