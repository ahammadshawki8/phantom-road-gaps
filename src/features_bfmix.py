"""Boundary-following TIGER length with railway boundaries out AND a per-region distance cut (E120, part A),
provided data only.

`features_bfrail` drops a boundary sample when a rail line is closer than the parallel core route. Rivers, canals,
right-of-way lines and section lines are not in the package, so a boundary that follows one of them 25-60 m from a
highway still passes. Seen on 28 Oct: 53037975202 runs 32-60 m from Canyon Rd along the Yakima River, 04025000616
holds a constant 27.3-27.7 m offset for 1 km with no road, path or rail within 60 m, and 40143003000 follows the
Arkansas River trail 38-51 m from the highway.

The distance itself separates them. A sample on TIGER's own highway line sits where Overture traces that highway,
a median 1.5 m away with an exponential tail. A coincidental parallel feature sits anywhere in the 60 m window. So
each region's followed samples (rail out, both sides of the boundary, 5-60 m) are fitted by maximum likelihood to
an exponential tail over a flat floor, and a sample is kept as the highway when it is more likely the tail than
the floor (posterior 1/2). Unfitted to anything published. The cuts come out at OK 21.4 m, TX 19.2 m and AZ 15.8 m,
where the rural OK/TX tail is long, and at WA 11.3 m and CA 10.6 m, where the floor starts early.

    bf_out_m                                         bf_out with railway boundaries out and the distance cut applied
    bf_x_multi_m / _local_m / _named_m / _desig_m    the boundary extras on the same samples
    cut_m                                            the region's cut (m)

Usage:  python src/features_bfmix.py [region ...]
Output: $BB_DATA/derived/<region>-bfmix.parquet
"""
import sys

import numpy as np
from scipy.optimize import minimize

import features as F
import features_bndfollow as FB
import features_desig as FD

LO = 5.0
COLS = ["x_multi", "x_local", "x_named", "x_desig"]


def cut(dc):
    """Distance (m) where the region's exponential tail and flat floor are equally likely."""
    x = dc[(dc >= LO) & (dc <= FB.D)]

    def nll(p):
        pi, tau = 1 / (1 + np.exp(-p[0])), np.exp(p[1])
        z = np.exp(-LO / tau) - np.exp(-FB.D / tau)
        return -np.log(pi * np.exp(-x / tau) / (tau * z) + (1 - pi) / (FB.D - LO)).sum()
    best = min((minimize(nll, [a, b], method="Nelder-Mead") for a in [1, 3, 5] for b in [0.5, 1.5, 2.5]),
               key=lambda o: o.fun)
    pi, tau = 1 / (1 + np.exp(-best.x[0])), np.exp(best.x[1])
    z = np.exp(-LO / tau) - np.exp(-FB.D / tau)
    grid = np.arange(LO, FB.D + 0.01, 0.1)
    tail = pi * np.exp(-grid / tau) / (tau * z)
    w = tail / (tail + (1 - pi) / (FB.D - LO))
    return float(grid[np.argmin(np.abs(w - 0.5))]), pi, tau


def build(r):
    _, s = FB.build(r, extra={"desig": FD.DESIG}, write=False, rail=True, samples=True)
    c, pi, tau = cut(s.dc.values)
    o = s[~s.inside.astype(bool) & (s.dc <= c)]
    agg = o.groupby("GEOID").agg(bf_out_m=("dc", "size"), **{f"bf_{k}_m": (k, "sum") for k in COLS}) * FB.STEP
    con = F.connect()
    F.load_tracts(con, r)
    df = con.sql("SELECT GEOID FROM tracts ORDER BY GEOID").df().merge(agg.reset_index(), on="GEOID", how="left")
    df = df.fillna(0.0).assign(cut_m=c)
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-bfmix.parquet", index=False)
    print(f"  {r}: tail share {pi:.3f}, tau {tau:.2f} m -> cut {c:.1f} m; bf_out {df.bf_out_m.sum() / 1e3:,.0f} km",
          flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
