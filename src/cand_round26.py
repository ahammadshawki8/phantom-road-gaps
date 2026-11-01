"""E114 / E115: the certified re-dose of the 27 Oct chain, and TIGER's one-record-per-name rule
re-applied on the spec vehicle with the boundary highways included.

**The convention correction (LEDGER 114a).** Every per-component mean the project has used as a target
(transport 0.194 burned / 0.104 unburned, POI 0.146 / 0.043, buildings 0.008 / 0.005; thread 63) is a mean
over ALL tracts with an undefined component written as 0 -- the README's CSV convention ("written as 0
with _defined = false"), and thread 63's own Camp Fire table shows it (06007001703: road gap 0.000 and
coverage 0.191 = (0.381 + 0.001)/2, i.e. the road is undefined there and printed as 0). The 16 published
composite cells decide it: under the all-tracts reading the component means imply composites of 0.118
(burned) / 0.0535 (unburned) against the cells' 0.1178 / 0.0530; under the defined-tracts reading 0.113 /
0.041. So the transport burned/unburned 1.87 is a ratio of ALL-tract means. The E110/E111 vehicle gives
2.665 on that reading (it was pinned to 1.87 on the defined reading, where the convention test of 13 Oct
had rejected the all-tracts reading only because the ring model of the day could not reach it).

**E115 -- the counting rules on the spec vehicle, boundary highways included.** TIGER keeps one record
per (edge, name) and the scorer does not deduplicate (E85 +0.083, E86 +0.093, E97 +0.045): a centreline
carrying two route numbers, a route number and a local street name, or an honorific highway name is
counted once per name. Those parts were measured on the old vehicle (ramp-free numerator, 15.49 m ring)
and enter our files as linear parts; the vehicle has since changed three times (E109 ramps in, E110 w,
E111 TIGER's own boundary line). And E111 added the boundary highways whose Overture copy lies in the
NEIGHBOURING tract -- their extra records were never counted anywhere, because cc_* is measured on the
copy. `features_bndfollow` now carries them (bf_x_multi / bf_x_local / bf_x_named, ~11,700 km against
17,139 km of bf_out). V2 = V0 + interior extras + boundary extras, k re-fitted; the part is V2 - V0.
Unfitted, on the corrected convention: transport burned/unburned 2.665 -> 1.962 (published 1.87),
interior extras alone 2.148, boundary extras alone 2.276.

    python src/cand_round26.py              # evidence: both conventions, traps, perp, dose table
    python src/cand_round26.py build        # e114_affine14.csv, e115_records.csv
"""
import sys

import numpy as np
import pandas as pd

import strata as CC
import cand_knn as CK
import cand_mm as CMM
import cand_round25 as R
import features as F
import lb
import table
import validate

PINNED = R.PINNED
AFFINE_FILES = ["e105_schoolbreadth.csv", "e109_ramp_in.csv", "e110_ringspec.csv", "e111_bndtiger.csv",
                "e112_toll.csv", "e113_sigma.csv", "e107_region.csv", "e102_recall.csv", "e98_schoolcat.csv",
                "const_0.20.csv", "e92_affine.csv", "e96_physics8.csv", "e95_tribal_roads.csv", "e87_affine.csv"]
RIDGE = 1e-6
E114 = "e114_affine14.csv"
E115 = ("e115_records.csv", 0.030, 0.040,
        "E115 TIGER keeps one record per name on the spec vehicle: the counting rules re-applied with the "
        "boundary highways' own extra records counted in the tract that holds TIGER's line")


def counting_extras():
    o = table.sample_order()[["GEOID"]]
    cc = o.merge(pd.concat([pd.read_parquet(F.OUT / f"{r}-concurrent.parquet") for r in F.REGIONS]),
                 on="GEOID", how="left").fillna(0)
    b = R.bndfollow()
    inner = (cc.cc_multi + cc.cc_local + cc.cc_named).values
    bound = (b.bf_x_multi + b.bf_x_local + b.bf_x_named).values
    return inner, bound


def describe(tag, d, X, o):
    rural, burn = CC.strata(d)
    fit = d.region.ne("eastern-wa").values
    dT, T = X["den"] > 0, o["T"]
    print(f"   {tag:10s} transport b/u all-tracts {T[fit & burn].mean() / T[fit & ~burn].mean():.3f} (pub 1.87) | "
          f"defined-only {T[dT & fit & burn].mean() / T[dT & fit & ~burn].mean():.3f} | T mean all-tracts "
          f"{T[fit].mean():.4f} (pub 0.112) | composite r/u {o['y'][fit & rural].mean() / o['y'][fit & ~rural].mean():.3f}"
          f" (pub 2.96)")


def part(L=None, quiet=False):
    L = lb.ledger() if L is None else L
    v0 = R.dens()["both"]
    inner, bound = counting_extras()
    d, X0, o0 = R.model(v0)
    _, X2, o2 = R.model(v0 + inner + bound)
    if not quiet:
        describe("V0", d, X0, o0)
        describe("V2", d, X2, o2)
    h = o2["y"] - o0["y"]
    col, perp = CMM.project(h, d, L)
    return d, h, col, perp, X0["den"] > 0


def affine(L=None):
    import final_model as FM
    L = lb.ledger() if L is None else L
    q, pred, err, lam = FM.affine(RIDGE, AFFINE_FILES, L)
    a = validate.analyse(q, L)
    g = lb.read_sub(AFFINE_FILES[0]).GEOID.values
    for geoid, v in PINNED.items():
        q[int(np.where(g == geoid)[0][0])] = v
    return g, q, pred, a, lam


def evidence(L=None):
    L = lb.ledger() if L is None else L
    g, q, pred, a, lam = affine(L)
    print(f"E114 affine over {len(AFFINE_FILES)} scored files, ridge {RIDGE:g}: identity {pred:.6f}, validator "
          f"{a['rmse_neutral']:.6f}, in span {a['share_explained_by_scored']:.4f}; best {validate.best_rmse(L):.6f}")
    print("   weights", lam)
    d, h, col, perp, dfn = part(L)
    name, W, chat, _ = E115
    print(f"\n{name}: reach {(np.abs(h) > 1e-6).mean():.1%}, sd {h.std():.5f}, max {np.abs(h).max():.3f}, perp {perp:.3f}")
    print(CK.traps({"records": col}, d, dfn).round(3).to_string())
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    for c in [0.0, W / 2, chat, 0.06]:
        print(f"   c {c:+.3f} -> {sd * np.sqrt(pred ** 2 / sd ** 2 + W * (W - 2 * c)):.6f} (on the E114 identity, before pins)")
    return g, q, col, sd


def build(L=None):
    L = lb.ledger() if L is None else L
    g, q, col, sd = evidence(L)
    lb.write(E114, g, q, f"E114 affine over {len(AFFINE_FILES)} scored files (ridge {RIDGE:g}) plus the two "
                         f"scorer-pinned water tracts", kind="model")
    name, W, _, comment = E115
    p = np.clip(q + sd * W * col, 0, 1)
    for geoid, v in PINNED.items():
        p[int(np.where(g == geoid)[0][0])] = v
    lb.write(name, g, p, comment, kind="model")


if __name__ == "__main__":
    build() if sys.argv[1:2] == ["build"] else evidence()
