"""E97: the honorific highway name is a THIRD TIGER record, and nobody has counted it.

TIGER files one linear feature per (centreline, name). E85 counted the extra records that come from
concurrent route NUMBERS (len x (nref - 1), read back c +0.083) and E86 the extra record that comes from a
route also carrying a LOCAL STREET name (c +0.093). Both took their name list from `routes[].ref` and from
`names.primary` when it does not look like a highway. That leaves one name class entirely uncounted: a
primary name that IS a highway name but names none of the segment's own route numbers --

    "Pearl Harbor Memorial Highway"  on I 10        "Eastex Freeway"      on US 59 / US 69
    "Arizona Veterans Highway"       on AZ 17       "Beeline Highway"     on AZ 87
    "Texas Vietnam Veterans Memorial Highway" on US 83   "Katy Freeway"   on US 90 / I 10

an honorific, historic or descriptive name. TIGER holds it as its own S1100/S1200 feature, so the
centreline is in the reference once per route number AND once more under the honorific; Overture keeps one
segment, so our numerator counts it once. E86's HWY regex threw exactly this class away as "already the
route record".

Disjoint from E86 by construction (cc_local is the primary names that do NOT match HWY, cc_named the ones
that do but name no ref of their own) and 1.65x its reach: named/core 0.277 OK, 0.666 AZ, 0.317 CA, 0.146 TX,
0.165 WA, 24.6% of the 93,933 km of core route pooled, against E86's 14.9%.

Same vehicle as E85/E86 so the read-back is directly comparable: a physics v4 difference, reference raised by
w = 1.0 x the extra-record length (the mechanism taken literally -- each extra name IS a full extra record),
k re-fitted to the same published composite means, minus v4 itself under common random numbers. The clip and
the defined-mask make it non-linear, which is why it is simulated rather than guessed. `CMM.project` removes
the whole scored span (E85/E86/E88 included, so this cannot be a re-reading of them) and the five region
dummies (AZ 0.67 vs TX 0.15 is a big region level, and E25 measured region z at -0.126).

    python src/features_concurrent.py      # adds cc_named
    python src/cand_round10.py             # verify E85/E86 rebuild, screens, traps, perp, dose
    python src/cand_round10.py build       # e97_named.csv
"""
import sys

import numpy as np
import pandas as pd

import physics4 as P4          # patches physics3 into v4
import physics3 as P3
import strata as CC
import cand_knn as CK
import cand_mm as CMM
import forecast as FC
import lb
import validate

BASE = "e96_physics8.csv"      # the best SCORED file (0.057738168); never build on an unscored base
NAME = "e97_named.csv"
NOTE = ("E97 TIGER keeps one record per name: the honorific highway name of a numbered route counted again, "
        "physics difference over E96")
W, CHAT, M = 1.0, 0.05, 0.85
DERIVED = lb.ROOT.parent / "data" / "derived"


def concurrent(d):
    cc = pd.concat([pd.read_parquet(DERIVED / f"{r}-concurrent.parquet") for r in sorted(d.region.unique())])
    return d[["GEOID"]].merge(cc, on="GEOID", how="left").fillna(0)


def physics_y(cols=None, quiet=False):
    """v4's y, optionally with the TIGER reference raised by sum(w * extra-record length)."""
    d, X = P4.inputs()
    if cols:
        cc = concurrent(d)
        X["unr"] = X["unr"] + sum(w * cc[c].values.astype(float) for c, w in cols.items()) / P3.LAM_U
    k, p_mis, p_bg, s, ms = P3.calibrate(d, X)
    o = P3.simulate(X, k, p_mis, p_bg, s, m_s=ms)
    if not quiet:
        rural, burn = CC.strata(d)
        fit, dT, T = d.region.ne("eastern-wa").values, X["den"] > 0, o["T"]
        print(f"   transport burned/unburned {T[dT & fit & burn].mean() / T[dT & fit & ~burn].mean():.2f} "
              f"(pub 1.87), rural/urban {o['y'][fit & rural].mean() / o['y'][fit & ~rural].mean():.2f} (pub 2.96)")
    return d, o["y"]


def verify(d, y0):
    """The cc_named column was added to an existing parquet: E85/E86 must still rebuild from it."""
    m1, m2 = lb.moments()
    sd = np.sqrt(m2 - m1 ** 2)
    base = lb.read_sub("e77_affine.csv").coverage_gap_score.values
    L85 = lb.ledger()
    sib = [lb.read_sub(f).coverage_gap_score.values - lb.read_sub(b).coverage_gap_score.values
           for f, b in [("e79_grad_urbbg.csv", "e74_physics2.csv"), ("e80_grad_school.csv", "e74_physics2.csv"),
                        ("e84_physics4.csv", "e77_affine.csv")]]
    L85 = L85[~L85.file.isin(["e85_multi.csv", "e86_local.csv", "e87_affine.csv", "e88_double2.csv",
                              "e89_affine.csv", "e90_physics5.csv", "e91_divided.csv", "e92_affine.csv",
                              "e93_physics6.csv", "e95_tribal_roads.csv", "e96_physics8.csv"])]
    H = {k: physics_y({c: 1.0}, quiet=True)[1] - y0 for k, c in [("multi", "cc_multi"), ("local", "cc_local")]}
    for k, f in [("multi", "e85_multi.csv"), ("local", "e86_local.csv")]:
        others = sib + [CMM.project(v, d, L85)[0] for j, v in H.items() if j != k]
        col, _ = CMM.project(H[k], d, L85, extra=others)
        p = np.clip(base + sd * M * 0.05 * col, 0, 1)
        got = lb.read_sub(f).coverage_gap_score.values
        print(f"   {f}: rebuild corr {np.corrcoef(p, got)[0, 1]:.6f}, max|diff| {np.abs(p - got).max():.2e}")
    return H


def candidates(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    print("v4 reference:")
    d, y0 = physics_y()
    print("E85/E86 rebuild check (features_concurrent.py gained a column):")
    H = verify(d, y0)
    print(f"named (w {W}):")
    h = physics_y({"cc_named": W})[1] - y0
    col, perp = CMM.project(h, d, L)
    base = lb.read_sub(BASE).coverage_gap_score.values
    p = np.clip(base + sd * M * CHAT * col, 0, 1)
    rho_b = lb.rho(float(L[L.file == BASE].rmse.iloc[0]), base, m1, m2)
    return d, h, col, perp, p, rho_b, H


def evidence(L=None):
    L = lb.ledger() if L is None else L
    d, h, col, perp, p, rho_b, H = candidates(L)
    _, dfn, _ = CMM.hypothesis(d)
    rural, burn = CC.strata(d)
    cc = concurrent(d)
    reach = cc.cc_named.values > 0
    print(f"\nnamed: reach {reach.mean():.1%} of tracts, named/core {cc.cc_named.sum() / cc.cc_core.sum():.3f}")
    print(f"named: mean shift {h.mean():+.5f}, burned/unburned {h[burn].mean():+.5f}/{h[~burn].mean():+.5f}, "
          f"rural/urban {h[rural].mean():+.5f}/{h[~rural].mean():+.5f}, perp {perp:.3f}")
    print(CK.traps({"named": col}, d, dfn).round(3).to_string())
    for k, v in H.items():
        print(f"   corr with E{'85' if k == 'multi' else '86'}'s raw part: "
              f"{np.corrcoef(h, v)[0, 1]:+.3f}, after projection {np.corrcoef(col, CMM.project(v, d, L)[0])[0, 1]:+.3f}")
    a = validate.analyse(p, L)
    print(f"\nneutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}] "
          f"best {validate.best_rmse(L):.6f}")
    FC.table(rho_b, {"named": CHAT}, M, L, label=NAME)
    sd = np.sqrt(lb.moments(L)[1] - lb.moments(L)[0] ** 2)
    print(f"\ndose W = {M * CHAT:.4f}  breakeven c > {M * CHAT / 2:.4f}")
    for c in [0.0, 0.025, 0.05, 0.083, 0.093]:
        print(f"   c {c:+.3f} -> {sd * np.sqrt(1 - rho_b ** 2 + M * CHAT * (M * CHAT - 2 * c)):.6f}")
    return d, p


def build(L=None):
    d, p = evidence(L)
    lb.write(NAME, d.GEOID, p, NOTE, kind="model")


if __name__ == "__main__":
    build() if "build" in sys.argv[1:] else evidence()
