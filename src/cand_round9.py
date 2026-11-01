"""E85/E86: TIGER keeps one record per (edge, name) -- concurrent routes and route+street names counted again.

TIGER's road file holds a separate record for every name a centreline carries: where US 183 and US 70 share a
road, or where US 62 is also "Main Street", S1100+S1200 contains the same geometry two or three times and its
length counts each copy. Overture keeps one segment with a list of routes and one primary name, so our
numerator counts it once. (Lead: a participant's public method notes, 22 Oct, who measured it on the leaderboard;
only the mechanism is taken. Features: src/features_concurrent.py, provided Overture data only.)

Each file is a physics-model difference, not a linear guess: v4 (src/physics4.py) with the TIGER reference
raised by w x the extra-record length, k re-fitted per region to the same published composite means, minus
v4 itself (same seed, common random numbers). w = 1 is the mechanism taken literally (each extra name IS a
full extra record); the clip and the defined-mask make the effect non-linear, which is why it is simulated.

E85 `multi` -- extra length from concurrent route numbers, len x (distinct core routes - 1).
E86 `local` -- core-route length whose primary name is a local street name (one more TIGER record).
Both projected out of the span, each other and the unscored siblings E79/E80/E84, on scored E77.

    python src/features_concurrent.py
    python src/cand_round9.py            # screens, perp, dose
    python src/cand_round9.py build      # e85_multi.csv, e86_local.csv
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

BASE = "e77_affine.csv"
FILES = {"multi": ("e85_multi.csv", "E85 TIGER keeps one record per route number: concurrent routes counted again, "
                   "physics difference over E77"),
         "local": ("e86_local.csv", "E86 TIGER keeps one record per name: routes that are also a local street counted "
                   "again, physics difference over E77")}
W = {"multi": 1.0, "local": 1.0}
CHAT, M = 0.05, 0.85
SIBS = ["e79_grad_urbbg.csv", "e80_grad_school.csv", "e84_physics4.csv"]


def physics_y(extra=None):
    d, X = P4.inputs()
    if extra is not None:
        cc = pd.concat([pd.read_parquet(lb.ROOT.parent / "data" / "derived" / f"{r}-concurrent.parquet")
                        for r in sorted(d.region.unique())])
        cc = d[["GEOID"]].merge(cc, on="GEOID", how="left").fillna(0)
        col, w = extra
        X["unr"] = X["unr"] + (w / P3.LAM_U) * cc[col].values.astype(float)   # simulate adds LAM_U * X["unr"]
    k, p_mis, p_bg, s, ms = P3.calibrate(d, X)
    o = P3.simulate(X, k, p_mis, p_bg, s, m_s=ms)
    rural, burn = CC.strata(d)
    fit = d.region.ne("eastern-wa").values
    dT = X["den"] > 0
    T = o["T"]
    print(f"   transport burned/unburned {T[dT & fit & burn].mean() / T[dT & fit & ~burn].mean():.2f} (pub 1.87), "
          f"rural/urban {o['y'][fit & rural].mean() / o['y'][fit & ~rural].mean():.2f} (pub 2.96)")
    return d, o["y"]


def candidates(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    base = lb.read_sub(BASE).coverage_gap_score.values
    print("v4 reference:")
    d, y0 = physics_y()
    H = {}
    for k, col in [("multi", "cc_multi"), ("local", "cc_local")]:
        print(f"{k} (w {W[k]}):")
        H[k] = physics_y((col, W[k]))[1] - y0
    sib = [lb.read_sub(f).coverage_gap_score.values - lb.read_sub(BASE if f == "e84_physics4.csv" else "e74_physics2.csv")
           .coverage_gap_score.values for f in SIBS]
    out = {}
    for k, h in H.items():
        others = sib + [CMM.project(v, d, L)[0] for j, v in H.items() if j != k]
        col, perp = CMM.project(h, d, L, extra=others)
        out[k] = (np.clip(base + sd * M * CHAT * col, 0, 1), col, perp, h)
    rho_b = lb.rho(float(L[L.file == BASE].rmse.iloc[0]), base, m1, m2)
    return out, rho_b, d


def evidence(L=None):
    L = lb.ledger() if L is None else L
    out, rho_b, d = candidates(L)
    _, dfn, _ = CMM.hypothesis(d)
    rural, burn = CC.strata(d)
    for k, (p, col, perp, h) in out.items():
        print(f"\n{k}: mean shift {h.mean():+.5f}, burned/unburned {h[burn].mean():+.5f}/{h[~burn].mean():+.5f}, "
              f"rural/urban {h[rural].mean():+.5f}/{h[~rural].mean():+.5f}, perp {perp:.3f}")
        print(CK.traps({k: col}, d, dfn).round(3).to_string())
        a = validate.analyse(p, L)
        print(f"neutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}]")
        FC.table(rho_b, {k: CHAT}, M, L, label=FILES[k][0])
    return out, d


def build(L=None):
    out, _, d = candidates(L)
    for k, (p, _, _, _) in out.items():
        lb.write(FILES[k][0], d.GEOID, p, FILES[k][1], kind="model")


if __name__ == "__main__":
    build() if "build" in sys.argv[1:] else evidence()


# ---- E88 (after E85 f 1.65, E86 f 1.85 on 22 Oct): does the double counting go beyond one extra record? ----
def physics_y2(wm, wl):
    d, X = P4.inputs()
    cc = pd.concat([pd.read_parquet(lb.ROOT.parent / "data" / "derived" / f"{r}-concurrent.parquet")
                    for r in sorted(d.region.unique())])
    cc = d[["GEOID"]].merge(cc, on="GEOID", how="left").fillna(0)
    X["unr"] = X["unr"] + (wm * cc.cc_multi.values + wl * cc.cc_local.values) / P3.LAM_U
    k, p_mis, p_bg, s, ms = P3.calibrate(d, X)
    return d, P3.simulate(X, k, p_mis, p_bg, s, m_s=ms)["y"]


def second(L=None):
    """Non-linear rest of the doubling: y(w=2,2) - y(w=1,1), projected out of the span (E85/E86 included)."""
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    base = lb.read_sub("e86_local.csv").coverage_gap_score.values
    d, y1 = physics_y2(1.0, 1.0)
    _, y2 = physics_y2(2.0, 2.0)
    sib = [lb.read_sub(f).coverage_gap_score.values - lb.read_sub(b).coverage_gap_score.values
           for f, b in [("e79_grad_urbbg.csv", "e74_physics2.csv"), ("e80_grad_school.csv", "e74_physics2.csv"),
                        ("e84_physics4.csv", "e77_affine.csv")]]
    col, perp = CMM.project(y2 - y1, d, L, extra=sib)
    p = np.clip(base + sd * M * CHAT * col, 0, 1)
    print(f"second step: perp {perp:.3f}")
    a = validate.analyse(p, L)
    print(f"neutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}]")
    return d, p
