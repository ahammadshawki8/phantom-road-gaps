"""E99: the JOINT recalibration of the three TIGER name-record rules, which no file holds.

`physics8.py screen` on the 42-file span (24 Oct) puts **`cc_named` at +0.158, the largest partial
correlation in the table** -- larger than tribal_legal_pct (+0.116), bfs_n (+0.111) or anything in the
heat cluster. That is not a new mechanism: E97 measured it at c +0.0446. It is there because the physics
model has **no cc_named parameter at all** (THETA carries W_MULTI and W_LOCAL from E85/E86, and nothing
for the honorific name), so the model misfits in that direction even though a submission holds it.

What is genuinely unowned is the JOINT effect. E85, E86 and E97 were each built as a one-at-a-time
difference with k re-fitted for that one change:

    h_multi = y(cc_multi=1) - y0     h_local = y(cc_local=1) - y0     h_named = y(cc_named=1) - y0

But the three rules act on the SAME centrelines, and the reference enters through `1 - min(1, num/ref)`
with a per-region k re-solved to a fixed published mean. A segment carrying three route numbers AND an
honorific name has four TIGER records, and the clip, the mask and the single shared recalibration make
that strictly different from the sum of the three separate corrections. This file is exactly that
remainder -- the second- and third-order interaction:

    h = y(1,1,1) - h_multi - h_local - h_named - y0
      = y(1,1,1) - y_multi - y_local - y_named + 2*y0

Nothing in the span holds it: E85/E86/E97 hold the three first-order terms and E88 holds the
`y(2,2) - y(1,1)` doubling of two of them, and all four are projected out here along with the rest of the
span and the five region dummies.

**Price, from the one direct precedent.** E88 is the same construction for two of the three rules and read
**+0.028** (f 0.55 of a 0.05 price). Three rules give a larger interaction than two, but the first-order
family has been halving (+0.083, +0.093, +0.0446), so c_hat **0.03** and W = 0.85*c_hat = 0.0255,
breakeven c > 0.0128. This is a small, cheap bet, and it is deliberately small: see LEDGER 99 for why
nothing larger was available today.

    python src/cand_round12.py          # screens, traps, perp, dose
    python src/cand_round12.py build    # e99_joint.csv
"""
import sys

import numpy as np

import physics4 as P4          # patches physics3 into v4
import physics3 as P3
import strata as CC
import cand_knn as CK
import cand_mm as CMM
import forecast as FC
import lb
import validate
from cand_round10 import concurrent

BASE = "e97_named.csv"         # the best SCORED file (0.057602037)
NAME = "e99_joint.csv"
NOTE = ("E99 the joint recalibration of the three TIGER one-record-per-name rules: the interaction the "
        "one-at-a-time files cannot hold, physics difference over E97")
CHAT, M = 0.03, 0.85
COLS = ["cc_multi", "cc_local", "cc_named"]


def physics_y(weights, quiet=True):
    """v4's y with the TIGER reference raised by sum(w_c * extra-record length) over the given columns."""
    d, X = P4.inputs()
    if weights:
        cc = concurrent(d)
        X["unr"] = X["unr"] + sum(w * cc[c].values.astype(float) for c, w in weights.items()) / P3.LAM_U
    k, p_mis, p_bg, s, ms = P3.calibrate(d, X)
    o = P3.simulate(X, k, p_mis, p_bg, s, m_s=ms)
    if not quiet:
        rural, burn = CC.strata(d)
        fit = d.region.ne("eastern-wa").values
        print(f"   rural/urban {o['y'][fit & rural].mean() / o['y'][fit & ~rural].mean():.2f} (pub 2.96)")
    return d, o["y"]


def candidates(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    print("v4 reference:")
    d, y0 = physics_y({}, quiet=False)
    ones = {}
    for c in COLS:
        print(f"  first order: {c}")
        ones[c] = physics_y({c: 1.0})[1]
    print("  joint: all three at 1.0")
    yj = physics_y({c: 1.0 for c in COLS})[1]
    # the remainder after the three one-at-a-time corrections are each taken out
    h = yj - sum(ones[c] - y0 for c in COLS) - y0
    col, perp = CMM.project(h, d, L)
    base = lb.read_sub(BASE).coverage_gap_score.values
    p = np.clip(base + sd * M * CHAT * col, 0, 1)
    rho_b = lb.rho(float(L[L.file == BASE].rmse.iloc[0]), base, m1, m2)
    return d, h, col, perp, p, rho_b, ones, y0


def evidence(L=None):
    L = lb.ledger() if L is None else L
    d, h, col, perp, p, rho_b, ones, y0 = candidates(L)
    _, dfn, _ = CMM.hypothesis(d)
    rural, burn = CC.strata(d)
    first = sum(np.abs(ones[c] - y0) for c in COLS)
    print(f"\njoint: reach {np.mean(np.abs(h) > 1e-6):.1%} of tracts, mean |h| {np.abs(h).mean():.6f}, "
          f"max {np.abs(h).max():.5f}")
    print(f"joint: |interaction| / sum|first order| = {np.abs(h).sum() / max(first.sum(), 1e-12):.3f}")
    print(f"joint: mean shift {h.mean():+.6f}, burned/unburned {h[burn].mean():+.6f}/{h[~burn].mean():+.6f}, "
          f"rural/urban {h[rural].mean():+.6f}/{h[~rural].mean():+.6f}, perp {perp:.3f}")
    print(CK.traps({"joint": col}, d, dfn).round(3).to_string())
    for f in ["e85_multi.csv", "e86_local.csv", "e97_named.csv", "e88_double2.csv"]:
        v = lb.read_sub(f).coverage_gap_score.values
        print(f"   corr of the projected part with {f}: {np.corrcoef(col, (v - v.mean()) / v.std())[0, 1]:+.3f}")
    a = validate.analyse(p, L)
    print(f"\nneutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}] "
          f"best {validate.best_rmse(L):.6f}")
    FC.table(rho_b, {"joint": CHAT}, M, L, label=NAME)
    sd = np.sqrt(lb.moments(L)[1] - lb.moments(L)[0] ** 2)
    W = M * CHAT
    print(f"\ndose W = {W:.4f}  breakeven c > {W / 2:.4f}")
    for c in [0.0, 0.0128, 0.028, 0.03, 0.045]:
        print(f"   c {c:+.4f} -> {sd * np.sqrt(1 - rho_b ** 2 + W * (W - 2 * c)):.6f}")
    return d, p


def build(L=None):
    d, p = evidence(L)
    lb.write(NAME, d.GEOID, p, NOTE, kind="model")


if __name__ == "__main__":
    build() if "build" in sys.argv[1:] else evidence()
