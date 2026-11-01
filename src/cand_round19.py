"""E105: the school reference's BREADTH is the region's taxonomy, not the tract's own listings.

**Where the day's five refutations lead.** 25 Oct refuted, free: a burn *level* term (E99 already
reproduces thread 63's eight published cells at 0.093); business/loop/spur routes (`CORE` in
`features_routes.py` has carried the state and US modifier networks since E19); the buildings
defined-mask (buildings are undefined in **2** tracts of 9,794); raising the school/EMS reference
rate to reach the published "fire 8.7x worse than schools" (at M_S >= 0.6 the facilities half alone
exceeds the published POI mean 0.052 and `s` has no root); and adding a taxonomy-breadth term on top
of the existing one (the breadth shortfall correlates **-0.778** with school count -- the E26 trap
class). **Together they say one thing: every reference RATE in the facilities half is pinned at the
ceiling the published POI mean allows.** What is not pinned is how a fixed amount of reference is
DISTRIBUTED, and that is what this file changes.

**The mechanism.** The README defines the facilities half as "mean over defined of fire, EMS, schools
(**6 school categories**)". Every model since E56 computes the school gap as ONE aggregate ratio,
`gs = 1 - min(1, sc/rs)`, and builds `rs` as the tract's own Overture school count plus misfiled
extras -- so a tract that lists only elementary schools is compared against a reference of only
elementary schools and scores a gap of zero. A real school roster is not shaped like that: a
community spans the categories communities of its size span. This file compares per category and
draws the reference's breadth from the **region's own** category-presence rates `q_c` (elementary
0.75-0.80, school 0.40-0.50, high 0.33-0.41, middle 0.19-0.31, private 0.15-0.23, public 0.14-0.29 --
measured, flat across all five regions, not fitted):

    r_c = scp_c + Bernoulli(lam_bg * q_c)        gs = mean over {c : r_c > 0} of (1 - min(1, scp_c/r_c))

`scp_c` is Overture's per-category count **by `categories.primary`**, which is E49's confirmed
counting rule (+0.120), and `sum(scp_*) == sc_n` **exactly** (max difference 0 over 9,794 tracts), so
the numerator is the one every model already uses, merely partitioned.

**E98 tested the other reading of that parenthesis and it is refuted.** E98 made the six categories
six *components of the facilities half*, running the divisor to 8, and read **-0.013**. This file
leaves `nf = (rf>0)+(re>0)+(rs>0)` untouched and changes only `gs`. The two are different objects.

**It is MEAN-PRESERVING, which is the whole point.** `lam_bg` is not a free parameter: it is solved
so the mean school gap is identical to the aggregate rule's, **0.01150 -> 0.01153** at
`lam_bg* = 0.0219`. The facilities half, the POI mean 0.052, `s`, the composite mean and every
published level therefore do not move at all. The file is a **pure re-ordering of which tracts carry
the school gap** -- from tracts with misfiled schools to tracts whose listed taxonomy is narrower
than their region's. Row 29a leaves 98.5% of the error within cell, and after four files that moved
levels and read -0.038, +0.013, -0.002 and 0.000, this one moves nothing else.

**Diagnostics.** Reach **85.2%**, mean |h| **0.0127** (13x E104's), max 0.205, perp **0.711**. Traps:
road-undefined +0.001, log density +0.012, rural -0.000, burned +0.018, E26 file +0.000,
**ss-on-undefined 0.279 against a 0.324 base rate** (under-concentrated, the opposite of E98's 0.472).
Correlation with every school sibling after projection -- E56, E59, E98 -- and with E99 and E104 is
**+-0.000**. The one thing stated against it: the raw breadth shortfall correlates -0.778 with school
count, i.e. narrow taxonomy largely means few schools; **after the span is projected out that falls to
-0.104**, because E56/E58/E59 already hold the count direction. If this file reads negative, that
residual count content is where to look.

**Price.** The school component is the best-paying vein we have (E56 misfiled schools **+0.124**,
E53 nfac **+0.044**) and counting rules are the best-paying family (E85 +0.083, E86 +0.093,
E97 +0.045, E99 +0.046) while every geometric refinement died. This is a counting rule on that
component, at 85% reach, mean-preserving. c_hat **0.07**.

**Dose, fixed by the hard goal.** From the best scored base E99 (0.05748527), beating **rival_F
(0.057299837)** needs (RMSE/SD)^2 to improve by 0.0026123, so the dose minimising the required
read-back is W* = sqrt(...) = **0.0505**, at which the minimum winning c is also 0.0505 -- below the
price, so the goal is reachable at c_hat rather than only above it. Dead null **0.057651** (slot cost
0.000166, still our second-best file).

    python src/cand_round19.py          # the solve, traps, perp, dose
    python src/cand_round19.py build    # e105_schoolbreadth.csv
"""
import io
import sys
import contextlib

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import physics4 as P4          # patches physics3 into v4
import physics3 as P3
import strata as CC
import cand_knn as CK
import cand_mm as CMM
import lb
import validate

BEST = "e99_joint.csv"          # 0.05748527
NAME = "e105_schoolbreadth.csv"
NOTE = ("E105 the school reference spans the region's six-category taxonomy, not the tract's own "
        "listings: per-category school gaps at a mean-preserving breadth, physics difference over E99")
RIVAL_F = 0.057299837
CHAT = 0.07
CATS = ["elementary_school", "middle_school", "high_school", "school", "private_school", "public_school"]
BG_SEED = 20261025


def catcounts(d):
    sc = pd.concat([pd.read_parquet(lb.ROOT.parent / "data" / "derived" / f"{r}-schoolcat.parquet")
                    for r in sorted(d.region.unique())])
    sc = d[["GEOID"]].merge(sc, on="GEOID", how="left").fillna(0)
    return np.asarray(sc[[f"scp_{c}" for c in CATS]].values, dtype=float)


def presence(d, prim):
    """Each region's own category-presence rates among school-having tracts. Measured, not fitted."""
    has = prim.sum(1) > 0
    q = np.zeros_like(prim)
    for r in sorted(d.region.unique()):
        m = (d.region == r).values
        q[m] = (prim[m & has] > 0).mean(0)
    return q, has


def make_simulate(prim, q, has, lam_bg, per_category):
    """physics3.simulate with the school gap per category; every other draw identical."""
    def sim(X, k, p_mis, p_bg, s, draws=P3.DRAWS, seed=P3.SEED, parts=False, p_comm=P3.P_COMM, m_s=1.0):
        rng = np.random.default_rng(seed)
        bg = np.random.default_rng(BG_SEED)
        n = len(X["num"])
        dT = X["den"] > 0
        lam_f = p_mis * X["fdm"] + P3.P_BLD * X["bfs"] + p_comm * X["miss"] + p_bg * X["bg"]
        lam_e = m_s * P3.P_EMS * X["emm"]
        lam_s = m_s * (P3.P_SCH * X["scm"] + P3.P_SCHB * X["scb"])
        cbp0 = s * X["r0"] * np.maximum(X["nost"], 0.02) ** P3.W_ST
        acc = {"y": np.zeros(n), "T": np.zeros(n), "P": np.zeros(n), "fblk": np.zeros(n),
               "fmis": np.zeros(n), "gs": np.zeros(n), "fac": np.zeros(n)}
        for _ in range(draws):
            tiger = k * X["den"] * np.exp(P3.SIGMA_T * rng.standard_normal(n) - P3.SIGMA_T ** 2 / 2) \
                + P3.LAM_U * X["unr"] + P3.LAM_T * X["ter"]
            T = np.where(dT, 1 - np.minimum(1, X["num"] / np.maximum(tiger, 1e-9)), 0.0)
            rf = X["fd"] + rng.poisson(lam_f)
            re = X["em"] + rng.poisson(lam_e)
            rs = X["sc"] + rng.poisson(lam_s)          # SAME draw in both variants
            gf = np.where(rf > 0, 1 - np.minimum(1, X["fd"] / np.maximum(rf, 1)), 0.0)
            ge = np.where(re > 0, 1 - np.minimum(1, X["em"] / np.maximum(re, 1)), 0.0)
            if per_category:
                r_c = prim + (bg.random(prim.shape) < (lam_bg * q)) * has[:, None]
                dfn = r_c > 0
                nd = dfn.sum(1)
                gap = np.where(dfn, 1 - np.minimum(1, prim / np.maximum(r_c, 1)), 0.0)
                gs = np.where(nd > 0, gap.sum(1) / np.maximum(nd, 1), 0.0)
            else:
                gs = np.where(rs > 0, 1 - np.minimum(1, X["sc"] / np.maximum(rs, 1)), 0.0)
            nf = (rf > 0).astype(float) + (re > 0) + (rs > 0)      # divisor UNCHANGED (E98 disjoint)
            fac = np.where(nf > 0, (gf + ge + gs) / np.maximum(nf, 1), 0.0)
            cbp_ref = cbp0 * np.exp(P3.SIGMA_C * rng.standard_normal(n) - P3.SIGMA_C ** 2 / 2)
            cbp = 1 - np.minimum(1, X["poi"] / np.maximum(cbp_ref, 1e-9))
            P = np.where(nf > 0, (fac + cbp) / 2, cbp)
            y = (T * dT + P3.B_CONST + P) / (dT + 2.0)
            acc["y"] += y; acc["T"] += T; acc["P"] += P; acc["gs"] += gs; acc["fac"] += fac
            acc["fblk"] += (rf > 0) & (X["fd"] == 0)
            acc["fmis"] += (rf > 0) & (X["fd"] == 0) & (X["fdm"] + X["bfs"] > 0)
        return {k_: v / draws for k_, v in acc.items()}
    return sim


def run(d, X, prim, q, has, lam_bg, per_category):
    sim = make_simulate(prim, q, has, lam_bg, per_category)
    old = P3.simulate
    P3.simulate = sim
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            k, p_mis, p_bg, s, ms = P3.calibrate(d, X)
            o = sim(X, k, p_mis, p_bg, s, m_s=ms)
    finally:
        P3.simulate = old
    return o


def candidates(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    d, X = P4.inputs()
    prim = catcounts(d)
    q, has = presence(d, prim)
    fit = d.region.ne("eastern-wa").values
    print(f"numerator partition exact (sum(scp_*) == sc_n): {np.abs(prim.sum(1) - X['sc']).max() == 0}")
    print(f"schools listed in {has.mean():.1%} of tracts, mean {(prim > 0).sum(1)[has].mean():.2f} of 6 "
          f"categories; region presence rates q_c measured, not fitted")

    # gs depends on neither k nor s, so lam_bg is solved on the school gap alone -- two full
    # calibrations in all, and the solve is exact rather than nested inside them.
    lam_s = P3.M_S * (P3.P_SCH * X["scm"] + P3.P_SCHB * X["scb"])

    def gs_only(lam_bg, per_category):
        rng = np.random.default_rng(P3.SEED)
        bg = np.random.default_rng(BG_SEED)
        a = np.zeros(len(prim))
        for _ in range(P3.DRAWS):
            rs = X["sc"] + rng.poisson(lam_s)
            if per_category:
                r_c = prim + (bg.random(prim.shape) < (lam_bg * q)) * has[:, None]
                dfn = r_c > 0
                a += np.where(dfn.sum(1) > 0,
                              np.where(dfn, 1 - np.minimum(1, prim / np.maximum(r_c, 1)), 0.0).sum(1)
                              / np.maximum(dfn.sum(1), 1), 0.0)
            else:
                a += np.where(rs > 0, 1 - np.minimum(1, X["sc"] / np.maximum(rs, 1)), 0.0)
        return a / P3.DRAWS

    tgt = gs_only(0.0, False)[fit].mean()
    print(f"\naggregate rule (every model since E56): mean school gap {tgt:.5f}")
    print("solving lam_bg so the mean school gap is UNCHANGED (mean-preserving):")
    for z in [0.01, 0.02, 0.03]:
        print(f"   lam_bg {z:.4f} -> school gap {gs_only(z, True)[fit].mean():.5f}")
    lam = brentq(lambda z: gs_only(z, True)[fit].mean() - tgt, 1e-4, 0.08, xtol=2e-4)
    print(f"   lam_bg* {lam:.4f} -> school gap {gs_only(lam, True)[fit].mean():.5f}  (HELD)")
    o0 = run(d, X, prim, q, has, 0.0, False)
    o1 = run(d, X, prim, q, has, lam, True)
    rural, burn = CC.strata(d)
    for lab, o in [("aggregate", o0), ("per-category", o1)]:
        print(f"   {lab:13s} school gap {o['gs'][fit].mean():.5f}  facilities {o['fac'][fit].mean():.4f}  "
              f"POI {o['P'][fit].mean():.4f} (pub 0.052)  y {o['y'][fit].mean():.4f} (pub 0.0585)  "
              f"rural/urban {o['y'][fit & rural].mean() / o['y'][fit & ~rural].mean():.2f} (pub 2.96)")
    h = o1["y"] - o0["y"]
    col, perp = CMM.project(h, d, L)
    base = lb.read_sub(BEST).coverage_gap_score.values
    rho_b = lb.rho(float(L[L.file == BEST].rmse.iloc[0]), base, m1, m2)
    need = (sd ** 2 * (1 - rho_b ** 2) - RIVAL_F ** 2) / sd ** 2
    W = round(float(np.sqrt(max(need, 1e-12))), 4)
    p = np.clip(base + sd * W * col, 0, 1)
    return d, prim, h, col, perp, p, rho_b, W, sd


def evidence(L=None):
    L = lb.ledger() if L is None else L
    d, prim, h, col, perp, p, rho_b, W, sd = candidates(L)
    _, dfn, _ = CMM.hypothesis(d)
    rural, burn = CC.strata(d)
    print(f"\npart: reach {np.mean(np.abs(h) > 1e-6):.1%}, mean |h| {np.abs(h).mean():.5f}, "
          f"max {np.abs(h).max():.4f}, perp {perp:.3f}")
    print(f"part: mean shift {h.mean():+.6f}, rural/urban {h[rural].mean():+.6f}/{h[~rural].mean():+.6f}")
    print(CK.traps({"breadth": col}, d, dfn).round(3).to_string())
    nb = (prim > 0).sum(1).astype(float); ns = prim.sum(1).astype(float)
    print(f"   after projection: corr with school COUNT "
          f"{np.corrcoef(col, (ns - ns.mean()) / ns.std())[0, 1]:+.3f}, with BREADTH "
          f"{np.corrcoef(col, (nb - nb.mean()) / nb.std())[0, 1]:+.3f}")
    for f_ in ["e56_schools.csv", "e58_schpart.csv", "e59_schclean.csv", "e98_schoolcat.csv",
               "e53_nfac.csv", "e49_fireblk.csv", "e99_joint.csv", "e104_burnlag.csv"]:
        try:
            v = lb.read_sub(f_).coverage_gap_score.values
        except Exception:
            continue
        print(f"   corr with {f_}: {np.corrcoef(col, (v - v.mean()) / v.std())[0, 1]:+.3f}")
    a = validate.analyse(p, L)
    print(f"\nneutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}] "
          f"best {validate.best_rmse(L):.6f}")
    print(f"\ndose W = {W:.4f} (minimises the c needed to beat rival_F {RIVAL_F:.9f}); "
          f"breakeven c > {W / 2:.4f}; c_hat {CHAT}")
    for c in [0.0, W / 2, 0.040, W, CHAT, 0.124]:
        r = sd * np.sqrt(1 - rho_b ** 2 + W * (W - 2 * c))
        tag = "   BEATS RIVAL_F" if r < RIVAL_F else ("   beats E99" if r < 0.05748527 else "")
        print(f"   c {c:+.4f} -> {r:.6f}{tag}")
    return d, p


def build(L=None):
    d, p = evidence(L)
    lb.write(NAME, d.GEOID, p, NOTE, kind="model")


if __name__ == "__main__":
    build() if "build" in sys.argv[1:] else evidence()
