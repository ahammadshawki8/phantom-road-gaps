"""E102/E103: Overture's recall of the road system is not uniform inside a region, and the roads
layer says so about itself.

**The gap in every model we have built.** The hidden TIGER primary+secondary length is estimated as
`k * (core-route length + boundary ring)` with **one k per region** (`physics3.calibrate`). That is
right only if Overture holds the same *fraction* of the route system everywhere inside a region.
Nothing in the project has ever carried a per-tract recall term on the road reference -- although
the CBP half has carried one since E66 (`cbp0 = s * r0 * (no-website share)`, c **+0.051**), which is
the best-paying completeness mechanism outside the record-counting family.

**The instrument, and why it is not a reference proxy.** `road_surface` and `speed_limits` are
linear-referenced LISTs in the provided roads layer: a non-empty list is Overture asserting
something about that stretch of road. A tract holding route metres on which Overture never recorded
a surface or a speed limit is a tract nobody surveyed, and an unsurveyed tract is also more likely
to be missing route length the reference holds. Built from Overture's own attributes only; it
estimates nothing about TIGER.

**The unfitted signature test, and it is the cleanest since E19.** a participant (board thread 34744,
reply of 25 Sep, provided data only, no external sources) published the tract-level version of this
measurement: across all five regions **2,739 tracts, 28.4%**, of those carrying scored-class highway
have not one metre with a name AND a surface AND a speed limit. `features_dispatch.py` rebuilt on
their class definition gives **0.284** -- exact to three decimals with no free parameter -- and
reproduces the per-region split to <= 0.005 in all five (AZ .083/.085, WA .092/.093, OK .257/.262,
CA .245/.248, TX .353/.357). Our length shares come out lower than theirs (name .673 vs .825)
because they count a route ref as a name and we use `names.primary`; the tract SET, which is what
this file rides on, is identical.

**The form.**  recall_t = mean of the name / surface / speed shares over the tract's own metres, and

    TIGER_t = k_r * den_t * cellnorm(1 + beta * (1 - recall_t))

  * `cellnorm` divides the multiplier by its mean inside the tract's **region x rural cell**, so the
    term redistributes STRICTLY WITHIN CELL -- which is where row 29a puts 98.5% of the remaining
    error, and which LEDGER 99b confirmed from outside our data (rival_I matches the target's card
    worse than we do on 7 of 8 rows and scores better). Without it the composite rural/urban ratio
    drifts 2.97 -> 3.24 against a published 2.96; with it, 3.00.
  * `beta` is pinned to ONE published number, the transport burned/unburned ratio **1.87** -- the
    method that identified E34's ring width (read +0.095) and E36's (read +0.086). v4 gives 1.597;
    beta solves it to 1.87. This is the IDENTIFYING use of a published aggregate, not the screening
    use that LEDGER 96f demoted.
  * `k_r` is re-solved per region to the same published region means every earlier model used, so
    the level is absorbed and only the within-cell re-ordering is sold -- E85/E86/E97's construction
    (+0.083, +0.093, +0.045).
  * The mask `den > 0` is untouched (the multiplier is strictly positive), so E19's match to the four
    published undefined rates cannot move.

**Two instruments, measured separately.**
  E102 `core`  -- the share on the tract's CORE-ROUTE metres: the direct measure, on the very
                  geometry the denominator is built from (median 4.6 km/tract).
  E103 `local` -- the share on residential/unclassified/tertiary metres (median 28.7 km/tract): the
                  same latent mapping effort, measured on six times the length, one step removed.
  Their projected parts correlate only **+0.38**, so they are two readings, not one.

**Price.** The one formal analogue -- a per-tract completeness multiplier inside a reference -- is
E66 at **+0.051**, on the CBP half; this is the same construction on transport, which carries the
larger variance budget (rho up to ~0.8 against our 0.764). Against that, road-quality instruments
entered as bare columns read +0.021 (E82), +0.024 (E67), +0.028 (E70). c_hat **0.045** for the core
instrument and **0.035** for the local one.

**Stated against it before upload.** (1) Attribute completeness is urbanity- and vulnerability-
aligned (a participant: rural 0.63, high-CVI 0.54, high-SVI 0.71 of the complement) and E26 taught that a
bare urbanity marker reads **-0.135**; the cell normalisation, the per-region k re-solve and the
span projection are the three things between this and that trap, and the trap battery is printed
below. (2) beta is pinned on a ratio v4 under-predicts (1.597 vs 1.87), and E78 showed part of that
shortfall belongs to the reference noise sigma_t (+0.021), so beta absorbs some of a different
mechanism; E78 is a scored file and is projected out, but if this reads low that is the reason.

    python src/features_dispatch.py
    python src/cand_round17.py              # beta, shares, reach, perp, traps, dose
    python src/cand_round17.py build core   # e102_recall.csv
    python src/cand_round17.py build local  # e103_localrecall.csv
"""
import io
import contextlib
import sys

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

VARIANT = {
    "core":  dict(name="e102_recall.csv", chat=0.045, beta=None,
                  note=("E102 Overture's recall of the route system is not uniform inside a region: the "
                        "road reference scaled within cell by the tract's own route-metre attribute share")),
    "local": dict(name="e103_localrecall.csv", chat=0.030, beta=1.0,
                  note=("E103 the same recall correction measured on the local road network, six times "
                        "the length and one step removed from the reference geometry")),
}
PUB_RATIO = 1.87               # published transport burned/unburned (the challenge discussion board)
BASE = "e99_joint.csv"
RIVAL_F = 0.057299837
DERIVED = lb.ROOT.parent / "data" / "derived"


def shares(d):
    """Attribute recall on core-route metres and on local-road metres, per tract."""
    regs = sorted(d.region.unique())
    dp = d[["GEOID", "region"]].merge(
        pd.concat([pd.read_parquet(DERIVED / f"{r}-dispatch.parquet") for r in regs]), on="GEOID", how="left").fillna(0.0)
    q = d[["GEOID"]].merge(
        pd.concat([pd.read_parquet(DERIVED / f"{r}-quality.parquet") for r in regs]), on="GEOID", how="left").fillna(0.0)
    loc = q.q_local.values.astype(float)
    num = {"core": dp.dp_core_name.values + dp.dp_core_surf.values + dp.dp_core_speed.values,
           "local": (loc - q.q_local_unnamed.values) + (loc - q.q_local_nosurf.values)
                    + (loc - q.q_local_nospeed.values)}
    den = {"core": dp.dp_core.values.astype(float), "local": loc}
    out = {}
    for kk in num:
        with np.errstate(invalid="ignore", divide="ignore"):
            a = num[kk] / (3 * den[kk])
        a = np.where(den[kk] > 0, a, np.nan)
        # where the instrument is not measured the tract gets its region's length-weighted share,
        # so the multiplier is neutral there and the part carries no signal it did not measure
        for r in regs:
            m = (d.region == r).values
            lvl = num[kk][m].sum() / (3 * den[kk][m].sum()) if den[kk][m].sum() > 0 else 1.0
            a = np.where(m & ~np.isfinite(a), lvl, a)
        out[kk] = np.clip(a, 0.0, 1.0)
    return out, dp, q


def _cells(d):
    rural, _ = CC.strata(d)
    return [((d.region == r).values & (rural == ru)) for r in sorted(d.region.unique()) for ru in (True, False)]


def multiplier(d, a, beta):
    """cellnorm(1 + beta*(1-a)): within region x rural, mean 1 by construction."""
    m = 1.0 + beta * (1.0 - a)
    out = m.copy()
    for c in _cells(d):
        if c.sum():
            out[c] = m[c] / m[c].mean()
    return out


def run(d, X0, mult):
    X = dict(X0)
    X["den"] = X0["den"] * mult
    with contextlib.redirect_stdout(io.StringIO()):
        k, p_mis, p_bg, s, ms = P3.calibrate(d, X)
        o = P3.simulate(X, k, p_mis, p_bg, s, m_s=ms)
    return o


def solve_beta(d, X0, a, lo=0.05, hi=12.0):
    rural, burn = CC.strata(d)
    fit = d.region.ne("eastern-wa").values
    dT = X0["den"] > 0
    def ratio(beta):
        T = run(d, X0, multiplier(d, a, beta))["T"]
        return T[dT & fit & burn].mean() / T[dT & fit & ~burn].mean()
    return brentq(lambda b: ratio(b) - PUB_RATIO, lo, hi, xtol=1e-3), ratio


def candidate(kind, L=None):
    L = lb.ledger() if L is None else L
    d, X0 = P4.inputs()
    A, _, _ = shares(d)
    a = A[kind]
    rural, burn = CC.strata(d)
    fit = d.region.ne("eastern-wa").values
    dT = X0["den"] > 0
    o0 = run(d, X0, np.ones(len(d)))
    y0, T0 = o0["y"], o0["T"]
    print(f"v4 baseline: transport burned/unburned {T0[dT & fit & burn].mean() / T0[dT & fit & ~burn].mean():.3f} "
          f"(pub {PUB_RATIO}), composite rural/urban {y0[fit & rural].mean() / y0[fit & ~rural].mean():.3f} (pub 2.96)")
    fixed = VARIANT[kind]["beta"]
    beta = fixed if fixed is not None else solve_beta(d, X0, a)[0]
    o = run(d, X0, multiplier(d, a, beta))
    y1, T1 = o["y"], o["T"]
    how = "held at 1 (unit elasticity, unfitted)" if fixed is not None else "pinned to the published ratio"
    print(f"[{kind}] beta {how}: {beta:.3f}  ->  transport ratio "
          f"{T1[dT & fit & burn].mean() / T1[dT & fit & ~burn].mean():.3f}; "
          f"composite rural/urban {y1[fit & rural].mean() / y1[fit & ~rural].mean():.3f} "
          f"(unfitted, held by the cell normalisation)")
    h = y1 - y0
    col, perp = CMM.project(h, d, L)
    return d, a, h, col, perp, beta


def evidence(kind="core", L=None, W=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    v = VARIANT[kind]
    d, a, h, col, perp, beta = candidate(kind, L)
    _, dfn, _ = CMM.hypothesis(d)
    rural, burn = CC.strata(d)
    print(f"\nrecall: mean {a.mean():.3f} sd {a.std():.3f} p10 {np.percentile(a, 10):.3f} "
          f"p90 {np.percentile(a, 90):.3f}")
    print(f"part: reach {np.mean(np.abs(h) > 1e-6):.1%} of tracts, mean |h| {np.abs(h).mean():.5f}, "
          f"max {np.abs(h).max():.4f}, mean shift {h.mean():+.6f}")
    print(f"part: burned/unburned {h[burn].mean():+.6f}/{h[~burn].mean():+.6f}, "
          f"rural/urban {h[rural].mean():+.6f}/{h[~rural].mean():+.6f}, perp {perp:.3f}")
    print(CK.traps({kind: col}, d, dfn).round(3).to_string())
    for f in ["e38_tiger.csv", "e66_web.csv", "e82_lo_import.csv", "e67_unrouted.csv", "e70_tertiary.csv",
              "e78_grad_sigt.csv", "e26_road_undefined.csv", "e30_rural_step.csv", "e99_joint.csv",
              "e101_cbpscope.csv"]:
        q = lb.read_sub(f).coverage_gap_score.values
        print(f"   corr of the projected part with {f:24s} {np.corrcoef(col, (q - q.mean()) / q.std())[0, 1]:+.3f}")
    print(f"   corr of the projected part with the recall share itself: "
          f"{np.corrcoef(col, (a - a.mean()) / a.std())[0, 1]:+.3f}")
    W = v["chat"] if W is None else W
    base = lb.read_sub(BASE).coverage_gap_score.values
    rho_b = lb.rho(float(L[L.file == BASE].rmse.iloc[0]), base, m1, m2)
    p = np.clip(base + sd * W * col, 0, 1)
    av = validate.analyse(p, L)
    print(f"\nbase {BASE} rho_b {rho_b:.5f};  neutral {av['rmse_neutral']:.6f} "
          f"range [{av['rmse_best']:.6f}, {av['rmse_worst']:.6f}]  best {validate.best_rmse(L):.6f}")
    # exact dose table on the REALISED delta (the clip moves both the mean and the scale)
    dd = p - base
    print(f"\ndose W = {W:.4f} (realised {dd.std() / sd:.4f} after the clip)   breakeven c > {W / 2:.4f}")
    mse_b = float(L[L.file == BASE].rmse.iloc[0]) ** 2
    for c in [0.0, W / 2, 0.030, v["chat"], 0.050, 0.060]:
        e_yd = sd * dd.std() * c + m1 * dd.mean()
        mse = mse_b - 2 * e_yd + 2 * float((base * dd).mean()) + float((dd ** 2).mean())
        r = np.sqrt(mse)
        print(f"   c {c:+.4f} -> {r:.6f}" + ("   BEATS rival_F" if r < RIVAL_F else
                                             ("   beats rival_I" if r < 0.057375614 else "")))
    return d, col, p


def build(kind="core", L=None):
    d, col, p = evidence(kind, L)
    lb.write(VARIANT[kind]["name"], d.GEOID, p, VARIANT[kind]["note"], kind="model")


if __name__ == "__main__":
    a = sys.argv[1:]
    k = a[1] if len(a) > 1 else "core"
    build(k) if a[:1] == ["build"] else evidence(a[0] if a else "core")
