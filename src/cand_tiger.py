"""E38: the reference measured instead of fitted, on three independent readings of TIGER.

**What changed today.** The transport component is `1 - min(1, overture / reference)` and the
numerator comes from the same Overture file the organisers used, so every error left in it is an
error in our estimate of the *reference* -- TIGER primary+secondary length per tract (CLAUDE.md
s5). E29 fitted that reference as one global constant `k`, E34 and E36 refitted `k` beside a
boundary ring width `w`. All three treat the reference as unobserved.

It is not unobserved. **The census tract polygons in the strata package are TIGER geometry**, and
TIGER draws tract boundaries along its own features -- which is the mechanism E19 rests on. So
where a tract boundary runs along a core route, that boundary polyline *is* TIGER's centreline
for that road, and `features_tiger.py` clips the two files against each other to measure, per
tract, both the length TIGER draws and the length Overture draws for the same stretch.

Three parts come out of it, and they are mutually orthogonal (|corr| < 0.33 after the scored
span is projected out), so they add in quadrature.

**T2 -- the boundary-coincidence reference, and the cleanest signature in the project.**
Reference = k * (core_in + tau * bnd15), where `bnd15` is the Overture core-route length lying
within 15 m of the tract boundary. A boundary road is shared by the two tracts either side of it,
so each tract's reference should claim **half** of it: tau = 1/2 is a mechanism, not a fit.
Fitting tau to the published burned/unburned transport ratio returns **0.514**. Pinning it at
exactly 1/2 and fitting only `k` to the published transport mean leaves that ratio entirely
unfitted, and it predicts **1.887 against a published 1.87** -- 0.9% out, with one free parameter
in the whole model.

**T1 -- the TIGER/Overture length ratio, measured per tract instead of fitted.** CLAUDE.md s3
records that this ratio spans 0.71-1.59 across regions, so the single constant `k` is known to be
wrong; psi = (TIGER boundary length near a core route) / (Overture core length near the boundary)
measures it where the two files overlap. It enters at **gamma = 1** -- a measured ratio is applied
as itself, not raised to a fitted power -- and it is robust to the matching tolerance (ratio
1.861-1.883 across coverage floors 200/500/1000 m, parts correlated 0.97-0.99). With psi applied,
the ring width re-identifies at **w* = 26.2 m**, which is where E34's independent fit put it
(27.3 m) before E36 moved it to 15.5 m.

**G1 -- geometric detail.** Reference = k * (core + ring50) * vden^gamma, vertices per metre of
the Overture trace, gamma fitted to the burned ratio. `w = 50 m` is not fitted here: it is where
the four published undefined rates are flattest (max error **0.013**, against 0.016 at E36's
15.5 m and 0.020 at T2's). Weakest of the three on mechanism -- psi says TIGER and Overture trace
a *boundary* highway to within 1% of each other (median psi 0.990), which is evidence against
generalisation being large -- but it has the best mask and it is only 0.32 correlated with T1.

**Q -- Overture attribute incompleteness** (three parts: named-highway attribute gaps, nameless
local road, surfaceless local road). No published aggregate screens these, which by the s4 rule
makes them weaker than anything reference-side; they are carried at a third of the dose because
the target's own card says the signal we are missing is SVI-aligned (our SVI contrast is 0.58 of
the target's) and this is the only family that points that way.

**Refuted on the way, at no quota cost** (LEDGER row 34a):
- *Non-route arterials in the reference.* 56% of the numerator is motorway-secondary road with no
  route number, and TIGER's S1200 is "main arteries". Fitting the admitted fraction to the burned
  ratio returns **lambda = 0.0001** -- the same flat zero county routes returned.
- *Under-construction road out of the numerator* (E36's exact mechanism, a road TIGER has not
  built yet). It is 0.69% of named length and moves the predicted ratio from 1.887 to 1.917,
  i.e. away from 1.87. Out, and it was 0.889 correlated with T2 anyway.
- *Overture's own road width*, which would have made E19's displacement per-tract instead of
  fitted: `width_rules` is populated on **10 of 57,516** Maricopa segments. Dead.
- *Sinuosity as the detail measure* rather than vertex density: it needs gamma = +1.47 to fit, and
  psi had already measured the length agreement at 1%.

    python src/cand_tiger.py            # identification, signatures, orthogonality, dose table
    python src/cand_tiger.py build      # write submissions/e38_tiger.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import cand_blend as CB
import cand_county as CY
import cand_level as CL
import cand_ramp as CP
import cand_road as CR
import features as F
import lb
import table
import validate

OUT = "e38_tiger.csv"
NOTE = ("E38 transport reference measured against TIGER's own tract-boundary geometry: "
        "boundary-coincidence at tau = 1/2, the per-tract TIGER/Overture length ratio, and "
        "geometric detail, over the least-squares blend of all scored files, then raked")
PUB_M, PUB_R = 0.112, 1.87
PSI_FLOOR = 500.0          # metres of matched core route below which psi is not measured
RING_G1 = 50.0             # where the four published undefined rates are flattest

# Dose per part = the correlation each is expected to read back, which is also the MSE-optimal
# weight for that belief. The reference-side family's read-backs decay (E19 +0.461, E29 +0.210,
# E34 +0.095, E36 +0.086), so T2 and T1 are priced one step below E36 rather than at it; G1 is
# discounted again for the weaker mechanism; the unscreened quality parts at half of that.
# Pricing at the belief rather than a round number dominates: against a fixed outcome of
# c = (0.07, 0.07, 0.045, 0.03 x3) these doses score 0.06323 where 0.095/0.075/0.055 score
# 0.06340, and the dead null improves from 0.06605 to 0.06553. Only in the optimistic tail
# (c ~ 0.11) does the larger dose win, and there by 0.0001.
DOSES = {"T2": 0.085, "T1": 0.085, "G1": 0.060,
         "Q_named": 0.045, "Q_local_unnamed": 0.045, "Q_local_nosurf": 0.045}
ORDER = list(DOSES)         # Gram-Schmidt order = order of evidential strength


def _join(name):
    t = pd.concat([pd.read_parquet(F.OUT / f"{r}-{name}.parquet") for r in F.REGIONS],
                  ignore_index=True)
    t = table.sample_order()[["GEOID"]].merge(t, on="GEOID", how="left").fillna(0.0)
    assert len(t) == 9794
    return t


def inputs(d=None):
    d = CL.table_all() if d is None else d
    rt, r, q = CY.routes_table(), CR.road_table(), CP.quality_table()
    return d, rt, r, q, _join("tiger"), _join("geom")


def psi(tg, floor=PSI_FLOOR):
    o, b = tg.tg_ovt15.values, tg.tg_bnd15.values
    return np.where(o > floor, np.clip(b / np.maximum(o, 1e-9), 0.3, 3.0), 1.0)


def _fit_k(num, den, s):
    return brentq(lambda k: CR.gap(num, den * k)[s].mean() - PUB_M, 1e-3, 1e4)


def _gap(num, den, fit):
    dfn = den > 0
    g = CR.gap(num, den * _fit_k(num, den, dfn & fit))
    return g, dfn


def hypotheses(d=None, parts=None):
    d, rt, r, q, tg, gm = inputs(d)
    num = CP.numerator(r, q)
    fit = (d.region != "eastern-wa").values
    burn = d.mtbs_wildfire_ever.astype(bool).values
    core = rt.rt_core.values.astype(float)
    ps = psi(tg)
    H = {}
    # T2: tau = 1/2 by mechanism; only k is fitted, the burned ratio is a prediction
    H["T2"] = _gap(num, core + 0.5 * tg.tg_ovt15.values, fit)[0]
    # T1: psi at gamma = 1 (measured), ring width re-identified against the burned ratio
    def rat_w(w):
        den = (core + CY.ring_at(rt, ["core"], w)) * ps
        g, dfn = _gap(num, den, fit)
        s = dfn & fit
        return g[s & burn].mean() / g[s & ~burn].mean()
    w1 = brentq(lambda w: rat_w(w) - PUB_R, 15.49, 50.0)
    H["T1"] = _gap(num, (core + CY.ring_at(rt, ["core"], w1)) * ps, fit)[0]
    # G1: vertex detail, ring pinned at the flattest undefined rates, exponent on the ratio
    ring = CY.ring_at(rt, ["core"], RING_G1)
    s0 = ((core + ring) > 0) & fit
    L = gm.gm_len_core.values
    v = np.where(L > 0, gm.gm_vden_core.values / np.maximum(L, 1.0), np.nan)
    lx = np.log(np.maximum(v, 1e-9))
    lx = np.clip(np.where(np.isfinite(lx), lx - np.nanmedian(lx[s0]), 0.0), -3, 3)
    def rat_g(gam):
        g, dfn = _gap(num, (core + ring) * np.exp(gam * lx), fit)
        s = dfn & fit
        return g[s & burn].mean() / g[s & ~burn].mean()
    g1 = brentq(lambda t: rat_g(t) - PUB_R, -3.0, 3.0)
    H["G1"] = _gap(num, (core + ring) * np.exp(g1 * lx), fit)[0]
    # Q: Overture attribute incompleteness, no published screen
    nm = np.maximum(r.rd_len_named.values.astype(float), 1.0)
    lo = np.maximum(q.q_local.values, 1.0)
    H["Q_named"] = np.where(r.rd_len_named.values > 0,
                            (q.q_unnamed_named.values + q.q_nosurf_named.values
                             + q.q_nospeed_named.values) / (3 * nm), 0.0)
    H["Q_local_unnamed"] = np.where(q.q_local.values > 0, q.q_local_unnamed.values / lo, 0.0)
    H["Q_local_nosurf"] = np.where(q.q_local.values > 0, q.q_local_nosurf.values / lo, 0.0)
    H["_params"] = {"w_T1": w1, "gamma_G1": g1}
    return {k: H[k] for k in (parts or ORDER)} | {"_params": H["_params"]}


def basis(H, L=None):
    """Unit-variance parts, each orthogonal to every scored file AND to the earlier parts."""
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    _, P, _ = validate.scored(L, m1, m2)
    Pc = P - P.mean(0)
    cols, perp = {}, {}
    B = Pc
    for k in ORDER:
        h = np.asarray(H[k], float)
        h = (h - h.mean()) / h.std()
        hp = h - B @ np.linalg.lstsq(B, h, rcond=validate.RCOND)[0]
        perp[k] = float(hp.std())
        cols[k] = hp / hp.std()
        B = np.column_stack([B, cols[k]])
    return cols, perp


def beta_measured(L=None):
    """Regression of the true group bias on an imposed published mean, re-read off E37.

    E33 measured beta = 0.620 on the 16 cell means alone. E37 applied the same rake -- cells plus
    the eight rank-1 rubric ratios -- at t = 0.62 on top of E36 and scored 0.064385 against
    E36's 0.064638, so the drop re-identifies beta on a second, independent file.
    """
    L = lb.ledger() if L is None else L
    try:
        d = CL.table_all()
        p36 = lb.read_sub("e36_ramp.csv").coverage_gap_score.values
        _, bhat = CB.rake(p36, d, CB.BETA)
        eb2 = float((bhat ** 2).mean())
        r36 = float(L[L.file == "e36_ramp.csv"].rmse.iloc[0])
        r37 = float(L[L.file == "e37_ramp_rake.csv"].rmse.iloc[0])
        t = CB.BETA
        return (t ** 2 * eb2 - (r37 ** 2 - r36 ** 2)) / (2 * t * eb2), eb2
    except Exception:
        return CB.BETA, float("nan")


def candidate(d=None, L=None, doses=None, beta=None):
    d = CL.table_all() if d is None else d
    L = lb.ledger() if L is None else L
    doses = DOSES if doses is None else doses
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    H = hypotheses(d)
    cols, perp = basis(H, L)
    p0, w, files, rho_b = CB.blend(L)
    p = p0 + sd * sum(doses[k] * cols[k] for k in ORDER)
    p = np.clip(p, 0, 1)
    b = CB.BETA if beta is None else beta
    pr, bhat = CB.rake(p, d, b)
    return pr, p, p0, rho_b, sd, perp, H, bhat


def forecast(rho_b, sd, doses, best, eb2_gain=0.0):
    base = 1 - rho_b ** 2
    sw2 = sum(v ** 2 for v in doses.values())
    def rmse(frac):
        s = sum(v * (v - 2 * frac * v) for v in doses.values())
        return sd * np.sqrt(max(base + s - eb2_gain, 0))
    print(f"\n   sum W^2 = {sw2:.5f};  RMSE = SD sqrt(1 - rho_b^2 + sum W(W - 2c)) - rake")
    print(f"   {'each part reads back':>26s} {'RMSE':>9s}")
    for frac, lab in [(0.0, "c = 0 (dead null)"), (0.25, "c = W/4"), (0.5, "c = W/2 (breakeven)"),
                      (0.75, "c = 3W/4"), (1.0, "c = W (as priced)"), (1.25, "c = 5W/4")]:
        print(f"   {lab:>26s} {rmse(frac):9.5f}")
    print(f"   current best scored {best:.6f}; target 0.063")


def evidence(d=None, L=None):
    raise NotImplementedError("diagnostic left out of the public release: it compared against a card derived from another team's published aggregates")
    d = CL.table_all() if d is None else d
    L = lb.ledger() if L is None else L
    pr, p, p0, rho_b, sd, perp, H, bhat = candidate(d, L)
    print(f"blend over {len(lb.ledger())} scored rows: rho_b {rho_b:.4f} "
          f"(certified {sd * np.sqrt(1 - rho_b ** 2):.5f})")
    print(f"identified: ring width for T1 w* = {H['_params']['w_T1']:.2f} m "
          f"(E34 fitted 27.3 m independently); G1 exponent {H['_params']['gamma_G1']:+.3f}")
    print("\npart            dose   perp   card-c   (card-c is ~0.002 for a within-cell part: "
          "E34 +0.0018 read +0.0954, E36 +0.0017 read +0.0855)")
    for k in ORDER:
        print(f"  {k:16s} {DOSES[k]:.3f}  {perp[k]:.3f}  {TC.measure(H[k], d, L)[0]:+.4f}")
    beta, eb2 = beta_measured(L)
    print(f"\nrake: beta re-measured on E37 = {beta:.3f} (E33 gave 0.620 on the cells alone); "
          f"E[b_hat^2] = {float((bhat ** 2).mean()):.3e}")
    forecast(rho_b, sd, DOSES, validate.best_rmse(L),
             beta ** 2 * float((bhat ** 2).mean()) / (sd ** 2))
    a = validate.analyse(pr, L)
    print(f"\nvalidate: neutral {a['rmse_neutral']:.5f} range [{a['rmse_best']:.5f}, "
          f"{a['rmse_worst']:.5f}] explained {a['share_explained_by_scored']:.2%}")
    print("\ncard, blend -> E38 -> target:")
    for k in TC.TRUTH:
        print(f"   {k:8s} {TC.card(p0, d)[k]:6.3f} -> {TC.card(pr, d)[k]:6.3f}   "
              f"target {TC.TRUTH[k]:.2f}")
    return pr


def build(d=None, L=None):
    d = CL.table_all() if d is None else d
    L = lb.ledger() if L is None else L
    beta, _ = beta_measured(L)
    pr = candidate(d, L, beta=beta)[0]
    lb.write(OUT, d.GEOID, pr, NOTE, kind="model")
    a = validate.analyse(pr, L)
    print(f"   neutral {a['rmse_neutral']:.5f} range [{a['rmse_best']:.5f}, {a['rmse_worst']:.5f}]"
          f"  best {validate.best_rmse(L):.6f}")
    return pr


if __name__ == "__main__":
    if "build" in sys.argv[1:]:
        build()
    else:
        evidence()
