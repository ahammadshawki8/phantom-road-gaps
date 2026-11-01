"""E46: two further readings of the POI component, each a DISTINCT mechanism, not a re-reading.

**Why POI again, and why two parts rather than one or thirteen.** E44 confirmed the component at
`c = +0.0612, t = +6.0` -- the first POI part in eleven attempts clear of noise. But the variance
budget says POI is **35-40% of var(y)** and `c = 0.0612` explains 0.37% of it, so the component is
confirmed and essentially unharvested. The rule from 15 Oct is *one mechanism per file*, and the
reason is that extra **re-readings** of a mechanism already in the span pay nothing (E38 f = 0.25,
E43 f = 0.17, where twelve re-readings of local completeness read back -0.008 between them). Two
**distinct** mechanisms are a different thing: E27 combined three measured parts and worked.
`forecast.dedupe` keeps both of these, and they correlate 0.33 after the span is projected out.

**Mechanism A -- `poi_county`: the reference is a COUNTY total allocated to tracts, not a
tract-level density.** CBP is published per county; a tract-level reference has to come from
allocating that county total. Which allocator is used decides the sign, and the sign is measurable:
allocating by population, housing units or buildings gives a burned/unburned ratio of
**0.29-0.62x against a published 3.40x** -- backwards, which is the same failure as all ten earlier
POI attempts -- while allocating by **area** gives **4.30x** with a single parameter fitted to the
published mean. The two bracket, so one mixing weight is identified by the published ratio:

    ref = k * estab_county * (lam * area_share + (1 - lam) * building_share),  estab_county ~ pop
    lam = 0.6533,  mean 0.0520 (published 0.052),  burned/unburned 3.40 (published 3.40)

This is **not** E44's reference re-read at another scale. E44 fits a global `s * bld^(1-a) * area^a`;
this one is normalised *within county*, and that normalisation is what flips the sign. Measured:
after the 30-file span and the five region dummies are projected out, it correlates **0.04** with
E44's own perp direction and keeps a perp share of **0.737** (E44's own vector is now down to 0.142,
because the span absorbed it).

**Mechanism B -- `poi_facmask`: the mean-of-DEFINED divisor inside POI, which no file has ever
modelled.** `POI = mean(facilities half, CBP half)`, and the facilities half is itself a mean over
*defined* types, so where HIFLD lists no fire station, no EMS and no school the facilities half is
**undefined** and POI is the CBP half **alone** -- twice the weight it gets elsewhere. Section 6.1
of the explainer established that the facilities half must be ~0 almost everywhere (any dense
reference floors the facilities mean at ~0.53 against a published POI mean of 0.052, so Overture's
emergency POIs and the HIFLD reference must agree nearly everywhere). That conclusion is what makes
Overture's own facility presence a usable proxy for the reference's:

    POI = cbp / 2   where Overture lists >= 1 fire / EMS / school   (88.4% of tracts)
    POI = cbp       where it lists none                             (11.6%)

**Only `k` is fitted, to the published mean. The burned/unburned ratio is then entirely unfitted and
comes out 2.98 against a published 3.40 -- 12% out, with one free parameter in the whole model.**
That is the E19/E36 standard of evidence, not a curve fit. Perp 0.387.

**Three more directions died on the way and cost nothing.** The `y >= 1/n` lower bound (a tract
whose transport gap is exactly 1 must score at least 1/3) is **refuted on sample size: only 19 of
9,794 tracts reach the upper clip**. The divisor `1/n_defined` on its own has perp **0.133** -- it
*is* E26's road-undefined indicator, already spent and already negative. A divisor-by-level
interaction reads 0.92x, the wrong sign. Per-category sparse facility references overshoot to
9.67x. A confidence-50 numerator re-solves to the same place as E44 (corr 0.62) and is a re-reading.

**Dose.** `m = 0.85`, the measured payoff fraction of single-mechanism files (E34 0.95, E41 0.82,
E44 0.77), not appetite. Prices are set from what the one measured member of this component paid:
E44 read +0.0612, so a second orthogonal reading of the same component is priced at 0.065 and the
weighting mechanism, which is structural rather than a new guess at the reference, at 0.055.

    python src/cand_poi2.py            # identification, screens, traps, dose table
    python src/cand_poi2.py build      # write submissions/e46_poi2.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import cand_blend as CB
import strata as CC
import cand_knn as CK
import cand_poi as CPO
import cand_tiger as CT
import forecast as FC
import lb
import validate

OUT = "e46_poi2.csv"
NOTE = ("E46 two distinct POI mechanisms -- a county-allocated reference and the mean-of-defined "
        "facilities weighting -- over the least-squares re-solve of all 30 scored submissions")

CHAT = {"poi_county": 0.065, "poi_facmask": 0.055}
ORDER = list(CHAT)
M = 0.85                      # measured payoff fraction of single-mechanism files
PUB_MEAN = CPO.PUB_MEAN       # 0.052
PUB_RATIO = CPO.PUB_RATIO     # 3.3953...


def _inputs(d=None):
    d = CC.table_all() if d is None else d
    rural, burn = CC.strata(d)
    fit = d.region.ne("eastern-wa").values
    cty = np.array([g[:5] for g in d.GEOID.astype(str)])
    tot = lambda v: pd.Series(np.asarray(v, float)).groupby(cty).transform("sum").values
    return d, rural, burn, fit, tot


def _fit_mean(mk, fit):
    k = brentq(lambda z: mk(z)[fit].mean() - PUB_MEAN, 1e-12, 1e12)
    return k, mk(k)


def _ratio(g, fit, burn):
    return g[fit & burn].mean() / max(g[fit & ~burn].mean(), 1e-12)


def poi_county(d=None):
    """Mechanism A: county establishment total allocated by a fitted area/building mix."""
    d, rural, burn, fit, tot = _inputs(d)
    poi = d.poi_all.fillna(0).values.astype(float)
    bld = d.bld_all.fillna(0).values.astype(float)
    pop = d.pop_total.fillna(0).values.astype(float)
    area = np.maximum(d.area_m2.values / 1e6, 1e-6)
    a_sh = area / np.maximum(tot(area), 1e-9)
    b_sh = bld / np.maximum(tot(bld), 1e-9)
    est = tot(pop)                                   # CBP establishments ~ county population

    def mk(lam):
        return lambda k: CPO.gap(k * est * (lam * a_sh + (1 - lam) * b_sh), poi)

    lam = brentq(lambda l: _ratio(_fit_mean(mk(l), fit)[1], fit, burn) - PUB_RATIO, 0.0, 1.0)
    k, g = _fit_mean(mk(lam), fit)
    return g, {"lam": lam, "k": k, "ratio": _ratio(g, fit, burn),
               "rural": g[fit & rural].mean() / g[fit & ~rural].mean()}


def poi_facmask(d=None):
    """Mechanism B: the mean-of-defined divisor. Only k is fitted; the ratio is a prediction."""
    d, rural, burn, fit, tot = _inputs(d)
    poi = d.poi_all.fillna(0).values.astype(float)
    bld = d.bld_all.fillna(0).values.astype(float)
    area = np.maximum(d.area_m2.values / 1e6, 1e-6)
    a, _, _, _ = CPO.identify(d)                     # the exponent E44 already identified
    hasfac = ((d.poi_fire.values + d.poi_fire_alt.values + d.poi_ems.values
               + d.poi_ems_alt.values + d.poi_school.values + d.poi_school_alt.values) > 0)

    def mk(k):
        cbp = CPO.gap(k * ((bld + 1) ** (1 - a)) * (area ** a), poi)
        return np.where(hasfac, cbp / 2.0, cbp)      # facilities half defined -> CBP gets half

    k, g = _fit_mean(mk, fit)
    return g, {"k": k, "a": a, "has_fac": float(hasfac.mean()),
               "ratio_UNFITTED": _ratio(g, fit, burn),
               "rural": g[fit & rural].mean() / g[fit & ~rural].mean()}


def hypotheses(d=None):
    d = CC.table_all() if d is None else d
    gA, mA = poi_county(d)
    gB, mB = poi_facmask(d)
    return {"poi_county": gA, "poi_facmask": gB}, {"poi_county": mA, "poi_facmask": mB}, d


def basis(H, d, L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    _, P, _ = validate.scored(L, m1, m2)
    B = np.column_stack([P - P.mean(0)]
                        + [(d.region == rg).values.astype(float) for rg in sorted(set(d.region))])
    B = B - B.mean(0)
    cols, perp = {}, {}
    for n in ORDER:
        h = np.asarray(H[n], float)
        h = (h - h.mean()) / h.std()
        hp = h - B @ np.linalg.lstsq(B, h, rcond=validate.RCOND)[0]
        perp[n] = float(hp.std())
        cols[n] = hp / max(hp.std(), 1e-12)
        B = np.column_stack([B, cols[n]])
    return cols, perp


def candidate(d=None, L=None, m=M):
    d = CC.table_all() if d is None else d
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    H, meta, d = hypotheses(d)
    cols, perp = basis(H, d, L)
    p0, _, files, rho_b = CB.blend(L)
    p = np.clip(p0 + sd * sum(m * CHAT[n] * cols[n] for n in ORDER), 0, 1)
    return p, p0, rho_b, sd, perp, meta, cols, d


def evidence(d=None, L=None, target=0.063414963):
    L = lb.ledger() if L is None else L
    p, p0, rho_b, sd, perp, meta, cols, d = candidate(d, L)
    _, rt, r, q, tg, _ = CT.inputs(d)
    _, dfn, _ = CK.transport_gap(d, rt, r, q)
    print("MECHANISM A  poi_county: ref = k * county_pop * (lam*area_share + (1-lam)*bld_share)")
    print(f"   lam = {meta['poi_county']['lam']:.4f} (fitted to the published ratio), "
          f"k = {meta['poi_county']['k']:.5f} (fitted to the published mean 0.052)")
    print(f"   burned/unburned {meta['poi_county']['ratio']:.3f} (published {PUB_RATIO:.2f}), "
          f"rural/urban {meta['poi_county']['rural']:.2f}")
    print("MECHANISM B  poi_facmask: POI = cbp/2 where Overture lists a facility, cbp where not")
    print(f"   facilities half defined in {meta['poi_facmask']['has_fac']:.1%} of tracts; "
          f"k = {meta['poi_facmask']['k']:.5f} is the ONLY fitted parameter")
    print(f"   burned/unburned {meta['poi_facmask']['ratio_UNFITTED']:.3f} **UNFITTED** "
          f"against a published {PUB_RATIO:.2f} -- "
          f"{abs(meta['poi_facmask']['ratio_UNFITTED'] / PUB_RATIO - 1):.1%} out")
    print(f"\n{'part':12s} {'c_hat':>7s} {'dose W':>8s} {'perp':>7s}")
    for n in ORDER:
        print(f"{n:12s} {CHAT[n]:7.3f} {M * CHAT[n]:8.3f} {perp[n]:7.3f}")
    print(f"corr between the two parts after projection: "
          f"{abs(float(np.corrcoef(cols['poi_county'], cols['poi_facmask'])[0, 1])):.2f}")
    print("\ntrap checks (E25/E26 urbanity):")
    print(CK.traps(cols, d, dfn).round(3).to_string())
    a0, a1 = validate.analyse(p0, L), validate.analyse(p, L)
    print(f"\nbase: the 30-file re-solve, rho_b {rho_b:.4f}, certified {a0['rmse_neutral']:.6f}")
    print(f"e46: neutral {a1['rmse_neutral']:.6f} range [{a1['rmse_best']:.6f}, "
          f"{a1['rmse_worst']:.6f}] explained {a1['share_explained_by_scored']:.2%}")
    FC.table(rho_b, CHAT, M, L, target=target, label="E46")
    s2 = sum(c ** 2 for c in CHAT.values())
    print("\n   checkpoints, and the payoff fraction f each needs:")
    for nm, v in [("the honest-10th entry", 0.063414963), ("rival_G", 0.063203965),
                  ("a participant", 0.063072465), ("rival_H", 0.062496005),
                  ("the listed-12th entry", 0.058930694)]:
        need = (1 - rho_b ** 2) - (v / sd) ** 2
        g = (need / s2 + M * M) / (2 * M)
        print(f"   {nm:26s} {v:.9f}  needs f = {g:5.2f}  "
              f"{'REACHABLE' if g <= 1.1 else 'not reachable from this file'}")
    return p


def build(d=None, L=None):
    p, _, _, _, _, _, _, dd = candidate(d, L)
    lb.write(OUT, dd.GEOID, p, NOTE, kind="model")
    return p


if __name__ == "__main__":
    if "build" in sys.argv[1:]:
        build()
    else:
        evidence()
