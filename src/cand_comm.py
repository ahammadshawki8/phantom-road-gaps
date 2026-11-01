"""E48: a COMMERCIAL-ACTIVITY reference for the CBP half of POI (H-P4), one new mechanism.

CBP is published per county / ZIP, and the standard way to put a ZIP total on tracts is by the
share of business addresses, so the tract reference follows where businesses actually are, not
population, housing or the plain building count (E44's basis). Overture carries two signals of
commercial activity that are not the POI numerator:

    big1k    buildings with footprint >= 1,000 m2 (stores, offices, warehouses)
    service  km of service road (parking aisles, drive-throughs, loading lanes; roads-unfiltered)

Free screen, scale fitted to the published POI mean 0.052 only (burned/unburned target 3.40):
    big1k alone 2.03x, big5k 1.27x (under)   service alone 9.29x (over)
    Overture's commercial building class 1.04x  -> REFUTED (flat), recorded
The two bracket, so one exponent is identified by the published ratio, as in E44:

    ref = s * (big1k + 1)^(1-a) * (service_km + 0.1)^a

    python src/features_commercial.py   # once, ~2.5 min
    python src/cand_comm.py             # identification, distinctness, traps, dose
    python src/cand_comm.py build       # write submissions/e48_commercial.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import cand_blend as CB
import strata as CC
import cand_knn as CK
import cand_mm as CMM
import cand_poi as CPO
import forecast as FC
import lb
import validate

OUT = "e48_commercial.csv"
NOTE = "E48 commercial-activity POI reference: s*(big1k+1)^(1-a)*(service_km+0.1)^a, projected"
CHAT = 0.06          # a second, distinct reading of a component E44 measured at +0.061
M = 0.85
PUB_MEAN, PUB_RATIO = CPO.PUB_MEAN, CPO.PUB_RATIO


def table(d=None):
    d = CC.table_all() if d is None else d
    c = pd.concat([pd.read_parquet(lb.ROOT.parent / "data" / "derived" / f"{r}-commercial.parquet")
                   for r in sorted(d.region.unique())])
    return d.merge(c, on="GEOID", how="left")


def identify(d):
    rural, burn = CC.strata(d)
    fit = d.region.ne("eastern-wa").values
    poi = d.poi_all.fillna(0).values.astype(float)
    big = d.bld_ge1k.fillna(0).values.astype(float) + 1
    svc = d.ru_len_service.fillna(0).values / 1000 + 0.1

    def at(a):
        r0 = big ** (1 - a) * svc ** a
        s = brentq(lambda z: CPO.gap(r0 * z, poi)[fit].mean() - PUB_MEAN, 1e-9, 1e9)
        return CPO.gap(r0 * s, poi), s

    ratio = lambda g: g[fit & burn].mean() / g[fit & ~burn].mean()
    a = brentq(lambda z: ratio(at(z)[0]) - PUB_RATIO, 0.0, 1.0)
    g, s = at(a)
    return g, {"a": a, "s": s, "ratio": ratio(g), "mean": g[fit].mean(),
               "rural": g[fit & rural].mean() / g[fit & ~rural].mean(), "zero": float(np.mean(g == 0))}


def candidate(d=None, L=None, chat=CHAT, m=M):
    d = table(d)
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    g, meta = identify(d)
    mm_col, _ = CMM.project(CMM.hypothesis(d)[0], d, L, extra=CMM.e46_cols(d, L))
    extra = CMM.e46_cols(d, L) + [mm_col]
    col, perp = CMM.project(g, d, L, extra=extra)
    e44_col, _ = CMM.project(CPO.identify(d)[2], d, L)
    p0, _, _, rho_b = CB.blend(L)
    return (np.clip(p0 + sd * m * chat * col, 0, 1), p0, rho_b, sd, perp, col, meta, e44_col, g, d)


def evidence(d=None, L=None):
    L = lb.ledger() if L is None else L
    p, p0, rho_b, sd, perp, col, meta, e44_col, g, d = candidate(d, L)
    print(f"identified a = {meta['a']:.4f}, s = {meta['s']:.5f}: mean {meta['mean']:.4f} (pub 0.052), "
          f"burned/unburned {meta['ratio']:.2f} (pub {PUB_RATIO:.2f}); unfitted rural/urban "
          f"{meta['rural']:.2f}; {meta['zero']:.1%} of tracts at exactly 0")
    print(f"perp share after span + regions + E46 + E47: {perp:.3f};  corr with E44's projected "
          f"direction {np.corrcoef(col, e44_col)[0, 1]:+.3f}")
    _, dfn, _ = CMM.hypothesis(d)
    print(CK.traps({"commercial": col}, d, dfn).round(3).to_string())
    a = validate.analyse(p, L)
    print(f"E48 neutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}]")
    FC.table(rho_b, {"commercial": CHAT}, M, L, label="E48")
    return p


def build(d=None, L=None):
    p, *_, dd = candidate(d, L)
    lb.write(OUT, dd.GEOID, p, NOTE, kind="model")
    return p


if __name__ == "__main__":
    build() if "build" in sys.argv[1:] else evidence()
