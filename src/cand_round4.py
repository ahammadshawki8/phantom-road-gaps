"""E62/E63: two new mechanisms, each on the SCORED best file E59 (rule from LEDGER 60b: never build
on an unscored base), so each read-back is exact.

E62 `settle` -- MISSING STATIONS FROM COMMUNITIES. The ~470 reference fire blackouts that no Overture
    place or building traces (a participant: 740 blackouts, 270 misfiled). A rural community keeps a
    volunteer fire department, which the facility reference lists, and a post office or town hall,
    which Overture lists reliably. So communities estimate reference stations:
        fire gap = 1 - min(1, fire_department / communities)
    on tracts E49/E52 do not already trace. Blackouts (no station at all) only where RUCA >= 4:
    an urban station covers several tracts, so an urban post office without a station is not a
    reference blackout (screen: urban-wide flag 0.83x, wrong sign; rural-only 157 tracts, burned
    5.97x). Partial gaps (fewer stations than communities): 327 tracts, burned 2.71x, rural 3.92x.
E63 `zipcbp` -- CBP IS ZIP-LEVEL, ALLOCATED TO TRACTS BY LAND AREA. Thread 35080 says the
    establishments reference moves ZIP totals onto tracts. Overture places carry their postcode
    (97%), so each ZIP's place volume is rebuilt and allocated by each tract's share of the ZIP's
    land area: ref = s * sum_z P_z * area(t, z) / A_z. Only s is fitted (to the published POI mean
    0.052, with the facilities half from our measured flags); the burned/unburned ratio is then
    unfitted at 3.13x against 3.40x. Allocation by housing gives 0.87x and is refuted. Perp 0.80,
    corr +0.00 with E48's part.

    python src/features_settle.py
    python src/cand_round4.py            # screens, traps, dose
    python src/cand_round4.py build      # e62_settle.csv, e63_zipcbp.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import strata as CC
import cand_fire as CF
import cand_knn as CK
import cand_mm as CMM
import cand_poi as CPO
import forecast as FC
import lb
import validate

BASE = "e59_schclean.csv"
FILES = {"settle": ("e62_settle.csv", "E62 missing fire stations from community counts (post offices, town halls): "
                    "rural blackouts and partial gaps on untraced tracts, over E59"),
         "zipcbp": ("e63_zipcbp.csv", "E63 CBP establishments as ZIP totals allocated to tracts by land area, "
                    "Overture postcodes, over E59")}
CHAT = {"settle": 0.05, "zipcbp": 0.05}
M = 0.85
RURAL = 4


def table():
    d = CF.table()
    s = pd.concat([pd.read_parquet(lb.ROOT.parent / "data" / "derived" / f"{r}-settle.parquet")
                   for r in sorted(d.region.unique())])
    return d.merge(s, on="GEOID", how="left")


def zip_reference(d):
    tz = pd.concat([pd.read_parquet(lb.ROOT.parent / "data" / "derived" / f"{r}-tract-zip.parquet")
                    for r in sorted(d.region.unique())])
    tz["w"] = tz.n / tz.groupby("GEOID").n.transform("sum")
    tz["ar"] = tz.w * tz.GEOID.map(d.set_index("GEOID").area_m2 / 1e6)
    tz["n_z"] = tz.groupby("zip").n.transform("sum")
    tz["ar_z"] = tz.groupby("zip").ar.transform("sum")
    v = tz.n_z * tz.ar / tz.ar_z.replace(0, np.nan)
    return v.groupby(tz.GEOID).sum().reindex(d.GEOID).fillna(0).values


def hypotheses(d):
    _, dfn, d = CF.hypotheses(d)
    g = lambda c: d[c].fillna(0).values.astype(float)
    fd, mis, bfs, po, th = g("fd_n"), g("fd_mis_n"), g("bfs_n"), g("n_po"), g("n_th")
    em, emm, sc, scm = g("em_n"), g("em_mis_n"), g("sc_n"), g("sc_mis_n")
    comm = np.maximum(po, th)
    traced = (mis > 0) | (bfs > 0)
    rural = d.ruca_primary.fillna(1).values >= RURAL
    w = 0.5 / (2.0 + dfn.astype(float))
    nfac = 1.0 + ((sc + scm) > 0) + ((em + emm) > 0)
    blk = (fd == 0) & ~traced & (comm >= 1) & rural
    part = np.where((fd >= 1) & ~traced & (comm > fd), 1 - fd / np.maximum(comm, 1), 0.0)
    settle = (blk.astype(float) + part) * w / nfac
    # E63: the facilities half from our measured flags, then s fitted to the POI mean only
    fit = d.region.ne("eastern-wa").values
    fac = ((((fd == 0) & traced) * 0.58 + ((em == 0) & (emm > 0)) * 0.3 + ((sc == 0) & (scm > 0)) * 0.5) / nfac)
    facdef = (fd + mis + bfs + em + emm + sc + scm) > 0
    poi = d.poi_all.fillna(0).values.astype(float)
    r0 = zip_reference(d)
    comp = lambda cbp: np.where(facdef, (fac + cbp) / 2, cbp)
    s = brentq(lambda z: comp(CPO.gap(r0 * z, poi))[fit].mean() - 0.052, 1e-6, 1e6)
    cbp = CPO.gap(r0 * s, poi)
    zipcbp = cbp * np.where(facdef, 0.5, 1.0) / (2.0 + dfn.astype(float))
    return {"settle": settle, "zipcbp": zipcbp}, dfn, d, comp(cbp)


def candidates(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    base = lb.read_sub(BASE).coverage_gap_score.values
    H, dfn, d, P = hypotheses(table())
    out = {}
    for k, h in H.items():
        others = [CMM.project(v, d, L)[0] for j, v in H.items() if j != k]
        col, perp = CMM.project(h, d, L, extra=others)
        out[k] = (np.clip(base + sd * M * CHAT[k] * col, 0, 1), col, perp)
    rho_b = lb.rho(float(L[L.file == BASE].rmse.iloc[0]), base, m1, m2)
    return out, base, rho_b, H, dfn, d, P


def evidence(L=None):
    L = lb.ledger() if L is None else L
    out, base, rho_b, H, dfn, d, P = candidates(L)
    rural, burn = CC.strata(d)
    fit = d.region.ne("eastern-wa").values
    print(f"base {BASE} scored {L[L.file == BASE].rmse.iloc[0]:.6f} (rho {rho_b:.4f}); "
          f"E63 composite POI burned/unburned {P[fit & burn].mean() / P[fit & ~burn].mean():.2f} (pub 3.40, unfitted)")
    for k, (p, col, perp) in out.items():
        h = np.abs(H[k])
        print(f"\n{k}: {int((h > 0)[fit].sum())} tracts > 0, burned/unburned "
              f"{h[fit & burn].mean() / h[fit & ~burn].mean():.2f}, rural/urban "
              f"{h[fit & rural].mean() / max(h[fit & ~rural].mean(), 1e-12):.2f}, perp {perp:.3f}")
        print(CK.traps({k: col}, d, dfn).round(3).to_string())
        a = validate.analyse(p, L)
        print(f"neutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}]")
        FC.table(rho_b, {k: CHAT[k]}, M, L, label=FILES[k][0])
    print(f"\ncorr of projected parts {np.corrcoef(out['settle'][1], out['zipcbp'][1])[0, 1]:+.3f}")


def build(L=None):
    out, *_, d, _ = candidates(L)
    for k, (p, _, _) in out.items():
        lb.write(FILES[k][0], d.GEOID, p, FILES[k][1], kind="model")


if __name__ == "__main__":
    build() if "build" in sys.argv[1:] else evidence()
