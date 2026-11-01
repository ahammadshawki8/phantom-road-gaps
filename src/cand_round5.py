"""E64-E66: from the field inventory of 18 Oct, each one mechanism, each on the SCORED best E62.

E64 `tribal_legal` -- legal tribal land (reservations / trust land, `tribal_legal_pct`), which E57's
    overall tribal share (mostly Oklahoma statistical areas) averaged away. 110 tracts, burned/
    unburned 16.9x, rural/urban 7.9x, perp 0.921. The legal-minus-statistical contrast is flat
    (0.85x) and is not used.
E65 `settle_loc` -- E62's mechanism (communities keep a volunteer fire department the reference
    lists) with communities counted as address LOCALITIES (>= 3 places), on the 869 tracts E62 and
    E49/E52 do not reach. Burned 2.65x, perp 0.834. E62 read back f 0.68 (c +0.034).
E66 `web_gap` -- record completeness of Overture places (share without a website or phone): the CBP
    half counts places, and thin records mark thinly mapped commerce. Burned 2.23x, rural 1.74x,
    perp 0.849; density corr -0.039 is stated as the weak point.
Screened and dropped the same day: closed share 0.39x, brand share 1.12x, BrightQuery share 1.10x,
Meta/Microsoft/Foursquare shares 0.67-1.11x, a BrightQuery-based CBP reference 0.76-2.02x vs 3.40x,
POI taxonomy/basic_category (identical to categories.primary), school expected counts (no age data).

    python src/features_places.py && python src/features_settle.py
    python src/cand_round5.py            # screens, traps, dose
    python src/cand_round5.py build      # e64_tribal_legal.csv, e65_settle_loc.csv, e66_web.csv
"""
import sys

import numpy as np
import pandas as pd

import strata as CC
import cand_fire as CF
import cand_knn as CK
import cand_mm as CMM
import forecast as FC
import lb
import validate

BASE = "e62_settle.csv"
FILES = {"tribal_legal": ("e64_tribal_legal.csv", "E64 legal tribal land share (reservations and trust land), over E62"),
         "settle_loc": ("e65_settle_loc.csv", "E65 missing fire stations from address-locality community counts on tracts "
                        "E62 does not reach, over E62"),
         "web_gap": ("e66_web.csv", "E66 Overture place record completeness (share without website or phone), over E62")}
CHAT = {"tribal_legal": 0.04, "settle_loc": 0.03, "web_gap": 0.03}
M = 0.85


def table():
    d = CF.table()
    for t in ["settle", "places"]:
        d = d.merge(pd.concat([pd.read_parquet(lb.ROOT.parent / "data" / "derived" / f"{r}-{t}.parquet")
                               for r in sorted(d.region.unique())]), on="GEOID", how="left")
    return d


def hypotheses(d):
    _, dfn, d = CF.hypotheses(d)
    g = lambda c: d[c].fillna(0).values.astype(float)
    fd, mis, bfs, po, th, loc = g("fd_n"), g("fd_mis_n"), g("bfs_n"), g("n_po"), g("n_th"), g("n_loc")
    sc, scm, em, emm = g("sc_n"), g("sc_mis_n"), g("em_n"), g("em_mis_n")
    ruca = d.ruca_primary.fillna(1).values
    traced = (mis > 0) | (bfs > 0)
    comm = np.maximum(po, th)
    w = 0.5 / (2.0 + dfn.astype(float))
    nfac = 1.0 + ((sc + scm) > 0) + ((em + emm) > 0)
    e62 = ((fd == 0) & ~traced & (comm >= 1) & (ruca >= 4)) | ((fd >= 1) & ~traced & (comm > fd))
    settle_loc = np.where(~e62 & ~traced,
                          np.where((fd == 0) & (loc >= 1) & (ruca >= 4), 1.0,
                                   np.where((fd >= 1) & (loc > fd), 1 - fd / np.maximum(loc, 1), 0.0)), 0.0) * w / nfac
    return {"tribal_legal": g("tribal_legal_pct"), "settle_loc": settle_loc, "web_gap": -g("web_share")}, dfn, d


def candidates(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    base = lb.read_sub(BASE).coverage_gap_score.values
    H, dfn, d = hypotheses(table())
    out = {}
    for k, h in H.items():
        others = [CMM.project(v, d, L)[0] for j, v in H.items() if j != k]
        col, perp = CMM.project(h, d, L, extra=others)
        out[k] = (np.clip(base + sd * M * CHAT[k] * col, 0, 1), col, perp)
    rho_b = lb.rho(float(L[L.file == BASE].rmse.iloc[0]), base, m1, m2)
    return out, base, rho_b, H, dfn, d


def evidence(L=None):
    L = lb.ledger() if L is None else L
    out, base, rho_b, H, dfn, d = candidates(L)
    rural, burn = CC.strata(d)
    fit = d.region.ne("eastern-wa").values
    print(f"base {BASE} scored {L[L.file == BASE].rmse.iloc[0]:.6f} (rho {rho_b:.4f})")
    for k, (p, col, perp) in out.items():
        h = np.abs(H[k]) if k == "settle_loc" else H[k] - H[k].min()
        print(f"\n{k}: {int((np.abs(H[k]) > 0)[fit].sum())} tracts != 0, burned/unburned "
              f"{h[fit & burn].mean() / h[fit & ~burn].mean():.2f}, rural/urban "
              f"{h[fit & rural].mean() / max(h[fit & ~rural].mean(), 1e-12):.2f}, perp {perp:.3f}")
        print(CK.traps({k: col}, d, dfn).round(3).to_string())
        a = validate.analyse(p, L)
        print(f"neutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}]")
        FC.table(rho_b, {k: CHAT[k]}, M, L, label=FILES[k][0])
    ks = list(out)
    print("\ncorr of projected parts:", {f"{a}-{b}": round(float(np.corrcoef(out[a][1], out[b][1])[0, 1]), 3)
                                         for i, a in enumerate(ks) for b in ks[i + 1:]})


def build(L=None):
    out, *_, d = candidates(L)
    for k, (p, _, _) in out.items():
        lb.write(FILES[k][0], d.GEOID, p, FILES[k][1], kind="model")


if __name__ == "__main__":
    build() if "build" in sys.argv[1:] else evidence()
