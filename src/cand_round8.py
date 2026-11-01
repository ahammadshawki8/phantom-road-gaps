"""E81/E82: road LINEAGE -- the 2007-08 TIGER import trace -- as exact road-gap derivatives, on E74.

Overture keeps each road's OSM `record_id` ('w<way id>@<version>', src/features_lineage.py). The
TIGER import mapped TIGER's feature codes to motorway/primary/secondary/residential and NEVER to
tertiary, so an import-era way that is tertiary today was reclassified by a mapper -- most often a
state/county highway demoted from secondary, i.e. exactly what TIGER S1200 counts. Rural county
highways imported as secondary were also often demoted to unclassified/residential.

E81 `te_import` -- TIGER holds import-era tertiary: d gap/d lam with lam * import-era tertiary
    length. Burned 2.52x, rural 3.72x against post-import tertiary 1.85x / 1.84x (E70 used all
    tertiary, +0.028). Perp 0.468 with E70 in the span: the import/new split is the new content.
E82 `lo_import_edit` -- TIGER holds edited import-era residential/unclassified (demoted county
    highways). Burned 2.89x, rural 4.64x, perp 0.678. (14 Oct refuted county ROUTES on a pooled
    published-ratio screen; that kind of screen also wrongly killed schools.)
Both projected out of the span, the region dummies, each other and the unscored siblings E76/E78.
Lead: a participant's public repository section 7 (TIGER import padding); OSM wiki Key:tiger:reviewed.

    python src/features_lineage.py
    python src/cand_round8.py            # traps, dose
    python src/cand_round8.py build      # e81_te_import.csv, e82_lo_import.csv
"""
import sys

import numpy as np
import pandas as pd

import strata as CC
import cand_knn as CK
import cand_level as CL
import cand_mm as CMM
import cand_road as CR
import forecast as FC
import lb
import validate

BASE = "e74_physics2.csv"
FILES = {"te_import": ("e81_te_import.csv", "E81 TIGER S1200 holds import-era (reclassified) tertiary: exact road-gap "
                       "derivative, over E74"),
         "lo_import_edit": ("e82_lo_import.csv", "E82 TIGER S1200 holds edited import-era local roads (demoted county "
                            "highways): exact road-gap derivative, over E74")}
CHAT, M = 0.04, 0.85


def hypotheses(d):
    g = lambda c: d[c].fillna(0).values.astype(float)
    r = CR.road_table()
    assert (r.GEOID.values == d.GEOID.values).all()
    den = 1.1957 * CL.road_denominator(r)
    named = g("rd_len_named")
    ratio = np.where(den > 0, named / np.maximum(den, 1e-9), np.inf)
    deriv = lambda extra: np.where((den > 0) & (ratio < 1), named * extra / np.maximum(den, 1e-9) ** 2, 0.0)
    return {"te_import": deriv(g("lin_te_raw") + g("lin_te_edit")), "lo_import_edit": deriv(g("lin_lo_edit"))}


def candidates(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    base = lb.read_sub(BASE).coverage_gap_score.values
    d = CC.table_all()
    d = d.merge(pd.concat([pd.read_parquet(lb.ROOT.parent / "data" / "derived" / f"{r}-lineage.parquet")
                           for r in sorted(d.region.unique())]), on="GEOID", how="left")
    H = hypotheses(d)
    sib = [lb.read_sub(f).coverage_gap_score.values - base for f in ["e76_physics3.csv", "e78_grad_sigt.csv"]]
    out = {}
    for k, h in H.items():
        others = sib + [CMM.project(v, d, L)[0] for j, v in H.items() if j != k]
        col, perp = CMM.project(h, d, L, extra=others)
        out[k] = (np.clip(base + sd * M * CHAT * col, 0, 1), col, perp)
    rho_b = lb.rho(float(L[L.file == BASE].rmse.iloc[0]), base, m1, m2)
    return out, rho_b, d


def evidence(L=None):
    L = lb.ledger() if L is None else L
    out, rho_b, d = candidates(L)
    _, dfn, _ = CMM.hypothesis(d)
    for k, (p, col, perp) in out.items():
        print(f"\n{k}: perp {perp:.3f}")
        print(CK.traps({k: col}, d, dfn).round(3).to_string())
        a = validate.analyse(p, L)
        print(f"neutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}]")
        FC.table(rho_b, {k: CHAT}, M, L, label=FILES[k][0])


def build(L=None):
    out, _, d = candidates(L)
    for k, (p, _, _) in out.items():
        lb.write(FILES[k][0], d.GEOID, p, FILES[k][1], kind="model")


if __name__ == "__main__":
    build() if "build" in sys.argv[1:] else evidence()
