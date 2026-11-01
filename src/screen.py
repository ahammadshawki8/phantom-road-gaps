"""In-span misfit screen: which per-tract signal does the physics model still miss? (22 Oct)

How a participant found concurrent routes (their method notes, "Road screen"): the part of the truth that our scored
files already reveal is known exactly -- the best combination of scored files, P_S(y) -- and so is the physics
model's own projection on the same span, P_S(physics). Their difference, the MISFIT, is the score-visible part of
everything the model gets wrong. Any signal that correlates with it points at a missing mechanism; when the
mechanism is added, its correlation should collapse (theirs fell +0.31 -> +0.09). Nothing new is learned from the
leaderboard: the misfit is built from the same scored files and the same algebra as the affine combinations.

    python src/screen.py            # physics with the E85/E86 doubling at w = 1.3, misfit, ranked signals

Correlations are within region (region means removed from both sides), on log1p of lengths/counts.
"""
import sys

import numpy as np
import pandas as pd

import final_model as FM
import lb

DERIVED = lb.ROOT.parent / "data" / "derived"
TABLES = ["features", "boundary", "routes", "quality", "geom", "tiger", "centroid", "commercial", "misfiled",
          "settle", "places", "lineage", "concurrent"]
SCORED = FM.FILES + ["e85_multi.csv", "e86_local.csv", "e87_affine.csv", "e88_double2.csv", "e84_physics4.csv"]


def table():
    d = pd.read_parquet(DERIVED / "all.parquet")
    regions = sorted(d.region.unique())
    for t in TABLES:
        x = pd.concat([pd.read_parquet(DERIVED / f"{r}-{t}.parquet") for r in regions])
        x = x.loc[:, ~x.columns.isin(d.columns.drop("GEOID"))]
        d = d.merge(x, on="GEOID", how="left")
    return d


def misfit(phys, L=None):
    L = lb.ledger() if L is None else L
    q, pred, _, _ = FM.affine(3e-6, SCORED, L)                      # P_S(y): best combination of scored files
    Q = np.column_stack([lb.read_sub(f).coverage_gap_score.values for f in SCORED])
    Qc = Q - Q.mean(0)
    pc = phys - phys.mean()
    beta = np.linalg.solve(Qc.T @ Qc + 1e-6 * len(Qc) * np.eye(Q.shape[1]), Qc.T @ pc)
    return q - (phys.mean() + Qc @ beta), pred                       # P_S(y) - P_S(physics)


def rank(d, m, top=45):
    reg = d.region.values
    dem = lambda v: v - pd.Series(v).groupby(reg).transform("mean").values
    mm = dem(m)
    rows = []
    for c in d.columns:
        if c in ("GEOID", "region") or not pd.api.types.is_numeric_dtype(d[c]) or d[c].dtype == bool:
            continue
        v = d[c].astype(float).fillna(0).values
        if v.std() == 0:
            continue
        v = np.log1p(np.maximum(v, 0)) if v.min() >= 0 and v.max() > 20 else v
        v = dem(v)
        if v.std() > 0:
            rows.append((c, float(np.corrcoef(v, mm)[0, 1])))
    r = pd.DataFrame(rows, columns=["signal", "corr"]).assign(a=lambda x: x["corr"].abs())
    return r.sort_values("a", ascending=False).drop(columns="a").head(top)


if __name__ == "__main__":
    import cand_round9 as R9
    w = float(sys.argv[1]) if len(sys.argv) > 1 else 1.3
    dp, phys = R9.physics_y2(w, w)
    d = table()
    d = d.set_index("GEOID").loc[dp.GEOID.values].reset_index()
    m, pred = misfit(phys)
    print(f"physics (doubling w={w}) vs P_S(y) (affine {pred:.6f}): misfit RMS {m.std():.4f}, "
          f"corr(physics, P_S y) {np.corrcoef(phys, m + phys)[0, 1]:.3f}")
    pd.DataFrame({"GEOID": dp.GEOID, "physics5": phys, "misfit": m}).to_parquet(DERIVED / "misfit.parquet", index=False)
    print(rank(d, m).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


def partial_rank(d, m, top=40):
    """Rank after removing the broad 'more road, more rural' factor: region dummies + rural + road presence
    + log core length + log density, regressed out of both the misfit and each signal."""
    g = lambda c: d[c].astype(float).fillna(0).values
    Z = np.column_stack([pd.get_dummies(d.region).values.astype(float), (g("ruca_primary") >= 4).astype(float),
                         (g("rt_core") + g("rg15_core") > 0).astype(float), np.log1p(g("rt_core")),
                         np.log1p(g("rg15_core")), np.log1p(g("hu_density") if "hu_density" in d else g("ALAND"))])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    mm = res(m)
    rows = []
    for c in d.columns:
        if c in ("GEOID", "region") or not pd.api.types.is_numeric_dtype(d[c]) or d[c].dtype == bool:
            continue
        v = g(c)
        if v.std() == 0:
            continue
        v = res(np.log1p(np.maximum(v, 0)) if v.min() >= 0 and v.max() > 20 else v)
        if v.std() > 1e-12:
            rows.append((c, float(np.corrcoef(v, mm)[0, 1])))
    r = pd.DataFrame(rows, columns=["signal", "pcorr"]).assign(a=lambda x: x["pcorr"].abs())
    return r.sort_values("a", ascending=False).drop(columns="a").head(top)
