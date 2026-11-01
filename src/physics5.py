"""Physics model v5 (E89+): the mechanism parameters calibrated on the score-visible part of the truth.

How a participant got their jumps (public method notes, 22 Oct): the best combination of scored files is the exact projection
P_S(y) of the truth on the scored span. A physics model makes its own projection P_S(phys). Fit the model's
MECHANISM parameters (not per-tract values) so the two agree, keeping every published-aggregate calibration
(per-region road level, 270/740 fire blackouts, POI mean) inside the loop. The fitted model then predicts the
truth OUTSIDE the span too, and that unscored part is the next file. The misfit screen (`src/screen.py`) is the
same quantity used diagnostically: it found concurrent routes for them at +0.31.

Parameters (13, each one named mechanism, bounded; defaults = v4 + E85/E86 doubling at one extra record):
  roads   SIGMA_T noise, LAM_U unrouted arterials, LAM_T post-import tertiary, LAM_LO edited import-era local,
          W_MULTI concurrent route numbers, W_LOCAL route + local street name, W_INTER interstate adjustment,
          W_DUAL divided highways (Overture draws two one-way carriageways)
  fire    P_BLD building-traced stations, URBAN_BG urban hidden-station rate, P_COMM community stations
  schools M_S extras multiplier
  cbp     W_ST completeness exponent, SIGMA_C noise

    python src/physics5.py fit [pre]      # fit (pre = only files scored before E85, the hold-out check)
    python src/physics5.py build NAME     # fitted model's unscored part on the best scored file
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import final_model as FM
import lb
import physics3 as P3
import physics4 as P4
import cand_mm as CMM
import validate

DERIVED = lb.ROOT.parent / "data" / "derived"
SCORED = FM.FILES + ["e85_multi.csv", "e86_local.csv", "e87_affine.csv", "e88_double2.csv", "e84_physics4.csv"]
SCORED6 = SCORED + ["e90_physics5.csv", "e91_divided.csv"]
PRE = FM.FILES                                    # the span before concurrent routes were measured
THETA0 = dict(SIGMA_T=0.22, LAM_U=0.20, LAM_T=0.05, LAM_LO=0.05, W_MULTI=1.0, W_LOCAL=1.0, W_INTER=0.0, W_DUAL=0.0, W_TRIB=0.0, W_BIG=0.0, W_FDX=0.0,
              P_BLD=0.8, URBAN_BG=0.1, P_COMM=0.5, M_S=0.12, W_ST=1.0, SIGMA_C=0.5)
BOUNDS = dict(SIGMA_T=(0.03, 1.0), LAM_U=(0, 1), LAM_T=(0, 0.6), LAM_LO=(0, 0.6), W_MULTI=(0, 3), W_LOCAL=(0, 3),
              W_INTER=(-1.0, 2.0), W_DUAL=(-1.0, 1.0), W_TRIB=(-0.8, 5.0), W_BIG=(-1.5, 1.5), W_FDX=(0.0, 1.0), P_BLD=(0, 1.5), URBAN_BG=(0, 1), P_COMM=(0, 1.5), M_S=(0, 1), W_ST=(0, 3),
              SIGMA_C=(0.1, 3.0))
KEYS = list(THETA0)
FIT = DERIVED / "physics5_theta.json"
BEST = "e87_affine.csv"
CHAT, M = 0.06, 0.85


def inputs():
    d, X = P4.inputs()
    cc = pd.concat([pd.read_parquet(DERIVED / f"{r}-concurrent.parquet") for r in sorted(d.region.unique())])
    cc = d[["GEOID"]].merge(cc, on="GEOID", how="left").fillna(0)
    lin = pd.concat([pd.read_parquet(DERIVED / f"{r}-lineage.parquet") for r in sorted(d.region.unique())])
    lin = d[["GEOID"]].merge(lin, on="GEOID", how="left").fillna(0)
    X0 = dict(X, unr0=X["unr"].copy(), te_new=lin.lin_te_new.values.astype(float),
              lo_edit=lin.lin_lo_edit.values.astype(float), multi=cc.cc_multi.values.astype(float),
              local=cc.cc_local.values.astype(float), inter=cc.cc_inter.values.astype(float), oneway=cc.cc_oneway.values.astype(float),
              named=cc.cc_named.values.astype(float),
              web=X["nost"].copy())
    # E92+ screen leads (neutral at 0, so E90/E91 rebuild unchanged): legal tribal land, large buildings, listed stations
    ex = d[["GEOID"]].merge(pd.read_parquet(DERIVED / "all.parquet")[["GEOID", "tribal_legal_pct"]], on="GEOID", how="left")
    big = pd.concat([pd.read_parquet(DERIVED / f"{r}-commercial.parquet")[["GEOID", "bld_ge1k"]]
                     for r in sorted(d.region.unique())])
    big = d[["GEOID"]].merge(big, on="GEOID", how="left").fillna(0)
    lb_ = np.log1p(big.bld_ge1k.values.astype(float))
    X0.update(trib=np.clip(ex.tribal_legal_pct.fillna(0).values.astype(float), 0, 1),   # a fraction (v6 wrongly /100)
              big=(lb_ - lb_.mean()) / lb_.std(), den0=X["den"].copy(), r00=X["r0"].copy(),
              fdm0=X["fdm"].copy(), bfs0=X["bfs"].copy())
    return d, X0


def model(d, X0, th, draws=200):
    """Set the mechanism parameters, re-run the published-aggregate calibration, return E[y]."""
    X = dict(X0)
    P3.SIGMA_T, P3.SIGMA_C, P3.P_BLD, P3.URBAN_BG, P3.P_COMM, P3.M_S, P3.W_ST = (
        th["SIGMA_T"], th["SIGMA_C"], th["P_BLD"], th["URBAN_BG"], th["P_COMM"], th["M_S"], th["W_ST"])
    P3.LAM_U, P3.LAM_T = 1.0, 1.0                 # the extra TIGER length is assembled here, in metres
    P3.simulate.__defaults__ = (P3.DRAWS, P3.SEED, False, th["P_COMM"], 1.0)   # calibrate() uses the default p_comm
    X["unr"] = np.maximum(th["LAM_U"] * X0["unr0"] + th["W_MULTI"] * X0["multi"] + th["W_LOCAL"] * X0["local"]
                          - th["W_INTER"] * X0["inter"] - th["W_DUAL"] * X0["oneway"], -0.9 * X0["den"])
    X["ter"] = th["LAM_T"] * X0["te_new"] + th["LAM_LO"] * X0["lo_edit"]
    X["den"] = X0["den0"] * (1 + th.get("W_TRIB", 0.0) * X0["trib"])          # BIA / tribal roads in TIGER
    X["r0"] = X0["r00"] * np.exp(th.get("W_BIG", 0.0) * X0["big"])            # establishments follow large buildings
    keep = 1 - th.get("W_FDX", 0.0) * (X0["fd"] > 0)                          # fewer extras where a station is listed
    X["fdm"], X["bfs"] = X0["fdm0"] * keep, X0["bfs0"] * keep
    X["bg"] = (X["fd"] == 0) * ((X["fdm"] + X["bfs"]) == 0) * (X["rural"] + th["URBAN_BG"] * (1 - X["rural"]))
    k, p_mis, p_bg, s, ms = P3.calibrate(d, X, draws=40)
    return P3.simulate(X, k, p_mis, p_bg, s, draws=draws, m_s=th["M_S"], p_comm=th["P_COMM"])["y"]


class Span:
    """P_S: least-squares projection on the scored files; P_S(y) is their best affine combination."""

    def __init__(self, files, L=None):
        L = lb.ledger() if L is None else L
        self.q, self.pred, _, _ = FM.affine(3e-6, files, L)
        Q = np.column_stack([lb.read_sub(f).coverage_gap_score.values for f in files])
        self.mu = Q.mean(0)
        self.U, sv, _ = np.linalg.svd(Q - self.mu, full_matrices=False)
        self.U = self.U[:, sv > sv[0] * 1e-4]

    def __call__(self, v):
        return v.mean() + self.U @ (self.U.T @ (v - v.mean()))

    def misfit(self, y):
        return self.q - self(y)


def fit(files=SCORED, maxiter=400, start=None, maxfev=600):
    d, X0 = inputs()
    S = Span(files)
    lo = np.array([BOUNDS[k][0] for k in KEYS])
    hi = np.array([BOUNDS[k][1] for k in KEYS])
    to_th = lambda z: dict(zip(KEYS, lo + (hi - lo) / (1 + np.exp(-z))))
    x0 = np.array([dict(THETA0, **(start or {}))[k] for k in KEYS])
    x0 = np.clip(x0, lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo))
    z0 = -np.log((hi - lo) / np.clip(x0 - lo, 1e-6, None) - 1 + 1e-9)
    hist = []
    import time
    t0 = time.time()

    def f(z):
        th = to_th(z)
        try:
            v = float((S.misfit(model(d, X0, th)) ** 2).mean()) * 1e6
        except ValueError:                        # the published aggregates cannot be met at these parameters
            v = 1e4
        hist.append(v)
        if len(hist) % 20 == 0:
            el = time.time() - t0
            print(f"   eval {len(hist)}/{maxfev}: best {min(hist):.3f}e-6, {el / 60:.1f} min elapsed, "
                  f"<= {(maxfev - len(hist)) * el / len(hist) / 60:.0f} min left", flush=True)
        return v

    v0 = f(z0)
    r = minimize(f, z0, method="Powell", options=dict(maxiter=maxiter, maxfev=maxfev, xtol=1e-2, ftol=1e-4))
    th = to_th(r.x)
    print(f"misfit MS {v0:.3f}e-6 -> {r.fun:.3f}e-6 after {len(hist)} evals (P_S(y) affine {S.pred:.6f})")
    for k in KEYS:
        print(f"   {k:9s} {THETA0[k]:7.3f} -> {th[k]:7.3f}")
    return th


def build(name, th, base_file=BEST, theta_out="physics5.parquet"):
    d, X0 = inputs()
    L = lb.ledger()
    y = model(d, X0, th, draws=400)
    col, perp = CMM.project(y, d, L)
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    base = lb.read_sub(base_file).coverage_gap_score.values
    p = np.clip(base + sd * M * CHAT * col, 0, 1)
    a = validate.analyse(p, L)
    print(f"perp {perp:.3f}; neutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}]")
    lb.write(name, d.GEOID, p, f"physics v5 (mechanism parameters fitted to the scored span) unscored part, over {base_file}",
             kind="model")
    pd.DataFrame({"GEOID": d.GEOID, "physics5": y}).to_parquet(DERIVED / theta_out, index=False)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "fit6":                      # 23 Oct: E90/E91 scored, + tribal / large buildings / listed stations
        out = DERIVED / "physics6_theta.json"
        th = fit(SCORED6, start=json.loads(out.read_text()) if out.exists() else json.loads(FIT.read_text()), maxfev=700)
        out.write_text(json.dumps(th, indent=1))
    elif a and a[0] == "fit7":                    # 23 Oct: tribal share fixed (fraction, not percent), E92/E93 in span
        th0 = json.loads((DERIVED / "physics6_theta.json").read_text())
        th0["W_TRIB"] = th0["W_TRIB"] / 100                                   # same model as v6 on the corrected scale
        th = fit(SCORED6 + ["e92_affine.csv", "e93_physics6.csv"], start=th0, maxfev=400)
        (DERIVED / "physics7_theta.json").write_text(json.dumps(th, indent=1))
    elif a and a[0] == "fit":
        pre = "pre" in a
        out = DERIVED / ("physics5_theta_pre.json" if pre else "physics5_theta.json")
        th = fit(PRE if pre else SCORED, start=json.loads(out.read_text()) if out.exists() else None)
        out.write_text(json.dumps(th, indent=1))
    elif a and a[0] == "build":
        build(a[1], json.loads(FIT.read_text()))


def build_part(name, th, key, comment):
    """One fitted mechanism alone: y(theta) - y(theta with that parameter at 0), out of span and E90's part, on BEST."""
    d, X0 = inputs()
    L = lb.ledger()
    h = model(d, X0, th, draws=400) - model(d, X0, dict(th, **{key: 0.0}), draws=400)
    e90 = lb.read_sub("e90_physics5.csv").coverage_gap_score.values - lb.read_sub(BEST).coverage_gap_score.values
    col, perp = CMM.project(h, d, L, extra=[e90])
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    p = np.clip(lb.read_sub(BEST).coverage_gap_score.values + sd * M * CHAT * col, 0, 1)
    a = validate.analyse(p, L)
    print(f"{key}: perp {perp:.3f}; neutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}]")
    lb.write(name, d.GEOID, p, comment, kind="model")
