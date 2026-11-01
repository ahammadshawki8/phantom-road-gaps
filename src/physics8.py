"""Physics model v8 (E96+): three new mechanisms on top of the v7 fit, each neutral at 0.

v7 (LEDGER 94) left three signals in the in-span misfit screen (`src/screen.py`, partial rank after
the road/rural factor), and E95 refuted the *road* reading of the largest one. The three parameters
added here are the facility / business-count readings the handoff (§4.1, §4.3, §4.4) asked for:

  W_TFAC  the reference's facility records on LEGAL TRIBAL LAND that Overture files under other
          primary categories (tribal government, community centre, IHS clinic, BIE school), so the
          fire / EMS / school components read as missing. Same scorer mechanic as E49/E56.
          It is a REALLOCATION, not a level: the hidden-station rate is multiplied by
          (1 + W_TFAC * tribal_legal_pct) on the fire blackout tracts, and p_bg is then re-solved so
          a participant's 740 blackouts still hold -- more of them land on tribal land, fewer elsewhere.
          EMS and schools get the same rate where Overture lists none of that type.
          E95 measured the ROAD reading of the tribal residual at c -0.017 (refuted), so with E95
          in the span the fit now has to choose, and W_TRIB is free to fall back to 0.

  W_AG    the fire reference counts one record per AGENCY, not per building. A city department with
          three stations is one reference record, so where Overture lists n > 1 stations the ratio
          num/ref >= 1 and the gap is exactly 0. v6/v7's W_FDX ("fewer hidden extras where a station
          is listed") fitted to 0 -- it scaled the extras; this scales the reference base itself:
              ref_base = 1 + (n - 1) * (1 - W_AG)      (W_AG = 0 -> n, the v7 model; 1 -> one agency)
          It cannot disturb the 270/740 blackout calibration, which lives on n = 0 tracts.

  W_RICH  Overture's recall of BUSINESSES is visible in its own record metadata: a tract whose places
          carry phone / website / email / social came from a business-listing feed, not a bulk import,
          so Overture lists nearly all of the CBP establishments there and the CBP half clips to 0.
          v3's single `nost = 1 - web_share` exponent W_ST hit its bound in every fit (the CBP half
          "is not identified"); this multiplies the reference by exp(-W_RICH * z_rich) on a composite
          of five richness shares, and W_ST / sigma_c are given wider bounds.

    python src/physics8.py fit [maxfev]     # refit all 20 parameters on the current scored span
    python src/physics8.py screen           # misfit screen on the fitted model
    python src/physics8.py part KEY NAME    # one fitted mechanism alone, as a submission
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

import cand_mm as CMM
import final_model as FM
import lb
import physics3 as P3
import physics5 as P5
import validate

DERIVED = lb.ROOT.parent / "data" / "derived"
FIT = DERIVED / "physics8_theta.json"
BEST = "e92_affine.csv"                       # the best SCORED file (rule: never build on an unscored base)
CHAT, M = 0.06, 0.85

# the whole scored span as of 23 Oct (39 files): v5's list + E90/E91 + E92/E93/E95
SPAN = P5.SCORED6 + ["e92_affine.csv", "e93_physics6.csv", "e95_tribal_roads.csv",
                    # 24 Oct: E96 (dead null, c -0.00003), E97 (c +0.0446), E98 (c -0.0133)
                    "e96_physics8.csv", "e97_named.csv", "e98_schoolcat.csv",
                    # 25 Oct: E99 (+0.046), E100 (-0.007), E101 (-0.038), E102 (+0.013)
                    "e99_joint.csv", "e100_block.csv", "e101_cbpscope.csv", "e102_recall.csv"]

# v7 (LEDGER 90a / 92a / 94) + the three new mechanisms at 0
THETA7 = dict(SIGMA_T=0.26, LAM_U=0.0, LAM_T=0.03, LAM_LO=0.05, W_MULTI=0.52, W_LOCAL=0.48,
              W_INTER=0.43, W_DUAL=0.28, W_TRIB=0.215, W_BIG=0.94, W_FDX=0.0,
              P_BLD=0.41, URBAN_BG=0.57, P_COMM=0.5, M_S=0.51, W_ST=1.0, SIGMA_C=0.5)
THETA0 = dict(THETA7, W_TFAC=0.0, W_AG=0.0, W_RICH=0.0, W_NAMED=0.0)
BOUNDS = dict(P5.BOUNDS, W_ST=(0, 6.0), SIGMA_C=(0.02, 3.0),
              W_TFAC=(0.0, 3.0), W_AG=(0.0, 1.0), W_RICH=(-1.5, 1.5), W_NAMED=(0.0, 3.0))
KEYS = list(THETA0)

TH = dict(THETA0)                              # set by model(); read by simulate()


def inputs():
    """v5's inputs plus the legal-tribal share (already there) and the record-richness composite."""
    d, X0 = P5.inputs()
    pl = pd.concat([pd.read_parquet(DERIVED / f"{r}-places.parquet") for r in sorted(d.region.unique())])
    pl = d[["GEOID"]].merge(pl, on="GEOID", how="left")
    cols = ["phone_share", "site_share", "email_share", "social_share", "conf_mean"]
    Z = np.column_stack([pl[c].fillna(pl[c].median()).values.astype(float) for c in cols])
    Z = (Z - Z.mean(0)) / Z.std(0)
    z = Z.mean(1)                              # one factor: the five shares move together
    X0["zrich"] = (z - z.mean()) / z.std()
    return d, X0


def simulate(X, k, p_mis, p_bg, s, draws=P3.DRAWS, seed=P3.SEED, parts=False, p_comm=None, m_s=1.0):
    """physics3.simulate with the three v8 mechanisms. Identical output at W_TFAC = W_AG = W_RICH = 0."""
    p_comm = P3.P_COMM if p_comm is None else p_comm
    w_tfac, w_ag, w_rich = TH["W_TFAC"], TH["W_AG"], TH["W_RICH"]
    trib, zrich = X["trib"], X["zrich"]
    rng = np.random.default_rng(seed)
    n = len(X["num"])
    dT = X["den"] > 0
    # tribal: a reference facility with no Overture trace because Overture files it under a tribal
    # government / community / IHS / BIE category. Fire is a reallocation of the calibrated 740
    # blackouts; EMS and schools get the same rate where Overture lists none of that type.
    lam_f = (p_mis * X["fdm"] + P3.P_BLD * X["bfs"] + p_comm * X["miss"]
             + p_bg * X["bg"] * (1.0 + w_tfac * trib))
    lam_e = m_s * (P3.P_EMS * X["emm"] + w_tfac * trib * (X["em"] == 0))
    lam_s = m_s * (P3.P_SCH * X["scm"] + P3.P_SCHB * X["scb"] + w_tfac * trib * (X["sc"] == 0))
    # the fire reference counts one record per agency, not per building
    fd_base = np.where(X["fd"] > 0, 1.0 + (X["fd"] - 1.0) * (1.0 - w_ag), 0.0)
    cbp0 = s * X["r0"] * np.maximum(X["nost"], 0.02) ** P3.W_ST * np.exp(-w_rich * zrich)
    acc = {"y": np.zeros(n), "T": np.zeros(n), "P": np.zeros(n), "fblk": np.zeros(n), "fmis": np.zeros(n)}
    for _ in range(draws):
        tiger = k * X["den"] * np.exp(P3.SIGMA_T * rng.standard_normal(n) - P3.SIGMA_T ** 2 / 2) \
            + P3.LAM_U * X["unr"] + P3.LAM_T * X["ter"]
        T = np.where(dT, 1 - np.minimum(1, X["num"] / np.maximum(tiger, 1e-9)), 0.0)
        rf = fd_base + rng.poisson(lam_f)
        re = X["em"] + rng.poisson(lam_e)
        rs = X["sc"] + rng.poisson(lam_s)
        gf = np.where(rf > 0, 1 - np.minimum(1, X["fd"] / np.maximum(rf, 1)), 0.0)
        ge = np.where(re > 0, 1 - np.minimum(1, X["em"] / np.maximum(re, 1)), 0.0)
        gs = np.where(rs > 0, 1 - np.minimum(1, X["sc"] / np.maximum(rs, 1)), 0.0)
        nf = (rf > 0).astype(float) + (re > 0) + (rs > 0)
        fac = np.where(nf > 0, (gf + ge + gs) / np.maximum(nf, 1), 0.0)
        cbp_ref = cbp0 * np.exp(P3.SIGMA_C * rng.standard_normal(n) - P3.SIGMA_C ** 2 / 2)
        cbp = 1 - np.minimum(1, X["poi"] / np.maximum(cbp_ref, 1e-9))
        P = np.where(nf > 0, (fac + cbp) / 2, cbp)
        y = (T * dT + P3.B_CONST + P) / (dT + 2.0)
        acc["y"] += y
        acc["T"] += T
        acc["P"] += P
        acc["fblk"] += (rf > 0) & (X["fd"] == 0)
        acc["fmis"] += (rf > 0) & (X["fd"] == 0) & (X["fdm"] + X["bfs"] > 0)
    return {k_: v / draws for k_, v in acc.items()}


def _solve(f, a, b):
    """brentq, but clamp to the bracket when the target is unreachable instead of raising.

    physics5.fit turned a ValueError here into a misfit of 1e4, which makes the whole parameter
    direction unreachable to Powell rather than merely bad. Found 23 Oct: the first tribal form
    pushed the misfiled-blackout count past 270 at every p_mis, so W_TFAC could never be explored.
    """
    fa, fb = f(a), f(b)
    if fa == 0 or fa * fb > 0:
        return a if abs(fa) <= abs(fb) else b
    return brentq(f, a, b)


def calibrate(d, X, draws=40):
    """physics3.calibrate with clamping brackets (same solution wherever brentq succeeded)."""
    import strata as CCm
    fit_ = d.region.ne("eastern-wa").values
    rural, burn = CCm.strata(d)
    reg = d.region.values

    def blk(pm, pb):
        o = simulate(X, 1.1, pm, pb, 1.0, draws=draws, m_s=P3.M_S)
        return o["fmis"][fit_].sum(), o["fblk"][fit_].sum()

    p_mis = _solve(lambda z: blk(z, 0.0)[0] - 270, 0.01, 1.0)
    p_bg = _solve(lambda z: blk(p_mis, z)[1] - 740, 0.0, 10.0)
    s_ = _solve(lambda z: simulate(X, 1.1, p_mis, p_bg, z, draws=draws, m_s=P3.M_S)["P"][fit_].mean() - 0.052,
                1e-4, 1e4)
    k = np.full(len(reg), 1.1)
    for r_, cells in CCm.PUB16.items():
        m = reg == r_
        w = [(m & rural & burn).sum(), (m & rural & ~burn).sum(), (m & ~rural & burn).sum(), (m & ~rural & ~burn).sum()]
        target = float(np.dot(w, cells) / sum(w))

        def f(z, m=m, target=target):
            kk = k.copy()
            kk[m] = z
            return simulate(X, kk, p_mis, p_bg, s_, draws=draws, m_s=P3.M_S)["y"][m].mean() - target

        k[m] = _solve(f, 0.3, 6.0)
    k[reg == "eastern-wa"] = np.median([k[reg == r_][0] for r_ in CCm.PUB16])
    return k, p_mis, p_bg, s_, P3.M_S


def model(d, X0, th, draws=200, quiet=True):
    """Set the mechanism parameters, re-run the published-aggregate calibration, return E[y]."""
    global TH
    TH = dict(THETA0, **th)
    X = dict(X0)
    P3.SIGMA_T, P3.SIGMA_C, P3.P_BLD, P3.URBAN_BG, P3.P_COMM, P3.M_S, P3.W_ST = (
        TH["SIGMA_T"], TH["SIGMA_C"], TH["P_BLD"], TH["URBAN_BG"], TH["P_COMM"], TH["M_S"], TH["W_ST"])
    P3.LAM_U, P3.LAM_T = 1.0, 1.0                 # the extra TIGER length is assembled in X["unr"]/["ter"]
    _p3sim = P3.simulate
    P3.simulate = simulate                        # calibrate() must see the v8 model
    X["unr"] = np.maximum(TH["LAM_U"] * X0["unr0"] + TH["W_MULTI"] * X0["multi"] + TH["W_LOCAL"] * X0["local"]
                          + TH["W_NAMED"] * X0["named"]          # E100: the honorific-name record (E97, c +0.0446)
                          - TH["W_INTER"] * X0["inter"] - TH["W_DUAL"] * X0["oneway"], -0.9 * X0["den"])
    X["ter"] = TH["LAM_T"] * X0["te_new"] + TH["LAM_LO"] * X0["lo_edit"]
    X["den"] = X0["den0"] * (1 + TH["W_TRIB"] * X0["trib"])
    X["r0"] = X0["r00"] * np.exp(TH["W_BIG"] * X0["big"])
    keep = 1 - TH["W_FDX"] * (X0["fd"] > 0)
    X["fdm"], X["bfs"] = X0["fdm0"] * keep, X0["bfs0"] * keep
    X["bg"] = (X["fd"] == 0) * ((X["fdm"] + X["bfs"]) == 0) * (X["rural"] + TH["URBAN_BG"] * (1 - X["rural"]))
    try:
        k, p_mis, p_bg, s, ms = calibrate(d, X, draws=40)
        out = simulate(X, k, p_mis, p_bg, s, draws=draws, m_s=TH["M_S"], p_comm=TH["P_COMM"])["y"]
    finally:
        P3.simulate = _p3sim                      # never leave physics3 patched: physics5 would
    return out                                    # then silently run the v8 model in the same process


def fit(files=SPAN, maxfev=500, start=None):
    d, X0 = inputs()
    S = P5.Span(files)
    lo = np.array([BOUNDS[k][0] for k in KEYS])
    hi = np.array([BOUNDS[k][1] for k in KEYS])
    to_th = lambda z: dict(zip(KEYS, lo + (hi - lo) / (1 + np.exp(-z))))
    x0 = np.array([dict(THETA0, **(start or {}))[k] for k in KEYS])
    x0 = np.clip(x0, lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo))
    z0 = -np.log((hi - lo) / np.clip(x0 - lo, 1e-6, None) - 1 + 1e-9)
    hist, t0 = [], time.time()

    def f(z):
        th = to_th(z)
        try:
            v = float((S.misfit(model(d, X0, th)) ** 2).mean()) * 1e6
        except ValueError:
            v = 1e4
        hist.append(v)
        if len(hist) % 10 == 0:
            el = time.time() - t0
            print(f"   eval {len(hist)}/{maxfev}: best {min(hist):.3f}e-6, {el / 60:.1f} min elapsed, "
                  f"<= {(maxfev - len(hist)) * el / len(hist) / 60:.0f} min left", flush=True)
        return v

    v0 = f(z0)
    r = minimize(f, z0, method="Powell", options=dict(maxiter=400, maxfev=maxfev, xtol=1e-2, ftol=1e-4))
    th = to_th(r.x)
    print(f"\nmisfit MS {v0:.3f}e-6 -> {r.fun:.3f}e-6 after {len(hist)} evals (P_S(y) affine {S.pred:.6f})")
    for k in KEYS:
        print(f"   {k:9s} {THETA0[k]:7.3f} -> {th[k]:7.3f}")
    return th


def evidence(th=None):
    """Published-aggregate checks that were never fitted, for the ledger row."""
    import strata as CC
    th = json.loads(FIT.read_text()) if th is None else th
    d, X0 = inputs()
    TH.update(dict(THETA0, **th))
    y = model(d, X0, th, draws=400)
    rural, burn = CC.strata(d)
    fit_ = d.region.ne("eastern-wa").values
    print(f"composite mean {y[fit_].mean():.4f} (pub 0.0585)  rural/urban "
          f"{y[fit_ & rural].mean() / y[fit_ & ~rural].mean():.2f} (pub 2.96, unfitted)  "
          f"burned/unburned {y[fit_ & burn].mean() / y[fit_ & ~burn].mean():.2f} (pub 2.23, unfitted)")
    return d, X0, y


def part(key, name, th=None, base_file=BEST, chat=CHAT, comment=None, extra_keys=()):
    """One fitted mechanism alone: y(theta) - y(theta with that parameter at 0), out of the span."""
    th = json.loads(FIT.read_text()) if th is None else th
    d, X0 = inputs()
    L = lb.ledger()
    off = {key: 0.0}
    off.update({k: 0.0 for k in extra_keys})
    h = model(d, X0, th, draws=400) - model(d, X0, dict(th, **off), draws=400)
    print(f"{key}: {np.mean(np.abs(h) > 1e-6):.1%} of tracts move, mean |h| "
          f"{np.abs(h).mean():.5f}, max {np.abs(h).max():.4f}")
    col, perp = CMM.project(h, d, L)
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    base = lb.read_sub(base_file).coverage_gap_score.values
    p = np.clip(base + sd * M * chat * col, 0, 1)
    a = validate.analyse(p, L)
    print(f"{key}: perp {perp:.3f}; neutral {a['rmse_neutral']:.6f} "
          f"range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}]")
    if name:
        lb.write(name, d.GEOID, p, comment or f"physics v8 mechanism {key} over {base_file}", kind="model")
    return h, col, perp, a


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "fit":
        start = json.loads(FIT.read_text()) if FIT.exists() else THETA7
        th = fit(maxfev=int(a[1]) if len(a) > 1 else 500, start=start)
        FIT.write_text(json.dumps(th, indent=1))
    elif a and a[0] == "screen":
        import screen as S
        th = json.loads(FIT.read_text())
        d, X0, y = evidence(th)
        Sp = P5.Span(SPAN)
        m = Sp.misfit(y)
        print(f"P_S(y) affine {Sp.pred:.6f}; misfit RMS {m.std():.5f}")
        t = S.table()
        t = t.set_index("GEOID").loc[d.GEOID.values].reset_index()
        print(S.partial_rank(t, m, top=40).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    elif a and a[0] == "part":
        part(a[1], a[2] if len(a) > 2 else None)
