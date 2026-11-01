"""Physics v9 (29 Oct): Ahnaf's spec vehicle V8 (E120's, `cand_round30.vehicles`) with its mechanism weights FITTED
to the score-visible part of the truth, the move that gave E90 (c +0.130) on physics v5.

V8 is calibrated only to published aggregates (k per region on the 16 cells, p_mis on 270, p_bg on 740, s on the POI
mean, M_S on 8.7); its in-span fit is corr 0.927. Its transport reference is a sum of measured terms,

    den8 = core + toll_in + (cc_multi + bf_x_multi_m) + (cc_local + bf_x_local_m) + (cc_named + bf_x_named_m)
           - (ds_in + bf_x_desig_m) + bf_out_m

each entered at weight 1 by mechanism. Here every term gets a weight (1 = V8), plus the physics5 mechanisms V8 does
not hold (interstate, divided carriageways, large buildings), the urban station-rate exponent, and the noise and
completeness parameters. With THETA0 the model reproduces V8 exactly (`check`). The road-defined mask is V8's.

The fit minimises the in-span misfit |P_S(y) - P_S(phys)|^2, P_S the truncated projection on the scored span
(`validate.scored`, which never reads e33/e37; this is the weights step). The bet is the fitted model's
out-of-span part. **Hold-out test before any slot is spent** (`holdout`): fit on the span up to a cut file, and
measure the covariance of the out-of-span part (relative to that early span) with y using the later scored files.

    python src/physics9.py check              # theta0 reproduces V8 (max diff must be ~0)
    python src/physics9.py fit [CUT] [MAXFEV] # fit on the span up to CUT (default: all); prints ETA
    python src/physics9.py holdout CUT        # needs fit CUT first: out-of-span part vs the later scored files
"""
import contextlib
import io
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

import cand_county as CY
import cand_round25 as R
import cand_round26 as R6
import cand_round28 as R8
import cand_round29 as R9
import cand_round30 as R30
import lb
import physics3 as P3
import validate

DERIVED = lb.ROOT.parent / "data" / "derived"
KEYS = ["SIGMA_T", "SIGMA_C", "W_ST", "P_BLD", "M_S", "A_TOLL", "A_MULTI", "A_LOCAL", "A_NAMED", "A_DES", "A_BND",
        "W_INTER", "W_DUAL", "ALPHA", "W_BIG", "LAM_U", "W_AG", "W_TFAC", "W_RICH"]
BOUNDS = dict(SIGMA_T=(0.03, 0.8), SIGMA_C=(0.1, 3.0), W_ST=(0.0, 3.0), P_BLD=(0.0, 1.5), M_S=(0.0, 1.0),
              A_TOLL=(0.0, 2.0), A_MULTI=(0.0, 2.0), A_LOCAL=(0.0, 2.0), A_NAMED=(0.0, 2.0), A_DES=(0.0, 2.0),
              A_BND=(0.0, 2.0), W_INTER=(-1.0, 1.5), W_DUAL=(-1.0, 1.0), ALPHA=(0.0, 1.5), W_BIG=(-1.5, 1.5),
              LAM_U=(0.0, 1.0), W_AG=(0.0, 1.0), W_TFAC=(0.0, 3.0), W_RICH=(-1.5, 1.5))


def theta0():
    return dict(SIGMA_T=P3.SIGMA_T, SIGMA_C=P3.SIGMA_C, W_ST=P3.W_ST, P_BLD=P3.P_BLD, M_S=R30.M_S_STAR,
                A_TOLL=1.0, A_MULTI=1.0, A_LOCAL=1.0, A_NAMED=1.0, A_DES=1.0, A_BND=1.0,
                W_INTER=0.0, W_DUAL=0.0, ALPHA=1.0, W_BIG=0.0, LAM_U=0.0, W_AG=0.0, W_TFAC=0.0, W_RICH=0.0)


def inputs():
    """V8's frame plus the separated reference terms (all in sample order)."""
    with contextlib.redirect_stdout(io.StringIO()):
        d, X4, o4, k4, X8, o8, k8, par = R30.vehicles()
    rt = CY.routes_table()
    bf, bm, ds, cc = R30.frame("bndfollow"), R30.frame("bfmix"), R30.frame("desig"), R30.frame("concurrent")
    big = R30.frame("commercial")
    lbig = np.log1p(big.bld_ge1k.values.astype(float))
    T = dict(core=rt.rt_core.values.astype(float), toll=bf.toll_in.values,
             multi=(cc.cc_multi + bm.bf_x_multi_m).values, local=(cc.cc_local + bm.bf_x_local_m).values,
             named=(cc.cc_named + bm.bf_x_named_m).values, des=(ds.ds_in + bm.bf_x_desig_m).values,
             bnd=bm.bf_out_m.values, inter=cc.cc_inter.values, oneway=cc.cc_oneway.values)
    free = (X4["fd"] == 0) & ((X4["fdm"] + X4["bfs"]) == 0)
    urban = X4["rural"] == 0
    area = d.area_m2.values / 1e6
    pl = R30.frame("places")
    Z = np.column_stack([pl[c].replace(0, np.nan).fillna(pl[c].median()).values.astype(float) if c == "conf_mean"
                         else pl[c].values.astype(float) for c in ["phone_share", "site_share", "email_share",
                                                                   "social_share", "conf_mean"]])
    z = ((Z - Z.mean(0)) / Z.std(0)).mean(1)
    trib = d[["GEOID"]].merge(pd.read_parquet(DERIVED / "all.parquet")[["GEOID", "tribal_legal_pct"]], on="GEOID",
                              how="left").tribal_legal_pct.fillna(0).clip(0, 1).values
    import published as DL
    bq = np.log((pl.n_bq.values.astype(float) + 0.5) / (pl.n_places.values.astype(float) + 1.0))  # registry share
    X0 = dict(X8, trib=trib, zrich=(z - z.mean()) / z.std(), zbq=(bq - bq.mean()) / bq.std(),
              pin=np.isin(d.GEOID.astype(str).values, DL.PIN_GEOIDS),
              den8=X8["den"].copy(), r08=X8["r0"].copy(), unr0=X4["unr"].copy(), T=T, free=free, urban=urban,
              area=area, zbig=(lbig - lbig.mean()) / lbig.std(), rural0=X4["rural"].copy(), X4=X4)
    return d, X0, o8


TH = {}


def simulate9(X, k, p_mis, p_bg, s, draws=P3.DRAWS, seed=P3.SEED, parts=False, p_comm=P3.P_COMM, m_s=1.0):
    """cand_round30.simulate (V8: misfiled blackouts need a misfiled PLACE) plus physics8's three mechanisms, read
    from TH: W_AG one fire record per agency, W_TFAC reference facilities on legal tribal land filed under other
    Overture categories, W_RICH business recall visible in the record metadata. Identical to V8 at 0."""
    w_ag, w_tfac, w_rich = TH.get("W_AG", 0.0), TH.get("W_TFAC", 0.0), TH.get("W_RICH", 0.0)
    u_sch = TH.get("U_SCH", 0.0)                   # physics12: misfiled schools weigh (1 + U_SCH) in urban tracts
    w_bq = TH.get("W_BQ", 0.0)                     # physics12: CBP reference rises with the BrightQuery share
    trib = X["trib"]
    rng = np.random.default_rng(seed)
    n = len(X["num"])
    dT = X["den"] > 0
    lam_f = p_mis * X["fdm"] + P3.P_BLD * X["bfs"] + p_comm * X["miss"] + p_bg * X["bg"] * (1.0 + w_tfac * trib)
    lam_e = m_s * (P3.P_EMS * X["emm"] + w_tfac * trib * (X["em"] == 0))
    lam_s = m_s * (P3.P_SCH * X["scm"] * (1.0 + u_sch * X["urban"]) + P3.P_SCHB * X["scb"]
                   + w_tfac * trib * (X["sc"] == 0))
    fd_base = np.where(X["fd"] > 0, 1.0 + (X["fd"] - 1.0) * (1.0 - w_ag), 0.0)
    cbp0 = s * X["r0"] * np.maximum(X["nost"], 0.02) ** P3.W_ST * np.exp(-w_rich * X["zrich"])
    if w_bq:
        cbp0 = cbp0 * np.exp(w_bq * X["zbq"])
    acc = {c: np.zeros(n) for c in ["y", "T", "P", "fblk", "fmis", "gf", "gs"]}
    for _ in range(draws):
        tiger = k * X["den"] * np.exp(P3.SIGMA_T * rng.standard_normal(n) - P3.SIGMA_T ** 2 / 2)             + P3.LAM_U * X["unr"] + P3.LAM_T * X["ter"]
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
        acc["y"] += (T * dT + P3.B_CONST + P) / (dT + 2.0)
        acc["T"] += T
        acc["P"] += P
        acc["fblk"] += (rf > 0) & (X["fd"] == 0)
        acc["fmis"] += (rf > 0) & (X["fd"] == 0) & (X["fdm"] > 0)
        acc["gf"] += gf
        acc["gs"] += gs
    return {c: v / draws for c, v in acc.items()}


def calibrate9(d, X, m_s):
    old = P3.simulate, P3.M_S
    P3.simulate, P3.M_S = simulate9, m_s
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return P3.calibrate(d, X)
    finally:
        P3.simulate, P3.M_S = old


def model(d, X0, th, draws=200, parts=False):
    T = X0["T"]
    X = {k: v for k, v in X0.items() if k not in ("T", "X4")}
    TH.clear()
    TH.update({k: th.get(k, 0.0) for k in ("W_AG", "W_TFAC", "W_RICH", "U_SCH", "W_BQ")})
    den = (T["core"] + th["A_TOLL"] * T["toll"] + th["A_MULTI"] * T["multi"] + th["A_LOCAL"] * T["local"]
           + th["A_NAMED"] * T["named"] - th["A_DES"] * T["des"] + th["A_BND"] * T["bnd"]
           - th["W_INTER"] * T["inter"] - th["W_DUAL"] * T["oneway"])
    den8 = X0["den8"]
    X["den"] = np.where(den8 > 0, np.maximum(den, 0.05 * den8), 0.0)          # V8's road-defined mask kept
    X["unr"] = X0["unr0"] * th["LAM_U"] / max(P3.LAM_U, 1e-9)                 # V8 runs the smear terms at 0
    X["ter"] = 0.0 * X0["ter"]
    X["r0"] = X0["r08"] * np.exp(th["W_BIG"] * X0["zbig"])
    P3.SIGMA_T, P3.SIGMA_C, P3.W_ST, P3.P_BLD = th["SIGMA_T"], th["SIGMA_C"], th["W_ST"], th["P_BLD"]
    # urban hidden stations in proportion to area^ALPHA (V8: 1), urban scale re-solved on the 740 blackouts
    free, urban, area, rural = X0["free"], X0["urban"], X0["area"], X0["rural0"]
    au = area ** th["ALPHA"]
    au = au / au[free & urban].mean()
    fit = d.region.ne("eastern-wa").values
    X4 = X0["X4"]
    with contextlib.redirect_stdout(io.StringIO()):
        _, p_mis4, p_bg4, _, _ = P3.calibrate(d, X4) if "cal4" not in X0 else X0["cal4"]
    X0.setdefault("cal4", (None, p_mis4, p_bg4, None, None))
    bg = lambda f: free * (rural + f * (1 - rural) * au)
    blk = lambda f: P3.simulate(dict(X4, bg=bg(f)), 1.1, p_mis4, p_bg4, 1.0, draws=60, m_s=P3.M_S)["fblk"][fit].sum()
    try:
        f = brentq(lambda z: blk(z) - R30.BLACKOUTS, 0.001, 5.0)
    except ValueError:
        f = 0.001 if blk(0.001) > R30.BLACKOUTS else 5.0
    X["bg"] = bg(f)
    k, p_mis, p_bg, s, ms = calibrate9(d, X, th["M_S"])
    o = simulate9(X, k, p_mis, p_bg, s, m_s=ms, draws=draws)
    o["y"] = np.where(X0["pin"], 1.0, o["y"])      # the two scorer-pinned water tracts (LEDGER 111a)
    return o if parts else o["y"]


class Span:
    """Truncated projection on the scored span up to `cut` (exactly validate.analyse's), and P_S(y) from the scores."""

    def __init__(self, cut=None, L=None):
        L = lb.ledger() if L is None else L
        if cut:
            L = L.loc[:L.index[L.file == cut][0]]
        self.L = L
        m1, m2 = lb.moments(L)
        self.files, P, c = validate.scored(L, m1, m2)
        Pc = P - P.mean(0)
        U, s, _ = np.linalg.svd(Pc, full_matrices=False)
        self.U = U[:, s > s[0] * validate.RCOND]
        self.psy = Pc @ np.linalg.lstsq(Pc.T @ Pc / len(Pc), c, rcond=validate.RCOND)[0]
        self.sd = np.sqrt(m2 - m1 ** 2)

    def proj(self, v):
        v = v - v.mean()
        return self.U @ (self.U.T @ v)

    def misfit(self, y):
        return self.psy - self.proj(y)


def fit(cut=None, maxfev=600, start=None, tag=""):
    d, X0, o8 = inputs()
    S = Span(cut)
    lo = np.array([BOUNDS[k][0] for k in KEYS])
    hi = np.array([BOUNDS[k][1] for k in KEYS])
    to_th = lambda z: dict(zip(KEYS, lo + (hi - lo) / (1 + np.exp(-z))))
    x0 = np.clip(np.array([dict(theta0(), **(start or {}))[k] for k in KEYS]), lo + 1e-3 * (hi - lo),
                 hi - 1e-3 * (hi - lo))
    z0 = -np.log((hi - lo) / (x0 - lo) - 1)
    hist, t0 = [], time.time()
    T0 = theta0()

    def f(z):
        try:
            v = float((S.misfit(model(d, X0, to_th(z))) ** 2).mean()) * 1e6
        except (ValueError, FloatingPointError):
            v = 1e4
        hist.append(v)
        if len(hist) % 10 == 0:
            el = time.time() - t0
            print(f"   eval {len(hist)}/{maxfev}: best {min(hist):.3f}e-6, {el / 60:.1f} min elapsed, "
                  f"<= {(maxfev - len(hist)) * el / len(hist) / 60:.0f} min left", flush=True)
        return v

    v0 = f(z0)
    r = minimize(f, z0, method="Powell", options=dict(maxiter=50, maxfev=maxfev, xtol=1e-2, ftol=1e-4))
    th = to_th(r.x)
    print(f"misfit MS {v0:.3f}e-6 -> {r.fun:.3f}e-6 after {len(hist)} evals (span: {len(S.files)} scored files"
          f"{' up to ' + cut if cut else ''})")
    for k in KEYS:
        print(f"   {k:8s} {dict(T0, **(start or {})).get(k, 0):7.3f} -> {th[k]:7.3f}")
    out = DERIVED / f"physics9_theta{tag}.json"
    out.write_text(json.dumps(th, indent=1))
    return th


def holdout(cut, tag):
    """The fitted (on the span up to `cut`) model's part outside that span, measured against y with the FULL span.
    Its covariance with y is exact for the component the later files cover; compare with V8 unfitted."""
    d, X0, o8 = inputs()
    early, full = Span(cut), Span()
    th = json.loads((DERIVED / f"physics9_theta{tag}.json").read_text())
    rows = []
    for name, y in [("V8 (unfitted)", o8["y"]), ("V9 fitted on early span", model(d, X0, th, draws=400))]:
        h = (y - y.mean()) - early.proj(y)                    # out of the early span
        hs = full.proj(h)                                     # the part of it the later files see
        c_seen = float(hs @ full.psy / len(h)) / (full.sd * h.std())       # cov(h, y) via the full span, as a corr
        rows.append((name, h.std() / full.sd, float(np.linalg.norm(hs) / np.linalg.norm(h)), c_seen))
    for n, size, seen, c in rows:
        print(f"{n:26s} out-of-early-span size {size:.3f} sd(y), share seen by later files {seen:.3f}, "
              f"corr with y via later files {c:+.4f}")


def bet(name, base, tag="", chat=0.06, m=0.85, comment=None):
    """base (a SCORED clean file) + W * (fitted V9's part outside the current scored span, unit sd), W = m * chat * sd(y):
    the E90 construction. Pins at 1. Prints the dose table (null, at c = chat, at c = 2 chat)."""
    import cand_mm as CMM
    import published as DL
    d, X0, o8 = inputs()
    L = lb.ledger()
    th = json.loads((DERIVED / f"physics9_theta{tag}.json").read_text())
    y = model(d, X0, th, draws=400)
    col, perp = CMM.project(y, d, L)
    b = lb.read_sub(base)
    assert (b.GEOID.astype(str).values == d.GEOID.astype(str).values).all()
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    W = m * chat
    q = np.clip(b.coverage_gap_score.values + sd * W * col, 0, 1)
    q[np.isin(b.GEOID.values, DL.PIN_GEOIDS)] = 1.0
    rb = float(L[L.file == base].rmse.iloc[0])
    rho2 = 1 - rb ** 2 / sd ** 2
    for c in (0.0, chat, 2 * chat):
        print(f"   if c = {c:+.3f}: {sd * np.sqrt(1 - rho2 + W * (W - 2 * c)):.7f}")
    print(f"{name}: perp {perp:.3f} (out-of-span size {perp:.3f} sd of the vehicle), W {W:.4f} on {base} ({rb:.7f})")
    lb.write(name, b.GEOID, q, comment or f"physics v9 (V8 with mechanism weights fitted to the scored span) "
                                         f"unscored part on {base}, W {W:.4f}", kind="model")
    np.save(DERIVED / f"physics9_y{tag}.npy", y)


def check():
    d, X0, o8 = inputs()
    y = model(d, X0, theta0(), draws=400)
    print(f"theta0 vs V8: max |diff| {np.abs(y - o8['y']).max():.2e}, den max |diff| "
          f"{np.abs(np.where(X0['den8'] > 0, X0['T']['core'] + X0['T']['toll'] + X0['T']['multi'] + X0['T']['local'] + X0['T']['named'] - X0['T']['des'] + X0['T']['bnd'], 0) - X0['den8']).max():.2e}")
    S = Span()
    print(f"V8 in-span misfit MS {(S.misfit(o8['y']) ** 2).mean() * 1e6:.3f}e-6 on {len(S.files)} scored files")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["check"]:
        check()
    elif a[:1] == ["fit"]:
        cut = a[1] if len(a) > 1 and a[1] != "all" else None
        start = json.loads((DERIVED / a[3]).read_text()) if len(a) > 3 else None
        fit(cut, int(a[2]) if len(a) > 2 else 600, start=start,
            tag=(("_" + cut.split("_")[0]) if cut else "") + (a[4] if len(a) > 4 else ""))
    elif a[:1] == ["holdout"]:
        holdout(a[1], "_" + a[1].split("_")[0])
    elif a[:1] == ["bet"]:                         # bet NAME BASE CHAT
        bet(a[1], a[2], chat=float(a[3]) if len(a) > 3 else 0.06)
