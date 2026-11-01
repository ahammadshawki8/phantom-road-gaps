"""Physics v12: v9b with the fire calibration freed (31 Oct, LEDGER 131d).

Every v9b fire rate is re-solved, on each evaluation, so that the simulated fire blackouts hit a competitor's
published totals (740, of which 270 misfiled). That ties the mechanisms together: raising the weight of stations
traced only as buildings (P_BLD) lowers the background rate to keep 740, so the in-span fit cannot raise the gap
where the misfit says it should be higher (tracts with no fire place and a building-traced station: +0.024 urban,
+0.034 rural), W_AG sits at its bound 1.0 and W_TFAC at 0. Here the two targets get free scales, F_BLK and F_MIS,
fitted with the fire mechanisms to the scored span; everything else stays at E127's theta. At F = 1 the model is
v9b exactly.

    python src/physics12.py check                 # v12(F = 1) == v9b(E127 theta)
    python src/physics12.py fit [CUT] [MAXFEV]    # fit the six fire keys on the span (up to CUT)
    python src/physics12.py holdout CUT           # the E110 hold-out test against v9b
    python src/physics12.py bet NAME BASE CHAT    # unscored part of the full fit on scored BASE
"""
import json
import sys
import time

import numpy as np
from scipy.optimize import minimize

import cand_round30 as R30
import lb
import physics3 as P3
import physics9 as P9

DERIVED = lb.ROOT.parent / "data" / "derived"
E127 = lb.ROOT / "params" / "physics9_e127_theta.json"
KEYS = ["P_BLD", "ALPHA", "W_AG", "W_TFAC", "F_BLK", "F_MIS"]
KEYS_S = ["M_S", "P_SCH", "U_SCH", "W_TFAC"]       # --school: misfiled schools freed, with an urban multiplier
KEYS_B = ["W_BQ", "SIGMA_C", "W_RICH", "W_ST"]     # --bq: the CBP reference rises with the BrightQuery share
BOUNDS = dict(P9.BOUNDS, F_BLK=(0.4, 2.5), F_MIS=(0.3, 2.5), P_SCH=(0.05, 2.0), U_SCH=(-0.9, 4.0), W_BQ=(-1.5, 3.0))


inputs = P9.inputs


def model(d, X0, th, draws=200):
    b0, t0, s0 = R30.BLACKOUTS, P3.FIRE_TARGETS, P3.P_SCH
    fb, fm = th.get("F_BLK", 1.0), th.get("F_MIS", 1.0)
    R30.BLACKOUTS, P3.FIRE_TARGETS, P3.P_SCH = b0 * fb, (t0[0] * fm, t0[1] * fb), th.get("P_SCH", s0)
    try:
        return P9.model(d, X0, th, draws=draws)
    finally:
        R30.BLACKOUTS, P3.FIRE_TARGETS, P3.P_SCH = b0, t0, s0


def start(cut=None):
    f = DERIVED / "physics9_theta_e110b.json" if cut else E127
    return dict(json.loads(f.read_text()), F_BLK=1.0, F_MIS=1.0, P_SCH=P3.P_SCH, U_SCH=0.0, W_BQ=0.0)


def fit(cut=None, maxfev=160, draws=200, keys=KEYS, tag2=""):
    d, X0, _ = P9.inputs()
    S = P9.Span(cut)
    th0 = start(cut)
    lo = np.array([BOUNDS[k][0] for k in keys])
    hi = np.array([BOUNDS[k][1] for k in keys])
    x0 = np.clip(np.array([th0[k] for k in keys]), lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo))
    to_th = lambda z: dict(th0, **dict(zip(keys, lo + (hi - lo) / (1 + np.exp(-z)))))
    t0, hist = time.time(), []

    def f(z):
        try:
            v = float((S.misfit(model(d, X0, to_th(z), draws=draws)) ** 2).mean()) * 1e6
        except (ValueError, FloatingPointError):
            v = 1e4
        hist.append(v)
        if len(hist) % 10 == 1:
            print(f"   eval {len(hist):4d}  misfit {v:.2f}e-6  best {min(hist):.2f}e-6  {time.time() - t0:.0f}s", flush=True)
        return v

    z0 = -np.log((hi - lo) / (x0 - lo) - 1)
    print(f"start misfit {f(z0):.2f}e-6 ({'span to ' + cut if cut else 'full span'})", flush=True)
    r = minimize(f, z0, method="Powell", options=dict(maxfev=maxfev, xtol=1e-3, ftol=1e-4))
    th = to_th(r.x)
    tag = tag2 + ("_" + cut.split("_")[0] if cut else "")
    (DERIVED / f"physics12_theta{tag}.json").write_text(json.dumps(th, indent=1))
    print(f"fit done: misfit {r.fun:.2f}e-6 after {len(hist)} evals, {time.time() - t0:.0f}s")
    print("  " + ", ".join(f"{k} {th0[k]:.3f} -> {th[k]:.3f}" for k in keys))
    return th


def holdout(cut, draws=2000, tag2=""):
    d, X0, _ = P9.inputs()
    early, full = P9.Span(cut), P9.Span()
    tag = tag2 + "_" + cut.split("_")[0]
    rows = [("v9b fitted on early span", model(d, X0, start(cut), draws=draws)),
            ("v12 fitted on early span", model(d, X0, json.loads((DERIVED / f"physics12_theta{tag}.json")
                                                                  .read_text()), draws=draws))]
    for name, y in rows:
        h = (y - y.mean()) - early.proj(y)
        hs = full.proj(h)
        c = float(hs @ full.psy / len(h)) / (full.sd * h.std())
        print(f"{name:26s} out-of-early-span size {h.std() / full.sd:.3f} sd(y), share seen by later files "
              f"{np.linalg.norm(hs) / np.linalg.norm(h):.3f}, corr with y via later files {c:+.4f}")


def combined():
    """E127's theta with every v12 full-span fit applied: fire (physics12_theta), school (thetas), BrightQuery (thetab)."""
    th = start()
    for tag, keys in (("", KEYS), ("s", KEYS_S), ("b", KEYS_B)):
        f = DERIVED / f"physics12_theta{tag}.json"
        if f.exists():
            th.update({k: json.loads(f.read_text())[k] for k in keys})
    return th


def bet(name, base, chat=0.04, m=0.85, draws=2000, comb=False):
    """The refit's change as one bet: h = y(v12 full fit) - y(same theta with the fitted keys back at E127's), its
    part outside the scored span, on scored BASE at W = m chat; writes params/NAME_final.json for final_build.
    comb=True takes the fire, school and BrightQuery fits together (`combined`)."""
    import cand_mm as CMM
    import published as PB
    d, X0, _ = inputs()
    L = lb.ledger()
    th = combined() if comb else json.loads((DERIVED / "physics12_theta.json").read_text())
    pfile = lb.ROOT / "params" / ("physics12_comb_theta.json" if comb else "physics12_theta.json")
    pfile.write_text(json.dumps(th, indent=1))
    off = {k: start()[k] for k in sorted(set(KEYS + KEYS_S + KEYS_B) if comb else KEYS)}
    h = model(d, X0, th, draws=draws) - model(d, X0, dict(th, **off), draws=draws)
    col, perp = CMM.project(h, d, L)
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    b = lb.read_sub(base)
    assert (b.GEOID.astype(str).values == d.GEOID.astype(str).values).all()
    W = m * chat
    q = np.clip(b.coverage_gap_score.values + sd * W * col, 0, 1)
    q[np.isin(b.GEOID.values, PB.PIN_GEOIDS)] = 1.0
    rb = float(L[L.file == base].rmse.iloc[0])
    rho2 = 1 - rb ** 2 / sd ** 2
    for c in (0.0, chat, 2 * chat, 4 * chat):
        print(f"   if c = {c:+.3f}: {sd * np.sqrt(1 - rho2 + W * (W - 2 * c)):.7f}")
    print(f"{name}: h sd {h.std():.5f} ({h.std() / sd:.3f} sd(y)), moves {(np.abs(h) > 1e-4).mean():.1%} of tracts, "
          f"perp {perp:.3f}; W {W:.4f} on {base} ({rb:.7f})")
    lb.write(name, b.GEOID, q, f"physics v12 {'fire+school+BrightQuery' if comb else 'fire'} change, unscored part on {base}, W {W:.4f}",
             kind="model")
    spec = {"base": "base", "components": {
        "base": {"kind": "final", "name": base.replace(".csv", "")},
        "fire": {"kind": "bet", "on": "base", "vehicle": "physics12", "theta": pfile.name, "theta_off": off,
                 "draws": draws, "span_upto": L.file.iloc[-1], "W": W}}, "combine": {"fire": 1.0}}
    (lb.ROOT / "params" / (name.replace(".csv", "") + "_final.json")).write_text(json.dumps(spec, indent=1))
    return col, h


def bet_refit(name, base, theta="physics9_theta_r31.json", chat=0.03, m=0.85, draws=2000):
    """A whole-vehicle refit as one bet (LEDGER 135): h = y(refit theta) - y(E127 theta), both with W_TFAC > 0 so
    the two runs share one random stream (a W_TFAC of exactly 0 skips Poisson draws and shifts it, LEDGER 132), its
    part outside the span of every scored file, on scored BASE at W = m chat."""
    import cand_mm as CMM
    import published as PB
    d, X0, _ = inputs()
    L = lb.ledger()
    th = json.loads((DERIVED / theta).read_text())
    th["W_TFAC"] = max(th.get("W_TFAC", 0.0), 1e-4)
    pfile = lb.ROOT / "params" / theta
    pfile.write_text(json.dumps(th, indent=1))
    off = {k: v for k, v in json.loads(E127.read_text()).items() if k in P9.KEYS}
    off["W_TFAC"] = max(off.get("W_TFAC", 0.0), 1e-4)
    h = model(d, X0, th, draws=draws) - model(d, X0, dict(th, **off), draws=draws)
    col, perp = CMM.project(h, d, L)
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    b = lb.read_sub(base)
    assert (b.GEOID.astype(str).values == d.GEOID.astype(str).values).all()
    W = m * chat
    q = np.clip(b.coverage_gap_score.values + sd * W * col, 0, 1)
    q[np.isin(b.GEOID.values, PB.PIN_GEOIDS)] = 1.0
    rb = float(L[L.file == base].rmse.iloc[0])
    rho2 = 1 - rb ** 2 / sd ** 2
    for c in (0.0, chat, 2 * chat, 4 * chat):
        print(f"   if c = {c:+.3f}: {sd * np.sqrt(1 - rho2 + W * (W - 2 * c)):.7f}")
    e134 = lb.read_sub("e134_learner.csv").coverage_gap_score.values - lb.read_sub("e132_robust.csv").coverage_gap_score.values
    print(f"{name}: h sd {h.std():.5f} ({h.std() / sd:.3f} sd(y)), moves {(np.abs(h) > 1e-4).mean():.1%} of tracts, "
          f"perp {perp:.3f}, corr with the learner step {np.corrcoef(col, e134)[0, 1]:+.3f}; W {W:.4f} on {base} ({rb:.7f})")
    lb.write(name, b.GEOID, q, f"physics v9b full refit on all scored files, its change's unscored part on {base}, "
                              f"W {W:.4f}", kind="model")
    spec = {"base": "base", "components": {
        "base": {"kind": "final", "name": base.replace(".csv", "")},
        "refit": {"kind": "bet", "on": "base", "vehicle": "physics12", "theta": pfile.name, "theta_off": off,
                  "draws": draws, "span_upto": L.file.iloc[-1], "W": W}}, "combine": {"refit": 1.0}}
    (lb.ROOT / "params" / (name.replace(".csv", "") + "_final.json")).write_text(json.dumps(spec, indent=1))
    return col, h


def check():
    d, X0, _ = P9.inputs()
    a = P9.model(d, X0, json.loads(E127.read_text()), draws=400)
    b = model(d, X0, start(), draws=400)
    print(f"v12(F = 1) vs v9b(E127 theta): max |diff| {np.abs(a - b).max():.2e}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["check"]:
        check()
    school, bq = "--school" in a, "--bq" in a
    a = [x for x in a if x not in ("--school", "--bq")]
    if a[:1] == ["fit"]:
        cut = a[1] if len(a) > 1 and a[1] != "all" else None
        fit(cut, int(a[2]) if len(a) > 2 else 160, keys=KEYS_S if school else KEYS_B if bq else KEYS,
            tag2="s" if school else "b" if bq else "")
    elif a[:1] == ["holdout"]:
        holdout(a[1], tag2="s" if school else "b" if bq else "")
    elif a[:1] == ["bet"]:                         # bet NAME BASE [CHAT] [--comb]
        comb = "--comb" in a
        a = [x for x in a if x != "--comb"]
        bet(a[1], a[2], float(a[3]) if len(a) > 3 else 0.04, comb=comb)
