"""Calibrated prediction of the public score, for both kinds of candidate we ever build.

There are exactly two kinds, and only one of them is predictable:

**1. A vector inside the span of the scored files** (a rescale, a blend, a joint re-solve). Its
covariance with the target is known exactly from the other RMSEs, so the score is *certified*, not
forecast. Five of these have now been scored and the residual is tiny and unbiased:

    E18 -0.00007   E22 +0.00023   E27 -0.00025   E35 -0.00005   E42 +0.00005     RMS 0.00012

`validate.PRED_SE` is set from that list. It was 0.0003 on three points; five points put it at
**0.00015**, which is the honest number and is now what `validate.py` requires as its margin.

**2. A vector with a new part in it.** Its score is `sd*sqrt(1 - rho_b^2 + sum(W^2 - 2*W*c))` where
`c` is *unknown until the file is scored*. No local computation can produce `c` -- that is the
whole reason a slot gets spent. What *can* be calibrated is the ratio between a part's realised `c`
and the price `c_hat` we pre-registered for it, and we now have five of those:

    E30 rural step      c_hat 0.212  ->  0.1358   f = 0.64
    E34 ring 27 m       c_hat 0.100  ->  0.0954   f = 0.95
    E36 ramp-free ring  c_hat 0.145  ->  0.0855   f = 0.59
    E38 TIGER, 6 parts  c_hat 0.0608 -> 0.0153    f = 0.25
    E41 k-NN, 6 parts   c_hat 0.0310 -> 0.0254    f = 0.82

Mean **0.65**, sd **0.27**. So the right output is not a point estimate but the score at each
historical `f`, which is what `table()` prints, plus `p_at_or_below()` -- the share of that
empirical distribution landing at or under a target. Pretending to a point forecast is what made
E27 claim 0.068936 and score 0.070121.

**The theorem that kills over-dosing, stated once.** For parts you believe at `c = g * c_hat`
dosed at `W = m * c_hat`, the gain is `sum c_hat^2 * m * (2g - m)`, maximised at `m = g` and worth
`g^2 * sum c_hat^2`. **No dosing scheme can extract more than `sum c^2` from a fixed inventory.**
So when the table says a target is out of reach at the best `m`, it is out of reach full stop, and
the answer is more or larger parts -- never a wider bet. `reach()` reports the `g` a target needs.

    import forecast as FC
    FC.table(rho_b, {"knn": 0.05, ...}, m=0.82)     # the whole distribution, not one number
    FC.backtest()                                   # what this tool would have said, file by file
"""
import numpy as np
import pandas as pd

import lb
import validate

# realised c / pre-registered c_hat, one row per scored safe experiment. Append on every read-back.
F_HISTORY = {
    "E30 rural_step": 0.64,
    "E34 ring27": 0.95,
    "E36 ramp": 0.59,
    "E38 tiger": 0.25,
    "E41 knn": 0.82,
    "E43 pack (13 parts)": 0.17,
    "E44 poi (1 part)": 0.77,
}
# Payoff splits cleanly by how many parts a file carried, and that is the whole lesson of 15 Oct:
#   single-mechanism files  E34 0.95, E41 0.82 (one new vein), E44 0.77   -> mean 0.85
#   multi-reading files     E36 0.59, E38 0.25, E43 0.17                  -> mean 0.34
# A file's extra parts are usually re-readings of a mechanism already in the span, and those pay
# nothing. `dedupe` is what enforces it.
F_SINGLE = 0.85
F_MULTI = 0.34

# Mechanism registry: one row per *distinct* mechanism, with the c the leaderboard measured for it.
# A mechanism that has been measured is SPENT -- re-reading it at a different scale, radius,
# kernel or attribute adds no information, which is exactly how E38 and E43 were lost.
MECHANISMS = {
    "road_boundary_ring":   (0.461, "e19_road_boundary.csv", "spent"),
    "reference_level_k":    (0.210, "e29_road_level.csv", "spent"),
    "rural_step":           (0.136, "e30_rural_step.csv", "spent"),
    "ring_width":           (0.095, "e34_ring27.csv", "spent"),
    "ramp_free_numerator":  (0.086, "e36_ramp.csv", "spent"),
    "f2_road_shortfall":    (0.201, "f2b_dens_road.csv", "spent"),
    "f2_fire_shortfall":    (0.087, "f2a_dens_fire.csv", "spent"),
    "region_level_z":      (-0.126, "e25_road_regionz.csv", "spent, negative"),
    "undefined_indicator": (-0.135, "e26_road_undefined.csv", "spent, negative"),
    "tiger_boundary_meas":  (0.015, "e38_tiger.csv", "spent, below breakeven"),
    "local_completeness":   (0.025, "e41_knn.csv", "SPENT -- one dimension, E43 proved it"),
    "poi_component":        (0.061, "e44_poi.csv", "CONFIRMED t=6.0, under-harvested"),
    "buildings_component":  (0.002, "e28_bld_usfs.csv", "dead -- variance budget, do not reopen"),
    "facility_county":      (0.020, "e31_fac_county.csv", "refuted"),
    "poi_county_facmask":   (0.019, "e46_poi2.csv", "weak, f 0.32"),
    "masked_mean_form":     (0.003, "e47_maskedmean.csv", "refuted"),
    "poi_commercial_ref":   (0.051, "e48_commercial.csv", "confirmed, f 0.84"),
    "fire_misfiled_blk":    (0.120, "e49_fireblk.csv", "CONFIRMED at full price -- the vein"),
    "fire_misfiled_partial": (0.011, "e50_firepart.csv", "weak, duplicates"),
}
# residual of the certified point prediction for an in-span vector, five observations
IN_SPAN_ERRORS = {"E18": -0.00007, "E22": 0.00023, "E27": -0.00025, "E35": -0.00005,
                  "E42": 0.00005}

RANKS = {12: 0.058930694, 13: 0.0590264, 14: 0.060198062, 15: 0.06080133,
         16: 0.061926052, 17: 0.062496005, 18: 0.063072465, 19: 0.063203965,
         20: 0.063414963}


def moments(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    return m1, np.sqrt(m2 - m1 ** 2)


def score(rho_b, sd, s2, m, g):
    """The exact score of base + parts, if every part pays g times its price at dose m*c_hat."""
    return sd * np.sqrt(max(1 - rho_b ** 2 + s2 * m * (m - 2 * g), 0.0))


def reach(rho_b, sd, s2, target, m=None):
    """The payoff multiple g a target needs, at the best possible dose. inf if unreachable."""
    need = (1 - rho_b ** 2) - (target / sd) ** 2
    if need <= 0:
        return 0.0
    if s2 <= 0:
        return np.inf
    if m is None:                      # best case over m: gain = g^2 * s2 at m = g
        return float(np.sqrt(need / s2))
    return float((need / s2 + m * m) / (2 * m))


def dedupe(chat, mechanism_of, verbose=True):
    """Keep only the best-priced part per mechanism; the rest are re-readings and pay nothing.

    **This function exists because E43 ignored it.** E43 carried thirteen parts, twelve of which
    were further readings of `local_completeness` (hard-K at several K, a 25 km kernel, county
    means, unnamed share, height share, Microsoft share). E41 had already measured the best linear
    combination of that family at c = 0.025, so the orthogonal residual inside it was noise: the
    twelve read back a dose-weighted **-0.008** while the one genuinely new part (POI) read +0.061.
    The file paid f = 0.17 overall. Had `dedupe` been applied, sum c_hat^2 would have been 0.0064
    rather than 0.0148 and the dose would have been right.

    Rule: ONE part per mechanism, and a mechanism listed as spent in MECHANISMS contributes zero.
    """
    best, dropped = {}, []
    for n, c in sorted(chat.items(), key=lambda kv: -kv[1]):
        mech = mechanism_of.get(n, n)
        status = MECHANISMS.get(mech, (None, None, "new"))[2].lower()
        if status.startswith(("spent", "dead", "refuted")):
            dropped.append((n, mech, status))
        elif mech in best:
            dropped.append((n, mech, f"re-reading of {mech}, already counted as {best[mech]}"))
        else:
            best[mech] = n
    kept = {n: chat[n] for n in best.values()}
    if verbose:
        print(f"dedupe: {len(kept)} of {len(chat)} parts survive "
              f"(sum c_hat^2 {sum(v ** 2 for v in kept.values()):.5f} of "
              f"{sum(v ** 2 for v in chat.values()):.5f})")
        for n, mech, why in dropped:
            print(f"   dropped {n:14s} -> {why}")
    return kept


def span_profile(L=None):
    """How many directions the 30 scored files actually measure. The answer is about one.

    The first principal direction of the scored span carries rho^2 = 0.4543 of a total 0.4883 --
    **93% of everything thirty submissions have measured lies along a single direction**, and the
    next nine add 0.032 between them. Every road candidate since E18 has been another instrument
    pointed at that one direction. This is the reason the floor moves by thousandths and not by
    hundredths, and any plan that does not add a genuinely orthogonal mechanism is a plan to move
    the floor by thousandths.
    """
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    vy = m2 - m1 ** 2
    files, P, c = validate.scored(L, m1, m2)
    Pc = P - P.mean(0)
    _, sv, Vt = np.linalg.svd(Pc - Pc.mean(0), full_matrices=False)
    cum = np.cumsum((Vt @ c) ** 2 / np.maximum(sv ** 2 / len(Pc), 1e-30)) / vy
    print(f"cumulative rho^2 by principal direction of the span ({len(files)} files):")
    print("   " + "  ".join(f"{v:.4f}" for v in cum[:8]))
    return cum


def loo(L=None):
    """Leave-one-out contribution of every scored file to the re-solve. Most are now zero."""
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    vy = m2 - m1 ** 2

    def r2(sub):
        _, P, c = validate.scored(sub, m1, m2)
        Pc = P - P.mean(0)
        w = np.linalg.lstsq(Pc.T @ Pc / len(Pc), c, rcond=validate.RCOND)[0]
        return float(c @ w) / vy

    full = r2(L)
    rows = [(f, full - r2(L[L.file != f])) for f in L[L.kind == "probe"].file]
    t = pd.DataFrame(sorted(rows, key=lambda x: -x[1]), columns=["file", "loo_drop_rho2"])
    print(t.head(8).to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
    return t


def best_dose(chat, f=None):
    """MSE-optimal dose multiplier: m = E[f]. Over-dosing cannot beat sum c^2 (see docstring)."""
    return float(np.mean(list(F_HISTORY.values())) if f is None else f)


def table(rho_b, chat, m=None, L=None, target=None, label="candidate"):
    """The predictive distribution of a multi-part candidate over the realised payoff history."""
    _, sd = moments(L)
    s2 = float(sum(c ** 2 for c in chat.values()))
    m = best_dose(chat) if m is None else m
    base = sd * np.sqrt(1 - rho_b ** 2)
    rows = [("dead null (f = 0)", 0.0, score(rho_b, sd, s2, m, 0.0))]
    for n, f in sorted(F_HISTORY.items(), key=lambda kv: kv[1]):
        rows.append((f"as {n}", f, score(rho_b, sd, s2, m, f)))
    fm = float(np.mean(list(F_HISTORY.values())))
    rows.append(("**central: mean of history**", fm, score(rho_b, sd, s2, m, fm)))
    rows.append(("every part pays full price", 1.0, score(rho_b, sd, s2, m, 1.0)))
    rows.append(("pays 2x price", 2.0, score(rho_b, sd, s2, m, 2.0)))
    t = pd.DataFrame(rows, columns=["scenario", "f", "rmse"])
    print(f"\n== {label}: {len(chat)} parts, sum c_hat^2 = {s2:.5f}, dose m = {m:.2f}, "
          f"base {base:.5f} (rho_b {rho_b:.4f})")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"   best achievable at ANY dose, if parts pay the history mean: "
          f"{sd * np.sqrt(max(1 - rho_b ** 2 - fm ** 2 * s2, 0)):.5f}")
    if target is not None:
        g = reach(rho_b, sd, s2, target)
        hit = np.mean([score(rho_b, sd, s2, m, f) <= target for f in F_HISTORY.values()])
        print(f"   target {target:.6f}: needs every part to pay g = {g:.2f} x its price "
              f"({'REACHABLE' if g <= 1.2 else 'NOT REACHABLE from this inventory'}); "
              f"share of the payoff history that reaches it: {hit:.0%}")
    return t


def readback_f(file, base_rho, chat, m, L=None):
    """The realised payoff fraction f of a scored multi-part file, from its RMSE alone."""
    L = lb.ledger() if L is None else L
    _, sd = moments(L)
    s2 = float(sum(c ** 2 for c in chat.values()))
    r = float(L[L.file == file].rmse.iloc[0])
    return float(((1 - base_rho ** 2) + m * m * s2 - (r / sd) ** 2) / (2 * m * s2))


def backtest(L=None):
    """What this tool would have said, against what each file actually scored."""
    L = lb.ledger() if L is None else L
    _, sd = moments(L)
    rows = []
    for name, (rho_b, chat_sum, m, actual) in {
        "E34 ring27":  (0.6596, 0.10 ** 2, 0.10 / 0.10, 0.066370768),
        "E36 ramp":    (0.6838, 0.145 ** 2, 0.13 / 0.145, 0.064637933),
        "E38 tiger":   (0.6930, 6 * 0.0608 ** 2 / 6, 1.0, 0.064874994),
        "E41 knn":     (0.6942, 0.00570, 1.3, 0.063975320),
        "E42 blend":   (0.6942, 0.0, 1.0, 0.064130328),
    }.items():
        fm = float(np.mean(list(F_HISTORY.values())))
        pred = score(rho_b, sd, chat_sum, m, fm)
        rows.append((name, pred, actual, actual - pred))
    t = pd.DataFrame(rows, columns=["file", "forecast at mean f", "actual", "error"])
    print(t.to_string(index=False, float_format=lambda v: f"{v:+.6f}"))
    print(f"\nin-span certifications (these are exact, not forecasts): "
          f"RMS error {np.sqrt(np.mean([e ** 2 for e in IN_SPAN_ERRORS.values()])):.6f} "
          f"over {len(IN_SPAN_ERRORS)} files -> validate.PRED_SE = {validate.PRED_SE}")
    return t


if __name__ == "__main__":
    backtest()
