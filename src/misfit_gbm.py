"""The misfit screen as a learner: what the simulator still gets wrong, predicted from every package feature (31 Oct).

`screen.py` ranks single signals against the in-span misfit M = P_S(y) - P_S(vehicle): the part of the vehicle's error
that the scored files already reveal (P_S(y) is the teacher, the ridge affine over the scored probes; the vehicle is
physics v9b at E127's theta). A strong signal is a missing mechanism, which so far had to be written by hand and
refitted (E90, E127). Here a gradient-boosted model learns M from all per-tract features at once, out of fold, and
the bet is the part of its prediction OUTSIDE the scored span: the learner's extrapolation of the misfit's pattern
(tribal land, listed facilities and their interactions) to the directions no file has measured. No score enters the
features; the teacher enters only as the target, as it does for the finals' pieces.

    python src/misfit_gbm.py fit                         # OOF fit, prints OOF R2 and the top features (~2 min)
    python src/misfit_gbm.py bet NAME BASE [CHAT]        # writes submissions/NAME on scored BASE at W = 0.85 CHAT
"""
import json
import sys

import lightgbm
import numpy as np
import pandas as pd

import cand_mm as CMM
import clean_geo as CG
import final_build as FB
import final_model as FM
import lb
import published as PB
import screen as SC

SEED = 20261031
RIDGE_T = 3e-5
NULL_BETS = {"e133_svi.csv"}                  # read at c ~ 0 (LEDGER 131): kept out of the teacher like REFUTED_BETS
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED, deterministic=True,
              force_col_wise=True, num_threads=8)
ROUNDS, FOLDS = 500, 5


def ledger(upto=None):
    """The ledger as it stood when `upto` was the last scored file (all of it by default)."""
    L = lb.ledger()
    return L.loc[:L.index[L.file == upto][0]] if upto else L


def misfit(upto=None):
    """Teacher, vehicle and their in-span difference, in the sample submission's tract order."""
    L = ledger(upto)
    files = [f for f in CG.teacher_files(L) if f not in CG.REFUTED_BETS | NULL_BETS]
    T, ident, _, _ = FM.affine(RIDGE_T, files, L)
    g, _ = FB._pieces(True)
    v = FB._extra(CG.V9B_FULL, g)
    Q = np.column_stack([lb.read_sub(f).coverage_gap_score.values for f in files])
    Qc = Q - Q.mean(0)
    beta = np.linalg.solve(Qc.T @ Qc + 1e-6 * len(Qc) * np.eye(Q.shape[1]), Qc.T @ (v - v.mean()))
    return T - (v.mean() + Qc @ beta), files, ident


def features(geoids, state=False):
    """Every per-tract package feature; with state=True also the vehicle's own expected components (v2, E135)."""
    d = SC.table().set_index("GEOID").loc[geoids].reset_index()
    X = d.drop(columns=["GEOID"]).copy()
    X["region"] = X.region.astype("category").cat.codes
    keep = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) or X[c].dtype == bool]
    X = X[keep].astype(float)
    if state:
        import physics9 as P9
        dv, X0, _ = P9.inputs()
        assert (dv.GEOID.astype(str).values == np.asarray(geoids)).all()
        o = P9.model(dv, X0, json.loads((lb.ROOT / "params" / "physics9_e127_theta.json").read_text()),
                     draws=400, parts=True)
        for k in ("y", "T", "P", "fblk", "fmis", "gf", "gs"):
            X["veh_" + k] = o[k]
    return d, X


def fit():
    m, files, ident = misfit()
    geo = lb.read_sub(files[0]).GEOID.astype(str).values
    d, X = features(geo)
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, FOLDS, len(m))
    oof, imp = np.zeros(len(m)), pd.Series(0.0, index=X.columns)
    for k in range(FOLDS):
        tr = fold != k
        mdl = lightgbm.train(PARAMS, lightgbm.Dataset(X[tr], m[tr]), num_boost_round=ROUNDS)
        oof[~tr] = mdl.predict(X[~tr])
        imp += pd.Series(mdl.feature_importance("gain"), index=X.columns)
    r2 = 1 - ((m - oof) ** 2).sum() / ((m - m.mean()) ** 2).sum()
    print(f"teacher identity {ident:.7f} over {len(files)} files; misfit sd {m.std():.5f}; "
          f"OOF R2 {r2:.3f}, corr {np.corrcoef(m, oof)[0, 1]:.3f}")
    print((imp / imp.sum()).sort_values(ascending=False).head(15).round(3).to_string())
    return d, geo, oof, r2


def bet(name, base, chat=0.04, mult=0.85):
    d, geo, g, r2 = fit()
    L = lb.ledger()
    col, perp = CMM.project(g, d, L)
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    b = lb.read_sub(base)
    assert (b.GEOID.astype(str).values == geo).all()
    W = mult * chat
    q = np.clip(b.coverage_gap_score.values + sd * W * col, 0, 1)
    q[np.isin(b.GEOID.values, PB.PIN_GEOIDS)] = 1.0
    rb = float(L[L.file == base].rmse.iloc[0])
    rho2 = 1 - rb ** 2 / sd ** 2
    for c in (0.0, chat, 2 * chat, 4 * chat):
        print(f"   if c = {c:+.3f}: {sd * np.sqrt(1 - rho2 + W * (W - 2 * c)):.7f}")
    print(f"{name}: learned misfit sd {g.std():.5f}, perp {perp:.3f}, OOF R2 {r2:.3f}; W {W:.4f} on {base} ({rb:.7f})")
    lb.write(name, b.GEOID, q, f"misfit learner (LightGBM, out of fold) unscored part on {base}, W {W:.4f}",
             kind="model")
    (lb.ROOT / "params" / (name.replace(".csv", "") + "_final.json")).write_text(json.dumps(
        {"base": base.replace(".csv", ""), "method": "misfit_gbm.bet", "chat": chat, "mult": mult, "W": W,
         "seed": SEED, "ridge_teacher": RIDGE_T, "params": PARAMS, "rounds": ROUNDS, "folds": FOLDS}, indent=1))
    return col, g


def learn(m, X, bags=1):
    """Out-of-fold LightGBM prediction of m from X, averaged over `bags` fold splits and seeds."""
    out = np.zeros(len(m))
    for b in range(bags):
        fold = np.random.default_rng(SEED + b).integers(0, FOLDS, len(m))
        oof = np.zeros(len(m))
        for k in range(FOLDS):
            tr = fold != k
            oof[~tr] = lightgbm.train(dict(PARAMS, seed=SEED + b), lightgbm.Dataset(X[tr], m[tr]),
                                      num_boost_round=ROUNDS).predict(X[~tr])
        out += oof / bags
    return out


def column(upto, bags=1, state=False):
    """The learner's bet column exactly as `bet` (bags=1, state=False) or `bet2` built it with the ledger ending at
    `upto`: unit-sd part outside that span, sd(y) from that ledger, tract order. final_build's "learner" kind."""
    m, files, _ = misfit(upto)
    geo = lb.read_sub(files[0]).GEOID.astype(str).values
    d, X = features(geo, state=state)
    g = learn(m, X, bags=bags)
    L = ledger(upto)
    col, _ = CMM.project(g, d, L)
    m1, m2 = lb.moments(L)
    return col, float(np.sqrt(m2 - m1 ** 2)), geo


def bet2(name, base, chat=0.03, mult=0.85, bags=5):
    """v2 (E135): the teacher now holds every scored probe since E46 (E134 included), the learner sees the vehicle's
    own expected components as features and is bagged over `bags` fold splits; the bet is its part outside the span
    of ALL scored files (E134 included), so it can only pay on what E134 did not already carry."""
    m, files, ident = misfit()
    geo = lb.read_sub(files[0]).GEOID.astype(str).values
    d, X = features(geo, state=True)
    g = learn(m, X, bags=bags)
    r2 = 1 - ((m - g) ** 2).sum() / ((m - m.mean()) ** 2).sum()
    print(f"teacher identity {ident:.7f} over {len(files)} files (E134 in: {'e134_learner.csv' in files}); "
          f"misfit sd {m.std():.5f}; bagged OOF R2 {r2:.3f}")
    L = lb.ledger()
    col, perp = CMM.project(g, d, L)
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    b = lb.read_sub(base)
    assert (b.GEOID.astype(str).values == geo).all()
    W = mult * chat
    q = np.clip(b.coverage_gap_score.values + sd * W * col, 0, 1)
    q[np.isin(b.GEOID.values, PB.PIN_GEOIDS)] = 1.0
    rb = float(L[L.file == base].rmse.iloc[0])
    rho2 = 1 - rb ** 2 / sd ** 2
    for c in (0.0, chat, 2 * chat, 4 * chat):
        print(f"   if c = {c:+.3f}: {sd * np.sqrt(1 - rho2 + W * (W - 2 * c)):.7f}")
    e134 = lb.read_sub("e134_learner.csv").coverage_gap_score.values - lb.read_sub("e132_robust.csv").coverage_gap_score.values
    print(f"{name}: learned misfit sd {g.std():.5f}, perp {perp:.3f}; corr of its unscored part with E134's step "
          f"{np.corrcoef(col, e134)[0, 1]:+.3f}; W {W:.4f} on {base} ({rb:.7f})")
    lb.write(name, b.GEOID, q, f"misfit learner v2 (bagged, vehicle state) unscored part on {base}, W {W:.4f}",
             kind="model")
    (lb.ROOT / "params" / (name.replace(".csv", "") + "_final.json")).write_text(json.dumps(
        {"base": base.replace(".csv", ""), "method": "misfit_gbm.bet2", "chat": chat, "mult": mult, "W": W,
         "bags": bags, "seed": SEED, "ridge_teacher": RIDGE_T, "params": PARAMS, "rounds": ROUNDS, "folds": FOLDS},
        indent=1))
    return col, g


def holdout(cut="e110_ringspec.csv", draws=2000):
    """physics10.holdout's test for the learner: everything (vehicle, misfit, learner) from the span up to `cut`;
    the out-of-early-span part of the prediction is read against the truth the LATER files reveal."""
    import physics9 as P9
    early, full = P9.Span(cut), P9.Span()
    v = P9.model(*P9.inputs()[:2], json.loads((lb.ROOT.parent / "data" / "derived" / "physics9_theta_e110b.json")
                                              .read_text()), draws=draws)
    geo = lb.read_sub(early.files[0]).GEOID.astype(str).values
    d, X = features(geo)
    me = early.misfit(v)
    g = learn(me, X)
    r2 = 1 - ((me - g) ** 2).sum() / ((me - me.mean()) ** 2).sum()

    def read(y):
        h = (y - y.mean()) - early.proj(y)
        hs = full.proj(h)
        return float(hs @ full.psy / len(h)) / (full.sd * h.std()), h.std() / full.sd, \
            np.linalg.norm(hs) / np.linalg.norm(h)

    print(f"early span {cut}: misfit sd {me.std():.5f}, learner OOF R2 {r2:.3f}")
    for name, y in [("vehicle (V9b, early fit)", v), ("vehicle + learner", v + g), ("learner alone", g)]:
        c, size, seen = read(y)
        print(f"{name:26s} out-of-early-span size {size:.3f} sd(y), share seen by later files {seen:.3f}, "
              f"corr with y via later files {c:+.4f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["fit"]:
        fit()
    elif a[:1] == ["holdout"]:
        holdout(*a[1:2])
    elif a[:1] == ["bet2"]:                        # bet2 NAME BASE [CHAT]
        bet2(a[1], a[2], float(a[3]) if len(a) > 3 else 0.03)
    elif a[:1] == ["bet"]:
        bet(a[1], a[2], float(a[3]) if len(a) > 3 else 0.04)
