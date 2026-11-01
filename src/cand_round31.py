"""E121: the E120 split's posterior-best direction (K1, `docs/oct29_primer.md` 6.2), on the certified 21-file affine.

**The base is certified, not modelled.** Ridge-3e-6 affine over E120's 20 base files + E120 (21 scored files):
identity 0.0555851, validator 0.0556267, lambda_e120 0.63, |lambda|_1 2.71. The two scorer-pinned water tracts are
set to 1.0 afterwards (111a).
**The part.** E120 sent V8 - V4 = A + B + C + D as one direction and read c +0.036 at 0.34 of its size (LEDGER 120c).
Priced by provenance, B (the urban hidden-station rate in proportion to area, exponent chosen in span) is the likely
null (P 0.74-0.82), and C was never measured (its out-of-span part ran against B's and D's: energy share -1%):
  A  railway / beyond-cut boundary samples out of the TIGER boundary length       (geometry, r ~ 0.48)
  B  urban hidden stations in proportion to area                                  (fitted in span, r ~ 0)
  C  a participant's 270 counted as misfiled PLACES (p_mis 0.255 -> 0.642)              (published count, r ~ 0.48)
  D  the school multiplier on the published fire/school 8.7 (M_S 0.12 -> 0.156)   (published ratio, r ~ 0.48)
With E120 in the span, the posterior-best mix is h = +0.10 A - 0.33 B + 1.00 C + 0.97 D (identity reading; the
validator reading gives +0.01/-0.32/+1.00/+0.91): size 0.066, E[c] +0.031, posterior sd 0.008, predictive sd 0.019.
It measures C and D and backs out a third of B. The sub-parts are rebuilt here exactly as `diag_lineage.split` builds
them (vehicles in order A, B, C, D), with no file lineage: this module reads only scored, non-excluded files.
**The dose** maximises P(pass rival_C 0.055459211), which is the brief for this slot: W = sqrt((R_b/sd)^2 -
(R_mz/sd)^2) on the identity is the dose at which the c needed to pass is smallest (c >= W). The expected-score
dose W = c costs < 1e-5 less in expectation and gives a lower P(pass rival_C) (`evidence` prints the table).

    python src/cand_round31.py            # evidence: base, sub-part sizes/perp, mix, traps, dose table (~90 s)
    python src/cand_round31.py build      # e121_split.csv
"""
import contextlib
import io
import sys

import numpy as np
from scipy.stats import norm

import cand_knn as CK
import cand_mm as CMM
import cand_round25 as R
import cand_round30 as R30
import final_model as FM
import lb
import physics3 as P3
import validate

BASE_FILES = R30.BASE_FILES + ["e120_vehicle.csv"]
RIDGE = 3e-6
NAME = "e121_split.csv"
COMMENT = ("E121 E120 split: a participant's 270 as misfiled places and the school multiplier on the published 8.7 measured, "
           "the in-span area exponent of the urban hidden stations backed out by a third")
MIX = {"A": 0.10, "B": -0.33, "C": 1.00, "D": 0.97}
E_C, S_PRED = 0.0307, 0.019          # posterior mean on the identity (120c), predictive sd incl. read-back noise
RIVALS = {"rival_A": 0.054404837, "rival_B": 0.055402428, "rival_C": 0.055459211, "rival_D": 0.055607098,
          "E120 (ours)": 0.055653812}  # public leaderboard, 29 Oct
TARGET = "rival_C"
W_BUILD = 0.0420                     # the pre-registered dose (LEDGER 121): sqrt((R_b/sd)^2 - (R_rival_C/sd)^2) on the identity
LAST = "e120_vehicle.csv"            # the span at build time: every scored row up to E120, so a rebuild is exact later


def span(L=None):
    L = lb.ledger() if L is None else L
    return L.iloc[:int(np.where(L.file.values == LAST)[0][0]) + 1].reset_index(drop=True)


def base(L):
    q, pred, err, lam = FM.affine(RIDGE, BASE_FILES, L)
    a = validate.analyse(q, L)
    g = lb.read_sub(BASE_FILES[0]).GEOID.values
    for geoid, v in R.PINNED.items():
        q[int(np.where(g == geoid)[0][0])] = v
    return g, q, pred, a["rmse_neutral"], lam


def parts():
    """The four E120 sub-parts as vehicle differences, applied in order A, B, C, D (as `diag_lineage.split`)."""
    with contextlib.redirect_stdout(io.StringIO()):
        d, X4, o4, k4, X8, o8, k8, par = R30.vehicles()
        XA = dict(X4, den=X8["den"])
        cA = P3.calibrate(d, XA)
        oA = P3.simulate(XA, *cA[:4], m_s=cA[4])
        XAB = dict(XA, bg=X8["bg"])
        cB = P3.calibrate(d, XAB)
        oAB = P3.simulate(XAB, *cB[:4], m_s=cB[4])
        cC = R30.calibrate(d, X8, P3.M_S)
        oABC = R30.simulate(X8, *cC[:4], m_s=cC[4])
    H = {"A": oA["y"] - o4["y"], "B": oAB["y"] - oA["y"], "C": oABC["y"] - oAB["y"], "D": o8["y"] - oABC["y"]}
    return d, X4, H


def dose(rb, sd):
    rt = RIVALS[TARGET]
    return float(np.sqrt((rb / sd) ** 2 - (rt / sd) ** 2))


def table(rb, sd, Ws):
    print(f"   dose table on the identity {rb:.7f}; c ~ N({E_C}, {S_PRED}^2); P(pass) per rival")
    print("   " + f"{'W':>6s} {'null':>10s} {'at E[c]':>10s} " + " ".join(f"{k:>14s}" for k in RIVALS))
    for W in Ws:
        null = sd * np.sqrt((rb / sd) ** 2 + W ** 2)
        at = sd * np.sqrt((rb / sd) ** 2 + W * (W - 2 * E_C))
        cells = []
        for k, rt in RIVALS.items():
            creq = (W ** 2 + (rb / sd) ** 2 - (rt / sd) ** 2) / (2 * W)
            cells.append(f"c>={creq:+.4f} {norm.sf((creq - E_C) / S_PRED):.2f}")
        print(f"   {W:6.4f} {null:10.7f} {at:10.7f} " + " ".join(f"{c:>14s}" for c in cells))


def evidence(L=None):
    L = span(L)
    m1, m2 = lb.moments(L)
    sd = float(np.sqrt(m2 - m1 ** 2))
    g, q, pred, val, lam = base(L)
    print(f"base: {len(BASE_FILES)}-file affine ridge {RIDGE:g}, identity {pred:.7f}, validator {val:.7f}, "
          f"|lambda|_1 {np.abs(list(lam.values())).sum():.2f}, lambda_e120 {lam.get('e120_vehicle.csv', np.nan):+.3f}")
    d, X4, H = parts()
    assert (g == d.GEOID.values).all()
    P = {}
    for k, h in H.items():
        col, perp = CMM.project(h, d, L)
        P[k] = col * h.std() * perp
        print(f"   {k}: sd(h) {h.std():.5f}, perp {perp:.3f}, out-of-span size {h.std() * perp / sd:.4f}")
    h = sum(w * H[k] for k, w in MIX.items())
    col, perp = CMM.project(h, d, L)
    e = col ** 2 / (col ** 2).sum()
    print(f"K1 mix {MIX}: sd(h) {h.std():.5f}, perp {perp:.3f}, out-of-span size {h.std() * perp / sd:.4f}, "
          f"reach {(np.abs(h) > 1e-4).mean():.3f}, effective tracts {1 / (e ** 2).sum():.0f}")
    print("   corr of the K1 column with each projected sub-part: "
          + ", ".join(f"{k} {np.corrcoef(col, v)[0, 1]:+.3f}" for k, v in P.items()))
    print(CK.traps({"K1": col}, d, X4["den"] > 0).round(3).to_string())
    W = dose(pred, sd)
    print(f"W = sqrt((R_b/sd)^2 - (R_{TARGET}/sd)^2) = {W:.4f} (P(pass {TARGET}) is largest here)")
    table(pred, sd, [0.0255, 0.0307, 0.035, W, 0.045, 0.050])
    return d, g, q, col, W, sd


def build(L=None):
    L = span(L)
    d, g, q0, col, W0, sd = evidence(L)
    W = W_BUILD
    q = np.clip(q0 + sd * W * col, 0, 1)
    for geoid, v in R.PINNED.items():
        q[int(np.where(g == geoid)[0][0])] = v
    a = validate.analyse(q, L)
    print(f"{NAME}: W {W:.4f} (root on today's identity {W0:.4f}), neutral {a['rmse_neutral']:.7f}, clipped at 0: {(q == 0).sum()}, "
          f"pins {[q[int(np.where(g == x)[0][0])] for x in R.PINNED]}")
    lb.write(NAME, lb.read_sub(BASE_FILES[0]).GEOID, q, COMMENT, kind="model")


if __name__ == "__main__":
    a = sys.argv[1:]
    build() if a == ["build"] else evidence()
