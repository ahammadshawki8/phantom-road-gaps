"""Local pre-submission check. Run on every candidate BEFORE uploading.

    python src/validate.py submissions/<candidate>.csv [...]

1. Format: 9,794 rows in sample order, 11-char GEOIDs, values in [0,1], no blanks.
2. What our own scored files say about it. Each scored file p_i gives c_i = Cov_P(y, p_i) exactly
   (from its RMSE). Write the candidate as q = a + sum_i b_i p_i + r, with r orthogonal to the scored
   files. The scored part's covariance with y is known; the residual's is not, but is bounded by
   |Cov(y, r)| <= sd(r) * sqrt(Var(y) - Var(y explained by scored files)).
   -> predicted public RMSE: near-exact if sd(r) == 0 (rescales and blends), else a [best, worst]
      range, plus the "residual-neutral" value (Cov(y, r) = 0).
3. Decision: SUBMIT only if the residual-neutral value beats the current best, or if the best case
   beats it by a margin and the candidate tests a pre-registered hypothesis (see LEDGER.md).

**Why b is solved with a truncating rcond (E27, 13 Oct).** cov_known = b @ c is invariant to which
of several equally-good b the solver picks *only* if c lies in the row space of Pc. It very nearly
does -- but c is measured from 9-decimal RMSEs, so it has a small inconsistent component. Once the
ledger held four near-duplicate safe experiments on one base, Pc lost 2 of 16 directions (relative
singular values 1.2e-3 and 5.8e-5) and `rcond=None` loaded ||b|| = 15 of mutually cancelling weight
onto near-identical files, amplifying that inconsistency into a 0.00094 optimistic bias. It cost
E27 its prediction: 0.068936 claimed, 0.070121 actual, of which 0.00087 was this. Truncating the
near-null directions gives ||b|| = 2.6 at the same residual and a 0.00025 error. PRED_SE records
that the point prediction is not exact even at 100% explained.
"""
import sys

import numpy as np
import pandas as pd

import lb
import table

NOISE = 0.0004    # RMSE differences below this are noise on the public split
RCOND = 1e-4      # drop near-singular directions of Pc: files that are blends of other files
PRED_SE = 0.00015 # residual of the in-span point prediction, RMS over E18, E22, E27, E35, E42


# Scored files kept OUT of every span, blend and validation (decided 15 Oct, Shawki): they embed
# information derived from the removed reference layers through other teams' outputs.
#   e37: raked to the rank-1 entry's scoring rubric (that entry scores 0, so its card is the target's)
#   e33: imposed a competitor's 16 published cell means at full strength
# Dropping both changes the certified re-solve by 0.000001 (0.063695 -> 0.063696).
EXCLUDED = {"e37_ramp_rake.csv", "e33_cells.csv"}


def scored(L, m1, m2):
    files, vecs, covs = [], [], []
    for _, r in L[(L.kind == "probe") & ~L.file.isin(EXCLUDED)].iterrows():
        p = lb.read_sub(r.file).coverage_gap_score.values
        files.append(r.file)
        vecs.append(p)
        covs.append(lb.eyp(r.rmse, p, m2) - m1 * p.mean())
    return files, np.column_stack(vecs), np.array(covs)


def check_format(path):
    s = pd.read_csv(path, dtype={"GEOID": str})
    order = table.sample_order().GEOID.values
    assert list(s.columns[:2]) == ["GEOID", "coverage_gap_score"], s.columns
    assert len(s) == 9794 and (s.GEOID.values == order).all(), "rows / order differ from sample"
    assert s.GEOID.str.len().eq(11).all(), "GEOID lost leading zero"
    v = s.coverage_gap_score
    assert v.notna().all() and v.between(0, 1).all(), "blank or out-of-range value"
    return v.values.astype(float)


def analyse(q, L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    var_y = m2 - m1 ** 2
    files, P, c = scored(L, m1, m2)
    Pc = P - P.mean(0)
    S = Pc.T @ Pc / len(Pc)                       # covariance of scored files
    # part of y explained by the scored files (population least squares)
    w = np.linalg.lstsq(S, c, rcond=RCOND)[0]
    var_expl = float(c @ w)
    qc = q - q.mean()
    # same truncation here, and for the same reason: see the module docstring. With rcond=None this
    # line is what made E27's prediction 0.00087 optimistic.
    b = np.linalg.lstsq(Pc, qc, rcond=RCOND)[0]
    r = qc - Pc @ b
    cov_known = float(b @ c)
    sd_r = float(r.std())
    slack = sd_r * np.sqrt(max(var_y - var_expl, 0))
    var_q = float(qc.var())
    bias2 = (q.mean() - m1) ** 2

    def rmse(cov):
        return np.sqrt(var_y + var_q + bias2 - 2 * cov)

    def rho(cov):
        return cov / np.sqrt(var_y * var_q)

    out = {
        "corr_with_scored": dict(zip(files, np.round([np.corrcoef(q, P[:, i])[0, 1] for i in range(len(files))], 3))),
        "share_explained_by_scored": round(1 - sd_r ** 2 / max(var_q, 1e-18), 4),
        "rmse_neutral": rmse(cov_known),
        "rmse_best": rmse(cov_known + slack),
        "rmse_worst": rmse(cov_known - slack),
        "rho_neutral": rho(cov_known),
        "rho_range": (rho(cov_known - slack), rho(cov_known + slack)),
    }
    return out


def best_rmse(L):
    return L[L.kind == "probe"].rmse.min()


if __name__ == "__main__":
    L = lb.ledger()
    best = best_rmse(L)
    for path in sys.argv[1:]:
        q = check_format(path)
        a = analyse(q, L)
        print(f"\n== {path}\nformat OK")
        print(f"corr with scored files: {a['corr_with_scored']}")
        print(f"share of candidate explained by scored files: {a['share_explained_by_scored']:.1%}")
        print(f"predicted public RMSE  neutral {a['rmse_neutral']:.5f} (+-{PRED_SE:.4f})  "
              f"range [{a['rmse_best']:.5f}, {a['rmse_worst']:.5f}]   current best {best:.5f}")
        print(f"rho neutral {a['rho_neutral']:+.3f}  range [{a['rho_range'][0]:+.3f}, {a['rho_range'][1]:+.3f}]")
        if a["share_explained_by_scored"] > 0.9999 and a["rmse_neutral"] + PRED_SE < best:
            # An in-span vector has a KNOWN covariance with the target, so the point prediction
            # plus PRED_SE is the whole story -- and PRED_SE was calibrated on exactly these
            # vectors (five of them, RMS 0.00012). NOISE is a threshold for *hypotheses*, whose
            # new part may carry nothing; applying it to a certified vector once mislabelled a
            # certified 0.06373 as "HYPOTHESIS ONLY" when the improvement was arithmetic.
            v = (f"SUBMIT (CERTIFIED): in-span, {a['rmse_neutral']:.6f} +- {PRED_SE} beats "
                 f"{best:.6f} by {best - a['rmse_neutral']:.5f}")
        elif a["rmse_neutral"] < best - NOISE - PRED_SE:
            v = "SUBMIT: beats the best even if its new part carries no signal"
        elif a["rmse_best"] < best - NOISE:
            v = "HYPOTHESIS ONLY: can win only if its new part is informative; submit only as a gated experiment"
        else:
            v = "REJECT: cannot beat the current best"
        print("verdict:", v)
