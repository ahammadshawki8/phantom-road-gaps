"""Structural stack: a certified base plus measured parts that were built ORTHOGONAL to it.

**Why the plain re-solve is not enough any more (found 17 Oct).** `cand_blend.blend` solves
w = S^-1 c over every scored file with RCOND = 1e-4. That truncation was set on 13 Oct to stop
near-duplicate files amplifying measurement noise (E27). But a well-designed part enters the ledger
as `base + small dose * part`, so its own direction carries a singular value of order (dose)^2:
after E51-E53 those sit at 4-7e-5, just under the cut, and the re-solve threw away exactly the new
information (0.062177 against E52 alone at 0.062150). Cutting at 1e-5 recovers them but re-breaks
the old near-duplicates (backtest: E27 off by 0.0012). No single global threshold serves both.

The fix is to use what we know by construction. Each part file is `clip(p0 + a_k * col_k)` where p0
is the certified re-solve it was built on and `col_k` is a unit vector projected out of p0's whole
span. So Cov(y, col_k) follows from that file's RMSE by algebra, exactly as for any scored vector:

    Cov(y, col_k) = (Cov(y, file_k) - Cov(y, p0)) / a_k
    stack = p0 + sum_k b_k col_k,   b = Sigma_cc^-1 Cov(y, col)   (cols are orthogonal to p0)
    MSE(stack) = MSE(p0) - b' Cov(y, col)

Every term is a measured covariance of a vector we scored, so this is certified to the same
`PRED_SE` as an in-span blend; `validate.analyse` still runs and is conservative (it truncates the
same directions and so treats the parts as unexplained).

    python src/cand_stack.py                 # prediction for the 17 Oct stack (E51-E53 on the 34-file base)
    python src/cand_stack.py build <name>    # write it
"""
import sys

import numpy as np

import cand_blend as CB
import cand_fire2 as CF2
import lb
import validate

PARTS = {"ems": "e51_ems.csv", "bldfire": "e52_bldfire.csv", "nfac": "e53_nfac.csv"}
BASE_UPTO = "e50_firepart.csv"      # the ledger the three parts were built against


def stack(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    var_y = m2 - m1 ** 2
    Lb = L.loc[:L.index[L.file == BASE_UPTO][0]]
    out, p0, rho_b, *_, d = CF2.candidates(L=Lb)
    p0 = np.asarray(p0, float)
    cov = lambda v, rmse: lb.eyp(rmse, v, m2) - m1 * v.mean()
    a0 = validate.analyse(p0, Lb)
    cov_p0 = (var_y + p0.var() + (p0.mean() - m1) ** 2 - a0["rmse_neutral"] ** 2) / 2
    cols, cyc = [], []
    for k, f in PARTS.items():
        p, col, _ = out[k]
        p, col = np.asarray(p, float), np.asarray(col, float)
        q = lb.read_sub(f).coverage_gap_score.values
        assert np.abs(p - q).max() < 1e-9, f"{f} does not rebuild"
        a = (q - p0) @ col / (col @ col)
        cols.append(col)
        cyc.append((cov(q, float(L[L.file == f].rmse.iloc[0])) - cov_p0) / a)
    C = np.column_stack(cols)
    Scc = C.T @ C / len(C)
    b = np.linalg.solve(Scc, np.array(cyc))
    v = np.clip(p0 + C @ b, 0, 1)
    mse = a0["rmse_neutral"] ** 2 - float(b @ np.array(cyc))
    return v, np.sqrt(mse), a0["rmse_neutral"], dict(zip(PARTS, b)), dict(zip(PARTS, cyc)), d


def evidence(L=None):
    v, pred, base, b, cyc, _ = stack(L)
    sd = np.sqrt(lb.moments()[1] - lb.moments()[0] ** 2)
    print(f"base (34-file re-solve) certified {base:.6f}")
    for k in PARTS:
        print(f"   {k:8s} Cov(y,col)/sd = {cyc[k] / sd:+.4f}   weight b/sd = {b[k] / sd:+.4f}")
    print(f"stack predicted {pred:.6f} (+-{validate.PRED_SE}); best scored "
          f"{validate.best_rmse(lb.ledger()):.6f}")
    a = validate.analyse(v, lb.ledger())
    print(f"validate (conservative, truncates the parts): neutral {a['rmse_neutral']:.6f} "
          f"range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}]")
    return v


def build(name, L=None):
    v, pred, *_, d = stack(L)
    lb.write(name, d.GEOID, v, f"structural stack: 34-file re-solve + E51/E52/E53 at measured weights "
             f"(predicted {pred:.6f})", kind="model")


if __name__ == "__main__":
    a = sys.argv[1:]
    build(a[1]) if a and a[0] == "build" else evidence()


# ---- stage 2 (17 Oct): E56/E57 were built over the stage-1 stack, so they stack on it the same way
PARTS2 = {"schools": "e56_schools.csv", "tribal": "e57_tribal.csv"}
BASE2_UPTO = "e53_nfac.csv"


def _cov_base(v, pred, m1, var_y):
    return (var_y + v.var() + (v.mean() - m1) ** 2 - pred ** 2) / 2


def stage(base, pred, out, parts, L, m1, m2):
    """Add parts built orthogonal to a certified base, each at its measured covariance."""
    var_y = m2 - m1 ** 2
    cb = _cov_base(base, pred, m1, var_y)
    cols, cyc = [], []
    for k, f in parts.items():
        p, col = np.asarray(out[k][0], float), np.asarray(out[k][1], float)
        q = lb.read_sub(f).coverage_gap_score.values
        assert np.abs(p - q).max() < 1e-9, f"{f} does not rebuild"
        a = (q - base) @ col / (col @ col)
        cols.append(col)
        cyc.append((lb.eyp(float(L[L.file == f].rmse.iloc[0]), q, m2) - m1 * q.mean() - cb) / a)
    C = np.column_stack(cols)
    b = np.linalg.solve(C.T @ C / len(C), np.array(cyc))
    return np.clip(base + C @ b, 0, 1), np.sqrt(pred ** 2 - float(b @ np.array(cyc))), dict(zip(parts, b))


def stack2(L=None):
    import cand_next as CN
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    L2 = L.loc[:L.index[L.file == BASE2_UPTO][0]]
    out, v1, pred1, *_, d = CN.candidates(L2)
    v2, pred2, b = stage(np.asarray(v1, float), pred1, out, PARTS2, L, m1, m2)
    return v2, pred2, pred1, b, d
