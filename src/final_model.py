"""The final model: a base plus every measured part, combined in a well-conditioned basis.

**Why this is needed (20 Oct).** Every part since E49 was measured as `file_k = base_k + Delta_k`
on a different base. The plain re-solve over raw files truncates the small Delta directions
(LEDGER 53a) and the unconstrained affine solve over raw files amplifies noise (sum |lam| 59). But
each Delta_k is an exact vector, and its covariance with the target is exact algebra:

    Cov(y, Delta_k) = Cov(y, file_k) - Cov(y, base_k)

where Cov(y, file_k) comes from file_k's own score and base_k is a scored file or an exact affine
combination of scored files (its covariance is that same combination of their covariances). With
Z = [base, Delta_1 .. Delta_K], unit-scaled, the best linear predictor is
beta = Cov(Z)^-1 Cov(y, Z), and the explained variance is Cov(y, Z) . beta -- the re-solve in a
basis where nothing is near-singular. A ridge term shrinks parts with small measured covariance.

**RESULT 20 Oct: the structural version FAILS where a base is an unscored stack.** E58-E60 sit on
the stage-2 stack, whose covariance is inflated (LEDGER 60b), so their Deltas read -0.11 to -0.22
and the solver "uses" that error: structural prediction 0.0558 vs the validator's 0.0606 for the
same vector. Use `affine()` below instead: it works on scored files only, where the unknown bases
cancel, with a ridge on lambda to stop noise amplification.

    python src/final_model.py              # prediction, weights, and the shrinkage sweep
    python src/final_model.py build NAME   # write it
"""
import sys

import numpy as np

import cand_blend as CB
import cand_stack as CS
import lb
import validate

POST = {"e37_ramp_rake.csv", "e33_cells.csv"}


def _cov(L, m1, m2, f):
    q = lb.read_sub(f).coverage_gap_score.values
    return q, lb.eyp(float(L[L.file == f].rmse.iloc[0]), q, m2) - m1 * q.mean()


def _blend_through(L, upto, m1, m2):
    """Re-solve over the ledger through `upto` (15 Oct exclusion set): vector and exact covariance."""
    validate.EXCLUDED = set(POST)
    Lb = L.loc[:L.index[L.file == upto][0]]
    files, P, c = validate.scored(Lb, m1, m2)
    Pc = P - P.mean(0)
    w = np.linalg.lstsq(Pc.T @ Pc / len(Pc), c, rcond=validate.RCOND)[0]
    v = m1 + Pc @ w
    return v, float(c @ w)            # unclipped: an exact linear combination of scored files


def parts(L=None):
    """(base vector, base covariance, [(name, Delta, Cov(y, Delta))])."""
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    var_y = m2 - m1 ** 2
    base48, cb48 = _blend_through(L, "e48_commercial.csv", m1, m2)
    base50, cb50 = _blend_through(L, "e50_firepart.csv", m1, m2)
    out = []

    def add(name, f, base, cbase):
        q, cq = _cov(L, m1, m2, f)
        out.append((name, q - base, cq - cbase))

    for f in ["e49_fireblk.csv", "e50_firepart.csv"]:
        add(f[:3], f, base48, cb48)
    for f in ["e51_ems.csv", "e52_bldfire.csv", "e53_nfac.csv"]:
        add(f[:3], f, base50, cb50)
    # stage-1 stack: base50 + sum b_k col_k, an exact combination of scored files (unclipped)
    v1 = base50 + sum(d for n, d, c in out if n in ("e51", "e52", "e53")) * 0   # placeholder, replaced below
    v1, _, _, b1, _, _ = CS.stack(L)
    v1 = np.asarray(v1, float)
    d1 = {n: (d, c) for n, d, c in out}
    # v1 = clip(base50 + C b); covariance from the parts it is made of (clip ignored: y >= 0 so it can only help)
    a = {n: float(np.sqrt((d1[n][0] ** 2).mean())) for n in ("e51", "e52", "e53")}
    cv1 = cb50 + sum(b1[k] / a[n] * d1[n][1] for k, n in zip(["ems", "bldfire", "nfac"], ["e51", "e52", "e53"]))
    for f in ["e56_schools.csv", "e57_tribal.csv"]:
        add(f[:3], f, v1, cv1)
    v2, _, _, b2, _ = CS.stack2(L)
    v2 = np.asarray(v2, float)
    d2 = {n: (d, c) for n, d, c in out}
    a2 = {n: float(np.sqrt((d2[n][0] ** 2).mean())) for n in ("e56", "e57")}
    cv2 = cv1 + sum(b2[k] / a2[n] * d2[n][1] for k, n in zip(["schools", "tribal"], ["e56", "e57"]))
    for f in ["e58_schpart.csv", "e59_schclean.csv", "e60_firededup.csv"]:
        add(f[:3], f, v2, cv2)
    q59, c59 = _cov(L, m1, m2, "e59_schclean.csv")
    for f in ["e62_settle.csv", "e63_zipcbp.csv"]:
        add(f[:3], f, q59, c59)
    q62, c62 = _cov(L, m1, m2, "e62_settle.csv")
    for f in ["e64_tribal_legal.csv", "e65_settle_loc.csv", "e66_web.csv", "e67_unrouted.csv", "e68_ppp.csv"]:
        add(f[:3], f, q62, c62)
    return base48, cb48, out, var_y, m1


def solve(ridge=0.0, L=None, drop=()):
    base, cb, out, var_y, m1 = parts(L)
    out = [o for o in out if o[0] not in drop]
    Z = np.column_stack([base] + [d for _, d, _ in out])
    c = np.array([cb] + [cc for _, _, cc in out])
    Zc = Z - Z.mean(0)
    s = Zc.std(0)
    Zs, cs = Zc / s, c / s
    G = Zs.T @ Zs / len(Zs)
    R = np.eye(len(c)) * ridge
    R[0, 0] = 0.0                         # never shrink the base
    beta = np.linalg.solve(G + R, cs)
    expl = float(cs @ beta) - float(beta @ R @ beta)   # exact explained variance of the shrunk fit
    expl = float(2 * cs @ beta - beta @ G @ beta)
    pred = np.sqrt(max(var_y - expl, 0))
    q = np.clip(m1 + Zs @ beta, 0, 1)
    return q, pred, dict(zip(["base"] + [n for n, _, _ in out], np.round(beta * 1, 4))), out


FILES = ["e45_blend.csv", "e48_commercial.csv", "e49_fireblk.csv", "e50_firepart.csv", "e51_ems.csv",
         "e52_bldfire.csv", "e53_nfac.csv", "e56_schools.csv", "e57_tribal.csv", "e58_schpart.csv",
         "e59_schclean.csv", "e60_firededup.csv", "e62_settle.csv", "e63_zipcbp.csv", "e64_tribal_legal.csv",
         "e65_settle_loc.csv", "e66_web.csv", "e67_unrouted.csv", "e68_ppp.csv",
         "e69_nostreet.csv", "e70_tertiary.csv", "e72_affine.csv", "e73_physics.csv", "e74_physics2.csv",
         "e76_physics3.csv", "e77_affine.csv", "e78_grad_sigt.csv", "e81_te_import.csv", "e82_lo_import.csv"]


def affine(ridge=3e-6, files=FILES, L=None):
    """min_lam  lam.MSE - 1/2 lam'D lam + ridge |lam|^2,  sum lam = 1  (all files scored, so exact)."""
    L = lb.ledger() if L is None else L
    Q = np.column_stack([lb.read_sub(f).coverage_gap_score.values for f in files])
    mse = np.array([L[L.file == f].rmse.iloc[0] ** 2 for f in files])
    D = ((Q[:, :, None] - Q[:, None, :]) ** 2).mean(0)
    n = len(files)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n], A[:n, n], A[n, :n] = -D + 2 * ridge * np.eye(n), 1, 1
    lam = np.linalg.solve(A, np.r_[-mse, 1])[:n]
    pred = float(np.sqrt(lam @ mse - 0.5 * lam @ D @ lam))
    err = 2 * 0.0612 * validate.PRED_SE * float(np.sqrt((lam ** 2).sum())) / (2 * pred)
    return np.clip(Q @ lam, 0, 1), pred, err, dict(zip(files, np.round(lam, 3)))


if __name__ == "__main__" and sys.argv[1:2] == ["affine"]:
    r = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-6
    q, pred, err, lam = affine(r)
    lb.write(sys.argv[2], lb.read_sub("e66_web.csv").GEOID, q,
             f"affine ridge {r:g} over {len(lam)} scored files (predicted {pred:.6f} +- {err:.6f})", kind="model")
    print(lam)
    sys.exit()

if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "build":
        q, pred, w, _ = solve(float(a[2]) if len(a) > 2 else 0.0)
        lb.write(a[1], lb.read_sub("e66_web.csv").GEOID, q, f"final model: base + measured parts (predicted {pred:.6f})", kind="model")
    else:
        for r in [0.0, 0.001, 0.003, 0.01, 0.03]:
            q, pred, w, out = solve(r)
            print(f"ridge {r:<6} predicted {pred:.6f}   validator {validate.analyse(q, lb.ledger())['rmse_neutral']:.6f}")
        print("weights at ridge 0:", solve(0.0)[2])
        print("measured Cov(y, Delta)/sd(Delta):", {n: round(c / max(d.std(), 1e-12) / np.sqrt(lb.moments()[1] - lb.moments()[0] ** 2), 4) for n, d, c in solve(0.0)[3]})
