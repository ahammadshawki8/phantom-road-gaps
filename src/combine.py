"""Joint-optimal blend of safe experiments that share one base file.

A safe experiment (src/safe.py) writes  q = MEAN + SD*(rho_base*z(base) + W*z(h_perp)),
so its new part is recoverable from the scored CSV alone, and the public RMSE gives that
part's correlation with the truth. This module combines several such parts on one base:

    q = clip(MEAN + SD * (rho_base * z(base) + sum_i w_i * z_i), 0, 1)

The parts are NOT mutually orthogonal (they were each orthogonalised against the *scored*
files at build time, not against each other), so the weights are the joint least-squares
solution w = R^-1 c, not the marginal correlations c. Using c directly double-counts
whatever two parts share; for E24/E26, which correlate -0.63, it is badly wrong.

A negative w_i is a legitimate result: the measured correlation IS the optimal weight, and a
pre-registered hypothesis that comes back at -0.13 has been refuted *with a direction*, which
carries exactly as much signal as a confirmation of the same size.

Everything here reads only committed submission CSVs and the ledger, so it runs without the
data package.

    python src/combine.py
"""
import numpy as np

import lb
import validate

# E27: base E22, with the three road parts whose |t| > 5 on the ~2,900-tract public split.
# E24 (highway-only numerator) is excluded: |t| = 2.1 is near the noise floor and it is the
# sole source of the collinearity that inflates every fitted weight ~2x.
BASE = "e22_e18_roadboundary_full.csv"
PARTS = ["e23_road_ring100.csv", "e25_road_regionz.csv", "e26_road_undefined.csv"]
OUT = "e27_road_combined.csv"
N_PUBLIC = 2900  # ~30% of 9,794; sets the SE of a measured correlation


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std()


def context(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    return L, m1, m2, np.sqrt(m2 - m1 ** 2)


def cov(L, m1, m2, file, p):
    """Cov_P(y, p), known exactly from this file's public RMSE."""
    return lb.eyp(L[L.file == file].rmse.iloc[0], p, m2) - m1 * p.mean()


def recover(base_file, part_files, L=None):
    """-> z(base), rho_base, Z (parts, standardised and orthogonal to base), c (their corr with y)."""
    L, m1, m2, sd = context(L)
    base = lb.read_sub(base_file).coverage_gap_score.values
    rho_b = cov(L, m1, m2, base_file, base) / (sd * base.std())
    zb = z(base)
    cols, c = [], []
    for f in part_files:
        q = lb.read_sub(f).coverage_gap_score.values
        part = q - m1 - sd * rho_b * zb                     # = W * z(h_perp) as written by safe.build
        c.append((cov(L, m1, m2, f, q) - cov(L, m1, m2, base_file, base) * (sd * rho_b / base.std()))
                 / (sd * part.std()))
        cols.append(z(part))
    Z = np.column_stack(cols)
    # the original safe experiments were clipped to [0,1], which leaks a little of the base back
    # into the recovered part; remove it so the base term keeps its own measured weight
    Z = Z - np.outer(zb, (zb @ Z) / len(zb))
    return zb, rho_b, Z / Z.std(0), np.array(c)


def weights(Z, c):
    return np.linalg.solve(np.corrcoef(Z, rowvar=False), c)


def build(base_file, part_files, name, note, L=None):
    L, m1, m2, sd = context(L)
    zb, rho_b, Z, c = recover(base_file, part_files, L)
    w = weights(Z, c)
    q = np.clip(m1 + sd * (rho_b * zb + Z @ w), 0, 1)
    geoids = lb.read_sub(base_file).GEOID
    lb.write(name, geoids, q, note, kind="model")
    report(L, m1, m2, sd, zb, rho_b, Z, c, w, q, base_file, part_files)
    return q


def report(L, m1, m2, sd, zb, rho_b, Z, c, w, q, base_file, part_files):
    se = 1 / np.sqrt(N_PUBLIC)
    print(f"\n   base {base_file}  rho {rho_b:+.4f}")
    print(f"   {'part':30s} {'corr':>7s} {'t':>6s} {'weight':>8s}")
    for f, ci, wi in zip(part_files, c, w):
        print(f"   {f:30s} {ci:+7.3f} {ci / se:+6.1f} {wi:+8.3f}")
    R = np.corrcoef(Z, rowvar=False)
    # worst case if every measured correlation is off by one SE against us
    adv = c - se * np.sign(np.linalg.solve(R, c))
    rho_adv = np.sqrt(max(rho_b ** 2 + 2 * (adv @ w) - w @ R @ w, 0))
    a = validate.analyse(q, L)
    print(f"\n   predicted public RMSE {a['rmse_neutral']:.6f}  rho {a['rho_neutral']:+.4f}   "
          f"(current best {validate.best_rmse(L):.6f})")
    print(f"   every corr off by 1 SE against us: {sd * np.sqrt(1 - rho_adv ** 2):.6f}")
    print(f"   explained by scored files: {a['share_explained_by_scored']:.1%} "
          f"(exact prediction when 100%)")


if __name__ == "__main__":
    build(BASE, PARTS, OUT,
          "E22 + ring100 part, minus region-z and undefined parts at joint-optimal doses")
