"""Leaderboard algebra. Each public RMSE R of a prediction vector p satisfies

    R^2 = E_P[y^2] - 2 E_P[y p] + E_P[p^2]

over the (unknown) public subset P. E_P[p^2] is approximated by the full-sample mean of p^2
(P is a ~30% random subset). Every scored submission therefore yields one linear functional
E_P[y p] of the hidden target, and a set of them yields least-squares coefficients.

The ledger (submissions/ledger.csv) is the single record of every scored file.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "submissions" / "ledger.csv"


def ledger():
    return pd.read_csv(LEDGER)


def read_sub(name):
    return pd.read_csv(ROOT / "submissions" / name, dtype={"GEOID": str})


def moments(L=None):
    """E_P[y], E_P[y^2] from the constant submissions (any two or more, least squares)."""
    L = ledger() if L is None else L
    c = L[L.kind == "const"]
    # R^2 - c^2 = m2 - 2 c m1
    A = np.c_[np.ones(len(c)), -2 * c.const.values]
    b = c.rmse.values ** 2 - c.const.values ** 2
    m2, m1 = np.linalg.lstsq(A, b, rcond=None)[0]
    return m1, m2


def eyp(rmse, p, m2):
    """E_P[y p] for a scored vector p."""
    return 0.5 * (m2 + np.mean(p ** 2) - rmse ** 2)


def rho(rmse, p, m1, m2):
    sd_y = np.sqrt(m2 - m1 ** 2)
    cov = eyp(rmse, p, m2) - m1 * np.mean(p)
    return cov / (sd_y * np.std(p))


def predicted_rmse(p, m1, m2, cov_fn):
    """RMSE of p given a function returning E_P[y p] (for offline what-ifs)."""
    return np.sqrt(m2 - 2 * cov_fn(p) + np.mean(p ** 2))


def check(p, n=9794):
    p = np.asarray(p, float)
    assert len(p) == n and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    return p


def write(name, geoids, p, note, kind="probe", overwrite_scored=False):
    """Write a candidate CSV. Refuses to overwrite a file whose RMSE is already in the ledger.

    A scored file is evidence, not an artefact: its RMSE is the only measurement we have of that
    vector, so rebuilding over it destroys the measurement and silently invalidates every weight
    read back from it. Re-running `cand_f2.py` or `combine.py` after a score came in used to do
    exactly that.
    """
    p = check(p)
    out = ROOT / "submissions" / name
    if not overwrite_scored and name in set(ledger().file):
        raise SystemExit(f"refusing to overwrite scored file {name} (RMSE is in the ledger); "
                         f"write under a new name, or pass overwrite_scored=True if you are sure")
    pd.DataFrame({"GEOID": geoids, "coverage_gap_score": p}).to_csv(out, index=False)
    print(f"wrote {out.name}  mean={p.mean():.5f} sd={p.std():.5f}  [{kind}] {note}")
    return out
