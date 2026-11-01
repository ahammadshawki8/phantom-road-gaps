"""Generic safe experiment on top of any scored base file.

    q = MEAN + SD * (rho_base * z(base) + W * z(h_perp))

h_perp = hypothesis vector with every scored file projected out, so the known part of the score
is unchanged. Breakeven: corr(h_perp, truth) > W/2. After scoring, `readback` returns that corr,
which is also the optimal weight for h_perp in the next combined model.
"""
import numpy as np
import pandas as pd

import lb
import validate


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std()


def context(L=None):
    """Pass L to reproduce an old experiment: `perp` projects out the files scored *then*."""
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    files, P, _ = validate.scored(L, m1, m2)
    return L, m1, m2, np.sqrt(m2 - m1 ** 2), P - P.mean(0)


def perp(h, Pc):
    h = z(h)
    return z(h - Pc @ np.linalg.lstsq(Pc, h, rcond=None)[0])


def vector(base_file, h, W, L=None):
    """The experiment vector itself, without writing a file (see `src/cand_road.py`)."""
    L, m1, m2, sd, Pc = context(L)
    base = lb.read_sub(base_file).coverage_gap_score.values
    rho_b = lb.rho(L[L.file == base_file].rmse.iloc[0], base, m1, m2)
    return np.clip(m1 + sd * (rho_b * z(base) + W * perp(h, Pc)), 0, 1)


def build(base_file, h, W, name, note, geoids, L=None):
    L = lb.ledger() if L is None else L
    q = vector(base_file, h, W, L)
    lb.write(name, geoids, q, note, kind="model")
    a = validate.analyse(q, L)
    print(f"   neutral {a['rmse_neutral']:.5f} range [{a['rmse_best']:.5f}, {a['rmse_worst']:.5f}] "
          f"best {validate.best_rmse(L):.5f}  breakeven corr > {W / 2:.3f}")
    return q


def readback(file, base_file, q=None, L=None):
    """corr(h_perp, truth) for a scored safe experiment built on base_file.

    Pass q to score a *rebuilt* vector against this file's RMSE, i.e. to ask what the read-back
    would have been had the rebuilt vector been the one submitted (`src/cand_road.py`).
    """
    L, m1, m2, sd, _ = context(L)
    q = lb.read_sub(file).coverage_gap_score.values if q is None else q
    base = lb.read_sub(base_file).coverage_gap_score.values
    rho_b = lb.rho(L[L.file == base_file].rmse.iloc[0], base, m1, m2)
    cov = lambda f, v: lb.eyp(L[L.file == f].rmse.iloc[0], v, m2) - m1 * v.mean()
    part = q - m1 - sd * rho_b * z(base)
    c_part = cov(file, q) - cov(base_file, base) * (sd * rho_b / base.std())
    return c_part / (sd * part.std())
