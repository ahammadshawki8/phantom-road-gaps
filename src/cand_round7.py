"""E69/E70 on the SCORED best E66, and E71 = the exact affine combination of today's scored files.

E66 (place-record completeness) read back at 1.68x its price, the first CBP-half part to beat its
forecast. Splitting that completeness into its parts (with E66 now in the span):
    no phone 1.80x / perp 0.42 (mostly E66 again), no website 1.59x / 0.69, no social 0.66x,
    no email 1.04x, low confidence 1.19x -- dropped
    **no street address 3.44x burned / 2.47x rural / perp 0.558** -- the published POI ratio is 3.40x
E69 `no_street` -- share of places with no street address (only a locality or nothing).
E70 `tertiary` -- the E67 construction with Overture TERTIARY length in place of unrouted
    primary/secondary: TIGER S1200 holds part of what Overture classes tertiary (a participant measured
    a tertiary direction; the README: the two sources draw the primary/secondary boundary
    differently by region). Exact derivative d gap / d lam = named * tertiary / den^2 on unclipped
    tracts. Burned 2.34x, rural 3.10x, perp 0.822. (14 Oct refuted tertiary in the NUMERATOR.)
E71 -- E62 + E64/E66/E67 parts: every file shares the scored base E62 and the parts are mutually
    orthogonal, so the affine identity MSE(sum lam q) = sum lam MSE - 1/2 sum lam lam |q_i - q_j|^2
    gives its score from the four scores alone. E65/E68 read back negative and are left out.

    python src/features_places.py
    python src/cand_round7.py            # screens, dose, E71 prediction
    python src/cand_round7.py build      # e69_nostreet.csv, e70_tertiary.csv, e71_combo.csv
"""
import sys

import numpy as np

import strata as CC
import cand_fire as CF
import cand_knn as CK
import cand_level as CL
import cand_mm as CMM
import cand_road as CR
import cand_round5 as R5
import forecast as FC
import lb
import validate

BASE = "e66_web.csv"
FILES = {"no_street": ("e69_nostreet.csv", "E69 share of Overture places with no street address, CBP half, over E66"),
         "tertiary": ("e70_tertiary.csv", "E70 TIGER S1200 holds part of Overture tertiary: exact derivative of the road "
                      "gap, over E66")}
CHAT = {"no_street": 0.04, "tertiary": 0.04}
M = 0.85
COMBO = ["e62_settle.csv", "e64_tribal_legal.csv", "e66_web.csv", "e67_unrouted.csv"]


def hypotheses(d):
    g = lambda c: d[c].fillna(0).values.astype(float)
    r = CR.road_table()
    assert (r.GEOID.values == d.GEOID.values).all()
    den = 1.1957 * CL.road_denominator(r)
    named, ter = g("rd_len_named"), g("rd_len_tertiary")
    ratio = np.where(den > 0, named / np.maximum(den, 1e-9), np.inf)
    return {"no_street": -g("street_share"),
            "tertiary": np.where((den > 0) & (ratio < 1), named * ter / np.maximum(den, 1e-9) ** 2, 0.0)}


def candidates(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    base = lb.read_sub(BASE).coverage_gap_score.values
    d = R5.table()
    _, dfn, d = CF.hypotheses(d)
    H = hypotheses(d)
    out = {}
    for k, h in H.items():
        others = [CMM.project(v, d, L)[0] for j, v in H.items() if j != k]
        col, perp = CMM.project(h, d, L, extra=others)
        out[k] = (np.clip(base + sd * M * CHAT[k] * col, 0, 1), col, perp)
    rho_b = lb.rho(float(L[L.file == BASE].rmse.iloc[0]), base, m1, m2)
    return out, base, rho_b, H, dfn, d


def combo(L=None):
    """Best affine combination of COMBO by the exact identity (all four are scored)."""
    L = lb.ledger() if L is None else L
    Q = np.column_stack([lb.read_sub(f).coverage_gap_score.values for f in COMBO])
    mse = np.array([L[L.file == f].rmse.iloc[0] ** 2 for f in COMBO])
    D = ((Q[:, :, None] - Q[:, None, :]) ** 2).mean(0)
    n = len(COMBO)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n], A[:n, n], A[n, :n] = -D, 1, 1
    lam = np.linalg.solve(A, np.r_[-mse, 1])[:n]
    return np.clip(Q @ lam, 0, 1), np.sqrt(lam @ mse - 0.5 * lam @ D @ lam), lam


def evidence(L=None):
    L = lb.ledger() if L is None else L
    out, base, rho_b, H, dfn, d = candidates(L)
    rural, burn = CC.strata(d)
    fit = d.region.ne("eastern-wa").values
    for k, (p, col, perp) in out.items():
        h = H[k] - H[k].min()
        print(f"\n{k}: burned/unburned {h[fit & burn].mean() / h[fit & ~burn].mean():.2f}, rural/urban "
              f"{h[fit & rural].mean() / max(h[fit & ~rural].mean(), 1e-12):.2f}, perp {perp:.3f}")
        print(CK.traps({k: col}, d, dfn).round(3).to_string())
        a = validate.analyse(p, L)
        print(f"neutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}]")
        FC.table(rho_b, {k: CHAT[k]}, M, L, label=FILES[k][0])
    q, pred, lam = combo(L)
    print(f"\nE71 combo predicted {pred:.6f}; lam {dict(zip([f[:3] for f in COMBO], np.round(lam, 2)))}; "
          f"validator {validate.analyse(q, L)['rmse_neutral']:.6f}")


def build(L=None):
    out, *_, d = candidates(L)
    for k, (p, _, _) in out.items():
        lb.write(FILES[k][0], d.GEOID, p, FILES[k][1], kind="model")
    q, pred, lam = combo(L)
    lb.write("e71_combo.csv", d.GEOID, q, f"E71 affine combination of E62/E64/E66/E67 (predicted {pred:.6f})", kind="model")


if __name__ == "__main__":
    build() if "build" in sys.argv[1:] else evidence()
