"""E47: the masked-mean FUNCTIONAL FORM of the target, as one new mechanism.

The target is not a sum of parts; it is a mean over the parts that are defined:

    y = (T * dT + P + B) / (2 + dT)        T transport gap, dT = 1 if transport is defined
                                           P POI gap, B buildings gap (~0.005-0.008)

Every scored file so far is LINEAR in its inputs, so the blend can hold T, P, dT and B as
separate directions but never their ratio. The ratio carries two things no file has: POI counts
**1/2** where transport is undefined and **1/3** where it is defined, and T and P compete for the
same divisor. Built from the best existing estimate of each part (E36's transport gap, E44's
identified POI reference, the published per-component building means), then projected onto the
scored span, the five region dummies and E46's two parts, what is left is exactly that
non-linearity. Distinct from E26 (the bare `1/n_defined` indicator, measured negative): here the
divisor multiplies the parts rather than standing alone.

    python src/cand_mm.py            # screens, traps, dose
    python src/cand_mm.py build      # write submissions/e47_maskedmean.csv
"""
import sys

import numpy as np

import cand_blend as CB
import strata as CC
import cand_knn as CK
import cand_poi as CPO
import cand_poi2 as CP2
import cand_tiger as CT
import forecast as FC
import lb
import validate

OUT = "e47_maskedmean.csv"
NOTE = "E47 masked-mean functional form: (T*dT + P + B)/(2 + dT) with E36 transport, E44 POI, projected"
CHAT = 0.05          # a structural correction of two measured parts, priced below E44's +0.061
M = 0.85


def hypothesis(d):
    y, dfn, d = CPO.composite(d)
    return np.asarray(y, float), dfn, d


def project(h, d, L, extra=()):
    """Standardise h, project out the scored span, region dummies and any extra columns."""
    m1, m2 = lb.moments(L)
    _, P, _ = validate.scored(L, m1, m2)
    B = np.column_stack([P - P.mean(0)]
                        + [(d.region == rg).values.astype(float) for rg in sorted(set(d.region))]
                        + list(extra))
    B = B - B.mean(0)
    z = (h - h.mean()) / h.std()
    hp = z - B @ np.linalg.lstsq(B, z, rcond=validate.RCOND)[0]
    return hp / max(hp.std(), 1e-12), float(hp.std())


def e46_cols(d, L):
    H, _, _ = CP2.hypotheses(d)
    cols, _ = CP2.basis(H, d, L)
    return [cols[n] for n in CP2.ORDER]


def candidate(d=None, L=None, chat=CHAT, m=M):
    d = CC.table_all() if d is None else d
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    h, dfn, d = hypothesis(d)
    col, perp = project(h, d, L, extra=e46_cols(d, L))
    p0, _, _, rho_b = CB.blend(L)
    return np.clip(p0 + sd * m * chat * col, 0, 1), p0, rho_b, sd, perp, col, h, dfn, d


def evidence(d=None, L=None):
    L = lb.ledger() if L is None else L
    p, p0, rho_b, sd, perp, col, h, dfn, d = candidate(d, L)
    rural, burn = CC.strata(d)
    fit = d.region.ne("eastern-wa").values
    print(f"composite mean {h[fit].mean():.4f} (published overall ~0.0585), "
          f"burned/unburned {h[fit & burn].mean() / h[fit & ~burn].mean():.2f} (published 2.23), "
          f"rural/urban {h[fit & rural].mean() / h[fit & ~rural].mean():.2f} (published 2.96)")
    print(f"perp share after span + regions + E46 parts: {perp:.3f}")
    print("traps:")
    print(CK.traps({"maskedmean": col}, d, dfn).round(3).to_string())
    a = validate.analyse(p, L)
    print(f"base certified {validate.analyse(p0, L)['rmse_neutral']:.6f} (rho_b {rho_b:.4f}); "
          f"E47 neutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}]")
    FC.table(rho_b, {"maskedmean": CHAT}, M, L, label="E47")
    return p


def build(d=None, L=None):
    p, *_, dd = candidate(d, L)
    lb.write(OUT, dd.GEOID, p, NOTE, kind="model")
    return p


if __name__ == "__main__":
    build() if "build" in sys.argv[1:] else evidence()
