"""E56/E57: two unmeasured directions, each one mechanism, built over the E54 structural stack.

E56 `schools` -- misfiled schools, the E49 rule on the school type. The facilities half matches
    Overture on six exact school categories (README); a real school filed as `education`, a
    college category, or with a school category only as alternate is in the reference and not in
    the numerator, so where it is the tract's only school the school gap is 1 and the facilities
    half rises by 1/n_defined. **Dropped twice on the screen (0.70x burned/unburned), and the
    screen was misapplied**: it compared one facility type against the WHOLE POI ratio 3.40x,
    which the fire half carries (fire flags alone 2.66x, 17 Oct). A school gap may legitimately be
    urban, so this is a measurement, priced low.
E57 `tribal`  -- AIANNH share of the tract. The one consistent pointer in three competitor
    scorecards (0.05715/0.05758/0.05776): their tribal/rural ratio is ~1.22, ours 1.13. Used as a
    pointer to an unmeasured direction only; nothing is fitted to any card. E32 built this in
    13 Oct and never uploaded it.

Each part is projected out of the 37-file span, the region dummies and the other part.

    python src/cand_next.py            # screens, traps, dose
    python src/cand_next.py build      # e56_schools.csv, e57_tribal.csv
"""
import sys

import numpy as np

import strata as CC
import cand_fire as CF
import cand_knn as CK
import cand_mm as CMM
import cand_stack as CS
import forecast as FC
import lb
import validate

FILES = {"schools": ("e56_schools.csv", "E56 misfiled schools: only school filed outside the six school "
                     "categories, school gap 1 over n defined, projected over E54"),
         "tribal": ("e57_tribal.csv", "E57 tribal-land share (AIANNH) as an unmeasured within-cell direction, "
                    "projected over E54")}
CHAT = {"schools": 0.04, "tribal": 0.04}
M = 0.85


def hypotheses(d):
    _, dfn, d = CF.hypotheses(d)
    g = lambda c: d[c].fillna(0).values.astype(float)
    sc, scm, em, emm, fd, mis = g("sc_n"), g("sc_mis_n"), g("em_n"), g("em_mis_n"), g("fd_n"), g("fd_mis_n")
    nfac = 1.0 + ((em + emm) > 0) + ((fd + mis) > 0)
    w = 0.5 / (2.0 + dfn.astype(float))
    return {"schools": ((sc == 0) & (scm > 0)).astype(float) * w / nfac,
            "tribal": d.tribal_pct.fillna(0).values.astype(float)}, dfn, d


def candidates(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    base, pred, *_ = CS.stack(L)
    H, dfn, d = hypotheses(CF.table())
    out = {}
    for k, h in H.items():
        others = [CMM.project(v, d, L)[0] for j, v in H.items() if j != k]
        col, perp = CMM.project(h, d, L, extra=others)
        out[k] = (np.clip(base + sd * M * CHAT[k] * col, 0, 1), col, perp)
    return out, base, pred, H, dfn, d


def evidence(L=None):
    L = lb.ledger() if L is None else L
    out, base, pred, H, dfn, d = candidates(L)
    rural, burn = CC.strata(d)
    fit = d.region.ne("eastern-wa").values
    m1, m2 = lb.moments(L)
    rho_b = np.sqrt(1 - pred ** 2 / (m2 - m1 ** 2))
    print(f"base E54 stack predicted {pred:.6f}")
    for k, (p, col, perp) in out.items():
        h = H[k]
        print(f"\n{k}: {int((h > 0)[fit].sum())} tracts > 0, burned/unburned "
              f"{h[fit & burn].mean() / h[fit & ~burn].mean():.2f}, rural/urban "
              f"{h[fit & rural].mean() / h[fit & ~rural].mean():.2f}, perp {perp:.3f}")
        print(CK.traps({k: col}, d, dfn).round(3).to_string())
        a = validate.analyse(p, L)
        print(f"neutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}]")
        FC.table(rho_b, {k: CHAT[k]}, M, L, label=FILES[k][0])
    print(f"\ncorr of projected parts {np.corrcoef(out['schools'][1], out['tribal'][1])[0, 1]:+.3f}")


def build(L=None):
    out, *_, d = candidates(L)
    for k, (p, _, _) in out.items():
        lb.write(FILES[k][0], d.GEOID, p, FILES[k][1], kind="model")


if __name__ == "__main__":
    build() if "build" in sys.argv[1:] else evidence()
