"""E58-E60: the schools vein after E56 paid 3.1x its price (c +0.124), over the stage-2 stack.

E58 `school_partial` -- tracts WITH properly filed schools plus a clean misfiled real school (a
    real-school name in a category a school can be misfiled under, not a same-name duplicate of a
    proper school): school gap = clean / (proper + clean). 502 tracts, disjoint from E56's.
E59 `school_clean` -- precision inside E56: of its 678 flags, those whose misfiled place has a
    strict real-school name versus the broad match (libraries, fields, alternates). Centred
    contrast on E56's tracts only.
E60 `fire_partial_dedup` -- E50's partial fire gap with same-name duplicates removed (423 of
    E50's 514 tracts). E50 read back +0.011 with the duplicates in.

Each is projected out of the 39-file span, the region dummies and the other two.

    python src/features_misfiled.py
    python src/cand_round3.py            # screens, traps, dose
    python src/cand_round3.py build      # e58_schpart.csv, e59_schclean.csv, e60_firededup.csv
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

FILES = {"school_partial": ("e58_schpart.csv", "E58 partial school gap: clean misfiled real schools beside "
                            "properly filed ones, deduplicated by name, projected over the E54+E56+E57 stack"),
         "school_clean": ("e59_schclean.csv", "E59 precision inside E56: strict real-school names versus the "
                          "broad match, centred contrast, projected over the stack"),
         "fire_partial_dedup": ("e60_firededup.csv", "E60 partial fire gap with same-name duplicates removed, "
                                "projected over the stack")}
CHAT = {"school_partial": 0.05, "school_clean": 0.03, "fire_partial_dedup": 0.03}
M = 0.85


def hypotheses(d):
    _, dfn, d = CF.hypotheses(d)
    g = lambda c: d[c].fillna(0).values.astype(float)
    sc, scm, scc = g("sc_n"), g("sc_mis_n"), g("sc_mis_clean_n")
    fd, fdm, fdc, em, emm = g("fd_n"), g("fd_mis_n"), g("fd_mis_clean_n"), g("em_n"), g("em_mis_n")
    w = 0.5 / (2.0 + dfn.astype(float))
    nfac = 1.0 + ((em + emm) > 0) + ((fd + fdm) > 0)
    nfac_f = 1.0 + ((em + emm) > 0) + ((sc + scm) > 0)
    blk = (sc == 0) & (scm > 0)
    strict = blk & (scc > 0)
    return {"school_partial": np.where((sc > 0) & (scc > 0), scc / (sc + scc), 0.0) * w / nfac,
            "school_clean": np.where(blk, strict.astype(float) - strict[blk].mean(), 0.0) * w / nfac,
            "fire_partial_dedup": np.where((fd > 0) & (fdc > 0), fdc / (fd + fdc), 0.0) * w / nfac_f}, dfn, d


def candidates(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    base, pred, *_ = CS.stack2(L)
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
    print(f"base stage-2 stack predicted {pred:.6f}")
    for k, (p, col, perp) in out.items():
        h = np.abs(H[k])
        print(f"\n{k}: {int((h > 0)[fit].sum())} tracts, burned/unburned "
              f"{h[fit & burn].mean() / h[fit & ~burn].mean():.2f}, rural/urban "
              f"{h[fit & rural].mean() / h[fit & ~rural].mean():.2f}, perp {perp:.3f}")
        print(CK.traps({k: col}, d, dfn).round(3).to_string())
        a = validate.analyse(p, L)
        print(f"neutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}]")
        FC.table(rho_b, {k: CHAT[k]}, M, L, label=FILES[k][0])
    ks = list(out)
    print("\ncorr of projected parts:", {f"{a}-{b}": round(float(np.corrcoef(out[a][1], out[b][1])[0, 1]), 3)
                                         for i, a in enumerate(ks) for b in ks[i + 1:]})


def build(L=None):
    out, *_, d = candidates(L)
    for k, (p, _, _) in out.items():
        lb.write(FILES[k][0], d.GEOID, p, FILES[k][1], kind="model")


if __name__ == "__main__":
    build() if "build" in sys.argv[1:] else evidence()
