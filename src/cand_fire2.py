"""E51-E53: the misfiled-facility vein after E49 confirmed it at FULL PRICE (f = 0.997, c = +0.120).

E49's flag (only fire station filed outside `fire_department`) read back the largest part since
E29. Three further readings, each a distinct mechanism on its own file, each projected out of the
34-file span AND the other two, so the three read-backs are independent:

  E51 `ems`     the same physics on the EMS type: no `ambulance_and_ems_services` primary, >= 1
                place named or alternate-tagged as EMS. 633 tracts, unfitted burned/unburned 2.54x,
                rural/urban 2.35x, perp 0.95. README says EMS "rarely binds" -> priced below fire.
  E52 `bldfire` a station on the map with NO place record: Overture buildings classed
                `fire_station` or named as a station, in tracts with no fire POI of any filing
                (disjoint from E49). 183 tracts, burned/unburned 2.12x, perp 0.98.
  E53 `nfac`    E49's step at its true size. The facilities half is a mean over the DEFINED types,
                so a missing station is 1, 1/2 or 1/3 of it depending on whether schools and EMS
                exist in the tract. E49 was flat; the re-weighted step keeps 33% new direction.
Schools fail the screen a second time (0.70x) and stay out.

    python src/features_misfiled.py
    python src/cand_fire2.py            # screens, traps, dose
    python src/cand_fire2.py build      # e51_ems.csv, e52_bldfire.csv, e53_nfac.csv
"""
import sys

import numpy as np

import cand_blend as CB
import strata as CC
import cand_fire as CF
import cand_knn as CK
import cand_mm as CMM
import forecast as FC
import lb
import validate

FILES = {"ems": ("e51_ems.csv", "E51 misfiled EMS stations: no ambulance_and_ems_services, EMS named "
                 "or alternate place present, EMS gap 1, projected"),
         "bldfire": ("e52_bldfire.csv", "E52 fire stations traced only as Overture buildings "
                     "(class fire_station or station name), no fire POI, fire gap 1, projected"),
         "nfac": ("e53_nfac.csv", "E53 E49 fire step weighted by the facilities-half divisor "
                  "(defined types fire/EMS/school), projected")}
CHAT = {"ems": 0.06, "bldfire": 0.06, "nfac": 0.05}
M = 0.85


def hypotheses(d):
    _, dfn, d = CF.hypotheses(d)
    g = lambda c: d[c].fillna(0).values
    fd, mis, bfs = g("fd_n"), g("fd_mis_n"), g("bfs_n")
    em, emm, sc, scm = g("em_n"), g("em_mis_n"), g("sc_n"), g("sc_mis_n")
    w = 0.5 / (2.0 + dfn.astype(float))
    nfac = 1.0 + ((sc + scm) > 0) + ((em + emm) > 0)
    blk = ((fd == 0) & (mis > 0)).astype(float)
    H = {"ems": ((em == 0) & (emm > 0)).astype(float) * w,
         "bldfire": ((fd == 0) & (mis == 0) & (bfs > 0)).astype(float) * w,
         "nfac": blk * w / nfac}
    return H, dfn, d


def candidates(d=None, L=None, m=M):
    d = CF.table(d)
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    H, dfn, d = hypotheses(d)
    p0, _, _, rho_b = CB.blend(L)
    out = {}
    for k, h in H.items():
        others = [CMM.project(v, d, L)[0] for j, v in H.items() if j != k]
        col, perp = CMM.project(h, d, L, extra=others)
        out[k] = (np.clip(p0 + sd * m * CHAT[k] * col, 0, 1), col, perp)
    return out, p0, rho_b, H, dfn, d


def evidence(d=None, L=None):
    L = lb.ledger() if L is None else L
    out, p0, rho_b, H, dfn, d = candidates(d, L)
    rural, burn = CC.strata(d)
    fit = d.region.ne("eastern-wa").values
    print(f"base re-solve certified {validate.analyse(p0, L)['rmse_neutral']:.6f} (rho_b {rho_b:.4f})")
    for k, (p, col, perp) in out.items():
        h = H[k]
        a = validate.analyse(p, L)
        print(f"\n{k}: {int((h > 0)[fit].sum())} tracts, burned/unburned "
              f"{h[fit & burn].mean() / h[fit & ~burn].mean():.2f}, rural/urban "
              f"{h[fit & rural].mean() / h[fit & ~rural].mean():.2f}, perp {perp:.3f}")
        print(CK.traps({k: col}, d, dfn).round(3).to_string())
        print(f"neutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}]")
        FC.table(rho_b, {k: CHAT[k]}, M, L, label=FILES[k][0])
    ks = list(out)
    print("\ncorr of projected parts:", {f"{a}-{b}": round(float(np.corrcoef(out[a][1], out[b][1])[0, 1]), 3)
                                         for i, a in enumerate(ks) for b in ks[i + 1:]})


def build(d=None, L=None):
    out, *_, dd = candidates(d, L)
    for k, (p, _, _) in out.items():
        lb.write(FILES[k][0], dd.GEOID, p, FILES[k][1], kind="model")


if __name__ == "__main__":
    build() if "build" in sys.argv[1:] else evidence()
