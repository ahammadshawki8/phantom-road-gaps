"""Physics model v3 (E76). v2 (E74) read back f 1.28, c +0.077 on top of v1. Two changes, both
from this week's measurements:
  - CBP completeness: E69 (no street address) read back NEGATIVE and E66 (no website or phone)
    POSITIVE (c +0.051), so the CBP reference now scales with the no-website/phone share.
  - v3a (road level per region x rural/urban) was tried first and dropped: rural and urban k came
    out nearly equal in every region (OK 1.61/1.64, TX 1.32/1.32), perp 0.044 -- v2 had it already.

Physics model v2 (E74). Changes from v1 (E73 read back f 0.91, c +0.054):
  - sigma_t 0.30 -> 0.15 (v1's transport burned/unburned came out 1.57 vs a published 1.87: the
    reference noise flattened it)
  - schools/EMS extras held at multiplier 0.35 instead of letting the POI ratio zero them (E56
    measured a large school part), so the POI ratio becomes an UNFITTED check again
  - a road level k PER REGION, each fitted to that region's published composite mean (four
    single constants; the region mean is the tract-weighted average of the post's cells)

Physics model: E[y | everything we know], by simulating the hidden reference layers.

Every file so far adds ONE deterministic part linearly. The target is not linear: each component
is 1 - min(1, overture / reference), undefined where the reference is 0, and y is the mean of the
defined components. With the references uncertain, E[y] is a smooth non-linear function of all
the tract's evidence at once (clip, divisor and Jensen effects together). This module draws the
references many times from a calibrated generative model and averages the exact formula.

Generative model per tract (all inputs from the provided package):
  roads     TIGER = k * den * LN(sigma_t) + lam_u * unrouted + lam_t * tertiary,  defined iff den > 0
            (den = E36's ramp-free core + 15.5 m ring; num = E36's numerator)
  fire      REF = fd + Poisson(p_mis * misfiled_clean + p_bld * bld_station + p_comm * rural_missing
                            + p_bg * (no trace at all; rural 1, urban 0.1))
            gap = 1 - min(1, fd / REF), defined iff REF > 0
  ems       REF = em + Poisson(p_ems * ems_misfiled)
  schools   REF = sc + Poisson(p_sch * school_misfiled_clean + p_schb * school_misfiled_broad)
  cbp       CBP = s * r0 * completeness * LN(sigma_c),  r0 = E44 reference (buildings^(1-a) area^a),
            completeness = (no-website/phone share)^w_st  (E66 confirmed, E69 refuted)
  buildings constant B = 0.0065, defined everywhere (published 0.005-0.008)
  y = (T dT + B + P dP) / (dT + 1 + dP),  P = mean of defined {facilities, cbp}

Calibrated ONLY to published aggregates (transport mean 0.112, POI mean 0.052, a participant's 740 fire
blackouts of which 270 misfiled), never to the leaderboard. The ratios (transport 1.87x, POI 3.40x,
composite rural 2.96x) are left unfitted as checks.

    python src/physics3.py              # calibrate, checks, perp share, dose table
    python src/physics3.py build NAME   # write NAME = scored base + dose * projected physics mean
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import strata as CC
import cand_fire as CF
import cand_knn as CK
import cand_mm as CMM
import cand_poi as CPO
import cand_tiger as CT
import forecast as FC
import lb
import validate

SEED = 20261020
DRAWS = 400
SIGMA_T, SIGMA_C = 0.15, 0.50
M_S = 0.12
LAM_U, LAM_T = 0.20, 0.05
P_BLD, P_SCH, P_SCHB, P_EMS = 0.8, 0.6, 0.25, 0.3
W_ST = 1.0
URBAN_BG = 0.1                 # assumed: an urban station covers several tracts
P_COMM = 0.5                    # E62's community term
FIRE_TARGETS = (270, 740)       # a participant's misfiled and total fire blackouts, the calibration targets
B_CONST = 0.0065
BASE = "e74_physics2.csv"
CHAT, M = 0.06, 0.85


def inputs():
    d = CF.table()
    for t in ["settle", "places"]:
        d = d.merge(pd.concat([pd.read_parquet(lb.ROOT.parent / "data" / "derived" / f"{r}-{t}.parquet")
                               for r in sorted(d.region.unique())]), on="GEOID", how="left")
    d, rt, r, q, tg, _ = CT.inputs(d)
    import cand_ramp as CP
    import cand_county as CY
    num = CP.numerator(r, q)
    den = rt.rt_core.values.astype(float) + CY.ring_at(rt, ["core"], CK.RING)
    g = lambda c: d[c].fillna(0).values.astype(float)
    X = dict(num=num, den=den, unr=g("rd_len_named_noroute"), ter=g("rd_len_tertiary"),
             fd=g("fd_n"), fdm=g("fd_mis_clean_n"), bfs=g("bfs_n"), em=g("em_n"), emm=g("em_mis_n"),
             sc=g("sc_n"), scm=g("sc_mis_clean_n"), scb=np.maximum(g("sc_mis_n") - g("sc_mis_clean_n"), 0),
             comm=np.maximum(g("n_po"), g("n_th")), rural=(d.ruca_primary.fillna(1).values >= 4).astype(float),
             poi=g("poi_all"), r0=(g("bld_all") + 1) ** (1 - 0.0621) * np.maximum(d.area_m2.values / 1e6, 1e-6) ** 0.0621,
             nost=1 - g("web_share"))
    X["miss"] = np.maximum(X["comm"] - X["fd"], 0) * X["rural"] * ((X["fdm"] + X["bfs"]) == 0)
    # background: a reference station with no Overture trace at all (rural full rate, urban a tenth)
    X["bg"] = (X["fd"] == 0) * ((X["fdm"] + X["bfs"]) == 0) * (X["rural"] + URBAN_BG * (1 - X["rural"]))
    return d, X


def simulate(X, k, p_mis, p_bg, s, draws=DRAWS, seed=SEED, parts=False, p_comm=P_COMM, m_s=1.0):
    rng = np.random.default_rng(seed)
    n = len(X["num"])
    dT = X["den"] > 0
    # the reference holds what Overture lists PLUS a random number of extras Overture does not
    # (a Poisson spread around Overture's own count would invent losses that the clip turns into gaps)
    lam_f = p_mis * X["fdm"] + P_BLD * X["bfs"] + p_comm * X["miss"] + p_bg * X["bg"]
    lam_e = m_s * P_EMS * X["emm"]
    lam_s = m_s * (P_SCH * X["scm"] + P_SCHB * X["scb"])
    cbp0 = s * X["r0"] * np.maximum(X["nost"], 0.02) ** W_ST
    acc = {"y": np.zeros(n), "T": np.zeros(n), "P": np.zeros(n), "fblk": np.zeros(n), "fmis": np.zeros(n)}
    for _ in range(draws):
        tiger = (k if np.isscalar(k) else k) * X["den"] * np.exp(SIGMA_T * rng.standard_normal(n) - SIGMA_T ** 2 / 2) \
            + LAM_U * X["unr"] + LAM_T * X["ter"]
        T = np.where(dT, 1 - np.minimum(1, X["num"] / np.maximum(tiger, 1e-9)), 0.0)
        rf = X["fd"] + rng.poisson(lam_f)
        re = X["em"] + rng.poisson(lam_e)
        rs = X["sc"] + rng.poisson(lam_s)
        gf = np.where(rf > 0, 1 - np.minimum(1, X["fd"] / np.maximum(rf, 1)), 0.0)
        ge = np.where(re > 0, 1 - np.minimum(1, X["em"] / np.maximum(re, 1)), 0.0)
        gs = np.where(rs > 0, 1 - np.minimum(1, X["sc"] / np.maximum(rs, 1)), 0.0)
        nf = (rf > 0).astype(float) + (re > 0) + (rs > 0)
        fac = np.where(nf > 0, (gf + ge + gs) / np.maximum(nf, 1), 0.0)
        cbp_ref = cbp0 * np.exp(SIGMA_C * rng.standard_normal(n) - SIGMA_C ** 2 / 2)
        cbp = 1 - np.minimum(1, X["poi"] / np.maximum(cbp_ref, 1e-9))
        P = np.where(nf > 0, (fac + cbp) / 2, cbp)
        y = (T * dT + B_CONST + P) / (dT + 2.0)
        acc["y"] += y
        acc["T"] += T
        acc["P"] += P
        acc["fblk"] += (rf > 0) & (X["fd"] == 0)
        acc["fmis"] += (rf > 0) & (X["fd"] == 0) & (X["fdm"] + X["bfs"] > 0)
    return {k_: v / draws for k_, v in acc.items()}


def calibrate(d, X, draws=60):
    import strata as CCm
    fit = d.region.ne("eastern-wa").values
    rural, burn = CC.strata(d)
    reg = d.region.values
    # fire first (independent of roads): misfiled blackouts 270, total 740 (FIRE_TARGETS; physics12 scales them)
    def blk(pm, pb):
        o = simulate(X, 1.1, pm, pb, 1.0, draws=draws, m_s=M_S)
        return o["fmis"][fit].sum(), o["fblk"][fit].sum()
    p_mis = brentq(lambda z: blk(z, 0.0)[0] - FIRE_TARGETS[0], 0.01, 1.0)
    p_bg = brentq(lambda z: blk(p_mis, z)[1] - FIRE_TARGETS[1], 0.0, 10.0)
    s_ = brentq(lambda z: simulate(X, 1.1, p_mis, p_bg, z, draws=draws, m_s=M_S)["P"][fit].mean() - 0.052, 1e-4, 1e4)
    # per-region road level to the region's published composite mean
    k = np.full(len(reg), 1.1)
    for r_, cells in CCm.PUB16.items():
        m = reg == r_
        w = [(m & rural & burn).sum(), (m & rural & ~burn).sum(), (m & ~rural & burn).sum(), (m & ~rural & ~burn).sum()]
        target = float(np.dot(w, cells) / sum(w))
        def f(z):
            kk = k.copy(); kk[m] = z
            return simulate(X, kk, p_mis, p_bg, s_, draws=draws, m_s=M_S)["y"][m].mean() - target
        try:
            k[m] = brentq(f, 0.3, 6.0)
        except ValueError:
            k[m] = 0.3 if f(0.3) > 0 else 6.0
        print(f"   {r_:17s} target {target:.4f}  k {k[m][0]:.3f}")
    k[reg == "eastern-wa"] = np.median([k[reg == r_][0] for r_ in CCm.PUB16])
    return k, p_mis, p_bg, s_, M_S


def evidence():
    d, X = inputs()
    k, p_mis, p_bg, s, ms = calibrate(d, X)
    o = simulate(X, k, p_mis, p_bg, s, m_s=ms)
    rural, burn = CC.strata(d)
    fit = d.region.ne("eastern-wa").values
    dT = X["den"] > 0
    y, T, P = o["y"], o["T"], o["P"]
    print(f"calibrated: k(median) {np.median(k):.3f}, p_mis {p_mis:.3f}, p_bg {p_bg:.3f}, school multiplier {ms:.3f}, s {s:.4f}")
    # the published component means are ALL-tract means with undefined = 0 (LEDGER 114a); defined-only is not 1.87's convention
    print(f"unfitted checks: transport burned/unburned {T[fit & burn].mean() / T[fit & ~burn].mean():.2f} all-tract (pub 1.87; "
          f"defined-only {T[dT & fit & burn].mean() / T[dT & fit & ~burn].mean():.2f}); "
          f"POI {P[fit & burn].mean() / P[fit & ~burn].mean():.2f} (pub 3.40, unfitted); composite mean {y[fit].mean():.4f} (pub 0.0585); "
          f"rural/urban {y[fit & rural].mean() / y[fit & ~rural].mean():.2f} (pub 2.96)")
    L = lb.ledger()
    a = validate.analyse(np.clip(y, 0, 1), L)
    print(f"physics mean as a standalone file: neutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}], "
          f"explained by scored files {a['share_explained_by_scored']:.1%}")
    col, perp = CMM.project(y, d, L)
    print(f"perp share after the whole span: {perp:.3f}")
    _, dfn, _ = CMM.hypothesis(d)
    print(CK.traps({"physics": col}, d, dfn).round(3).to_string())
    m1, m2 = lb.moments(L)
    base = lb.read_sub(BASE).coverage_gap_score.values
    rho_b = lb.rho(float(L[L.file == BASE].rmse.iloc[0]), base, m1, m2)
    FC.table(rho_b, {"physics": CHAT}, M, L, label="physics part over " + BASE)
    return d, y, col


def build(name):
    d, y, col = evidence()
    L = lb.ledger()
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    base = lb.read_sub(BASE).coverage_gap_score.values
    lb.write(name, d.GEOID, np.clip(base + sd * M * CHAT * col, 0, 1),
             f"physics model (Monte-Carlo of the hidden references, exact formula) projected, over {BASE}", kind="model")
    pd.DataFrame({"GEOID": d.GEOID, "physics": y}).to_parquet(lb.ROOT.parent / "data" / "derived" / "physics3.parquet", index=False)


if __name__ == "__main__":
    a = sys.argv[1:]
    build(a[1]) if a and a[0] == "build" else evidence()
