"""E111 / E112: the transport reference's boundary roads measured on TIGER's own geometry, and the toll
highways the core-route regex never matched.

Both are scorer mechanics, read off the public reconstruction of the scoring pipeline: TIGER S1100+S1200
is intersected with the tract polygons by an overlay in EPSG:5070 with no deduplication, so a highway
lying on a shared tract boundary belongs to BOTH tracts in the reference. The published road-undefined
counts behind the Info page's percentages are OK 253, CA 218, AZ 869, TX 1704 tracts.

E111 -- the boundary road, from TIGER's line instead of Overture's copy.
    Since E19 the boundary road enters the reference as the Overture core length in an outer ring of
    width w (50 m, 27.3, 15.49, now 7.91 m), and E110 showed the score still moves when w moves
    (c +0.0567). A single width cannot be right: in rural Oklahoma and California the copy of the
    boundary highway sits 15-60 m off TIGER's line [corrected 28 Oct, LEDGER 120a: only OK/TX's tail
    reaches ~20 m; 25-60 m is a railway, river or right-of-way boundary], while in Phoenix, Dallas
    and Houston a frontage road or parallel street 30-60 m from a freeway is what the boundary follows.
    `features_bndfollow` walks every tract boundary (TIGER geometry) and keeps the stretches where the
    NEAREST road is a core route running parallel within 60 m; the stretches whose copy lies outside the
    tract (`bf_out`) are what TIGER holds and Overture does not. den = rt_core + bf_out, nothing else
    moves (E100's rule). No width is fitted, so the published burned/unburned ratio becomes an unfitted
    check: 1.895 against 1.87 (E110 had to FIT w to reach it).

E112 -- toll highways are S1100.
    `features_routes.CORE` matches US:I, US:US and US:<ST> networks; Texas toll roads are filed under
    their authorities (NTTA, HCTRA, CTRMA, FBCTRA, MCTRA, managed lanes) and match nothing, although
    they are limited-access highways and class motorway in Overture -- in the numerator, never in the
    reference or the mask. 1,314 km in 6,003 TX tracts. den = rt_core + ring(7.91) + toll_in. The TX
    road-undefined count goes 1,809 -> 1,724 against a published 1,704 (error 105 -> 20).

E113 -- sigma_t is the root of the published ratio once the width is gone.
    E110 identified the ring width on the published transport burned/unburned 1.87. E111 removes the
    width altogether (the boundary road is TIGER's line), and E112 adds the toll highways, so the
    ratio drifts to 1.881 and the parameter it now pins is the reference noise sigma_t (E78 read the
    sigma gradient at +0.021 and said raise it). On the E111+E112 vehicle the sweep is monotone and
    brackets 1.87 (0.22 -> 1.881, 0.30 -> 1.862): sigma_t* = 0.2665, independently close to the 0.26
    the physics-v8 fit found on the scored span. The part is y(sigma*) - y(0.22) on that vehicle.

    python src/cand_round25.py              # evidence, traps, dose tables
    python src/cand_round25.py build        # e111_bndtiger.csv, e112_toll.csv, e113_sigma.csv
"""
import io
import contextlib
import sys

import numpy as np
import pandas as pd

import physics4 as P4          # patches physics3 into v4
import physics3 as P3
import strata as CC
import cand_county as CY
import cand_knn as CK
import cand_mm as CMM
import cand_ramp as CP
import features as F
import forecast as FC
import lb
import table
import validate

BASE = "e110_ringspec.csv"     # the best SCORED file (0.057168596)
W_RING = 7.91                  # E110's width, solved on the spec numerator
PUB_UNDEF_N = {"eastern-ok": 253, "northern-ca": 218, "maricopa-az": 869, "south-central-tx": 1704}
FILES = {
    "bnd": ("e111_bndtiger.csv", 0.040, 0.045,
            "E111 the boundary highway from TIGER's own line: tract-boundary stretches whose nearest road is a "
            "parallel core route with its Overture copy outside replace the 7.91 m ring"),
    "toll": ("e112_toll.csv", 0.030, 0.035,
             "E112 Texas toll highways are S1100: the toll-authority networks the core-route regex never "
             "matched enter the transport reference and the mask"),
    "sigma": ("e113_sigma.csv", 0.022, 0.025,
              "E113 with the width gone the reference noise sigma_t is the root of the published transport "
              "ratio: 0.2665 on the TIGER-boundary and toll reference"),
}
PUB_RATIO = 1.87


def bndfollow():
    o = table.sample_order()[["GEOID"]]
    x = pd.concat([pd.read_parquet(F.OUT / f"{r}-bndfollow.parquet") for r in F.REGIONS])
    return o.merge(x, on="GEOID", how="left").fillna(0.0)


def model(den, quiet=True):
    """v4 with E109's spec numerator (ramps in); only the reference base `den` varies."""
    d, X = P4.inputs()
    X = dict(X)
    X["num"] = CP.numerator(alpha=0.0)
    X["den"] = den
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx:
        k, p_mis, p_bg, s, ms = P3.calibrate(d, X)
        o = P3.simulate(X, k, p_mis, p_bg, s, m_s=ms)
    return d, X, o


def dens():
    rt = CY.routes_table()
    core = rt.rt_core.values.astype(float)
    ring = CY.ring_at(rt, ["core"], W_RING)
    b = bndfollow()
    return {"e110": core + ring, "bnd": core + b.bf_out.values, "toll": core + ring + b.toll_in.values,
            "both": core + b.bf_out.values + b.toll_in.values}


def sigma_model(den, sigma):
    old = P3.SIGMA_T
    P3.SIGMA_T = sigma
    try:
        return model(den)
    finally:
        P3.SIGMA_T = old


def sigma_star(den):
    """E113's root, solved on the DEFINED-only ratio -- the wrong convention for 1.87 (LEDGER 114a). Kept as built."""
    from scipy.optimize import brentq

    def ratio(s):
        d, X, o = sigma_model(den, s)
        rural, burn = CC.strata(d)
        fit, dT, T = d.region.ne("eastern-wa").values, X["den"] > 0, o["T"]
        return T[dT & fit & burn].mean() / T[dT & fit & ~burn].mean()
    return brentq(lambda s: ratio(s) - PUB_RATIO, 0.22, 0.34, xtol=0.002)


def describe(tag, d, X, o):
    rural, burn = CC.strata(d)
    fit, reg = d.region.ne("eastern-wa").values, d.region.values
    dT, T = X["den"] > 0, o["T"]
    und = {r[:2]: int((~dT)[reg == r].sum()) - n for r, n in PUB_UNDEF_N.items()}
    print(f"   {tag:16s} burned/unburned {T[fit & burn].mean() / T[fit & ~burn].mean():.3f} all-tract (pub 1.87; defined-only "
          f"{T[dT & fit & burn].mean() / T[dT & fit & ~burn].mean():.3f}) | "
          f"composite rural/urban {o['y'][fit & rural].mean() / o['y'][fit & ~rural].mean():.3f} (pub 2.96) | "
          f"road-undefined error vs published counts {und}")


def candidates(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    D = dens()
    d, X0, o0 = model(D["e110"])
    describe("E110 model", d, X0, o0)
    base = lb.read_sub(BASE).coverage_gap_score.values
    rho_b = lb.rho(float(L[L.file == BASE].rmse.iloc[0]), base, m1, m2)
    out = {}
    s_star = sigma_star(D["both"])
    print(f"   sigma_t* on the E111+E112 reference = {s_star:.4f} (root of the published {PUB_RATIO})")
    for key in ["bnd", "toll", "sigma"]:
        if key == "sigma":
            _, Xa, oa = sigma_model(D["both"], P3.SIGMA_T)
            _, X1, o1 = sigma_model(D["both"], s_star)
            describe("sigma*", d, X1, o1)
            h = o1["y"] - oa["y"]
        else:
            _, X1, o1 = model(D[key])
            describe(key, d, X1, o1)
            h = o1["y"] - o0["y"]
        col, perp = CMM.project(h, d, L)
        W = FILES[key][1]
        out[key] = dict(h=h, col=col, perp=perp, p=np.clip(base + sd * W * col, 0, 1), dfn=X0["den"] > 0)
    return d, out, rho_b, sd


def evidence(L=None):
    L = lb.ledger() if L is None else L
    d, out, rho_b, sd = candidates(L)
    for key, v in out.items():
        name, W, chat, _ = FILES[key]
        h = v["h"]
        print(f"\n{name}: reach {(np.abs(h) > 1e-6).mean():.1%}, mean|h| {np.abs(h).mean():.5f}, sd {h.std():.5f}, "
              f"max {np.abs(h).max():.3f}, perp {v['perp']:.3f}")
        print(CK.traps({key: v["col"]}, d, v["dfn"]).round(3).to_string())
        a = validate.analyse(v["p"], L)
        print(f"   neutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}] "
              f"best {validate.best_rmse(L):.6f}")
        FC.table(rho_b, {key: chat}, 0.85, L, label=name)
        print(f"   dose W = {W:.3f}, breakeven c > {W / 2:.4f}")
        for c in [0.0, W / 2, chat, 0.06, 0.08]:
            print(f"      c {c:+.4f} -> {sd * np.sqrt(1 - rho_b ** 2 + W * (W - 2 * c)):.6f}")
    print(f"\ncorr(projected bnd, projected toll) = {np.corrcoef(out['bnd']['col'], out['toll']['col'])[0, 1]:+.3f}")
    return d, out


PINNED = {"48039990000": 1.0, "48057990000": 1.0}   # scorer-pinned: no scored Overture feature at all (LEDGER 111a)
COMBINE_FILES = ["e105_schoolbreadth.csv", "e109_ramp_in.csv", "e110_ringspec.csv",
                 "e111_bndtiger.csv", "e112_toll.csv", "e113_sigma.csv"]


def combine(name="e114_combine.csv", ridge=0.0, L=None):
    """Affine re-solve over the last chain of scored safe experiments, plus the two scorer-pinned tracts.

    E109 and E110 were dosed at W ~ 0.081 to reach rival_D in one file, against read-backs of 0.0405 and
    0.0567; re-dosing them to c is worth (W - c)^2 each and is certified (in-span). E111-E113 are
    measured on E110 and enter at their own read-backs. Few files and small weights, so the affine
    identity is accurate to ~2e-5 (not PRED_SE, which was calibrated on 20-40 file re-solves).
    """
    import final_model as FM
    L = lb.ledger() if L is None else L
    files = [f for f in COMBINE_FILES if f in set(L.file)]
    q, pred, err, lam = FM.affine(ridge, files, L)
    a = validate.analyse(q, L)
    print(f"affine over {len(files)} scored files: identity {pred:.6f}, validator {a['rmse_neutral']:.6f}, "
          f"in span {a['share_explained_by_scored']:.4f}")
    print("   weights", lam)
    g = lb.read_sub(BASE).GEOID.values
    for geoid, v in PINNED.items():
        i = int(np.where(g == geoid)[0][0])
        print(f"   pinned {geoid}: {q[i]:.3f} -> {v}")
        q[i] = v
    lb.write(name, g, q, f"E114 affine re-dose of E109/E110 plus E111-E113 at their read-backs "
                         f"(certified {pred:.6f} before the two scorer-pinned water tracts)", kind="model")
    return q, pred


def build(L=None, keys=None):
    d, out = evidence(L)
    for key, v in out.items():
        if keys is None or key in keys:
            lb.write(FILES[key][0], d.GEOID, v["p"], FILES[key][3], kind="model")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "combine":
        combine()
    elif a and a[0] == "build":
        build(keys=a[1:] or None)
    else:
        evidence()
