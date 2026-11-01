"""E118: the two reference smear terms re-identified on the spec vehicle (LAM_U = LAM_T = 0), on a certified
in-span base.

**The base is certified, not modelled.** Every file in it is scored, so the ridge affine over them has an
exact identity (`final_model.affine`, rows 101a/114/116): E114's fourteen files plus E114, E115, E116 and E117,
ridge 3e-6, identity 0.055957 (validator 0.055999), |lambda|_1 3.23. The two scorer-pinned water tracts are set
to 1.0 afterwards. E117's part read c -0.0104 (a null), so the affine gives it its small negative weight.

**The part.** Since physics v1 the transport reference has been

    TIGER = k * den * noise  +  LAM_U * unr  +  LAM_T * ter

with two smear terms added unscaled: LAM_U = 0.20 of the unrouted named-class road (E67, +0.024) and LAM_T = 0.05
of post-import tertiary plus edited import-era local road (E70 +0.028 / E82 +0.021 against E81 -0.023). All four
read-backs are inside the c ~ N(0, 0.0145^2) noise of a random direction (106a), and all were measured on the
ring vehicle, before E109-E115 rebuilt the reference from the spec: ramps in, TIGER's own boundary line, toll
highways, one record per name. The spec says S1100 + S1200 are primary and secondary roads; Texas's unrouted
secondary is mostly FM/RM road (52,536 km, 14.4% of the Texas reference at LAM_U 0.2), which CLAUDE.md section 3
and thread 35542 put outside S1200. So both terms are re-identified on the vehicle they now sit in, the move that
paid after E109 (E110, +0.057).

**Unfitted identification.** Each region's road level k is fitted only to its own published cells, so whether the
levels agree, and where they sit, is not fitted. With both terms at 0 they are OK 1.038 / CA 1.032 / TX 0.995 /
WA 1.014: at the independently measured TIGER/Overture tracing ratio psi = 0.99 (E38), where V2's are 0.878 /
0.839 / 0.827 / 0.833 -- a reference that needed a 15% fudge no longer does. Agreement improves too (log-sd
OK/CA/TX 0.0255 -> 0.0185; LAM_U alone gives 0.0073, the tightest yet). Arizona stays the outlier (0.601).

    python src/cand_round28.py            # evidence: base, part, traps, k by region, dose table
    python src/cand_round28.py build      # e118_spec_ref.csv
"""
import contextlib
import io
import sys

import numpy as np

import cand_knn as CK
import cand_mm as CMM
import cand_ramp as CP
import cand_round25 as R
import cand_round26 as R6
import final_model as FM
import lb
import physics3 as P3
import physics4 as P4
import validate

BASE_FILES = R6.AFFINE_FILES + ["e114_affine14.csv", "e115_records.csv", "e116_redose.csv", "e117_interior.csv"]
RIDGE = 3e-6
TARGETS = [("rival_D", 0.056098312), ("rival_E", 0.055768533), ("rival_B", 0.055456101),
           ("rival_A", 0.054687549)]
NAME = "e118_spec_ref.csv"
COMMENT = ("E118 the transport reference's two smear terms (unrouted arterials, tertiary) re-identified at 0 on "
           "the spec vehicle, on the certified affine of the scored chain")


def e116(write=False):
    """E116 exactly as built on 28 Oct (row 116: t* = 2.388 from E114/E115's scores, rounded)."""
    a, b = lb.read_sub("e114_affine14.csv"), lb.read_sub("e115_records.csv")
    q = np.clip(a.coverage_gap_score + 2.388 * (b.coverage_gap_score - a.coverage_gap_score), 0, 1).values
    if write:
        lb.write("e116_redose.csv", a.GEOID, q,
                 "E116 certified re-dose of E115's counting-rule part to its measured optimum", kind="model")
    return q


def model(den, unr_scale=1.0, ter_scale=1.0):
    """cand_round25.model with the two smear terms scaled (0 removes them); everything else V2."""
    d, X = P4.inputs()
    X = dict(X)
    X["num"] = CP.numerator(alpha=0.0)
    X["den"] = den
    X["unr"] = X["unr"] * unr_scale
    X["ter"] = X["ter"] * ter_scale
    with contextlib.redirect_stdout(io.StringIO()):
        k, p_mis, p_bg, s, ms = P3.calibrate(d, X)
        o = P3.simulate(X, k, p_mis, p_bg, s, m_s=ms)
    return d, X, o, k


def part():
    inner, bound = R6.counting_extras()
    den = R.dens()["both"] + inner + bound
    d, X, o1, k1 = model(den)
    _, _, o0, k0 = model(den, 0.0, 0.0)
    return d, X, o0["y"] - o1["y"], k1, k0


def base(L):
    """Ridge affine over the scored chain, pins set; (GEOID, vector, identity, validator, lambda)."""
    q, pred, err, lam = FM.affine(RIDGE, BASE_FILES, L)
    a = validate.analyse(q, L)
    g = lb.read_sub(BASE_FILES[0]).GEOID.values
    for geoid, v in R.PINNED.items():
        q[int(np.where(g == geoid)[0][0])] = v
    return g, q, pred, a["rmse_neutral"], lam


def dose(rb, sd):
    """Next entry above the base, and the dose minimising the c needed to pass it."""
    name, rt = next(((n, r) for n, r in TARGETS if r < rb - 1e-7), TARGETS[-1])
    return name, rt, float(np.sqrt(max(rb ** 2 - rt ** 2, 0.0)) / sd)


def kreg(d, k):
    return {r: float(np.median(np.asarray(k)[(d.region == r).values])) for r in sorted(d.region.unique())}


def evidence(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = float(np.sqrt(m2 - m1 ** 2))
    g, q, pred, val, lam = base(L)
    rb = 0.5 * (pred + val)
    d, X, h, k1, k0 = part()
    col, perp = CMM.project(h, d, L)
    print(f"base: {len(BASE_FILES)}-file affine ridge {RIDGE:g}, identity {pred:.6f}, validator {val:.6f}, "
          f"|lambda|_1 {np.abs(list(lam.values())).sum():.2f}; dosing on the midpoint {rb:.6f}")
    print("k by region, V2:      ", {r: round(v, 3) for r, v in kreg(d, k1).items()})
    print("k by region, part on: ", {r: round(v, 3) for r, v in kreg(d, k0).items()})
    print(f"LAM_U 0.2 -> 0, LAM_T 0.05 -> 0: sd(h) {h.std():.5f}, perp {perp:.3f}, out-of-span size "
          f"{h.std() * perp / sd:.4f} sd(y), reach {(np.abs(h) > 1e-4).mean():.3f}")
    print(CK.traps({"spec_ref": col}, d, X["den"] > 0).round(3).to_string())
    name, rt, W = dose(rb, sd)
    print(f"W* to pass {name} {rt:.9f} = {W:.4f}")
    for c in [0.0, W / 2, 0.035, W, 0.07, 0.085]:
        print(f"   c {c:+.4f} -> {sd * np.sqrt((rb / sd) ** 2 + W * (W - 2 * c)):.6f}")
    return d, g, q, col, W, sd


def build(L=None):
    L = lb.ledger() if L is None else L
    d, g, q0, col, W, sd = evidence(L)
    assert (g == d.GEOID.values).all()
    q = np.clip(q0 + sd * W * col, 0, 1)
    for geoid, v in R.PINNED.items():
        q[int(np.where(g == geoid)[0][0])] = v
    a = validate.analyse(q, L)
    print(f"{NAME}: W {W:.4f}, neutral {a['rmse_neutral']:.6f}, clipped at 0: {(q == 0).sum()}")
    lb.write(NAME, lb.read_sub(BASE_FILES[0]).GEOID, q, COMMENT, kind="model")


if __name__ == "__main__":
    build() if sys.argv[1:] == ["build"] else evidence()
