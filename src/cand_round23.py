"""E109: the scorer filters Overture on CLASS only. A motorway ramp is class 'motorway'.

Two organiser documents define the road numerator, and neither mentions `subclass`:

    README, "Format & conventions":
      "The road gap compares named-highway classes only: TIGER/Line S1100 + S1200 against
       Overture motorway / trunk / primary / secondary."
    Zindi Info page, "Evaluation":
      "TIGER MTFCC in (S1100, S1200) against Overture class in (motorway, trunk, primary,
       secondary)."

Overture files a motorway ramp as `class='motorway'`, `subclass='link'`, and the shipped
`overture-roads` layer keeps it ("Drivable road segments ... motorway, trunk, primary, secondary,
tertiary, residential, unclassified"). So a ramp IS in the scorer's numerator. E36 took it OUT --
`cand_ramp.numerator = rd_len_named - alpha * q_link_named` at alpha = 1 -- on the argument that
TIGER files a ramp as S1630, which is not in S1100+S1200. That argument is about the REFERENCE, and
it is correct: TIGER has no ramp. But the fix was applied to the NUMERATOR, which models a scorer
that excludes ramps. The spec says the scorer does not. The asymmetry is real and it belongs where
the scorer puts it: ramp length in the numerator, no ramp in the reference.

    Overture class motorway|trunk|primary|secondary   192,503 km   <- the spec numerator
      ... minus ramps, which is what we feed physics    180,056 km
      ... ramps (q_link_named), 6.5% of class length     12,448 km   <- restored here

E36 read +0.0855, so this is deliberately NOT a sign-flip of a confirmed mechanism: E36's scored
file stays in the span and its direction is held. Measured, this part's projected correlation with
e36_ramp.csv is **+0.034** -- it is a new direction, not the negative of that one, because E36 also
re-identified the displacement width w at 15.49 m and most of its content is the ring.

ONE change, nothing else touched (E100's rule: never move the vehicle and the operating point in
one file). The ring stays at E36's w = 15.49 m, which E19's mechanism identifies physically as half
a road width; k re-fits per region to the same published 16 composite cell means. The defined mask
is `den > 0` and is untouched, so the four published undefined rates -- which the Info page now
states per region (OK 21%, CA 37%, TX 28%, AZ 55%) -- cannot move.

THE UNFITTED CHECK, and it is the project's longest-standing miss. Nothing here is fitted to the
published transport burned/unburned ratio 1.87. `physics3`'s own docstring has flagged the
transport half since v2 and nothing has touched it since:

    v4 (ramps out)      burned/unburned 1.597    15% low
    ramps restored      burned/unburned 1.802     3.6% low    <- one change, unfitted

STATED AGAINST IT. The composite rural/urban drifts 2.965 -> 3.228 against a published 2.96 (9%
high), where v4 matches it exactly. Pinning the ring on rural/urban instead (w = 40 m) recovers
2.963 but puts burned/unburned back to 1.618, so the two published numbers disagree about w once
ramps are in; w is held at 15.49 m on mechanism rather than re-fitted to either. Second: per law
98b -- now seven confirmations, E108's card matched the target's on 8 of 8 rows and read -0.001 --
the burned/unburned improvement is reported as evidence the level structure is not broken, NEVER as
support for the ordering or for the dose.

    python src/cand_round23.py          # evidence, traps, perp, dose table
    python src/cand_round23.py build    # e109_ramp_in.csv
"""
import sys

import numpy as np

import physics4 as P4          # patches physics3 into v4
import physics3 as P3
import strata as CC
import cand_knn as CK
import cand_mm as CMM
import cand_ramp as CP
import forecast as FC
import lb
import validate

BASE = "e105_schoolbreadth.csv"   # the best SCORED file (0.057353743); never build on an unscored base
NAME = "e109_ramp_in.csv"
NOTE = ("E109 the scorer filters Overture on class only and a motorway ramp is class motorway: "
        "the 6.5% of named-class length E36 removed, restored to the numerator where the spec puts it")
W, CHAT, M = 0.0808, 0.05, 0.85
SIBS = ["e36_ramp.csv", "e19_road_boundary.csv", "e34_ring27.csv", "e67_unrouted.csv",
        "e102_recall.csv", "e107_region.csv", "e105_schoolbreadth.csv", "e99_joint.csv",
        "e97_named.csv", "e108_juris.csv", "e106_five.csv"]


def physics_y(alpha=1.0, quiet=False):
    """v4's y; alpha = 1 removes ramps from the numerator (v4), alpha = 0 restores them (the spec)."""
    d, X = P4.inputs()
    if alpha != 1.0:
        X = dict(X)
        X["num"] = CP.numerator(alpha=alpha)
    k, p_mis, p_bg, s, ms = P3.calibrate(d, X)
    o = P3.simulate(X, k, p_mis, p_bg, s, m_s=ms)
    if not quiet:
        rural, burn = CC.strata(d)
        fit, dT, T = d.region.ne("eastern-wa").values, X["den"] > 0, o["T"]
        print(f"   burned/unburned {T[dT & fit & burn].mean() / T[dT & fit & ~burn].mean():.3f} (pub 1.87, UNFITTED)"
              f" | transport mean {T[dT & fit].mean():.4f} | composite rural/urban "
              f"{o['y'][fit & rural].mean() / o['y'][fit & ~rural].mean():.3f} (pub 2.96) | composite mean "
              f"{o['y'][fit].mean():.4f} (pub 0.0585) | undefined {1 - dT.mean():.3f}")
    return d, o, X


def candidates(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = np.sqrt(m2 - m1 ** 2)
    print("v4 reference (ramps OUT of the numerator, E36's alpha = 1):")
    d, o0, X = physics_y(1.0)
    print("spec numerator (ramps RESTORED, alpha = 0):")
    _, o1, _ = physics_y(0.0)
    h = o1["y"] - o0["y"]
    col, perp = CMM.project(h, d, L)
    base = lb.read_sub(BASE).coverage_gap_score.values
    p = np.clip(base + sd * W * col, 0, 1)
    rho_b = lb.rho(float(L[L.file == BASE].rmse.iloc[0]), base, m1, m2)
    return d, X, h, col, perp, p, rho_b


def evidence(L=None):
    L = lb.ledger() if L is None else L
    d, X, h, col, perp, p, rho_b = candidates(L)
    _, dfn, _ = CMM.hypothesis(d)
    ramp = CP.quality_table().q_link_named.values
    full = CP.numerator(alpha=0.0)
    print(f"\nramp length {ramp.sum() / 1e3:,.0f} km of {full.sum() / 1e3:,.0f} km class length "
          f"({ramp.sum() / full.sum():.1%}); present in {(ramp > 0).mean():.1%} of tracts")
    print(f"part: reach {(np.abs(h) > 1e-9).mean():.1%}, mean shift {h.mean():+.5f}, "
          f"mean|h| {np.abs(h).mean():.5f}, max {np.abs(h).max():.4f}, perp {perp:.3f}")
    print(CK.traps({"ramp_in": col}, d, dfn).round(3).to_string())
    print("\nprojected correlation with siblings:")
    for s in SIBS:
        if not (lb.ROOT / "submissions" / s).exists():
            continue
        v = lb.read_sub(s).coverage_gap_score.values
        print(f"   {s:28s} {np.corrcoef(col, CMM.project(v, d, L)[0])[0, 1]:+.3f}")
    a = validate.analyse(p, L)
    print(f"\nneutral {a['rmse_neutral']:.6f} range [{a['rmse_best']:.6f}, {a['rmse_worst']:.6f}] "
          f"best {validate.best_rmse(L):.6f}")
    FC.table(rho_b, {"ramp_in": CHAT}, M, L, label=NAME)
    sd = np.sqrt(lb.moments(L)[1] - lb.moments(L)[0] ** 2)
    print(f"\ndose W = {W:.4f}  breakeven c > {W / 2:.4f}   (rival_D 0.056901223 needs c >= {W:.4f})")
    for c in [0.0, 0.0404, 0.05, 0.06, 0.07, 0.0808, 0.0855, 0.10]:
        v = sd * np.sqrt(1 - rho_b ** 2 + W * (W - 2 * c))
        tag = "  <= BEATS RIVAL_D" if v < 0.056901223 else ("  (new best)" if v < 0.057353743 else "")
        print(f"   c {c:+.4f} -> {v:.6f}{tag}")
    return d, p


def build(L=None):
    d, p = evidence(L)
    lb.write(NAME, d.GEOID, p, NOTE, kind="model")


if __name__ == "__main__":
    build() if "build" in sys.argv[1:] else evidence()
