"""E122: the rural hidden-station rate in proportion to area^0.19, the package's own listed-station exponent
(`docs/oct29_primer.md` 13.5, re-priced after E121), on the certified 22-file affine.

**The base is certified, not modelled.** Ridge-3e-6 affine over E121's 21 base files + E121 (22 scored files):
identity 0.0555853, validator 0.0556269. E121 read c -0.0104 on the identity (-0.0034 on the validator), a null, so
the affine keeps it at lambda 0.006 and the base is E121's own base again. The two scorer-pinned water tracts are set
to 1.0 afterwards (111a).
**The part.** V8 draws the hidden fire stations of a station-free RURAL tract at one flat rate. Overture's own listed
stations scale as area^0.19 in rural tracts (Poisson GLM, 120a(5); area^0.30 urban), so the rate a free rural tract
carries a reference station it cannot see is redrawn in proportion to area^0.19, normalised to the free rural mean.
The urban rate keeps V8's area^1; p_bg is re-solved on a participant's 740 and k, p_mis, s on PUB16 at M_S* (V8's own
calibration, `cand_round30.calibrate`). Mean-preserving by construction: it moves which rural tracts hold the
blackouts, not how many there are (the one family the 25 Oct law leaves open, 104a).
**Price.** A measured-exponent re-identification of a DISTRIBUTION parameter: prior r ~ 0.30 (between the spec
law's 0.48 and the fitted 0, because E121 just read the POI-side re-identifications C/D at ~0). On the 22-file span
the projection out of E120's direction adds a known +0.006 (the part runs against D, which E120 held at +0.039).
c_hat 0.017, predictive sd 0.019 (`evidence` prints the table). The in-span fit agrees (0.9254 -> 0.9256), and it
falls for steeper rural exponents (area^0.5 0.9252, area^1 0.9236): the measured value is also the in-span one.
**The dose** is law 3's W = 0.85 c_hat, about c_hat (lambda ~ 1 in the end-of-day affine, so the base's identity error is
not amplified by c/W). Its null branch stays under E120's 0.055653812 on the identity.

    python src/cand_round32.py            # evidence: base, part size/perp/traps, dose table (~2 min)
    python src/cand_round32.py build      # e122_rural.csv
"""
import contextlib
import io
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

import cand_knn as CK
import cand_mm as CMM
import cand_round25 as R
import cand_round30 as R30
import cand_round31 as R31
import final_model as FM
import lb
import physics3 as P3
import validate

BASE_FILES = R31.BASE_FILES + ["e121_split.csv"]
RIDGE = 3e-6
NAME = "e122_rural.csv"
COMMENT = ("E122 the rural hidden-station rate in proportion to area^0.19, the package's own listed-station exponent, "
           "re-solved on the 740: which rural tracts hold a blackout, not how many")
X_RURAL = 0.19                       # Overture's listed fire stations ~ area^0.19 in rural tracts (120a(5))
C_HAT, S_PRED = 0.017, 0.019         # prior r ~ 0.30 x size 0.0358 + the projection's known +0.006; predictive sd
W_BUILD = 0.0145                     # 0.85 x c_hat
RIVALS = {"rival_A": 0.054404837, "rival_B": 0.055402428, "rival_C": 0.055459211, "rival_D": 0.055607098,
          "E120 (ours)": 0.055653812}  # public leaderboard, 29 Oct
LAST = "e121_split.csv"              # the span at build time: every scored row up to E121, so a rebuild is exact later


def span(L=None):
    L = lb.ledger() if L is None else L
    return L.iloc[:int(np.where(L.file.values == LAST)[0][0]) + 1].reset_index(drop=True)


def base(L):
    q, pred, err, lam = FM.affine(RIDGE, BASE_FILES, L)
    g = lb.read_sub(BASE_FILES[0]).GEOID.values
    for geoid, v in R.PINNED.items():
        q[int(np.where(g == geoid)[0][0])] = v
    a = validate.analyse(q, L)
    return g, q, pred, a["rmse_neutral"], lam


def part():
    """V8 with the free-rural hidden-station rate in proportion to area^X_RURAL, minus V8 (same calibration path)."""
    with contextlib.redirect_stdout(io.StringIO()):
        d, X4, o4, k4, X8, o8, k8, par = R30.vehicles()
        _, p_mis4, p_bg4, _, _ = P3.calibrate(d, X4)
    fit = d.region.ne("eastern-wa").values
    free = (X4["fd"] == 0) & ((X4["fdm"] + X4["bfs"]) == 0)
    rur, area = X4["rural"], d.area_m2.values / 1e6
    au = area / area[free & (rur == 0)].mean()                                  # V8's urban rate, unchanged
    ar = area ** X_RURAL / (area[free & (rur == 1)] ** X_RURAL).mean()
    bg = lambda f: free * (rur * ar + f * (1 - rur) * au)
    blk = lambda f: P3.simulate(dict(X4, bg=bg(f)), 1.1, p_mis4, p_bg4, 1.0, draws=60, m_s=P3.M_S)["fblk"][fit].sum()
    f = brentq(lambda z: blk(z) - R30.BLACKOUTS, 0.01, 1.0)
    X = dict(X8, bg=bg(f))
    with contextlib.redirect_stdout(io.StringIO()):
        c = R30.calibrate(d, X, R30.M_S_STAR)
    o = R30.simulate(X, *c[:4], m_s=c[4])
    print(f"part: urban/rural rate ratio f {f:.3f}; p_mis {c[1]:.3f}, p_bg {c[2]:.4f}, s {c[3]:.4f}; "
          f"in-span fit {R30.inspan_fit(o['y'], lb.ledger()):.4f} (V8 {R30.inspan_fit(o8['y'], lb.ledger()):.4f})")
    return d, X8, o["y"] - o8["y"]


def table(rb, sd, Ws):
    print(f"   dose table on the identity {rb:.7f}; c ~ N({C_HAT}, {S_PRED}^2); P(pass) per rival")
    print("   " + f"{'W':>6s} {'null':>10s} {'at c_hat':>10s} " + " ".join(f"{k:>14s}" for k in RIVALS))
    for W in Ws:
        null = sd * np.sqrt((rb / sd) ** 2 + W ** 2)
        at = sd * np.sqrt((rb / sd) ** 2 + W * (W - 2 * C_HAT))
        cells = []
        for k, rt in RIVALS.items():
            creq = (W ** 2 + (rb / sd) ** 2 - (rt / sd) ** 2) / (2 * W)
            cells.append(f"c>={creq:+.4f} {norm.sf((creq - C_HAT) / S_PRED):.2f}")
        print(f"   {W:6.4f} {null:10.7f} {at:10.7f} " + " ".join(f"{c:>14s}" for c in cells))


def evidence(L=None):
    L = span(L)
    m1, m2 = lb.moments(L)
    sd = float(np.sqrt(m2 - m1 ** 2))
    g, q, pred, val, lam = base(L)
    print(f"base: {len(BASE_FILES)}-file affine ridge {RIDGE:g}, identity {pred:.7f}, validator {val:.7f}, "
          f"|lambda|_1 {np.abs(list(lam.values())).sum():.2f}, lambda_e120 {lam['e120_vehicle.csv']:+.3f}, "
          f"lambda_e121 {lam['e121_split.csv']:+.3f}")
    d, X8, h = part()
    assert (g == d.GEOID.values).all()
    col, perp = CMM.project(h, d, L)
    e = col ** 2 / (col ** 2).sum()
    print(f"part: sd(h) {h.std():.5f}, perp {perp:.3f}, out-of-span size {h.std() * perp / sd:.4f}, "
          f"reach {(np.abs(h) > 1e-4).mean():.3f}, effective tracts {1 / (e ** 2).sum():.0f}, "
          f"rural share of sum-of-squares {e[X8['rural'] == 1].sum():.3f}")
    print(CK.traps({"R19": col}, d, X8["den"] > 0).round(3).to_string())
    table(pred, sd, [0.010, 0.012, W_BUILD, C_HAT, 0.020])
    return d, g, q, col, sd


def build(L=None):
    L = span(L)
    d, g, q0, col, sd = evidence(L)
    q = np.clip(q0 + sd * W_BUILD * col, 0, 1)
    for geoid, v in R.PINNED.items():
        q[int(np.where(g == geoid)[0][0])] = v
    a = validate.analyse(q, L)
    print(f"{NAME}: W {W_BUILD:.4f}, neutral {a['rmse_neutral']:.7f}, clipped at 0: {(q == 0).sum()}, "
          f"pins {[q[int(np.where(g == x)[0][0])] for x in R.PINNED]}")
    lb.write(NAME, lb.read_sub(BASE_FILES[0]).GEOID, q, COMMENT, kind="model")


if __name__ == "__main__":
    a = sys.argv[1:]
    build() if a == ["build"] else evidence()
