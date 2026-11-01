"""E35/E36 family: county-numbered roads belong in the transport reference, and the published
undefined rates say they are allowed in.

**The hypothesis (H-C1).** The transport reference is TIGER's primary+secondary road file. TIGER's
own definition of MTFCC S1200 is "main arteries, usually in the U.S. Highway, State Highway **or
County Highway** system". Every road model we have built since E19 uses only the first two:
`features.py`'s `core` regex matches `US:I`, `US:US` and `US:<ST>` and their Business/Loop/Spur
variants, and drops everything else. Overture carries the county systems explicitly and at scale --
`US:OK:Washita` (2,351 segments), `US:OK:Garfield`, `US:OK:Rogers`, `US:TX:Brazoria` and ~60 more
-- 3.6 km per tract on average against 10.5 km of core route. They were excluded by a regex, never
by a measurement.

**Why this is a WITHIN-CELL hypothesis** (the constraint from CLAUDE.md s4). County routes are
present in 34% of eastern-ok tracts, 5% of south-central-tx and ~2% elsewhere. They therefore
re-order tracts *inside* region x rural x burned cells -- two neighbouring rural Oklahoma tracts,
one crossed by a county highway and one not, move apart. No level shift does that.

**The screen it had to pass first, and did.** Admitting a road family into the reference changes
the undefined mask (`den > 0`), and the organisers' four published "at least one component
undefined" rates are the aggregate that pins the mask. Adding county routes moves them
0.224 -> 0.219 (OK), 0.541 -> 0.530 (AZ), 0.300 -> 0.293 (TX), 0.384 unchanged (CA): max error
against the published 0.21 / 0.37 / 0.55 / 0.28 stays **0.020**, exactly where the core-only mask
sits. The family is admissible on the evidence that admitted the ring.

**And the same screen refutes the obvious companion.** Texas FM/RM roads are TxDOT-maintained
state highways, so they look like an even better S1200 candidate than county roads. Adding them
drops the Texas undefined rate to **0.190** against a published 0.28 -- a 0.090 error, four times
the worst error of any mask we have accepted. FM stays out, which independently re-confirms the
13 Oct note in CLAUDE.md s3.

Two parameters, the same two published numbers E34 used: `k` from the transport mean 0.112, `w`
(ring width) from the burned/unburned ratio 1.87x. The two *levels* and the four undefined rates
are then predictions, not fits.

    python src/cand_county.py          # the signature table for every route family
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

import cand_level as CL
import cand_road as CR
import features as F
import table

BUFFERS = (5, 10, 15, 30, 50)
PUB_TRANSPORT = 0.112
PUB_BURN_RATIO = 1.87
PUB_BURN = (0.194, 0.104)
PUB_UNDEF = {"eastern-ok": 0.21, "northern-ca": 0.37,
             "maricopa-az": 0.55, "south-central-tx": 0.28}
FAMILIES = {
    "core": ["core"],
    "core+county": ["core", "county"],
    "core+fm": ["core", "fm"],
    "core+county+fm": ["core", "county", "fm"],
}


def routes_table():
    rt = pd.concat([pd.read_parquet(F.OUT / f"{r}-routes.parquet") for r in F.REGIONS],
                   ignore_index=True)
    o = table.sample_order()[["GEOID"]]
    rt = o.merge(rt, on="GEOID", how="left").fillna(0.0)
    assert len(rt) == 9794
    return rt


def ring_at(rt, fams, w):
    """Ring length of the chosen families at width w, linear between the measured buffers."""
    R = {b: sum(rt[f"rg{b}_{f}"].values.astype(float) for f in fams) for b in BUFFERS}
    lo = max([b for b in R if b <= w], default=min(R))
    hi = min([b for b in R if b >= w], default=max(R))
    return R[lo] if lo == hi else R[lo] + (w - lo) / (hi - lo) * (R[hi] - R[lo])


def gap_at(rt, num, fams, w, fit=None):
    inside = sum(rt[f"rt_{f}"].values.astype(float) for f in fams)
    den = inside + ring_at(rt, fams, w)
    dfn = den > 0
    sel = dfn if fit is None else (dfn & fit)
    k = brentq(lambda kk: CR.gap(num, den * kk)[sel].mean() - PUB_TRANSPORT, 0.2, 60.0)
    return CR.gap(num, den * k), dfn, k


def width(rt, num, fams, fit, burn):
    def ratio(w):
        g, dfn, _ = gap_at(rt, num, fams, w, fit)
        s = dfn & fit
        return g[s & burn].mean() / g[s & ~burn].mean()
    try:
        return brentq(lambda w: ratio(w) - PUB_BURN_RATIO, 15.0, 50.0)
    except ValueError:
        return None


def evidence(d=None, r=None, rt=None):
    d = CL.table_all() if d is None else d
    r = CR.road_table() if r is None else r
    rt = routes_table() if rt is None else rt
    num = r.rd_len_named.values.astype(float)
    fit = (d.region != "eastern-wa").values
    burn = d.mtbs_wildfire_ever.astype(bool).values
    reg = d.region.values
    print(f"{'family':16s} {'w(m)':>6s} {'k':>7s} {'burned':>7s} {'unburn':>7s} {'ratio':>6s}"
          f"   {'undefined OK/CA/AZ/TX':>25s} {'maxerr':>7s}")
    out = {}
    for lab, fams in FAMILIES.items():
        w = width(rt, num, fams, fit, burn)
        if w is None:
            g, dfn, k = gap_at(rt, num, fams, 15.0, fit)
            s = dfn & fit
            print(f"{lab:16s} {'none':>6s} {k:7.4f} {g[s & burn].mean():7.4f} "
                  f"{g[s & ~burn].mean():7.4f} {g[s & burn].mean() / g[s & ~burn].mean():6.2f}"
                  f"   no width reproduces 1.87x")
            continue
        g, dfn, k = gap_at(rt, num, fams, w, fit)
        s = dfn & fit
        u = {rg: float((~dfn)[reg == rg].mean()) for rg in PUB_UNDEF}
        err = max(abs(u[rg] - PUB_UNDEF[rg]) for rg in PUB_UNDEF)
        print(f"{lab:16s} {w:6.1f} {k:7.4f} {g[s & burn].mean():7.4f} {g[s & ~burn].mean():7.4f} "
              f"{g[s & burn].mean() / g[s & ~burn].mean():6.2f}   "
              + " ".join(f"{u[rg]:5.3f}" for rg in PUB_UNDEF) + f" {err:7.3f}")
        out[lab] = (g, dfn, k, w)
    print(f"{'published':16s} {'':6s} {'':7s} {PUB_BURN[0]:7.3f} {PUB_BURN[1]:7.3f} "
          f"{PUB_BURN_RATIO:6.2f}   " + " ".join(f"{PUB_UNDEF[rg]:5.2f}" for rg in PUB_UNDEF))
    return out


if __name__ == "__main__":
    evidence()
