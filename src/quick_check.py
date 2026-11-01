"""Check Part A's key numbers in seconds, without running the simulator.

Reads only the package's per-region strata tables, our published per-tract results (docs/writeup/tract_results.csv,
written by `writeup_numbers.py scorecard`) and our final submission, and recomputes: the bookkeeping share of the road
gap (Section 3), the region and urbanity table (Table 2), the worst-tenth worklist and the hidden-tract profile
(Table 4), the scorecard ratios as scored and counted once (Section 4.1) and the Census and FEMA comparison of Section
4.5. "As scored" must equal the card Zindi's API printed for the final. With --geometry it also re-measures the Yuba
City worked example from Overture's roads.

    python src/download.py --strata        # a few MB: sample submissions and strata tables
    python src/quick_check.py              # seconds
    python src/download.py northern-ca     # only for the worked example's geometry
    python src/quick_check.py --geometry
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"
REGIONS = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
COLS = ["GEOID", "ur_class", "tribal_any", "ruca_primary", "svi_overall", "cvi_overall", "usfs_WHP_mean",
        "epht_heat_days_summer", "usdm_winter_dsci", "usdm_summer_dsci", "mtbs_wildfire_ever", "pop_total"]


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'BAD'} {label:58s} {got:10.3f}   (paper {want})")
    return ok


def main():
    s = pd.concat([pd.read_parquet(DATA / "strata" / r / f"{r}-strata-tract-table.parquet", columns=COLS)
                   for r in REGIONS]).drop_duplicates("GEOID")
    t = pd.read_csv(ROOT / "docs" / "writeup" / "tract_results.csv", dtype={"GEOID": str})
    y = pd.read_csv(ROOT / "submissions" / "e136_finalB.csv", dtype={"GEOID": str})
    x = y.merge(s, on="GEOID", how="left").merge(t, on="GEOID", how="left")
    h = x[x.ref_m.fillna(0) > 0]
    oks = []
    print("Section 3: the road gap of road-scored tracts")
    oks.append(check("road-scored tracts", len(h), 6593, 0))
    oks.append(check("mean road gap as scored", h.road_gap_as_scored.mean(), 0.166, 0.0006))
    oks.append(check("mean road gap counted once", h.road_gap_counted_once.mean(), 0.047, 0.0006))
    oks.append(check("share from bookkeeping", 1 - h.road_gap_counted_once.mean() / h.road_gap_as_scored.mean(),
                     0.72, 0.006))
    rural = h.ruca_primary.fillna(1) >= 4
    ok_urb = h[(h.region == "eastern-ok") & ~rural]
    oks.append(check("urban Oklahoma, as scored (Table 2)", ok_urb.road_gap_as_scored.mean(), 0.32, 0.006))
    oks.append(check("urban Oklahoma, counted once (Table 2)", ok_urb.road_gap_counted_once.mean(), 0.09, 0.006))
    big = h.added_by_bookkeeping >= 0.2
    oks.append(check("tracts where bookkeeping adds at least 0.2", big.sum(), 1548, 0))
    print("Section 4: the worst-tenth worklist")
    top_s = h.road_gap_as_scored >= h.road_gap_as_scored.quantile(0.9)
    top_o = h.road_gap_counted_once >= h.road_gap_counted_once.quantile(0.9)
    hid, flag = h[top_o & ~top_s], h[top_s & ~top_o]
    oks.append(check("worst tenth as scored", top_s.sum(), 660, 0))
    oks.append(check("of them, out of the worst tenth when counted once", (top_s & ~top_o).sum(), 321, 0))
    whp = s.usfs_WHP_mean.median()
    for lab, f, want in [("hidden tracts rural (RUCA >= 4)", lambda z: (z.ruca_primary >= 4).mean(), 0.71),
                         ("hidden tracts on tribal land", lambda z: (z.tribal_any > 0).mean(), 0.58),
                         ("hidden tracts above-median wildfire hazard", lambda z: (z.usfs_WHP_mean > whp).mean(), 0.83),
                         ("flagged tracts rural", lambda z: (z.ruca_primary >= 4).mean(), 0.38)]:
        oks.append(check(lab, f(hid if "hidden" in lab else flag), want, 0.006))
    print("Section 4.1: the scorecard on our final, as scored and counted once")
    x["y_once"] = np.where(x.ref_m.fillna(0) > 0, np.clip(x.coverage_gap_score - x.added_by_bookkeeping.fillna(0) / 3,
                                                          0, 1), x.coverage_gap_score)
    want = json.loads((ROOT / "docs" / "writeup" / "numbers.json").read_text())["scorecard"]
    groups = {"rural_vs_urban": (x.ur_class.eq("Rural"), x.ur_class.ne("Rural")),
              "tribal_vs_nontribal": (x.tribal_any.gt(0), x.tribal_any.le(0))}
    for k, c in [("high_svi", "svi_overall"), ("high_cvi", "cvi_overall"), ("wildfire_hazard", "usfs_WHP_mean"),
                 ("summer_heat", "epht_heat_days_summer"), ("winter_drought", "usdm_winter_dsci")]:
        v = x[c]
        ok = v.notna()
        groups[k] = (v.ge(v[ok].median()) & ok, v.lt(v[ok].median()) & ok)
    for k, (a, b) in groups.items():
        for col, key in (("coverage_gap_score", "as_scored"), ("y_once", "counted_once")):
            oks.append(check(f"{k}, {key.replace('_', ' ')}", x[col][a].mean() / x[col][b].mean(), want[k][key], 0.006))
    print("Section 4.5: the same two groups on Census and FEMA data (docs/writeup/external_tracts.csv)")
    e = pd.read_csv(ROOT / "docs" / "writeup" / "external_tracts.csv", dtype={"GEOID": str})
    z = h.assign(top_s=top_s, top_o=top_o).merge(e, on="GEOID", how="left")
    a, b = z[z.top_o & ~z.top_s], z[z.top_s & ~z.top_o]
    high = z.WFIR_RISKR.isin(["Relatively High", "Very High"])
    oks.append(check("hidden tracts: homes that are mobile homes", a.hu_mobile.sum() / a.hu_total.sum(), 0.165, 0.0006))
    oks.append(check("flagged tracts: homes that are mobile homes", b.hu_mobile.sum() / b.hu_total.sum(), 0.093, 0.0006))
    oks.append(check("hidden tracts rated high for wildfire risk", high[a.index].mean(), 0.336, 0.0006))
    oks.append(check("flagged tracts rated high for wildfire risk", high[b.index].mean(), 0.112, 0.0006))
    oks.append(check("high wildfire risk on the worklist as scored", (high & z.top_s).sum(), 110, 0))
    oks.append(check("high wildfire risk on the worklist counted once", (high & z.top_o).sum(), 182, 0))
    if "--geometry" in sys.argv:
        oks.append(geometry())
    print(f"\n{sum(oks)} of {len(oks)} checks match the paper.")
    return all(oks)


def geometry(geoid="06101050503", region="northern-ca"):
    """The worked example: State Route 99 along tract 505.03's east edge, measured from Overture's roads."""
    import geopandas as gpd
    t = gpd.read_parquet(DATA / "strata" / region / f"{region}-census-tracts.parquet", columns=["GEOID", "geometry"])
    tr = t[t.GEOID == geoid].to_crs(5070)
    x0, y0, x1, y1 = t[t.GEOID == geoid].total_bounds
    rd = gpd.read_parquet(DATA / "reference" / region / f"{region}-overture-roads.parquet",
                          columns=["class", "routes", "geometry"], bbox=(x0 - .006, y0 - .006, x1 + .006, y1 + .006))
    rd = rd[rd["class"].isin(["motorway", "trunk", "primary", "secondary"])].to_crs(5070)
    hw = rd[rd.routes.apply(lambda r: r is not None and len(r) > 0)]
    poly = tr.geometry.iloc[0]
    along = hw[hw.distance(poly.boundary) < 10].union_all().intersection(poly.boundary.buffer(5))
    print("Worked example (Yuba City 505.03)")
    return check("State Route 99 along the tract line (m)", along.length, 1643.5, 5)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
