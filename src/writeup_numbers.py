"""Every number in the writeup's Part A (the road-reference bookkeeping finding), recomputed from the challenge package.

The road component compares Overture's motorway-secondary length inside a tract with the reference's primary and
secondary road length inside it. Three properties of how the reference is counted, all visible in the package:

  1. a highway that runs ALONG a tract boundary is counted in full inside both tracts it separates (the organisers'
     overlay; tract boundaries are drawn on road centrelines), while Overture's own copy of that highway lies a few
     metres to one side, so one of the two tracts sees a road the map "lacks" (`features_bndfollow`, `features_bfmix`);
  2. the reference keeps one record per route number and per name, so a stretch carrying two routes, or a route that
     is also a named street, is counted twice (`features_concurrent`);
  3. motorway ramps are in Overture's counted classes but not in the reference's (`features_quality`).

Writes docs/writeup/numbers.json and docs/writeup/tracts.csv. Needs the derived feature tables; no removed layer.

    python src/writeup_numbers.py          # ~1 min
"""
import json
import sys

import numpy as np
import pandas as pd

import lb

DERIVED = lb.ROOT.parent / "data" / "derived"
OUT = lb.ROOT / "docs" / "writeup"
REGIONS = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
LABEL = {"eastern-ok": "Oklahoma", "eastern-wa": "Washington", "maricopa-az": "Arizona", "northern-ca": "California",
         "south-central-tx": "Texas"}
SEED = 20261030


def frame():
    d = pd.read_parquet(DERIVED / "all.parquet")[["GEOID", "region", "county", "housing_units", "ur_class",
                                                  "ruca_primary", "tribal_legal_pct", "aiannh_name", "uhe_city_name",
                                                  "area_m2"]]
    for t in ["routes", "bndfollow", "bfmix", "concurrent", "quality"]:
        x = pd.concat([pd.read_parquet(DERIVED / f"{r}-{t}.parquet") for r in REGIONS])
        d = d.merge(x[[c for c in x.columns if c == "GEOID" or c not in d.columns]], on="GEOID", how="left")
    names = pd.concat([pd.read_parquet(lb.ROOT.parent / "data" / "strata" / r / f"{r}-census-tracts.parquet",
                                       columns=["GEOID", "NAMELSAD"]) for r in REGIONS])
    d = d.merge(names, on="GEOID", how="left")
    f = lambda c: d[c].fillna(0).astype(float).values
    d["core_m"] = f("rt_core")                                         # Overture route-tagged highway inside
    d["bnd_m"] = f("bf_out_m")                                         # boundary highway, Overture's copy outside
    d["dup_m"] = (f("cc_multi") + f("bf_x_multi_m") + f("cc_local") + f("bf_x_local_m")
                  + f("cc_named") + f("bf_x_named_m"))                 # extra records: route numbers and names
    d["ramp_m"] = f("q_link_named")
    d["ref_m"] = d.core_m + d.bnd_m + d.dup_m                          # the reference length, as bookkept
    d["book_share"] = np.where(d.ref_m > 0, (d.bnd_m + d.dup_m) / d.ref_m.where(d.ref_m > 0, 1), 0.0)
    d["boundary_only"] = (d.core_m == 0) & (d.bnd_m > 0)
    d["rural"] = d.ruca_primary.fillna(1) >= 4
    return d


def boot(x, stat=np.mean, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    s = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def _calibrated(draws=2000):
    """physics v9b at E127's fitted weights, run once to capture its final calibrated inputs (road scale k, facility
    and business rates), plus the separated road-reference terms and E127's theta."""
    import physics9 as P9
    d, X0, _ = P9.inputs()
    th = json.loads((lb.ROOT / "params" / "physics9_e127_theta.json").read_text())
    seen = {}
    raw = P9.simulate9

    def spy(X, k, p_mis, p_bg, s, draws=400, **kw):          # keep the final call's calibrated inputs
        if draws == 2000:
            seen.update(X=X, args=(k, p_mis, p_bg, s), kw=kw)
        return raw(X, k, p_mis, p_bg, s, draws=draws, **kw)
    P9.simulate9 = spy
    try:
        P9.model(d, X0, th, draws=draws)
    finally:
        P9.simulate9 = raw
    T, den8 = X0["T"], X0["den8"]
    base = T["core"] + th["A_TOLL"] * T["toll"] - th["A_DES"] * T["des"] - th["W_INTER"] * T["inter"] - th["W_DUAL"] * T["oneway"]
    dup = th["A_MULTI"] * T["multi"] + th["A_LOCAL"] * T["local"] + th["A_NAMED"] * T["named"]
    road = lambda v: raw(dict(seen["X"], den=np.where(den8 > 0, np.maximum(v, 0.05 * den8), 0.0)), *seen["args"],
                         draws=draws, **seen["kw"])["T"]
    return d, th, T, base, dup, road


def attribution(draws=2000):
    """The road gap the bookkeeping itself adds to each tract: the calibrated simulator (physics v9b, E127's fitted
    weights) with every term, against the same simulator, same calibration, with the boundary terms or the
    duplicate-record terms switched off. Returns GEOID and the expected road gaps (full, no boundary, no duplicates,
    neither)."""
    d, th, T, base, dup, road = _calibrated(draws)
    bnd = th["A_BND"] * T["bnd"]
    out = {name: road(v) for name, v in [("full", base + bnd + dup), ("no_boundary", base + dup),
                                         ("no_duplicates", base + bnd), ("neither", base)]}
    return pd.DataFrame(dict(GEOID=d.GEOID.astype(str).values, **out))


def sensitivity(draws=2000):
    """How the bookkeeping share of the mean road gap depends on the counting weights: the same calibrated simulator
    with the boundary weight and the duplicate-record weights scaled from their fitted values (0.92 and ~0.54) to
    face value (1, 1), half, and each rule alone. Adds N["sensitivity"] to numbers.json."""
    d, th, T, base, dup, road = _calibrated(draws)
    w = pd.read_parquet(DERIVED / "writeup_tracts.parquet").set_index("GEOID")
    has = (w.ref_m.reindex(d.GEOID.astype(str).values).fillna(0) > 0).values     # main()'s road-scored tracts
    once = road(base)[has].mean()
    rows = {}
    for name, wb, wd in [("fitted", th["A_BND"], 1.0), ("face_value", 1.0, 1.0 / max(th["A_MULTI"], 1e-9)),
                         ("half", th["A_BND"] / 2, 0.5), ("boundary_only", th["A_BND"], 0.0),
                         ("duplicates_only", 0.0, 1.0)]:
        full = road(base + wb * T["bnd"] + wd * dup)[has].mean()
        rows[name] = {"boundary_weight": round(float(wb), 3), "duplicate_scale": round(float(wd), 3),
                      "as_scored": round(float(full), 4), "counted_once": round(float(once), 4),
                      "share_from_bookkeeping": round(float(1 - once / full), 3)}
        print(f"   {name:16s} boundary w {wb:.2f}, duplicates x{wd:.2f}: road gap {full:.3f} vs {once:.3f} counted once, "
              f"share {1 - once / full:.2f}", flush=True)
    N = json.loads((OUT / "numbers.json").read_text())
    N["sensitivity"] = rows
    (OUT / "numbers.json").write_text(json.dumps(N, indent=1))
    return rows


def edge_cases(d):
    """Counts for Part B's edge-case table (the judges' three cases plus ours)."""
    t = pd.concat([pd.read_parquet(lb.ROOT.parent / "data" / "strata" / r / f"{r}-census-tracts.parquet",
                                   columns=["GEOID", "pop_total", "ALAND"]) for r in REGIONS])
    a = pd.read_parquet(DERIVED / "all.parquet")[["GEOID", "bld_all", "poi_all", "rd_len_all", "rd_len_named",
                                                  "housing_units"]]
    x = d[["GEOID", "ref_m", "region"]].merge(t, on="GEOID", how="left").merge(a, on="GEOID", how="left",
                                                                              suffixes=("", "_a"))
    f = lambda c: x[c].fillna(0).astype(float)
    return {"zero_population": int((f("pop_total") == 0).sum()),
            "zero_housing_units": int((f("housing_units") == 0).sum()),
            "zero_land_area": int((f("ALAND") == 0).sum()),
            "no_overture_buildings": int((f("bld_all") == 0).sum()),
            "no_overture_places": int((f("poi_all") == 0).sum()),
            "no_overture_roads": int((f("rd_len_all") == 0).sum()),
            "no_overture_data_at_all": int(((f("bld_all") == 0) & (f("poi_all") == 0) & (f("rd_len_all") == 0)).sum()),
            "road_component_undefined": int((x.ref_m.fillna(0) == 0).sum()),
            "eastern_wa_tracts": int((x.region == "eastern-wa").sum())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = frame()
    pop = pd.concat([pd.read_parquet(lb.ROOT.parent / "data" / "strata" / r / f"{r}-census-tracts.parquet",
                                     columns=["GEOID", "pop_total"]) for r in REGIONS])
    d = d.merge(pop, on="GEOID", how="left")
    a = attribution()
    d = d.merge(a, on="GEOID", how="left")
    d["att_boundary"] = d.full - d.no_boundary
    d["att_duplicates"] = d.full - d.no_duplicates
    d["att_total"] = d.full - d.neither
    cut = pd.concat([pd.read_parquet(DERIVED / f"{r}-bfmix.parquet", columns=["GEOID", "cut_m"]) for r in REGIONS])
    d = d.merge(cut.groupby("GEOID").cut_m.first().rename("cut"), on="GEOID", how="left")
    N = {"n_tracts": int(len(d))}
    km = lambda v: round(float(v) / 1000, 0)
    N["km"] = {"core": km(d.core_m.sum()), "boundary_outside": km(d.bnd_m.sum()), "duplicate_records": km(d.dup_m.sum()),
               "ramps": km(d.ramp_m.sum())}
    has = d.ref_m > 0
    N["tracts_with_reference"] = int(has.sum())
    N["share_reference_bookkept"] = round(float((d.bnd_m.sum() + d.dup_m.sum()) / d.ref_m.sum()), 3)
    reg = {}
    for r in REGIONS:
        m = (d.region == r).values
        h = m & has.values
        bo = d.boundary_only.values & m
        lo, hi = boot(d.boundary_only.values[h].astype(float))
        reg[r] = {"label": LABEL[r], "tracts": int(m.sum()), "with_reference": int(h.sum()),
                  "cut_m": round(float(d.cut[m].iloc[0]), 1),
                  "boundary_only": int(bo.sum()), "boundary_only_share": round(float(bo.sum() / max(h.sum(), 1)), 3),
                  "boundary_only_ci": [round(lo, 3), round(hi, 3)],
                  "boundary_only_homes": int(d.housing_units[bo].fillna(0).sum()),
                  "book_share_mean": round(float(d.book_share[h].mean()), 3),
                  "book_share_ci": [round(x, 3) for x in boot(d.book_share.values[h])],
                  "bookkept_majority": int((d.book_share.values[h] >= 0.5).sum()),
                  "km_boundary_outside": km(d.bnd_m[m].sum()), "km_duplicate": km(d.dup_m[m].sum()),
                  "km_core": km(d.core_m[m].sum()), "km_ramps": km(d.ramp_m[m].sum())}
        for nm, g in [("rural", d.rural.values), ("urban", ~d.rural.values)]:
            hh = h & g
            reg[r][f"book_share_{nm}"] = round(float(d.book_share[hh].mean()), 3) if hh.any() else None
            reg[r][f"boundary_only_share_{nm}"] = (round(float(d.boundary_only.values[hh].mean()), 3)
                                                   if hh.any() else None)
    N["regions"] = reg
    # what the bookkeeping adds to the road gap (calibrated simulator, same calibration with the terms switched off)
    att = {}
    for r in REGIONS:
        for nm, g in [("rural", d.rural.values), ("urban", ~d.rural.values)]:
            h = has.values & (d.region == r).values & g
            big = h & (d.att_total.values >= 0.2)
            att[f"{r}|{nm}"] = {"n": int(h.sum()), "gap_full": round(float(d.full[h].mean()), 3),
                                "gap_without": round(float(d.neither[h].mean()), 3),
                                "added_total": round(float(d.att_total[h].mean()), 3),
                                "added_ci": [round(x, 3) for x in boot(d.att_total.values[h])],
                                "added_boundary": round(float(d.att_boundary[h].mean()), 3),
                                "added_duplicates": round(float(d.att_duplicates[h].mean()), 3),
                                "tracts_added_ge_0.2": int(big.sum()), "homes_added_ge_0.2":
                                    int(d.housing_units[big].fillna(0).sum())}
    N["attribution"] = att
    hv = has.values
    q_full = d.full[hv].quantile(0.9)
    q_wo = d.neither[hv].quantile(0.9)
    top_full = hv & (d.full.values >= q_full)
    top_wo = hv & (d.neither.values >= q_wo)
    N["top_decile"] = {"n": int(top_full.sum()), "leave_without_bookkeeping": int((top_full & ~top_wo).sum()),
                       "homes_in_leavers": int(d.housing_units[top_full & ~top_wo].fillna(0).sum()),
                       "people_in_leavers": int(d.pop_total[top_full & ~top_wo].fillna(0).sum())}
    big = hv & (d.att_total.values >= 0.2)
    N["added_ge_0.2_total"] = {"tracts": int(big.sum()), "homes": int(d.housing_units[big].fillna(0).sum()),
                               "people": int(d.pop_total[big].fillna(0).sum())}
    N["road_gap_mean"] = {"as_scored": round(float(d.full[hv].mean()), 3), "counted_once": round(float(d.neither[hv].mean()), 3),
                          "share_from_bookkeeping": round(float(d.att_total[hv].mean() / d.full[hv].mean()), 3)}
    N["edge_cases"] = edge_cases(d)
    bo = d.boundary_only.values
    N["boundary_only_total"] = int(bo.sum())
    N["boundary_only_homes_total"] = int(d.housing_units[bo].fillna(0).sum())
    N["bookkept_majority_total"] = int((d.book_share.values[has.values] >= 0.5).sum())
    N["bookkept_majority_homes"] = int(d.housing_units[has.values & (d.book_share.values >= 0.5)].fillna(0).sum())
    trib = has.values & (d.tribal_legal_pct.fillna(0).values > 0.5)
    N["tribal"] = {"tracts_with_reference": int(trib.sum()),
                   "book_share_mean": round(float(d.book_share[trib].mean()), 3),
                   "boundary_only": int((bo & trib).sum())}
    # the hidden-target tests: our own submissions that isolated each mechanism (agreement read from the score change)
    L = lb.ledger()
    tests = {"e19_road_boundary.csv": "boundary highways counted inside (first test)",
             "e34_ring27.csv": "boundary displacement width",
             "e36_ramp.csv": "ramps removed from the numerator",
             "e85_multi.csv": "one record per route number",
             "e86_local.csv": "one record per local street name",
             "e97_named.csv": "one record per highway name",
             "e109_ramp_in.csv": "ramps restored (the scorer filters on class)",
             "e110_ringspec.csv": "displacement width re-identified",
             "e120_vehicle.csv": "regional displacement cut, railway boundaries out"}
    N["tests"] = {f: {"what": w, "public_rmse": float(L[L.file == f].rmse.iloc[0])} for f, w in tests.items()
                  if f in set(L.file)}
    (OUT / "numbers.json").write_text(json.dumps(N, indent=1))
    cols = ["GEOID", "region", "county", "NAMELSAD", "aiannh_name", "uhe_city_name", "housing_units", "ur_class",
            "core_m", "bnd_m", "dup_m", "full", "neither", "att_total", "att_boundary", "att_duplicates"]
    ex = d[has & (d.housing_units.fillna(0) > 0)].sort_values("att_total", ascending=False)
    pick = pd.concat([ex[ex.region == r].head(3) for r in REGIONS])
    pick[cols].to_csv(OUT / "tracts.csv", index=False)   # GEOID is a string column in d
    d[["GEOID", "region", "core_m", "bnd_m", "dup_m", "ramp_m", "ref_m", "book_share", "boundary_only", "rural",
       "housing_units", "full", "no_boundary", "no_duplicates", "neither", "att_total", "att_boundary",
       "att_duplicates"]].to_parquet(DERIVED / "writeup_tracts.parquet", index=False)
    print(json.dumps({k: v for k, v in N.items() if k != "tests"}, indent=1))
    print(pick[cols].to_string(index=False))


def scorecard(final="e136_finalB.csv"):
    """The organisers' bias scorecard computed on our final, as scored and with each highway counted once. The scorecard's
    ratios are the mean score of one group over the other (rural vs urban by `ur_class`, tribal vs not, and median splits
    of SVI, CVI, USFS wildfire hazard, EPHT summer heat and USDM winter drought); "as scored" reproduces the card Zindi's
    API prints for the file. Counted once, a road-scored tract's score loses the road gap the bookkeeping adds, divided by
    its three defined components. Adds N["scorecard"] to numbers.json and writes docs/writeup/tract_results.csv."""
    import strata
    d = strata.table_all()
    w = pd.read_parquet(DERIVED / "writeup_tracts.parquet")
    x = d.merge(lb.read_sub(final).rename(columns={"coverage_gap_score": "y"}), on="GEOID")
    x = x.merge(w[["GEOID", "ref_m", "full", "neither", "att_total"]], on="GEOID", how="left")
    has = x.ref_m.fillna(0) > 0
    x["y_once"] = np.where(has, np.clip(x.y - x.att_total.fillna(0) / 3, 0, 1), x.y)
    groups = {"rural_vs_urban": (x.ur_class.eq("Rural"), x.ur_class.ne("Rural")),
              "tribal_vs_nontribal": (x.tribal_any.gt(0), x.tribal_any.le(0))}
    for k, c in [("high_svi", "svi_overall"), ("high_cvi", "cvi_overall"), ("wildfire_hazard", "usfs_WHP_mean"),
                 ("summer_heat", "epht_heat_days_summer"), ("winter_drought", "usdm_winter_dsci"),
                 ("summer_drought", "usdm_summer_dsci")]:
        v = x[c]
        ok = v.notna()
        groups[k] = (v.ge(v[ok].median()) & ok, v.lt(v[ok].median()) & ok)
    out = {k: {"as_scored": round(float(x.y[a].mean() / x.y[b].mean()), 3),
               "counted_once": round(float(x.y_once[a].mean() / x.y_once[b].mean()), 3)} for k, (a, b) in groups.items()}
    h = x[has]
    for k, (a, b) in [("road_rural_vs_urban", (h.ur_class.eq("Rural"), h.ur_class.ne("Rural"))),
                      ("road_tribal_vs_nontribal", (h.tribal_any.gt(0), h.tribal_any.le(0)))]:
        out[k] = {"as_scored": round(float(h.full[a].mean() / h.full[b].mean()), 3),
                  "counted_once": round(float(h.neither[a].mean() / h.neither[b].mean()), 3)}
    N = json.loads((OUT / "numbers.json").read_text())
    N["scorecard"] = dict(out, final=final)
    (OUT / "numbers.json").write_text(json.dumps(N, indent=1))
    cols = ["GEOID", "region", "core_m", "bnd_m", "dup_m", "ref_m", "book_share", "full", "neither", "att_total",
            "att_boundary", "att_duplicates"]
    t = pd.read_parquet(DERIVED / "writeup_tracts.parquet")[cols].rename(
        columns={"full": "road_gap_as_scored", "neither": "road_gap_counted_once", "att_total": "added_by_bookkeeping",
                 "att_boundary": "added_by_boundary", "att_duplicates": "added_by_duplicates"})
    t.round(6).to_csv(OUT / "tract_results.csv", index=False)
    for k, v in out.items():
        print(f"   {k:26s} as scored {v['as_scored']:.2f}   counted once {v['counted_once']:.2f}")
    return out


def displaced():
    """Whose gaps the bookkeeping swaps on a worst-tenth worklist: the tracts that join the worst tenth when each
    highway is counted once (their gap is coverage by the score's own definition) against the tracts they replace.
    Reads main()'s per-tract table, so it needs no simulator run; adds N["displaced"] to numbers.json."""
    import strata
    w = pd.read_parquet(DERIVED / "writeup_tracts.parquet")
    s = strata.table_all()[["GEOID", "pop_total", "ruca_primary", "tribal_any", "usfs_WHP_mean", "mtbs_wildfire_ever"]]
    whp = s.usfs_WHP_mean.median()
    h = w[w.ref_m > 0].merge(s, on="GEOID", how="left")
    top_full = h.full >= h.full.quantile(0.9)
    top_once = h.neither >= h.neither.quantile(0.9)
    out = {}
    for k, m in [("hidden", top_once & ~top_full), ("false_flags", top_full & ~top_once),
                 ("all_road_scored", pd.Series(True, index=h.index))]:
        x = h[m]
        out[k] = {"tracts": int(m.sum()), "people": int(x.pop_total.fillna(0).sum()),
                  "rural": round(float((x.ruca_primary >= 4).mean()), 3),
                  "tribal": round(float((x.tribal_any > 0).mean()), 3),
                  "whp_above_median": round(float((x.usfs_WHP_mean > whp).mean()), 3),
                  "burned_since_1984": round(float(x.mtbs_wildfire_ever.mean()), 3),
                  "bookkeeping_share_median": round(float(((x.bnd_m + x.dup_m) / x.ref_m).median()), 3),
                  "regions": {r: int(v) for r, v in x.region.value_counts().items()}}
    h["rank_scored"] = h.full.rank(ascending=False, method="min").astype(int)
    h["rank_once"] = h.neither.rank(ascending=False, method="min").astype(int)
    ex = h[top_once & ~top_full].sort_values("neither", ascending=False).head(6)
    out["named"] = [{"GEOID": g, "rank_as_scored": int(a), "rank_counted_once": int(b), "people": int(p)}
                    for g, a, b, p in zip(ex.GEOID, ex.rank_scored, ex.rank_once, ex.pop_total.fillna(0))]
    N = json.loads((OUT / "numbers.json").read_text())
    N["displaced"] = out
    (OUT / "numbers.json").write_text(json.dumps(N, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"displaced": displaced, "scorecard": scorecard, "sensitivity": sensitivity}.get((sys.argv[1:] or [""])[0], main)()
