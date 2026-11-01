"""Part A's outside check: who lives in the tracts the bookkeeping swaps, from public data outside the challenge package.

For the Best Bias Discovery analysis only, which the prize form allows. No scored file reads anything here, no build path
imports this module, and `final_build.py` does not touch its output.

    python src/external_context.py fetch   # ~6 min: streams 575 MB, keeps our 9,794 tracts -> docs/writeup/external_*
    python src/external_context.py         # seconds: the comparison -> numbers.json["external"]

Sources:
  Census ACS 2020-2024 5-year detailed tables (public domain), the bulk table-based summary files, which need no key.
  FEMA National Risk Index v1.20, December 2025, FEMA's own ArcGIS service. Not endorsed by FEMA; see FEMA_STATEMENT.

The twelve measures below were fixed, and committed, before any comparison was computed. All twelve are reported.
Groups are the paper's: "hidden" joins the worst tenth of road gap only when each highway is counted once, "flagged" is in
it only as scored. Each measure is a pooled num/den over a group's tracts, so a share of households is a share of the
group's households; a share of tracts has den 1. Intervals are a tract bootstrap, seeded.
"""
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "writeup"
SF = ("https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/5YRData/"
      "acsdt5y2024-{t}.dat")
NRI = ("https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/National_Risk_Index_Census_Tracts/"
       "FeatureServer/0/query")
FEMA_STATEMENT = ("This product uses the Federal Emergency Management Agency's National Risk Index dataset API or "
                  "downloadable datasets but is not endorsed by FEMA. The Federal Government or FEMA cannot vouch for "
                  "the data or analyses derived from these data after the data have been retrieved from the Agency's "
                  "website(s).")
STATES = ("AZ", "CA", "OK", "TX", "WA")
HEADERS = {"User-Agent": "phantom-road-gaps/1.0"}
ACS = {"B25044": {"hh_total": ["001"], "hh_no_vehicle": ["003", "010"]},             # owner + renter, no vehicle
       "B25024": {"hu_total": ["001"], "hu_mobile": ["010"]},
       "B01001": {"pop_acs": ["001"], "pop_65": ["020", "021", "022", "023", "024", "025",
                                                 "044", "045", "046", "047", "048", "049"]},
       "B18101": {"dis_univ": ["001"], "pop_dis": ["004", "007", "010", "013", "016", "019",
                                                   "023", "026", "029", "032", "035", "038"]},
       "B28002": {"net_univ": ["001"], "hh_no_internet": ["013"]},
       "B17001": {"pov_univ": ["001"], "pop_poor": ["002"]}}
NRI_FIELDS = ["TRACTFIPS", "NRI_VER", "POPULATION", "RISK_RATNG", "WFIR_RISKR", "WFIR_RISKS", "IFLD_RISKR", "EAL_VALT",
              "SOVI_SCORE", "RESL_SCORE"]
HIGH = ("Relatively High", "Very High")
MEASURES = [("no_vehicle", "Households with no vehicle", "hh_no_vehicle", "hh_total"),   # key, label, numerator, denominator
            ("mobile_home", "Homes that are mobile homes", "hu_mobile", "hu_total"),
            ("age_65", "People aged 65 or over", "pop_65", "pop_acs"),
            ("disability", "People with a disability", "pop_dis", "dis_univ"),
            ("no_internet", "Households with no internet access", "hh_no_internet", "net_univ"),
            ("poverty", "People below the poverty line", "pop_poor", "pov_univ"),
            ("wildfire_high", "FEMA wildfire risk high (tracts)", "wfir_high", "one"),
            ("flood_high", "FEMA inland flood risk high (tracts)", "ifld_high", "one"),
            ("risk_high", "FEMA overall risk high (tracts)", "risk_high", "one"),
            ("resilience", "FEMA community resilience score (mean)", "RESL_SCORE", "one"),
            ("social_vuln", "FEMA social vulnerability score (mean)", "SOVI_SCORE", "one"),
            ("loss_per_person", "FEMA expected annual loss per person ($)", "EAL_VALT", "POPULATION")]
BOOT, SEED = 2000, 20261101


def _open(url, tries=4):
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=180)
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def _acs(table, parts, geoids):
    """One bulk table, streamed: keeps our tracts and sums each column group; nothing is written to disk."""
    with _open(SF.format(t=table.lower())) as r:
        rows = io.TextIOWrapper(r, encoding="utf-8", errors="replace")
        head = next(rows).rstrip("\r\n").split("|")
        ix = {k: [head.index(f"{table}_E{c}") for c in cs] for k, cs in parts.items()}
        out = {}
        for ln in rows:
            if ln.startswith("1400000US") and ln[9:20] in geoids:
                p = ln.rstrip("\r\n").split("|")
                out[ln[9:20]] = {k: sum(float(p[i]) if p[i] not in ("", "null") else np.nan for i in c)
                                 for k, c in ix.items()}
    print(f"  {table}: {len(out)} tracts")
    return pd.DataFrame.from_dict(out, orient="index")


def _nri():
    """Every tract of our five states from FEMA's service, attributes only, 2,000 rows a page."""
    rows, at = [], 0
    states = ",".join(f"'{s}'" for s in STATES)
    while True:
        q = urllib.parse.urlencode({"where": f"STATEABBRV IN ({states})", "outFields": ",".join(NRI_FIELDS), "f": "json",
                                    "returnGeometry": "false", "orderByFields": "TRACTFIPS", "resultOffset": at,
                                    "resultRecordCount": 2000})
        with _open(f"{NRI}?{q}") as r:
            d = json.load(r)
        got = [f["attributes"] for f in d["features"]]
        rows += got
        at += len(got)
        if not got or not d.get("exceededTransferLimit"):
            break
    print(f"  NRI: {len(rows)} tracts in {','.join(STATES)}")
    return pd.DataFrame(rows).rename(columns={"TRACTFIPS": "GEOID"}).set_index("GEOID")


def fetch():
    t = pd.read_csv(OUT / "tract_results.csv", dtype={"GEOID": str}, usecols=["GEOID"])
    y = pd.read_csv(ROOT / "submissions" / "e136_finalB.csv", dtype={"GEOID": str}, usecols=["GEOID"])
    ids = sorted(set(y.GEOID))
    print(f"{len(ids)} scored tracts ({len(t)} in tract_results.csv)")
    x = pd.DataFrame(index=ids)
    x = x.join(pd.concat([_acs(k, v, set(ids)) for k, v in ACS.items()], axis=1)).join(_nri())
    x.index.name = "GEOID"
    x.to_csv(OUT / "external_tracts.csv")
    (OUT / "external_sources.json").write_text(json.dumps({
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "acs": {"name": "American Community Survey 2020-2024 5-year detailed tables, table-based summary file",
                "tables": list(ACS), "url": SF.format(t="<table>"), "licence": "US Census Bureau, public domain"},
        "nri": {"name": "FEMA National Risk Index, census tracts", "version": str(x.NRI_VER.dropna().iloc[0]),
                "url": NRI, "statement": FEMA_STATEMENT}}, indent=1))
    print(f"{len(x)} rows; ACS matched {x.hh_total.notna().sum()}, NRI matched {x.POPULATION.notna().sum()}")


def table():
    """The tract table with every numerator and denominator of MEASURES, plus the two groups."""
    e = pd.read_csv(OUT / "external_tracts.csv", dtype={"GEOID": str})
    t = pd.read_csv(OUT / "tract_results.csv", dtype={"GEOID": str})
    h = t[t.ref_m.fillna(0) > 0].merge(e, on="GEOID", how="left")
    h["one"] = 1.0
    for k, c in [("wfir_high", "WFIR_RISKR"), ("ifld_high", "IFLD_RISKR"), ("risk_high", "RISK_RATNG")]:
        h[k] = h[c].isin(HIGH).astype(float).where(h[c].notna())
    h["top_s"] = h.road_gap_as_scored >= h.road_gap_as_scored.quantile(0.9)
    h["top_o"] = h.road_gap_counted_once >= h.road_gap_counted_once.quantile(0.9)
    return h


def _stat(g, num, den):
    return float(g[num].sum() / g[den].sum())


def compare(h):
    hid, flag = h[h.top_o & ~h.top_s], h[h.top_s & ~h.top_o]
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, n, (BOOT, n)) for n in (len(hid), len(flag))]
    weights = hid.region.value_counts(normalize=True)
    out = {"tracts": {"hidden": len(hid), "flagged": len(flag)}, "measures": {}}
    for key, label, num, den in MEASURES:
        a, b = hid.dropna(subset=[num, den]), flag.dropna(subset=[num, den])
        r = {"label": label, "hidden": _stat(a, num, den), "flagged": _stat(b, num, den),
             "all_road_scored": _stat(h.dropna(subset=[num, den]), num, den),
             "tracts_used": {"hidden": len(a), "flagged": len(b)}}
        d = []
        for g, ix in ((hid, idx[0]), (flag, idx[1])):
            n, m = g[num].to_numpy(float), g[den].to_numpy(float)
            ok = ~(np.isnan(n) | np.isnan(m))                     # a missing value drops that tract from that draw
            d.append(np.where(ok, n, 0)[ix].sum(1) / np.where(ok, m, 0)[ix].sum(1))
        r["diff_ci95"] = [float(v) for v in np.percentile(d[0] - d[1], [2.5, 97.5])]
        ratio = d[0] / d[1]
        r["ratio"], r["ratio_ci95"] = r["hidden"] / r["flagged"], [float(v) for v in np.percentile(ratio, [2.5, 97.5])]
        by = [(w, _stat(b[b.region == k], num, den)) for k, w in weights.items() if (b.region == k).any()]
        r["flagged_at_hidden_region_mix"] = sum(w * v for w, v in by) / sum(w for w, _ in by)
        draws = np.zeros((BOOT, 2))                               # the same contrast inside each region: a region's
        for k, w in weights.items():                              # tracts are redrawn within it, weights held at hidden's
            for j, g in enumerate((hid, flag)):
                z = g[(g.region == k) & g[num].notna() & g[den].notna()]
                ix = rng.integers(0, len(z), (BOOT, len(z)))
                draws[:, j] += w * z[num].to_numpy(float)[ix].sum(1) / z[den].to_numpy(float)[ix].sum(1)
        r["ratio_within_region"] = float(np.median(draws[:, 0] / draws[:, 1]))
        r["ratio_within_region_ci95"] = [float(v) for v in np.percentile(draws[:, 0] / draws[:, 1], [2.5, 97.5])]
        out["measures"][key] = r
    count = {"hh_no_vehicle": "households with no vehicle", "hu_mobile": "mobile homes", "pop_65": "people aged 65+",
             "pop_dis": "people with a disability", "hh_no_internet": "households with no internet", "pop_poor":
             "people below the poverty line", "pop_acs": "people"}
    out["counts"] = {lab: {"hidden": int(hid[c].sum()), "flagged": int(flag[c].sum())} for c, lab in count.items()}
    out["worklist"] = {k: {"road_scored": int(h[k].sum()), "on_list_as_scored": int(h[k][h.top_s].sum()),
                           "on_list_counted_once": int(h[k][h.top_o].sum())}
                       for k in ("wfir_high", "ifld_high", "risk_high")}
    out["spearman_wildfire"] = {c: float(h[c].corr(h.WFIR_RISKS, method="spearman"))
                                for c in ("road_gap_as_scored", "road_gap_counted_once")}
    out["spearman_wildfire_within_region"] = {c: float(h.groupby("region").apply(
        lambda g: g[c].corr(g.WFIR_RISKS, method="spearman"), include_groups=False).mean())
        for c in ("road_gap_as_scored", "road_gap_counted_once")}
    out["regions"] = {"hidden": {k: int(v) for k, v in hid.region.value_counts().items()},
                      "flagged": {k: int(v) for k, v in flag.region.value_counts().items()}}
    return out


def main():
    out = compare(table())
    out["sources"] = json.loads((OUT / "external_sources.json").read_text())
    N = json.loads((OUT / "numbers.json").read_text())
    N["external"] = out
    (OUT / "numbers.json").write_text(json.dumps(N, indent=1))
    print(f"{'measure':42s} {'hidden':>9s} {'flagged':>9s} {'ratio':>6s} {'95% CI':>14s}   {'within region':>13s} {'95% CI':>14s}")
    for r in out["measures"].values():
        print(f"{r['label']:42s} {r['hidden']:9.3f} {r['flagged']:9.3f} {r['ratio']:6.2f} "
              f"[{r['ratio_ci95'][0]:.2f}, {r['ratio_ci95'][1]:.2f}]   {r['ratio_within_region']:13.2f} "
              f"[{r['ratio_within_region_ci95'][0]:.2f}, {r['ratio_within_region_ci95'][1]:.2f}]")
    print(json.dumps({k: out[k] for k in ("tracts", "regions", "counts", "worklist", "spearman_wildfire",
                                          "spearman_wildfire_within_region")}, indent=1))


if __name__ == "__main__":
    {"fetch": fetch}.get((sys.argv[1:] or [""])[0], main)()
