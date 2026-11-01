"""Concurrent routes and route-plus-street names per tract (E85+), provided data only.

TIGER's road file keeps ONE RECORD PER (edge, name): where US 183 and US 70 share a road, or where US 62
is also "Main Street", the same centreline appears two or three times, and the reference length
S1100+S1200 counts it each time. Overture keeps one segment with a list of routes, so our numerator
counts it once. (Lead: a participant's public method notes, 22 Oct; mechanism checked against Overture
here, where concurrent segments carry e.g. [US:US/183, US:US/70] with primary name "South Locust Street".)

Per tract, length (m) of core-route segments (CORE as in features_routes.py), in the tract:
    cc_core    all core-route length
    cc_multi   sum of len x (distinct core route numbers - 1)   (extra TIGER records from concurrency)
    cc_local   core-route length whose primary name is a local street name (an extra TIGER record)
    cc_named   core-route length whose primary name is a HIGHWAY name carrying none of the segment's own route
               numbers -- an honorific / historic / descriptive highway name ("Pearl Harbor Memorial Highway"
               on I 10, "Eastex Freeway" on US 59/69, "Beeline Highway" on AZ 87). TIGER files that name as its
               own linear feature, so S1100+S1200 holds the centreline once per route number AND once more
               under the honorific. Disjoint from cc_local by construction: cc_local is the primary names that
               do NOT match HWY, cc_named the ones that do but name no ref of their own. (E97)
    cc_inter   core-route length on an Interstate
    cc_oneway  core-route length drawn one-way (a divided highway's two carriageways; TIGER may draw one line)

Usage:  python src/features_concurrent.py [region ...]
Output: $BB_DATA/derived/<region>-concurrent.parquet
"""
import sys
import time

import features as F
from features_routes import CORE

HWY = (r"(?i)(highway|hwy|interstate|^i-? ?[0-9]|^us |^u\.s\.|state (route|road|hwy|highway)|^sh |^s\.h\.|route|"
       r"turnpike|expressway|freeway|parkway|pkwy|tollway|^loop [0-9]|^spur [0-9]|^fm |^rm |^business |^bus |^bypass|"
       r"^ca[- ][0-9]|^sr[- ][0-9]|^az[- ][0-9]|^ok[- ][0-9]|^tx[- ][0-9]|^wa[- ][0-9]|^nm[- ][0-9])")


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    con.sql(f"""
        CREATE OR REPLACE TABLE seg AS
        WITH s AS (
            SELECT names.primary AS nm, geometry,
                   list_distinct(list_transform(list_filter(coalesce(routes, []), rr -> list_transform([rr.network], x -> {CORE})[1]),
                                                r -> r.network || '/' || coalesce(r.ref, ''))) AS refs,
                   len(list_filter(coalesce(routes, []), r -> r.network = 'US:I')) > 0 AS inter,
                   list_bool_or(list_transform(coalesce(access_restrictions, []),
                                               a -> a.access_type = 'denied' AND a.when.heading = 'backward')) AS oneway,
                   list_distinct(list_filter(list_transform(list_filter(coalesce(routes, []), rr -> list_transform([rr.network], x -> {CORE})[1]),
                                                            r -> r.ref), v -> v IS NOT NULL AND v <> '')) AS nums
            FROM '{F.ref(r, "overture-roads")}')
        SELECT geometry, len(refs) AS nref, inter, coalesce(oneway, false) AS oneway,
               nm IS NOT NULL AND NOT regexp_matches(nm, '{HWY}') AS local,
               nm IS NOT NULL AND regexp_matches(nm, '{HWY}')
                   AND len(list_filter(nums, f -> regexp_matches(nm, '(^|[^0-9])' || f || '([^0-9]|$)'))) = 0
                   AS named
        FROM s WHERE len(refs) > 0""")
    con.sql(f"""
        CREATE OR REPLACE TABLE f_cc AS
        WITH x AS (
            SELECT t.GEOID, s.nref, s.inter, s.local, s.named, s.oneway,
                   ST_Length(ST_Transform(ST_Intersection(t.geom, s.geometry), 'EPSG:4326', 'EPSG:5070',
                                          always_xy := true)) AS len
            FROM tracts t JOIN seg s ON ST_Intersects(t.geom, s.geometry))
        SELECT GEOID, sum(len) AS cc_core, sum(len * (nref - 1)) AS cc_multi,
               sum(len) FILTER (WHERE local) AS cc_local, sum(len) FILTER (WHERE named) AS cc_named,
               sum(len) FILTER (WHERE inter) AS cc_inter,
               sum(len) FILTER (WHERE oneway) AS cc_oneway
        FROM x GROUP BY GEOID""")
    df = con.sql("SELECT t.GEOID, f_cc.* EXCLUDE (GEOID) FROM tracts t LEFT JOIN f_cc USING (GEOID) "
                 "ORDER BY GEOID").df().fillna(0)
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-concurrent.parquet", index=False)
    print(f"  {r}: {df.shape} in {time.time() - t0:.0f}s, multi/core {df.cc_multi.sum() / df.cc_core.sum():.3f}, "
          f"local/core {df.cc_local.sum() / df.cc_core.sum():.3f}, "
          f"named/core {df.cc_named.sum() / df.cc_core.sum():.3f}", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
