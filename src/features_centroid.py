"""Per-tract road length with segments assigned WHOLE, by centroid, instead of clipped.

Every model since E18 measures the transport numerator as `ST_Length(ST_Intersection(segment,
tract))` -- the segment is cut at the tract boundary and each tract gets its own piece. That is
one of two conventions, and the other is at least as common in practice: assign the **whole**
segment to the tract containing its centroid. The Bias-Discovery writeup on the challenge board
states its own measure that way ("Segments are assigned to a tract by centroid and measured in
EPSG:5070 metres"), which is evidence the convention is in the air for this data.

The two differ most where segments are long relative to tracts -- rural highway, exactly where
the gap is largest -- and the difference is a tract-level property that varies *inside* a
region x rural x burned cell, which is the standing constraint (CLAUDE.md s4, row 29a).

It also changes E19's mechanism rather than refining it: under centroid assignment a boundary
road is not split between two tracts at all, it lands wholly in one, so the 50 m ring that has
carried every model since E19 would be the wrong correction rather than an under-sized one.

Usage:  python src/features_centroid.py [region ...]
Output: $BB_DATA/derived/<region>-centroid.parquet
"""
import sys
import time

import features as F
import features_routes as FR

XY = "'EPSG:4326', 'EPSG:5070', always_xy := true"


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    con.sql(f"""
        CREATE OR REPLACE TABLE cseg AS
        SELECT is_core, is_link, ST_Length(gp) AS L, ST_Centroid(gp) AS c
        FROM (
            SELECT len(list_filter(coalesce(list_transform(routes, x -> x.network), []),
                                   x -> {FR.CORE})) > 0 AS is_core,
                   coalesce(subclass, '') = 'link' AS is_link,
                   ST_Transform(geometry, {XY}) AS gp
            FROM '{F.ref(r, "overture-roads")}' WHERE class IN {FR.NAMED})""")
    con.sql(f"CREATE OR REPLACE TABLE tp AS SELECT GEOID, ST_Transform(geom, {XY}) AS g FROM tracts")
    con.sql("""
        CREATE OR REPLACE TABLE f_c AS
        WITH j AS (SELECT t.GEOID, s.* EXCLUDE (c) FROM tp t JOIN cseg s ON ST_Within(s.c, t.g))
        SELECT GEOID,
               sum(L)                                        AS cn_named,
               sum(L) FILTER (WHERE is_link)                 AS cn_link,
               sum(L) FILTER (WHERE is_core AND NOT is_link) AS cn_core,
               count(*) FILTER (WHERE is_core AND NOT is_link) AS cn_core_n
        FROM j GROUP BY GEOID""")
    df = con.sql("SELECT * EXCLUDE (geom, area_m2) FROM tracts t LEFT JOIN f_c USING (GEOID) "
                 "ORDER BY GEOID").df().fillna(0.0)
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-centroid.parquet", index=False)
    print(f"  {r}: {df.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
