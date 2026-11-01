"""Per-tract GEOMETRIC DETAIL of the core-route road family.

Why. The transport numerator is Overture's own length, so the only remaining error in the
transport component is our estimate of the *reference* -- TIGER primary+secondary length per
tract (CLAUDE.md s5). Every correction so far has been a question of which segments belong
(E19 ring, E29 level, E36 ramps). This layer asks a different question: given the same road,
do the two files report the same LENGTH?

They do not, and the discrepancy is not noise. A polyline's measured length rises with the
number of vertices used to trace it: TIGER's primary+secondary geometry is a generalised
centreline, Overture's is OSM's traced one. Where a road is straight the two agree; where it
curves, Overture's length exceeds TIGER's by roughly the ratio of traced to chorded length.
That ratio is a property of terrain -- mountain vs plain -- and terrain varies *inside* a
region x rural x burned cell, which is the only kind of variation left to exploit.

Measured per tract, length-weighted over the segments that intersect it, so no geometry is
clipped and the weights are exactly the lengths the numerator already counts:

    gm_sinu_*   sum(len_in_tract * segment_length / endpoint_distance)   -- traced / chorded
    gm_vden_*   sum(len_in_tract * (npoints - 1) / segment_length)       -- vertices per metre
    gm_cden_*   sum(len_in_tract * n_connectors / segment_length)        -- junctions per metre
    gm_brg_*, gm_con_*  length flagged is_bridge / is_under_construction

`*` is `core` (the route family in the numerator, ramps excluded, which is E36's basis) and
`all` (every motorway-secondary segment). Divide by `gm_len_*` to get the weighted mean.

Usage:  python src/features_geom.py [region ...]
Output: $BB_DATA/derived/<region>-geom.parquet
"""
import sys
import time

import features as F
import features_routes as FR

XY = "'EPSG:4326', 'EPSG:5070', always_xy := true"


def segments(con, r):
    con.sql(f"""
        CREATE OR REPLACE TABLE gseg AS
        SELECT is_core, is_link, named,
               ST_Length(gp) AS L,
               greatest(ST_Distance(ST_StartPoint(gp), ST_EndPoint(gp)), 1.0) AS chord,
               ST_NPoints(gp) AS npts,
               nconn,
               list_contains(flags, 'is_bridge') AS is_bridge,
               list_contains(flags, 'is_under_construction') AS is_con,
               geometry
        FROM (
            SELECT len(list_filter(nets, x -> {FR.CORE})) > 0 AS is_core,
                   coalesce(subclass, '') = 'link' AS is_link,
                   class IN {FR.NAMED} AS named,
                   coalesce(len(connectors), 0) AS nconn,
                   coalesce(list_distinct(flatten(list_transform(road_flags, x -> x.values))),
                            []) AS flags,
                   ST_Transform(geometry, {XY}) AS gp,
                   geometry
            FROM (SELECT *, coalesce(list_transform(routes, x -> x.network), []) AS nets
                  FROM '{F.ref(r, "overture-roads")}'
                  WHERE class IN {FR.NAMED})
        )
    """)


def agg(p, cond):
    f = f"FILTER (WHERE {cond})"
    return [f"sum(len) {f} AS gm_len_{p}",
            f"sum(len * L / chord) {f} AS gm_sinu_{p}",
            f"sum(len * (npts - 1) / greatest(L, 1.0)) {f} AS gm_vden_{p}",
            f"sum(len * nconn / greatest(L, 1.0)) {f} AS gm_cden_{p}",
            f"sum(len) FILTER (WHERE {cond} AND is_bridge) AS gm_brg_{p}",
            f"sum(len) FILTER (WHERE {cond} AND is_con) AS gm_con_{p}"]


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    segments(con, r)
    cols = agg("core", "is_core AND NOT is_link") + agg("all", "TRUE")
    con.sql(f"""
        CREATE OR REPLACE TABLE f_g AS
        WITH j AS (SELECT t.GEOID, s.* EXCLUDE (geometry),
                          ST_Length(ST_Transform(ST_Intersection(s.geometry, t.geom), {XY})) AS len
                   FROM tracts t JOIN gseg s ON ST_Intersects(t.geom, s.geometry))
        SELECT GEOID, {", ".join(cols)} FROM j GROUP BY GEOID""")
    df = con.sql("SELECT * EXCLUDE (geom, area_m2) FROM tracts t LEFT JOIN f_g USING (GEOID) "
                 "ORDER BY GEOID").df().fillna(0.0)
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-geom.parquet", index=False)
    print(f"  {r}: {df.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
