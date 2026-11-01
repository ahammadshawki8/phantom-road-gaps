"""Per-tract DIRECT measurement of the TIGER / Overture highway length ratio.

The whole transport model turns on one number we have never observed: TIGER's primary+secondary
road length inside a tract. E29 fitted it as a single global constant `k` against the published
transport mean, E34 and E36 refitted it alongside a boundary ring width. CLAUDE.md s3 records
that the Overture/TIGER ratio spans 0.71-1.59 across regions, so a single constant is known to
be wrong -- but nothing in the package looked like a per-tract measurement of it.

Something does. **The census tract polygons in the strata package are TIGER geometry**, and
TIGER draws tract boundaries along its own features -- which is the mechanism E19 is built on.
So wherever a tract boundary runs along a core route, that boundary polyline *is* TIGER's
centreline for that road. Clip the boundary to a buffer around the Overture core segments and
clip those segments to a buffer around the boundary, and the two lengths measure the same
stretch of the same road in the two files:

    tg_bnd  length of the tract boundary within BUF of a core route segment   (TIGER's trace)
    tg_ovt  length of core route segments within BUF of the tract boundary    (Overture's trace)

Their ratio is `k` measured locally instead of fitted globally. It is a within-cell quantity --
two tracts in the same region x rural x burned cell can sit on opposite sides of it -- which is
the only kind of variation left to exploit (CLAUDE.md s4, row 29a).

Also recorded, because it is free once the boundary is in hand and it measures TIGER's
generalisation directly:

    tg_blen  boundary length;  tg_bpts  boundary vertices  -> vertices per metre of TIGER line

Usage:  python src/features_tiger.py [region ...]
Output: $BB_DATA/derived/<region>-tiger.parquet
"""
import sys
import time

import features as F
import features_routes as FR

XY = "'EPSG:4326', 'EPSG:5070', always_xy := true"
BUFS = (15, 30, 60)


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    con.sql(f"""
        CREATE OR REPLACE TABLE tp AS
        SELECT GEOID, ST_Transform(geom, {XY}) AS g,
               ST_Boundary(ST_Transform(geom, {XY})) AS b FROM tracts""")
    con.sql(f"""
        CREATE OR REPLACE TABLE cseg AS
        SELECT ST_Transform(geometry, {XY}) AS g FROM (
            SELECT geometry, coalesce(list_transform(routes, x -> x.network), []) AS nets,
                   coalesce(subclass, '') AS sc, class
            FROM '{F.ref(r, "overture-roads")}' WHERE class IN {FR.NAMED})
        WHERE sc <> 'link' AND len(list_filter(nets, x -> {FR.CORE})) > 0""")
    cols = ["ST_Length(b) AS tg_blen", "ST_NPoints(b) AS tg_bpts"]
    sel = ["t.GEOID", *cols]
    # one pass per buffer: boundary near road, and road near boundary
    joins = []
    for bu in BUFS:
        con.sql(f"""
            CREATE OR REPLACE TABLE m{bu} AS
            WITH j AS (
                SELECT t.GEOID, t.b, s.g
                FROM tp t JOIN cseg s ON ST_Intersects(ST_Buffer(t.b, {bu}), s.g)),
            u AS (SELECT GEOID, any_value(b) AS b, ST_Union_Agg(g) AS gs FROM j GROUP BY GEOID)
            SELECT GEOID,
                   ST_Length(ST_Intersection(b, ST_Buffer(gs, {bu})))  AS tg_bnd{bu},
                   ST_Length(ST_Intersection(gs, ST_Buffer(b, {bu})))  AS tg_ovt{bu}
            FROM u""")
        joins.append(f"m{bu}")
    j = " ".join(f"LEFT JOIN {t} USING (GEOID)" for t in joins)
    df = con.sql(f"SELECT {', '.join(sel)}, * EXCLUDE (GEOID, g, b) FROM tp t {j} "
                 f"ORDER BY GEOID").df().fillna(0.0)
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-tiger.parquet", index=False)
    print(f"  {r}: {df.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
