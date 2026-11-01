"""Per-tract Overture ROAD ATTRIBUTE RECALL on the scored route geometry (E102).

The transport reference is TIGER's primary+secondary file and every model we have built since
E19 estimates it as `k * (Overture core-route length + boundary ring)`, with k a single constant
per region. That is only right if Overture's recall of the route system is uniform inside a
region. It is not, and the provided roads layer says so about itself: `road_surface` and
`speed_limits` are linear-referenced lists, so a non-empty list is Overture asserting something
about that stretch of road, and the share of scored metres carrying one varies from tract to
tract far more than k does.

a participant (board thread 34744, reply of 25 Sep, provided data only) published the tract-level
version of exactly this measurement across all five regions: 82.5% of scored highway metres
carry a name, 54.0% a surface, 53.8% a speed limit, and **2,739 tracts (28.4%) carry not one
metre with all three**. Those are the published shares this module reproduces as an unfitted
check, on the class definition they used (motorway/trunk/primary/secondary).

What it is for: a tract where Overture never recorded a surface or a speed limit on the road it
*does* hold is a tract where Overture is also more likely to be missing route length the
reference holds -- the same argument E66 confirmed for places (`web_share` -> CBP reference,
c +0.051), applied to the component with the largest variance budget. The instrument is a RECALL
proxy, not a reference proxy, so it enters as a per-tract multiplier on k with the exponent
pinned at 1 and k re-solved per region to the published region mean.

Both geometries are measured because the model's denominator is route-based and the published
check is class-based:
    dp_core*   core-route segments (CORE as in features_routes.py) -- what the model uses
    dp_cls*    class in (motorway, trunk, primary, secondary) -- what a participant published

Usage:  python src/features_dispatch.py [region ...]
Output: $BB_DATA/derived/<region>-dispatch.parquet
"""
import sys
import time

import features as F
from features_routes import CORE

NAMED = "('motorway','trunk','primary','secondary')"


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    xy = "'EPSG:4326', 'EPSG:5070', always_xy := true"
    con.sql(f"""
        CREATE OR REPLACE TABLE seg AS
        SELECT len(list_filter(nets, x -> {CORE})) > 0            AS is_core,
               class IN {NAMED}                                   AS is_cls,
               nm IS NOT NULL                                     AS has_name,
               coalesce(len(road_surface), 0) > 0                 AS has_surf,
               coalesce(len(speed_limits), 0) > 0                 AS has_speed,
               geometry
        FROM (SELECT class, names.primary AS nm, road_surface, speed_limits, geometry,
                     coalesce(list_transform(routes, x -> x.network), []) AS nets
              FROM '{F.ref(r, "overture-roads")}')
        WHERE class IN {NAMED} OR len(list_filter(nets, x -> {CORE})) > 0
    """)
    con.sql(f"""
        CREATE OR REPLACE TABLE f_dp AS
        WITH j AS (
            SELECT t.GEOID, s.is_core, s.is_cls, s.has_name, s.has_surf, s.has_speed,
                   ST_Length(ST_Transform(ST_Intersection(s.geometry, t.geom), {xy})) AS len
            FROM tracts t JOIN seg s ON ST_Intersects(t.geom, s.geometry))
        SELECT GEOID,
               coalesce(sum(len) FILTER (WHERE is_core), 0)                                AS dp_core,
               coalesce(sum(len) FILTER (WHERE is_core AND has_name), 0)                   AS dp_core_name,
               coalesce(sum(len) FILTER (WHERE is_core AND has_surf), 0)                   AS dp_core_surf,
               coalesce(sum(len) FILTER (WHERE is_core AND has_speed), 0)                  AS dp_core_speed,
               coalesce(sum(len) FILTER (WHERE is_core AND has_name AND has_surf
                                               AND has_speed), 0)                          AS dp_core_all,
               coalesce(sum(len) FILTER (WHERE is_cls), 0)                                 AS dp_cls,
               coalesce(sum(len) FILTER (WHERE is_cls AND has_name), 0)                    AS dp_cls_name,
               coalesce(sum(len) FILTER (WHERE is_cls AND has_surf), 0)                    AS dp_cls_surf,
               coalesce(sum(len) FILTER (WHERE is_cls AND has_speed), 0)                   AS dp_cls_speed,
               coalesce(sum(len) FILTER (WHERE is_cls AND has_name AND has_surf
                                               AND has_speed), 0)                          AS dp_cls_all
        FROM j GROUP BY GEOID""")
    df = con.sql("SELECT t.GEOID, f_dp.* EXCLUDE (GEOID) FROM tracts t LEFT JOIN f_dp USING (GEOID) "
                 "ORDER BY GEOID").df().fillna(0.0)
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-dispatch.parquet", index=False)
    print(f"  {r}: {df.shape} in {time.time() - t0:.0f}s  class shares "
          f"name {df.dp_cls_name.sum() / df.dp_cls.sum():.3f} surf {df.dp_cls_surf.sum() / df.dp_cls.sum():.3f} "
          f"speed {df.dp_cls_speed.sum() / df.dp_cls.sum():.3f} all {df.dp_cls_all.sum() / df.dp_cls.sum():.3f}; "
          f"zero-dispatch tracts {(df.dp_cls_all.eq(0) & df.dp_cls.gt(0)).mean():.3f}", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
