"""Per-tract Overture ROAD ATTRIBUTE QUALITY, and the ramp length the reference does not contain.

Two things this measures, both from the provided roads layer only.

1. **Ramp length** (`subclass = 'link'`). The transport reference is TIGER's primary+secondary
   road file; TIGER files ramps separately as MTFCC S1630 and they are **not** in that file.
   Overture, on the other hand, classes a motorway ramp as `class = 'motorway'`, so every model
   we have built since E18 counts ramps in the numerator and nothing in the denominator. That
   inflates the numerator exactly where interchanges are dense, which is a within-tract property
   that varies inside a region x rural x burned cell.

2. **Attribute completeness of local roads** -- the share of `residential` + `unclassified`
   length with no `names.primary`, and the share with no surface and no speed limit. This is not
   a reference proxy; it is a proxy for *how well Overture was built here*. Two Bias-Discovery
   posts on the challenge board document that it tracks household connectivity and social
   vulnerability (nameless local-road share rises 1.8-2.6x from the most- to the least-connected
   quintile of rural tracts). Our own deficit profile against the target's published disparity
   card says the signal we are missing is exactly SVI-aligned: our SVI contrast is 0.58 of the
   target's where rural is 0.91 and heat 0.95. A tract where Overture never got round to naming
   the local roads is a tract where Overture is also likely to have missed highway length.

Usage:  python src/features_quality.py [region ...]
Output: $BB_DATA/derived/<region>-quality.parquet
"""
import sys
import time

import features as F

LOCAL = "('residential','unclassified','tertiary')"
NAMED = "('motorway','trunk','primary','secondary')"


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    xy = "'EPSG:4326', 'EPSG:5070', always_xy := true"
    con.sql(f"""
        CREATE OR REPLACE TABLE seg AS
        SELECT class,
               coalesce(subclass, '') = 'link' AS is_link,
               names.primary IS NULL AS unnamed,
               coalesce(len(road_surface), 0) = 0 AS no_surface,
               coalesce(len(speed_limits), 0) = 0 AS no_speed,
               geometry
        FROM '{F.ref(r, "overture-roads")}'
    """)
    con.sql(f"""
        CREATE OR REPLACE TABLE f_q AS
        WITH j AS (
            SELECT t.GEOID, s.class, s.is_link, s.unnamed, s.no_surface, s.no_speed,
                   ST_Length(ST_Transform(ST_Intersection(s.geometry, t.geom), {xy})) AS len
            FROM tracts t JOIN seg s ON ST_Intersects(t.geom, s.geometry))
        SELECT GEOID,
               sum(len) FILTER (WHERE class IN {NAMED} AND is_link)          AS q_link_named,
               sum(len) FILTER (WHERE class IN {NAMED} AND unnamed)          AS q_unnamed_named,
               sum(len) FILTER (WHERE class IN {NAMED} AND no_surface)       AS q_nosurf_named,
               sum(len) FILTER (WHERE class IN {NAMED} AND no_speed)         AS q_nospeed_named,
               sum(len) FILTER (WHERE class IN {LOCAL})                      AS q_local,
               sum(len) FILTER (WHERE class IN {LOCAL} AND unnamed)          AS q_local_unnamed,
               sum(len) FILTER (WHERE class IN {LOCAL} AND no_surface)       AS q_local_nosurf,
               sum(len) FILTER (WHERE class IN {LOCAL} AND no_speed)         AS q_local_nospeed
        FROM j GROUP BY GEOID""")
    df = con.sql("SELECT * EXCLUDE (geom, area_m2) FROM tracts t LEFT JOIN f_q USING (GEOID) "
                 "ORDER BY GEOID").df().fillna(0.0)
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-quality.parquet", index=False)
    print(f"  {r}: {df.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
