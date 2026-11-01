"""Scored-class road with no core route, split by class (28 Oct).

The transport reference is TIGER S1100 + S1200. S1100 is defined by access ("limited-access highways ...
distinguished by the presence of interchanges"), not by route number; S1200 by route system ("main
arteries, usually in the U.S. Highway, State Highway, or County Highway system"). Every vehicle since E19
builds the reference from Overture's core ROUTE length (+ E112's Texas toll authorities) and lets one
constant LAM_U carry all unrouted scored-class road, motorway and secondary alike. This splits that
length by Overture class so the two TIGER definitions can be applied separately: an unnumbered motorway
mainline is S1100 by definition, an unnumbered city arterial is S1400.

Usage:  python src/features_unrouted.py [region ...]
Output: $BB_DATA/derived/<region>-unrouted.parquet  (GEOID, nr_motorway, nr_trunk, nr_primary, nr_secondary)
"""
import sys
import time

import features as F
import features_bndfollow as FB
import features_routes as FR

XY = FB.XY
CLASSES = ["motorway", "trunk", "primary", "secondary"]


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    con.sql(f"CREATE OR REPLACE TABLE tp AS SELECT GEOID, ST_Transform(geom, {XY}) AS g FROM tracts")
    con.sql(f"""
        CREATE OR REPLACE TABLE rd AS
        SELECT class, ST_Transform(geometry, {XY}) AS g
        FROM (SELECT class, coalesce(subclass, '') = 'link' AS link,
                     coalesce(list_transform(routes, x -> x.network), []) AS nets, geometry
              FROM '{F.ref(r, "overture-roads")}')
        WHERE class IN {F.sql_list(CLASSES)} AND NOT link
          AND len(list_filter(nets, x -> {FR.CORE})) = 0
          AND len(list_filter(nets, x -> list_contains({FB.TOLL}, x))) = 0""")
    agg = ", ".join(f"coalesce(sum(len) FILTER (WHERE class = '{c}'), 0) AS nr_{c}" for c in CLASSES)
    df = con.sql(f"""
        WITH j AS (SELECT t.GEOID, s.class, ST_Length(ST_Intersection(s.g, t.g)) AS len
                   FROM tp t JOIN rd s ON ST_Intersects(t.g, s.g))
        SELECT GEOID, {agg} FROM j GROUP BY GEOID""").df()
    tr = con.sql("SELECT GEOID FROM tracts").df()
    df = tr.merge(df, on="GEOID", how="left").fillna(0.0)
    df.to_parquet(F.OUT / f"{r}-unrouted.parquet", index=False)
    print(f"  {r}: {len(df)} tracts, " + ", ".join(f"{c} {df[f'nr_{c}'].sum() / 1e3:,.0f} km" for c in CLASSES)
          + f" ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    for r in sys.argv[1:] or F.REGIONS:
        build(r)
