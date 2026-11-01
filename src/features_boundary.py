"""Boundary features: Overture roads and facilities just OUTSIDE each tract's edge.

Hypothesis (E19): census tract boundaries follow TIGER road centrelines, so a highway on a tract's
edge counts toward that tract's TIGER length, while Overture's copy of the road (a few metres off)
falls in the neighbour. That tract then has TIGER road but little Overture road: road gap near 1.
The same mismatch can move a fire station or school across the line.

Usage:  python src/features_boundary.py [region ...]
Output: $BB_DATA/derived/<region>-boundary.parquet
"""
import sys
import time

import features as F

BUFFERS = (15, 30, 50, 100)  # metres


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    xy = "'EPSG:4326', 'EPSG:5070', always_xy := true"
    con.sql(f"CREATE OR REPLACE TABLE tp AS SELECT GEOID, ST_Transform(geom, {xy}) AS g FROM tracts")
    rings = []
    for b in BUFFERS:
        con.sql(f"""CREATE OR REPLACE TABLE ring{b} AS
                    SELECT GEOID, ST_Difference(ST_Buffer(g, {b}), g) AS g FROM tp""")
        rings.append(b)
    core = (r"len(list_filter(nets, x -> regexp_matches(x, "
            r"'^US:(I|US)(:(Business|Business:Loop|Business:Spur|Alternate|Bypass|Truck|Spur|Loop|Future))?$')"
            r" OR regexp_matches(x, '^US:(OK|WA|AZ|CA|TX|NM)(:(Turnpike|Loop|Spur|Business|Bypass|Alternate|Toll|Beltway|Park))?$'))) > 0")
    con.sql(f"""
        CREATE OR REPLACE TABLE seg AS
        SELECT class IN ('motorway','trunk','primary','secondary') AS named, {core} AS is_hw,
               ST_Transform(geometry, {xy}) AS g
        FROM (SELECT class, coalesce(list_transform(routes, x -> x.network), []) AS nets, geometry
              FROM '{F.ref(r, "overture-roads")}')
        WHERE class IN ('motorway','trunk','primary','secondary') OR {core}
    """)
    parts = []
    for b in rings:
        con.sql(f"""
            CREATE OR REPLACE TABLE fr{b} AS
            WITH j AS (SELECT q.GEOID, s.named, s.is_hw, ST_Length(ST_Intersection(s.g, q.g)) AS len
                       FROM ring{b} q JOIN seg s ON ST_Intersects(q.g, s.g))
            SELECT GEOID, sum(len) FILTER (WHERE named) AS ring{b}_named,
                          sum(len) FILTER (WHERE is_hw) AS ring{b}_hw
            FROM j GROUP BY GEOID""")
        parts.append(f"fr{b}")
    # facilities in the 200 m ring outside the tract
    con.sql(f"""CREATE OR REPLACE TABLE ring200 AS
                SELECT GEOID, ST_Difference(ST_Buffer(g, 200), g) AS g FROM tp""")
    con.sql(f"""
        CREATE OR REPLACE TABLE ffac AS
        WITH p AS (SELECT categories.primary AS cat, coalesce(categories.alternate, []) AS alt,
                          ST_Transform(geometry, {xy}) AS g
                   FROM '{F.ref(r, "overture-pois")}'
                   WHERE categories.primary IN {F.sql_list(F.FIRE + F.SCHOOL + F.EMS)}
                      OR list_has_any(coalesce(categories.alternate, []), {list(F.FIRE + F.SCHOOL + F.EMS)}))
        SELECT q.GEOID,
               count(*) FILTER (WHERE p.cat IN {F.sql_list(F.FIRE)} OR list_has_any(p.alt, {list(F.FIRE)})) AS ring200_fire,
               count(*) FILTER (WHERE p.cat IN {F.sql_list(F.SCHOOL)} OR list_has_any(p.alt, {list(F.SCHOOL)})) AS ring200_school,
               count(*) FILTER (WHERE p.cat IN {F.sql_list(F.EMS)} OR list_has_any(p.alt, {list(F.EMS)})) AS ring200_ems
        FROM ring200 q JOIN p ON ST_Intersects(q.g, p.g) GROUP BY q.GEOID""")
    parts.append("ffac")
    joins = " ".join(f"LEFT JOIN {t} USING (GEOID)" for t in parts)
    df = con.sql(f"SELECT * EXCLUDE (g), ST_Length(ST_Boundary(tp.g)) AS perimeter_m FROM tp {joins} ORDER BY GEOID").df()
    df = df.fillna(0)
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-boundary.parquet", index=False)
    print(f"  {r}: {df.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
