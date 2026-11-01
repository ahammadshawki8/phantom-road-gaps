"""Per-tract Overture road length by ROUTE FAMILY, in the tract and in the boundary ring.

Why this layer exists. The transport reference is TIGER's primary+secondary road file, and the
published "at least one component undefined" rates (0.21 / 0.37 / 0.55 / 0.28) say it is absent
from a fifth to a half of tracts, so it is the *numbered route* system and not all roads (E19,
CLAUDE.md s3). `features.py` builds that reference from one route family only: interstate, US and
state networks. But TIGER's own definition of S1200 is "main arteries, usually in the U.S.
Highway, State Highway **or County Highway** system", and Overture carries county networks
explicitly -- `US:OK:Washita`, `US:OK:Garfield`, `US:TX:Brazoria` and ~60 more, 2,351 segments in
Washita County alone. They are excluded today by a regex, never by a measurement.

Three families are measured separately so each can be admitted or refused on its own evidence:

    core    US:I, US:US, US:<ST> and their Business/Loop/Spur/... variants   (what we use today)
    county  US:<ST>:<CountyName>                                            (TIGER says S1200)
    fm      US:TX:FM, US:TX:RM, US:TX:PR                                    (TxDOT state highways)

Each is measured inside the tract and inside the 15/30/50 m outer boundary ring, because E19's
displacement correction applies to whichever family turns out to belong in the reference.

Usage:  python src/features_routes.py [region ...]
Output: $BB_DATA/derived/<region>-routes.parquet
"""
import sys
import time

import features as F

BUFFERS = (5, 10, 15, 30, 50)

CORE = (r"regexp_matches(x, '^US:(I|US)(:(Business|Business:Loop|Business:Spur|Alternate|Bypass"
        r"|Truck|Spur|Loop|Future))?$') OR regexp_matches(x, "
        r"'^US:(OK|WA|AZ|CA|TX|NM)(:(Turnpike|Loop|Spur|Business|Bypass|Alternate|Toll|Beltway|Park))?$')")
COUNTY = r"regexp_matches(x, '^US:(OK|WA|AZ|CA|TX|NM):[A-Z][a-z]+( [A-Z][a-z]+)*$')"
FM = r"x IN ('US:TX:FM', 'US:TX:RM', 'US:TX:PR')"
NAMED = "('motorway','trunk','primary','secondary')"


def segments(con, r):
    con.sql(f"""
        CREATE OR REPLACE TABLE seg AS
        SELECT class IN {NAMED} AS named,
               len(list_filter(nets, x -> {CORE})) > 0 AS is_core,
               len(list_filter(nets, x -> {COUNTY})) > 0 AS is_county,
               len(list_filter(nets, x -> {FM})) > 0 AS is_fm,
               geometry
        FROM (SELECT class, coalesce(list_transform(routes, x -> x.network), []) AS nets, geometry
              FROM '{F.ref(r, "overture-roads")}')
    """)


AGG = ["sum(len) FILTER (WHERE is_core) AS {p}core",
       "sum(len) FILTER (WHERE is_county AND NOT is_core) AS {p}county",
       "sum(len) FILTER (WHERE is_fm AND NOT is_core) AS {p}fm",
       "sum(len) FILTER (WHERE is_county AND NOT is_core AND named) AS {p}county_named",
       "sum(len) FILTER (WHERE (is_core OR is_county OR is_fm)) AS {p}any"]


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    segments(con, r)
    xy = "'EPSG:4326', 'EPSG:5070', always_xy := true"
    con.sql(f"""
        CREATE OR REPLACE TABLE f_in AS
        WITH j AS (SELECT t.GEOID, s.named, s.is_core, s.is_county, s.is_fm,
                          ST_Length(ST_Transform(ST_Intersection(s.geometry, t.geom), {xy})) AS len
                   FROM tracts t JOIN seg s ON ST_Intersects(t.geom, s.geometry))
        SELECT GEOID, {", ".join(a.format(p="rt_") for a in AGG)} FROM j GROUP BY GEOID""")
    con.sql(f"CREATE OR REPLACE TABLE tp AS SELECT GEOID, ST_Transform(geom, {xy}) AS g FROM tracts")
    con.sql(f"CREATE OR REPLACE TABLE segp AS SELECT named, is_core, is_county, is_fm, "
            f"ST_Transform(geometry, {xy}) AS g FROM seg WHERE is_core OR is_county OR is_fm")
    parts = ["f_in"]
    for b in BUFFERS:
        con.sql(f"""CREATE OR REPLACE TABLE rg{b} AS
                    SELECT GEOID, ST_Difference(ST_Buffer(g, {b}), g) AS g FROM tp""")
        con.sql(f"""
            CREATE OR REPLACE TABLE f_r{b} AS
            WITH j AS (SELECT q.GEOID, s.named, s.is_core, s.is_county, s.is_fm,
                              ST_Length(ST_Intersection(s.g, q.g)) AS len
                       FROM rg{b} q JOIN segp s ON ST_Intersects(q.g, s.g))
            SELECT GEOID, {", ".join(a.format(p=f"rg{b}_") for a in AGG)} FROM j GROUP BY GEOID""")
        parts.append(f"f_r{b}")
    joins = " ".join(f"LEFT JOIN {t} USING (GEOID)" for t in parts)
    df = con.sql(f"SELECT * EXCLUDE (geom) FROM tracts t {joins} ORDER BY GEOID").df()
    df = df.drop(columns=["area_m2"]).fillna(0.0)
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-routes.parquet", index=False)
    print(f"  {r}: {df.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
