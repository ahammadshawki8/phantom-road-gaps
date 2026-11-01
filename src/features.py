"""Per-tract features from the PROVIDED challenge data only (Overture extracts, ACS, strata).

Usage:  python src/features.py [region ...]      (default: all five)
Input:  $BB_DATA (default ../data) laid out like the bucket: reference/<r>/..., strata/<r>/...
Output: $BB_DATA/derived/<region>-features.parquet, one row per scored GEOID.

Nothing here reads TIGER, Microsoft, HIFLD or CBP. Those layers are not in the package and
must not be used for the scored submission.
"""
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

DATA = Path(os.environ.get("BB_DATA", Path(__file__).resolve().parents[2] / "data"))
OUT = DATA / "derived"
REGIONS = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
STATE_NET = {"eastern-ok": "US:OK", "eastern-wa": "US:WA", "maricopa-az": "US:AZ",
             "northern-ca": "US:CA", "south-central-tx": "US:TX"}

# categories.primary values the organiser's HIFLD matching uses (bucket README)
FIRE = ["fire_department"]
EMS = ["ambulance_and_ems_services"]
SCHOOL = ["elementary_school", "middle_school", "high_school", "school", "private_school",
          "public_school"]
NAMED = ["motorway", "trunk", "primary", "secondary"]
ROAD_CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential",
                "unclassified"]


def sql_list(xs):
    return "(" + ",".join(f"'{x}'" for x in xs) + ")"


def connect():
    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial;")
    con.sql(f"SET threads={os.cpu_count()};")
    return con


def ref(r, layer):
    return (DATA / "reference" / r / f"{r}-{layer}.parquet").as_posix()


def strata(r, layer):
    return (DATA / "strata" / r / f"{r}-{layer}.parquet").as_posix()


def load_tracts(con, r):
    sub = (DATA / "reference" / r / f"{r}-sample-submission.csv").as_posix()
    con.sql(f"""
        CREATE OR REPLACE TABLE tracts AS
        SELECT t.GEOID, t.geometry AS geom,
               ST_Area(ST_Transform(t.geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true)) AS area_m2
        FROM '{strata(r, "census-tracts")}' t
        WHERE t.GEOID IN (SELECT GEOID FROM read_csv('{sub}', types={{'GEOID':'VARCHAR'}}))
    """)
    n = con.sql("SELECT count(*) FROM tracts").fetchone()[0]
    print(f"  {r}: {n} scored tracts")
    return n


def poi_features(con, r):
    cats = {"fire": FIRE, "ems": EMS, "school": SCHOOL, "hospital": ["hospital"]}
    agg = [f"count(*) FILTER (WHERE p.cat IN {sql_list(v)}) AS poi_{k}" for k, v in cats.items()]
    # same types where they only appear as an alternate category (Overture often files a
    # station under e.g. 'government_office' with 'fire_department' as alternate)
    agg += [f"count(*) FILTER (WHERE p.cat NOT IN {sql_list(v)} AND list_has_any(p.alt, {list(v)})) AS poi_{k}_alt"
            for k, v in cats.items() if k != "hospital"]
    agg += ["count(*) AS poi_all",
            "count(*) FILTER (WHERE p.conf >= 0.5) AS poi_conf50",
            "count(*) FILTER (WHERE p.conf >= 0.8) AS poi_conf80",
            "count(*) FILTER (WHERE p.cat IS NULL) AS poi_nocat",
            "count(*) FILTER (WHERE p.has_addr) AS poi_addr",
            "count(*) FILTER (WHERE p.n_src >= 2) AS poi_multisrc"]
    con.sql(f"""
        CREATE OR REPLACE TABLE f_poi AS
        WITH p AS (
            SELECT categories.primary AS cat, coalesce(categories.alternate, []) AS alt,
                   confidence AS conf, len(addresses) > 0 AS has_addr, len(sources) AS n_src,
                   geometry
            FROM '{ref(r, "overture-pois")}')
        SELECT t.GEOID, {", ".join(agg)}
        FROM tracts t JOIN p ON ST_Intersects(t.geom, p.geometry)
        GROUP BY t.GEOID
    """)


def road_features(con, r, layer="overture-roads", prefix="rd"):
    # Core highway system = what TIGER S1100/S1200 holds: interstates, US routes, state
    # routes (+ their business/loop/spur variants). County, forest, historic, scenic and
    # NHS overlay networks are excluded. Texas FM/RM roads are kept apart (is_fm), since
    # whether TIGER files them as S1200 is unknown.
    core = (r"len(list_filter(nets, x -> regexp_matches(x, "
            r"'^US:(I|US)(:(Business|Business:Loop|Business:Spur|Alternate|Bypass|Truck|Spur|Loop|Future))?$')"
            r" OR regexp_matches(x, '^US:(OK|WA|AZ|CA|TX|NM)(:(Turnpike|Loop|Spur|Business|Bypass|Alternate|Toll|Beltway|Park))?$'))) > 0")
    fm = r"len(list_filter(nets, x -> x IN ('US:TX:FM', 'US:TX:RM', 'US:TX:PR'))) > 0"
    con.sql(f"""
        CREATE OR REPLACE TABLE seg AS
        SELECT class, nets, {core} AS is_hw, {fm} AS is_fm, geometry
        FROM (SELECT class, coalesce(list_transform(routes, x -> x.network), []) AS nets, geometry
              FROM '{ref(r, layer)}')
    """)
    hw = "is_hw"
    classes = ROAD_CLASSES if prefix == "rd" else \
        [c for (c,) in con.sql("SELECT class FROM seg GROUP BY 1 HAVING count(*) > 500").fetchall()]
    agg = [f"sum(len) FILTER (WHERE class = '{c}') AS {prefix}_len_{c}" for c in classes]
    if prefix == "rd":
        agg += [f"sum(len) FILTER (WHERE class IN {sql_list(NAMED)}) AS rd_len_named",
                f"sum(len) FILTER (WHERE is_hw) AS rd_len_hwroute",
                f"sum(len) FILTER (WHERE is_hw AND class IN {sql_list(NAMED)}) AS rd_len_hw_named",
                f"sum(len) FILTER (WHERE is_hw AND class NOT IN {sql_list(NAMED)}) AS rd_len_hw_unnamed",
                f"sum(len) FILTER (WHERE class IN {sql_list(NAMED)} AND NOT is_hw) AS rd_len_named_noroute",
                "sum(len) FILTER (WHERE is_fm AND NOT is_hw) AS rd_len_fm",
                f"sum(len) FILTER (WHERE is_fm AND NOT is_hw AND class IN {sql_list(NAMED)}) AS rd_len_fm_named",
                "sum(len) FILTER (WHERE list_has_any(nets, ['US:I']) OR len(list_filter(nets, x -> x LIKE 'US:I%')) > 0) AS rd_len_interstate",
                "sum(len) FILTER (WHERE len(list_filter(nets, x -> x LIKE 'US:US%')) > 0) AS rd_len_ushwy",
                f"sum(len) FILTER (WHERE len(list_filter(nets, x -> x LIKE '{STATE_NET[r]}%')) > 0) AS rd_len_statehwy",
                "sum(len) FILTER (WHERE len(nets) > 0) AS rd_len_anyroute",
                "count(*) AS rd_nseg"]
    agg.append(f"sum(len) AS {prefix}_len_all")
    con.sql(f"""
        CREATE OR REPLACE TABLE f_{prefix} AS
        WITH j AS (
            SELECT t.GEOID, s.class, s.nets, s.is_hw, s.is_fm,
                   ST_Length(ST_Transform(ST_Intersection(s.geometry, t.geom),
                             'EPSG:4326', 'EPSG:5070', always_xy := true)) AS len
            FROM tracts t JOIN seg s ON ST_Intersects(t.geom, s.geometry))
        SELECT GEOID, {", ".join(agg)} FROM j GROUP BY GEOID
    """)


def building_features(con, r):
    # bbox centre as the representative point: buildings are small, and this avoids
    # decoding 4M+ polygons. A building is assigned to exactly one tract.
    con.sql(f"""
        CREATE OR REPLACE TABLE f_bld AS
        WITH b AS (
            SELECT ST_Point((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2) AS pt,
                   sources[1].dataset AS src,
                   list_transform(sources, x -> x.dataset) AS srcs,
                   height IS NOT NULL AS has_h, class IS NOT NULL AS has_class,
                   (bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin) AS bb_area
            FROM '{ref(r, "overture-buildings")}')
        SELECT t.GEOID,
               count(*) AS bld_all,
               count(*) FILTER (WHERE src = 'OpenStreetMap') AS bld_osm,
               count(*) FILTER (WHERE src = 'Microsoft ML Buildings') AS bld_msft,
               count(*) FILTER (WHERE src = 'Google Open Buildings') AS bld_google,
               count(*) FILTER (WHERE src = 'Esri Community Maps') AS bld_esri,
               count(*) FILTER (WHERE list_contains(srcs, 'Microsoft ML Buildings')) AS bld_msft_any,
               count(*) FILTER (WHERE has_h) AS bld_height,
               count(*) FILTER (WHERE has_class) AS bld_class,
               median(bb_area) * 1e10 AS bld_bbarea_med
        FROM tracts t JOIN b ON ST_Intersects(t.geom, b.pt)
        GROUP BY t.GEOID
    """)


def point_layer_features(con, r, layer, name):
    con.sql(f"""
        CREATE OR REPLACE TABLE f_{name} AS
        SELECT t.GEOID, count(*) AS {name}_n
        FROM tracts t JOIN '{ref(r, layer)}' x ON ST_Intersects(t.geom, x.geometry)
        GROUP BY t.GEOID
    """)


def build(r):
    t0 = time.time()
    con = connect()
    load_tracts(con, r)
    steps = [("poi", lambda: poi_features(con, r)),
             ("rd", lambda: road_features(con, r)),
             ("ru", lambda: road_features(con, r, "overture-roads-unfiltered", "ru")),
             ("infra", lambda: point_layer_features(con, r, "overture-infrastructure", "infra")),
             ("bld", lambda: building_features(con, r))]
    tables = []
    for name, fn in steps:
        if name == "bld" and not Path(ref(r, "overture-buildings")).exists():
            print(f"  {r}: buildings not downloaded yet, skipping")
            continue
        t1 = time.time()
        fn()
        tables.append(f"f_{name}")
        print(f"  {r}: {name} done in {time.time() - t1:.0f}s")
    joins = " ".join(f"LEFT JOIN {t} USING (GEOID)" for t in tables)
    df = con.sql(f"SELECT * EXCLUDE (geom) FROM tracts t {joins} ORDER BY GEOID").df()
    acs = pd.read_parquet(ref(r, "census-acs-housing"), columns=["GEOID", "housing_units"])
    df = df.merge(acs, on="GEOID", how="left")
    df.insert(1, "region", r)
    num = df.columns.drop(["GEOID", "region"])
    cnt = [c for c in num if c not in ("area_m2", "housing_units", "bld_bbarea_med")]
    df[cnt] = df[cnt].fillna(0)  # no intersecting feature = zero length / count
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{r}-features.parquet", index=False)
    print(f"  {r}: {df.shape} written in {time.time() - t0:.0f}s")
    return df


if __name__ == "__main__":
    for r in (sys.argv[1:] or REGIONS):
        build(r)
