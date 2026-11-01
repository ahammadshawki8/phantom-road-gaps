"""Road lineage per tract (E81+), provided data only: the 2007-08 TIGER import trace in Overture.

US OpenStreetMap roads were bulk-imported from TIGER in 2007-08, and Overture inherits them. Every
Overture road keeps its OSM provenance in `sources[].record_id` = 'w<way id>@<version>'. A way from
the import era has a low id; a version of 1-2 means nobody has edited it since. Where the import is
untouched, the road CLASS still comes mechanically from TIGER's own feature codes and should agree
with TIGER's S1100/S1200; where mappers reclassified it (US convention demoted many state/county
routes from secondary to tertiary), Overture and TIGER disagree -- the README's "the two draw the
primary/secondary boundary differently". (Lead: a participant's public repository, section 7.)

Per tract, length (m) of class x {import-era untouched, import-era edited, post-import}:
    lin_<class>_raw   way id < IMPORT_MAX and version <= 2
    lin_<class>_edit  way id < IMPORT_MAX and version >= 3
    lin_<class>_new   way id >= IMPORT_MAX
for class in motorway/trunk/primary/secondary/tertiary/residential+unclassified.

Usage:  python src/features_lineage.py [region ...]
Output: $BB_DATA/derived/<region>-lineage.parquet
"""
import sys
import time

import features as F

IMPORT_MAX = 60_000_000          # OSM way ids below this were created by late 2008 (TIGER import era)
CLASSES = {"mw": "('motorway')", "tr": "('trunk')", "pr": "('primary')", "se": "('secondary')",
           "te": "('tertiary')", "lo": "('residential', 'unclassified')"}


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    con.sql(f"""
        CREATE OR REPLACE TABLE seg AS
        WITH s AS (
            SELECT class, geometry,
                   coalesce(list_filter(sources, x -> x.dataset = 'OpenStreetMap')[1].record_id, '') AS rid
            FROM '{F.ref(r, "overture-roads")}'
            WHERE class IN ('motorway','trunk','primary','secondary','tertiary','residential','unclassified'))
        SELECT class, geometry,
               TRY_CAST(regexp_extract(rid, '^w([0-9]+)@', 1) AS BIGINT) AS wid,
               TRY_CAST(regexp_extract(rid, '@([0-9]+)$', 1) AS INTEGER) AS ver
        FROM s""")
    agg = []
    for k, cl in CLASSES.items():
        for lab, cond in [("raw", f"wid < {IMPORT_MAX} AND ver <= 2"), ("edit", f"wid < {IMPORT_MAX} AND ver >= 3"),
                          ("new", f"wid >= {IMPORT_MAX}")]:
            agg.append(f"sum(len) FILTER (WHERE class IN {cl} AND {cond}) AS lin_{k}_{lab}")
    con.sql(f"""
        CREATE OR REPLACE TABLE f_lin AS
        WITH x AS (
            SELECT t.GEOID, seg.class, seg.wid, seg.ver,
                   ST_Length(ST_Transform(ST_Intersection(t.geom, seg.geometry), 'EPSG:4326', 'EPSG:5070',
                                          always_xy := true)) AS len
            FROM tracts t JOIN seg ON ST_Intersects(t.geom, seg.geometry))
        SELECT GEOID, {", ".join(agg)} FROM x GROUP BY GEOID""")
    df = con.sql("SELECT t.GEOID, f_lin.* EXCLUDE (GEOID) FROM tracts t LEFT JOIN f_lin USING (GEOID) "
                 "ORDER BY GEOID").df().fillna(0)
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-lineage.parquet", index=False)
    print(f"  {r}: {df.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
