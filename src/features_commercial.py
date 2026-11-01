"""Commercial-activity features per tract (E48 input), provided data only.

Hypothesis H-P4: the CBP half's tract reference follows where businesses actually are (a ZIP total
split by business-address share is the standard way to put CBP on tracts), so the reference
tracks commercial activity, not population, housing or plain building count. Overture-side
signals of commercial activity that are NOT the POI numerator:
  - large building footprints (stores, offices, warehouses: bbox area >= 1,000 / 5,000 m2)
  - buildings Overture classes as commercial / retail / office / industrial / warehouse
  - service-road length (parking aisles, drive-throughs, loading lanes), already in features.py

Usage:  python src/features_commercial.py [region ...]
Output: $BB_DATA/derived/<region>-commercial.parquet
"""
import sys
import time

import features as F

COMMERCIAL = ["commercial", "retail", "office", "industrial", "warehouse", "supermarket", "hotel",
              "service", "manufacture", "factory"]


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    con.sql(f"""
        CREATE OR REPLACE TABLE f_com AS
        WITH b AS (
            SELECT ST_Point((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2) AS pt,
                   -- bbox area in m2 at the tract latitude (degrees -> metres)
                   (bbox.xmax - bbox.xmin) * 111320 * cos(radians((bbox.ymin + bbox.ymax) / 2))
                   * (bbox.ymax - bbox.ymin) * 110540 AS a_m2,
                   class, subtype
            FROM '{F.ref(r, "overture-buildings")}')
        SELECT t.GEOID,
               count(*) FILTER (WHERE a_m2 >= 1000) AS bld_ge1k,
               count(*) FILTER (WHERE a_m2 >= 5000) AS bld_ge5k,
               sum(a_m2) FILTER (WHERE a_m2 >= 1000) AS bld_area_ge1k,
               count(*) FILTER (WHERE class IN {F.sql_list(COMMERCIAL)}
                                   OR subtype IN ('commercial', 'industrial')) AS bld_commercial
        FROM tracts t JOIN b ON ST_Intersects(t.geom, b.pt)
        GROUP BY t.GEOID""")
    df = con.sql("SELECT t.GEOID, f_com.* EXCLUDE (GEOID) FROM tracts t LEFT JOIN f_com USING (GEOID) "
                 "ORDER BY GEOID").df().fillna(0)
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-commercial.parquet", index=False)
    print(f"  {r}: {df.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
