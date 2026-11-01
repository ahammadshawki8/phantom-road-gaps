"""Per-tract count of Overture places that County Business Patterns does not count (E101).

The POI component's CBP half compares an Overture place count with an establishment reference derived
from CBP. Every model we have built puts `poi_all` in that numerator -- every Overture place, whatever it
is. CBP counts **establishments**, and its published universe excludes, by NAICS:

    92      public administration        (city hall, courthouse, police, fire, public schools, libraries)
    8131    religious organizations      (churches, mosques, synagogues, temples)
    814     private households
    491     Postal Service
    111/112 crop and animal production   (farms, ranches)

and Overture additionally carries records that are not establishments in any taxonomy at all: ATMs, parks,
playgrounds, landmarks, monuments, cemeteries, transit stops and trailheads.

Removing them is a **numerator** correction, so it is observable end to end -- no hidden reference is
involved -- and it is deterministic. Measured over the five regions: **14.0%** of 2,001,201 places are
out of scope, the per-tract share has mean 0.164 and sd 0.089 (p10 0.080, p90 0.273), region means are
flat (0.151-0.191) and **97.1% of the spread is WITHIN region x rural cell**, which is the only place
row 29a leaves anything to find.

    poi_out   places whose primary category CBP does not count
    poi_in    poi_all - poi_out

Usage:  python src/features_cbpscope.py [region ...]
Output: $BB_DATA/derived/<region>-cbpscope.parquet
"""
import sys
import time

import features as F

# a place is out of CBP's universe if its primary category matches any of these
OUT = """cat IS NULL OR cat ILIKE '%church%' OR cat ILIKE '%religio%' OR cat ILIKE '%mosque%'
 OR cat ILIKE '%synagog%' OR cat ILIKE '%temple%' OR cat = 'park' OR cat ILIKE '%national_park%'
 OR cat ILIKE '%state_park%' OR cat ILIKE '%playground%' OR cat ILIKE '%landmark%'
 OR cat ILIKE '%monument%' OR cat ILIKE '%cemeter%' OR cat ILIKE '%government%'
 OR cat ILIKE '%city_hall%' OR cat ILIKE '%courthouse%' OR cat ILIKE '%post_office%'
 OR cat ILIKE '%police%' OR cat = 'fire_department' OR cat ILIKE '%public_service%' OR cat = 'atms'
 OR cat ILIKE '%public_school%' OR cat ILIKE '%library%' OR cat ILIKE '%farm%' OR cat ILIKE '%ranch%'
 OR cat ILIKE '%agricultur%' OR cat ILIKE '%bus_stop%' OR cat ILIKE '%transit%' OR cat ILIKE '%trail%'"""


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    con.sql(f"""
        CREATE OR REPLACE TABLE f_cbp AS
        WITH p AS (SELECT categories.primary AS cat, geometry FROM '{F.ref(r, "overture-pois")}')
        SELECT t.GEOID, count(*) AS poi_all_chk, count(*) FILTER (WHERE {OUT}) AS poi_out
        FROM tracts t JOIN p ON ST_Intersects(t.geom, p.geometry)
        GROUP BY t.GEOID""")
    df = con.sql("SELECT t.GEOID, f_cbp.* EXCLUDE (GEOID) FROM tracts t LEFT JOIN f_cbp USING (GEOID) "
                 "ORDER BY GEOID").df().fillna(0)
    df["poi_in"] = (df.poi_all_chk - df.poi_out).clip(lower=0)
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-cbpscope.parquet", index=False)
    print(f"  {r}: {df.shape} in {time.time() - t0:.0f}s, out-of-scope share "
          f"{df.poi_out.sum() / max(df.poi_all_chk.sum(), 1):.3f}", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
