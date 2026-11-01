"""Settlement and ZIP features per tract (E62/E63 inputs), provided data only.

E62 (missing stations): every rural community keeps a volunteer fire department, which the facility
reference lists, and also a post office or town hall, which Overture lists reliably. So the number
of communities in a tract estimates how many stations the reference holds there, and against
Overture's `fire_department` count it gives a fire gap that no place record traces (a participant: 740
reference blackouts, only 270 of them misfiled).
    n_po    distinct post offices (by normalised name)
    n_th    distinct town / city halls
    n_loc   distinct address localities with >= 3 places in the tract
E63 (ZIP allocation of CBP): Overture places carry `addresses[1].postcode` (97%); the per-tract
place count by 5-digit ZIP is written to `tract_zip.parquet` for the allocation model.

Usage:  python src/features_settle.py [region ...]
Output: $BB_DATA/derived/<region>-settle.parquet, $BB_DATA/derived/<region>-tract-zip.parquet
"""
import sys
import time

import features as F


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    con.sql(f"""
        CREATE OR REPLACE TABLE tp AS
        SELECT t.GEOID, p.categories.primary AS cat,
               regexp_replace(lower(coalesce(p.names.primary, '')), '[^a-z]|postoffice|post office|usps|united states', '', 'g') AS k,
               lower(coalesce(p.addresses[1].locality, '')) AS loc,
               substr(coalesce(p.addresses[1].postcode, ''), 1, 5) AS zip
        FROM tracts t JOIN '{F.ref(r, "overture-pois")}' p ON ST_Intersects(t.geom, p.geometry)""")
    df = con.sql("""
        WITH l AS (SELECT GEOID, loc, count(*) n FROM tp WHERE loc <> '' GROUP BY 1, 2)
        SELECT t.GEOID,
               coalesce((SELECT count(DISTINCT k) FROM tp WHERE tp.GEOID = t.GEOID AND cat = 'post_office'), 0) AS n_po,
               coalesce((SELECT count(DISTINCT k) FROM tp WHERE tp.GEOID = t.GEOID AND cat = 'town_hall'), 0) AS n_th,
               coalesce((SELECT count(*) FROM l WHERE l.GEOID = t.GEOID AND l.n >= 3), 0) AS n_loc
        FROM tracts t ORDER BY GEOID""").df()
    z = con.sql("""SELECT GEOID, zip, count(*) AS n FROM tp WHERE regexp_matches(zip, '^[0-9]{5}$')
                   GROUP BY 1, 2 ORDER BY 1, 2""").df()
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-settle.parquet", index=False)
    z.to_parquet(F.OUT / f"{r}-tract-zip.parquet", index=False)
    print(f"  {r}: {df.shape}, {len(z)} tract-zip rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
