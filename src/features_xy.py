"""Tract centroid coordinates in metres (EPSG:5070), for the neighbourhood parts of E41.

The census-tract polygons in the strata package are TIGER geometry (CLAUDE.md §3), so their
centroids are the only positional information the package carries about where a tract *is*.
`cand_knn.py` uses them to read Overture completeness over the K nearest tracts rather than over
the county, which is the geography the local-mapping-community mechanism actually names.

Usage:  python src/features_xy.py        (~20 s)
Output: $BB_DATA/derived/tract_xy.parquet -- GEOID, cx, cy (metres, CONUS Albers)
"""
import duckdb
import pandas as pd

import features as F


def build():
    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial;")
    out = []
    for r in F.REGIONS:
        sub = (F.DATA / "reference" / r / f"{r}-sample-submission.csv").as_posix()
        tr = (F.DATA / "strata" / r / f"{r}-census-tracts.parquet").as_posix()
        pt = ("ST_Centroid(ST_Transform(t.geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true))")
        out.append(con.sql(f"""
            SELECT t.GEOID, ST_X({pt}) AS cx, ST_Y({pt}) AS cy
            FROM '{tr}' t
            WHERE t.GEOID IN (SELECT GEOID FROM read_csv('{sub}', types={{'GEOID':'VARCHAR'}}))
        """).df())
        print(f"  {r}: {len(out[-1])} centroids")
    df = pd.concat(out, ignore_index=True)
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / "tract_xy.parquet", index=False)
    print(f"wrote {F.OUT / 'tract_xy.parquet'}  {len(df)} rows")


if __name__ == "__main__":
    build()
