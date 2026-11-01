"""Per-category school counts per tract (E98), provided data only.

The bucket README defines the facilities half as the mean over DEFINED of fire (`fire_department`),
EMS and schools -- and names SIX school categories. Everything we have built treats "schools" as one
slot, so the facilities divisor is at most 3 (`physics3.simulate`: nf = (rf>0)+(re>0)+(rs>0)). If the
scorer instead treats the six categories as six components of the mean-over-defined, the divisor runs
to 8 and a fire or EMS blackout is diluted by however broad the school taxonomy is in that tract.

Per tract, for each of the six categories in `features.SCHOOL`, the number of Overture places whose
`categories.primary` is that category or that carry it as an alternate, and:
    sc_ncat       how many of the six are present at all (primary or alternate)
    sc_ncat_prim  the same counting `categories.primary` only (robustness check)

Usage:  python src/features_schoolcat.py [region ...]
Output: $BB_DATA/derived/<region>-schoolcat.parquet
"""
import sys
import time

import features as F


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    per = [f"count(*) FILTER (WHERE cat = '{c}' OR list_contains(alt, '{c}')) AS sc_{c}" for c in F.SCHOOL]
    prim = [f"count(*) FILTER (WHERE cat = '{c}') AS scp_{c}" for c in F.SCHOOL]
    con.sql(f"""
        CREATE OR REPLACE TABLE f_sc AS
        WITH p AS (
            SELECT categories.primary AS cat, coalesce(categories.alternate, []) AS alt, geometry
            FROM '{F.ref(r, "overture-pois")}')
        SELECT t.GEOID, {", ".join(per + prim)}
        FROM tracts t JOIN p ON ST_Intersects(t.geom, p.geometry)
        GROUP BY t.GEOID""")
    df = con.sql("SELECT t.GEOID, f_sc.* EXCLUDE (GEOID) FROM tracts t LEFT JOIN f_sc USING (GEOID) "
                 "ORDER BY GEOID").df().fillna(0)
    df["sc_ncat"] = sum((df[f"sc_{c}"] > 0).astype(int) for c in F.SCHOOL)
    df["sc_ncat_prim"] = sum((df[f"scp_{c}"] > 0).astype(int) for c in F.SCHOOL)
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-schoolcat.parquet", index=False)
    print(f"  {r}: {df.shape} in {time.time() - t0:.0f}s, mean ncat {df.sc_ncat.mean():.2f}, "
          f"share >=2 {(df.sc_ncat >= 2).mean():.3f}", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
