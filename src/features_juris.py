"""Jurisdiction x tract place counts (E108 input), provided data only.

A fire department serves a JURISDICTION -- an incorporated place, a township, a fire-protection
district -- and the reference (HIFLD) lists one permanent building per station, in ONE of the
jurisdiction's tracts. Every file so far has asked "does this tract contain a community?"
(E62 post offices / town halls, E65 within-tract address localities). Nothing has ever asked the
question the reference actually answers: **given that this jurisdiction has a station somewhere,
WHICH of its tracts holds it?** That is a competition between tracts, and it turns a diffuse
per-tract probability into a step.

The package carries two jurisdiction keys on every Overture place, in `addresses[1]`:
    postcode   5-digit ZIP, the postal routing unit (present on ~97% of places)
    locality   the town name as written in the address
Both are emitted per (tract, key) with the counts needed to (a) size the jurisdiction, (b) decide
whether Overture already traces a station anywhere inside it, and (c) say which tract is its centre.

    jz_*   keyed on postcode      jl_*   keyed on locality
    n      places in this tract with this key
    nfire  places whose categories.primary is fire_department
    nfalt  places with fire_department among categories.alternate, or a station-shaped name under
           another primary category (E49's misfiling rule, applied here at jurisdiction level)

Usage:  python src/features_juris.py [region ...]
Output: $BB_DATA/derived/<region>-juris.parquet   (GEOID, key, kind, n, nfire, nfalt)
"""
import sys
import time

import features as F
import features_misfiled as FM


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    con.sql(f"""
        CREATE OR REPLACE TABLE tp AS
        SELECT t.GEOID,
               substr(coalesce(p.addresses[1].postcode, ''), 1, 5) AS zip,
               regexp_replace(lower(coalesce(p.addresses[1].locality, '')), '[^a-z ]', '', 'g') AS loc,
               (p.categories.primary IN {F.sql_list(F.FIRE)}) AS is_fire,
               (list_contains(coalesce(p.categories.alternate, []), 'fire_department')
                OR (regexp_matches(lower(coalesce(p.names.primary, '')), '{FM.FIRE_NAME}')
                    AND regexp_matches(lower(coalesce(p.names.primary, '')), '{FM.FIRE_STRONG}')
                    AND NOT regexp_matches(lower(coalesce(p.names.primary, '')), '{FM.FIRE_NOT}')
                    AND NOT regexp_matches(lower(coalesce(p.categories.primary, '')), '{FM.CAT_NOT}'))) AS is_falt
        FROM tracts t JOIN '{F.ref(r, "overture-pois")}' p ON ST_Intersects(t.geom, p.geometry)""")
    df = con.sql("""
        SELECT GEOID, zip AS key, 'zip' AS kind, count(*) AS n,
               sum(is_fire::INT) AS nfire, sum((is_falt AND NOT is_fire)::INT) AS nfalt
        FROM tp WHERE regexp_matches(zip, '^[0-9]{5}$') GROUP BY 1, 2
        UNION ALL
        SELECT GEOID, loc AS key, 'loc' AS kind, count(*) AS n,
               sum(is_fire::INT) AS nfire, sum((is_falt AND NOT is_fire)::INT) AS nfalt
        FROM tp WHERE trim(loc) <> '' GROUP BY 1, 2
        ORDER BY kind, key, GEOID""").df()
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-juris.parquet", index=False)
    print(f"  {r}: {df.shape} rows, {df[df.kind == 'zip'].key.nunique()} zips, "
          f"{df[df.kind == 'loc'].key.nunique()} localities in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
