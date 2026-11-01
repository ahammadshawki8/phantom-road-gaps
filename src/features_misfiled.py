"""Misfiled facilities per tract (E49 input), provided data only.

The POI facilities half counts an Overture place only by `categories.primary` (bucket README:
fire = `fire_department`, EMS = `ambulance_and_ems_services`, schools = six school categories).
A real fire station that Overture files as `fire_protection_service`, `police_department`, a
government office, or with `fire_department` only as an *alternate* category is in the reference
and not in the numerator, so where it is the tract's only station the fire gap is exactly 1.
A competitor with the reference in hand published the count (discussion thread 34925, a participant):
**740 fire-station blackouts, 270 of them on the map but misfiled.** That count is the signature
test; nothing here is fitted to it.

Usage:  python src/features_misfiled.py [region ...]
Output: $BB_DATA/derived/<region>-misfiled.parquet
"""
import sys
import time

import features as F

# a station, not a business that sells or services fire equipment
FIRE_NAME = (r"(fire|f\.d\.|vfd|v\.f\.d)\s*(dept|department|station|district|hall|company|co\b|"
             r"rescue|&\s*rescue|and rescue|protection district|authority|\#|no\.|\d)|volunteer fire|"
             r"\bvfd\b|\bfpd\b|\bfire\s*$|^fire station|^station \d+|fire\s*-\s*rescue|fire ems")
# names that say "this is a station" outright, rather than merely mentioning fire
FIRE_STRONG = r"station|volunteer|\bvfd\b|v\.f\.d|fire (dept|department|district|rescue|company)|fire & rescue|fire and rescue|\bfpd\b"
CAT_NOT = r"restaurant|shop|food|bar\b|cafe|pub|builders|event|landscap|beverage|concrete|store|retail|contractor"
FIRE_NOT = r"firehouse|fire house|subs\b|extinguisher|sprinkler|alarm|equipment|supply|safety|inspection|llc|\binc\b|systems|training|marshal|museum|insurance|restoration|hydrant|pit\b|wood|place|grill|brew|pizza|bbq"
SCHOOL_NAME = r"elementary|middle school|high school|junior high|intermediate school|\bisd\b|primary school|\bk-?8\b|charter school|\bschool$"
SCHOOL_NOT = r"driving|dance|martial|karate|music|art school|beauty|cosmetology|swim|gymnastics|cdl|truck|flight|barber|massage|nursing|language|tutor|preschool|day ?care|montessori|bible|sunday"
SCHOOL_REAL = r"elementary|middle school|high school|junior high|intermediate school|primary school|\bisd\b|\bk-?8\b|charter school|catholic school|christian school|preparatory|\bprep\b"
SCHOOL_MISCAT = ["education", "college_university", "religious_school", "", "community_services_non_profits",
                 "public_service_and_government", "specialty_school", "vocational_and_technical_school",
                 "church_cathedral", "christian_church", "catholic_church"]
EMS_NAME =r"ambulance|\bems\b|emergency medical|medic \d|life ?flight|air evac"


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    fire, ems, sch = F.sql_list(F.FIRE), F.sql_list(F.EMS), F.sql_list(F.SCHOOL)
    con.sql(f"""
        CREATE OR REPLACE TABLE f_mis AS
        WITH p AS (
            SELECT categories.primary AS cat, coalesce(categories.alternate, []) AS alt,
                   lower(coalesce(names.primary, '')) AS nm, geometry
            FROM '{F.ref(r, "overture-pois")}'),
        q AS (
            SELECT *,
                   cat IN {fire} AS fd,
                   coalesce(cat NOT IN {fire}, true) AND (list_has_any(alt, {list(F.FIRE)})
                       OR (regexp_matches(nm, '{FIRE_NAME}') AND NOT regexp_matches(nm, '{FIRE_NOT}')
                            AND NOT regexp_matches(coalesce(cat, ''), '{CAT_NOT}')))
                       AS fd_mis,
                   cat IN {ems} AS em,
                   coalesce(cat NOT IN {ems}, true) AND (list_has_any(alt, {list(F.EMS)})
                       OR regexp_matches(nm, '{EMS_NAME}')) AND coalesce(cat NOT IN {fire}, true) AS em_mis,
                   cat IN {sch} AS sc,
                   coalesce(cat NOT IN {sch}, true) AND (list_has_any(alt, {list(F.SCHOOL)})
                       OR (regexp_matches(nm, '{SCHOOL_NAME}') AND NOT regexp_matches(nm, '{SCHOOL_NOT}')))
                       AS sc_mis
            FROM p)
        SELECT t.GEOID,
               count(*) FILTER (WHERE fd) AS fd_n, count(*) FILTER (WHERE fd_mis) AS fd_mis_n,
               count(*) FILTER (WHERE em) AS em_n, count(*) FILTER (WHERE em_mis) AS em_mis_n,
               count(*) FILTER (WHERE sc) AS sc_n, count(*) FILTER (WHERE sc_mis) AS sc_mis_n,
               count(*) FILTER (WHERE cat = 'fire_protection_service') AS fps_n,
               count(*) FILTER (WHERE fd_mis AND (list_has_any(alt, {list(F.FIRE)})
                                OR regexp_matches(nm, '{FIRE_STRONG}'))) AS fd_mis_strong_n
        FROM tracts t JOIN q ON ST_Intersects(t.geom, q.geometry)
        GROUP BY t.GEOID""")
    # E51: a building Overture classes as a fire station (or names as one) is a station on the
    # map with no place record at all -- a third, independent trace of the same misfiling
    con.sql(f"""
        CREATE OR REPLACE TABLE f_bfs AS
        WITH b AS (
            SELECT ST_Point((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2) AS pt
            FROM '{F.ref(r, "overture-buildings")}'
            WHERE class = 'fire_station'
               OR (regexp_matches(lower(coalesce(names.primary, '')), '{FIRE_NAME}')
                   AND NOT regexp_matches(lower(coalesce(names.primary, '')), '{FIRE_NOT}')))
        SELECT t.GEOID, count(*) AS bfs_n
        FROM tracts t JOIN b ON ST_Intersects(t.geom, b.pt)
        GROUP BY t.GEOID""")
    # E59: CLEAN misfiled schools for the partial gap -- a real-school name in a category a school
    # can plausibly be misfiled under, and not a same-name duplicate of a properly filed school
    con.sql(f"""
        CREATE OR REPLACE TABLE f_scc AS
        WITH p AS (
            SELECT categories.primary AS cat, lower(coalesce(names.primary, '')) AS nm,
                   regexp_replace(lower(coalesce(names.primary, '')), '[^a-z]|school', '', 'g') AS k,
                   geometry
            FROM '{F.ref(r, "overture-pois")}'),
        q AS (SELECT t.GEOID, p.* FROM tracts t JOIN p ON ST_Intersects(t.geom, p.geometry)),
        prop AS (SELECT GEOID, k FROM q WHERE cat IN {sch}),
        mis AS (SELECT * FROM q
                WHERE coalesce(cat, '') IN {F.sql_list(SCHOOL_MISCAT)}
                  AND regexp_matches(nm, '{SCHOOL_REAL}') AND NOT regexp_matches(nm, '{SCHOOL_NOT}')
                  AND length(k) > 3
                  AND NOT EXISTS (SELECT 1 FROM prop WHERE prop.GEOID = q.GEOID
                                  AND (prop.k = q.k OR strpos(prop.k, q.k) > 0 OR strpos(q.k, prop.k) > 0)))
        SELECT GEOID, count(DISTINCT k) AS sc_mis_clean_n FROM mis GROUP BY GEOID""")
    # E60: misfiled fire stations that are NOT a same-name duplicate of a fire_department in the
    # tract (E50's partial gap was weak because most were duplicates); distinct names only
    con.sql(f"""
        CREATE OR REPLACE TABLE f_fdc AS
        WITH p AS (
            SELECT categories.primary AS cat, coalesce(categories.alternate, []) AS alt,
                   lower(coalesce(names.primary, '')) AS nm,
                   regexp_replace(lower(coalesce(names.primary, '')),
                                  '[^a-z0-9]|fire|department|dept|station|volunteer|rescue', '', 'g') AS k,
                   geometry
            FROM '{F.ref(r, "overture-pois")}'),
        q AS (SELECT t.GEOID, p.* FROM tracts t JOIN p ON ST_Intersects(t.geom, p.geometry)),
        prop AS (SELECT GEOID, k FROM q WHERE cat IN {fire}),
        mis AS (SELECT * FROM q
                WHERE coalesce(cat NOT IN {fire}, true)
                  AND (list_has_any(alt, {list(F.FIRE)})
                       OR (regexp_matches(nm, '{FIRE_NAME}') AND NOT regexp_matches(nm, '{FIRE_NOT}')
                           AND NOT regexp_matches(coalesce(cat, ''), '{CAT_NOT}')))
                  AND NOT EXISTS (SELECT 1 FROM prop WHERE prop.GEOID = q.GEOID
                                  AND (prop.k = q.k OR (length(q.k) > 2 AND (strpos(prop.k, q.k) > 0
                                       OR strpos(q.k, prop.k) > 0)))))
        SELECT GEOID, count(DISTINCT k) AS fd_mis_clean_n FROM mis GROUP BY GEOID""")
    df = con.sql("SELECT t.GEOID, f_mis.* EXCLUDE (GEOID), coalesce(f_bfs.bfs_n, 0) AS bfs_n, "
                 "coalesce(f_scc.sc_mis_clean_n, 0) AS sc_mis_clean_n, "
                 "coalesce(f_fdc.fd_mis_clean_n, 0) AS fd_mis_clean_n "
                 "FROM tracts t LEFT JOIN f_mis USING (GEOID) LEFT JOIN f_bfs USING (GEOID) "
                 "LEFT JOIN f_scc USING (GEOID) LEFT JOIN f_fdc USING (GEOID) ORDER BY GEOID").df().fillna(0)
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-misfiled.parquet", index=False)
    print(f"  {r}: {df.shape} in {time.time() - t0:.0f}s; fire blackout-by-misfiling "
          f"{int(((df.fd_n == 0) & (df.fd_mis_n > 0)).sum())}", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
