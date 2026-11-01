"""Honorific highway records that are route-log DESIGNATIONS, not names TIGER carries (E119), provided data only.

`features_concurrent` counts a third TIGER record (cc_named, E97 +0.045) wherever a core route's primary name is a
highway name naming none of its own route numbers -- "Katy Freeway" on I 10, "Will Rogers Turnpike" on I 44,
"Black Canyon Freeway" on I 17. TIGER keeps one record per feature name, and its feature names are the ones used
for addressing, which those are. Arizona's OpenStreetMap highways also carry ADOT's route-log names as their
primary name -- "Casa Grande–Tucson Highway" on I 10, "Benson–Steins Pass Highway", "Globe–Lordsburg Highway",
"Payson-Show Low Highway": one per state-highway section, a place-to-place designation that no address uses, so
TIGER holds no record under it. They are 58% of Arizona's counted honorific length and 0-2% anywhere else, which
is why Arizona's road level k sits at 0.60 on the spec vehicle where OK/CA/TX/WA sit at 1.00-1.04 (psi 0.99).

Rule: a counted honorific (the cc_named / bf_x_named rule) whose name ends "<place> - <place> Highway/Hwy", the
place names letters only, memorial names excepted (a hyphenated surname is not a route log, and removing memorial
names breaks Oklahoma's road level, so they are TIGER records). The list is route-log names only: 81 in Arizona,
and Mount Shasta - Mount Lassen (CA), Brownsville-Port Isabel (TX), The Dalles-California (WA). Same geometry as the rule it
qualifies: interior length on the core segment (EPSG:5070), and the boundary length whose Overture copy lies
outside the tract (`features_bndfollow`, STEP-sampled).

    ds_in   interior core-route length carrying a designation honorific   (subset of cc_named)
    ds_bd   boundary-following length, copy outside, carrying one           (subset of bf_x_named)

Usage:  python src/features_desig.py [region ...]
Output: $BB_DATA/derived/<region>-desig.parquet
"""
import sys
import time

import features as F
import features_bndfollow as FB
import features_concurrent as FC
import features_routes as FR

NAMED = (f"regexp_matches(nm, '{FC.HWY}') "
         f"AND len(list_filter(nums, f -> regexp_matches(nm, '(^|[^0-9])' || f || '([^0-9]|$)'))) = 0")
DESIG = (f"nm IS NOT NULL AND {NAMED} AND regexp_matches(nm, '[A-Za-z. ]+ ?[–—-] ?[A-Za-z. ]+ (Highway|Hwy)$') "
         f"AND NOT regexp_matches(nm, '(?i)memorial')")


def interior(con, r):
    return con.sql(f"""
        WITH s AS (
            SELECT names.primary AS nm, geometry,
                   len(list_filter(coalesce(routes, []), rr -> list_transform([rr.network], x -> {FR.CORE})[1])) AS ncore,
                   list_distinct(list_filter(list_transform(list_filter(coalesce(routes, []),
                                                                        rr -> list_transform([rr.network], x -> {FR.CORE})[1]),
                                                            q -> q.ref), v -> v IS NOT NULL AND v <> '')) AS nums
            FROM '{F.ref(r, "overture-roads")}'),
        x AS (SELECT t.GEOID, ST_Length(ST_Transform(ST_Intersection(t.geom, s.geometry), 'EPSG:4326', 'EPSG:5070',
                                                     always_xy := true)) AS len
              FROM tracts t JOIN (SELECT * FROM s WHERE ncore > 0 AND {DESIG}) s ON ST_Intersects(t.geom, s.geometry))
        SELECT t.GEOID, coalesce(sum(x.len), 0) AS ds_in FROM tracts t LEFT JOIN x USING (GEOID) GROUP BY 1""").df()


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    a = interior(con, r)
    b = FB.build(r, extra={"desig": DESIG}, write=False)[["GEOID", "bf_x_desig"]].rename(columns={"bf_x_desig": "ds_bd"})
    df = a.merge(b, on="GEOID", how="left").fillna(0.0).sort_values("GEOID")
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-desig.parquet", index=False)
    print(f"  {r}: designation records interior {df.ds_in.sum() / 1e3:,.0f} km, boundary {df.ds_bd.sum() / 1e3:,.0f} km, "
          f"{(df.ds_in + df.ds_bd > 0).sum()} tracts ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
