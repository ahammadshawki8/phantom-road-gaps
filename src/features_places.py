"""Place-record quality features per tract (E64+ screens), provided data only.

The CBP half compares ALL Overture places with employer establishments. Where Overture's place
records are stale or thin, it counts fewer places than there are businesses, whatever the true
density. The place layer carries fields no file has used:
    closed_share   operating_status = 'permanently_closed' (a stale record still counted)
    brand_share    places with a brand (chains are mapped everywhere; independents are not)
    web_share      places with a website or phone (record completeness)
    one_src_share  places backed by a single source dataset
    old_share      places whose newest source update is before 2024
    phone/site/social/email/street_share, conf_mean   E66 follow-ups: which part of record completeness
    src_meta / src_msft / src_fsq / src_bq   share of places whose sources include Meta / Microsoft /
                   Foursquare / BrightQuery (a business-registry dataset: the CBP-like universe)

Usage:  python src/features_places.py [region ...]
Output: $BB_DATA/derived/<region>-places.parquet
"""
import sys
import time

import features as F


def build(r):
    t0 = time.time()
    con = F.connect()
    F.load_tracts(con, r)
    df = con.sql(f"""
        WITH p AS (
            SELECT geometry,
                   operating_status = 'permanently_closed' AS closed,
                   brand.names.primary IS NOT NULL AS branded,
                   (len(coalesce(websites, [])) > 0 OR len(coalesce(phones, [])) > 0) AS web,
                   len(coalesce(sources, [])) <= 1 AS one_src,
                   coalesce(list_max(list_transform(sources, s -> s.update_time)), '') < '2024' AS old,
                   list_contains(list_transform(sources, s -> lower(s.dataset)), 'meta') AS meta,
                   list_contains(list_transform(sources, s -> lower(s.dataset)), 'microsoft') AS msft,
                   list_has_any(list_transform(sources, s -> lower(s.dataset)), ['foursquare', 'fsq']) AS fsq,
                   list_contains(list_transform(sources, s -> lower(s.dataset)), 'brightquery') AS bq,
                   len(coalesce(phones, [])) > 0 AS phone, len(coalesce(websites, [])) > 0 AS site,
                   len(coalesce(socials, [])) > 0 AS social, len(coalesce(emails, [])) > 0 AS email,
                   coalesce(addresses[1].freeform, '') <> '' AS street, confidence AS conf
            FROM '{F.ref(r, "overture-pois")}')
        SELECT t.GEOID, count(p.geometry) AS n_places,
               avg(closed::INT) AS closed_share, avg(branded::INT) AS brand_share,
               avg(web::INT) AS web_share, avg(one_src::INT) AS one_src_share, avg(old::INT) AS old_share,
               avg(meta::INT) AS src_meta, avg(msft::INT) AS src_msft, avg(fsq::INT) AS src_fsq, avg(bq::INT) AS src_bq, sum(bq::INT) AS n_bq,
               avg(phone::INT) AS phone_share, avg(site::INT) AS site_share, avg(social::INT) AS social_share,
               avg(email::INT) AS email_share, avg(street::INT) AS street_share, avg(conf) AS conf_mean
        FROM tracts t LEFT JOIN p ON ST_Intersects(t.geom, p.geometry)
        GROUP BY t.GEOID ORDER BY t.GEOID""").df()
    F.OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(F.OUT / f"{r}-places.parquet", index=False)
    print(f"  {r}: {df.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
