"""Per-tract TIGER boundary length that FOLLOWS a core route, measured on TIGER's own geometry (E111/E112).

**Why.** E19's mechanism: TIGER draws tract boundaries along its own road centrelines, and the scorer's
overlay intersection (EPSG:5070, no deduplication) gives a road lying on a shared boundary to BOTH
tracts. Overture's copy of that road sits a few metres to one side, so only one tract holds it in the
numerator. Every model since E19 has estimated the boundary road from Overture's copy -- the ring of
width w outside the tract (50 m, 27.3 m, 15.49 m, 7.91 m as the numerator moved). But the boundary
road's TIGER geometry is in the package: it *is* the tract polygon's edge. This module measures it
directly.

**How.** Sample every tract boundary every STEP m (EPSG:5070). For each sample point within D m of an
Overture core route, find the nearest core segment and the nearest non-core road (any class in the
scored roads layer). The point follows the core route when

    the core route is the nearest road (no non-core road is closer),
    it lies within D m, and it runs parallel to the boundary (angle <= ANG degrees).

The nearest-road test is what a single ring width cannot do: in metro areas a frontage road or a
parallel street 30-60 m from a freeway is what the boundary follows. [Corrected 28 Oct, LEDGER 120a:
the copy of the boundary highway itself sits a median 0.9-1.6 m off TIGER's line, with an exponential
tail that reaches ~20 m only in Oklahoma and Texas. A boundary 25-60 m from a highway follows a railway,
river, canal or right-of-way line, which `rail=True` and `features_bfmix` take out.] The side of the
matched Overture point (inside / outside the tract) splits the length:

    bf_in   boundary length whose Overture copy lies inside  (already in the numerator and in rt_core)
    bf_out  boundary length whose Overture copy lies outside (TIGER holds it, Overture does not)

**Toll authorities.** `features_routes.CORE` matches the interstate, US and state networks, but Texas
toll roads are filed under their authorities -- `US:TX:NTTA`, `US:TX:Harris:HCTRA`, `US:TX:CTRMA`,
`US:TX:Fort_Bend:FBCTRA`, `US:TX:Montgomery:MCTRA` and the managed-lane networks -- and match nothing.
They are limited-access highways (TIGER S1100, "may include some toll highways") and class `motorway`
in Overture, so they sit in the numerator and never in the reference or the mask. TOLL counts them
as core here; `toll_in` is their mainline length inside the tract (ramps excluded, named classes only).

Usage:  python src/features_bndfollow.py [region ...]
Output: $BB_DATA/derived/<region>-bndfollow.parquet
"""
import sys
import time

import numpy as np
import pandas as pd

import features as F
import features_concurrent as FC
import features_routes as FR

STEP, D, ANG = 10.0, 60.0, 20.0
TOLL = ("['US:TX:NTTA','US:TX:Harris:HCTRA','US:TX:CTRMA','US:TX:Fort_Bend:FBCTRA','US:TX:Montgomery:MCTRA',"
        "'US:I:Express:Toll','US:TX:Express:Toll','US:TX:CTRMA:Express','US:TX:Loop:Express:Toll']")
XY = "'EPSG:4326', 'EPSG:5070', always_xy := true"
CORE_OR_TOLL = (f"list_transform([rr.network], x -> ({FR.CORE}) OR list_contains({TOLL}, x))[1]")


def build(r, extra=None, write=True, rail=False, samples=False):
    """`extra` maps a rule name to an SQL condition on the core segment's `nm` / `nums` (features_desig); each adds a
    bf_x_<name> column of boundary length whose outside copy carries that rule. Without it the output is unchanged.
    `rail=True` (features_bfrail, E120) also rejects a sample whose nearest rail line is closer than the core route:
    tract boundaries follow railways too, and a railway beside a highway is not in the roads layer, so the
    nearest-road test alone cannot tell which of the two the boundary is. `samples=True` (features_bfmix) also
    returns the followed samples themselves (GEOID, dc, inside, x_*), one row per STEP m of boundary."""
    t0 = time.time()
    extra = extra or {}
    xsel = "".join(f"({cond})::INT AS x_{k},\n                   " for k, cond in extra.items())
    con = F.connect()
    F.load_tracts(con, r)
    con.sql(f"CREATE OR REPLACE TABLE tp AS SELECT GEOID, ST_Transform(geom, {XY}) AS g FROM tracts")
    # every scored road, single-part; ramps are never core (TIGER files them as S1630)
    con.sql(f"""
        CREATE OR REPLACE TABLE rd AS
        SELECT row_number() OVER () AS rid, * FROM (
            SELECT class, coalesce(subclass, '') = 'link' AS link,
                   (len(list_filter(nets, x -> {FR.CORE})) > 0
                    OR len(list_filter(nets, x -> list_contains({TOLL}, x))) > 0)
                     AND coalesce(subclass, '') <> 'link' AS core,
                   len(list_filter(nets, x -> list_contains({TOLL}, x))) > 0
                     AND len(list_filter(nets, x -> {FR.CORE})) = 0 AS toll,
                   -- TIGER's extra records on this centreline (features_concurrent's three rules)
                   greatest(len(refs) - 1, 0) AS x_multi,
                   (nm IS NOT NULL AND NOT regexp_matches(nm, '{FC.HWY}'))::INT AS x_local,
                   (nm IS NOT NULL AND regexp_matches(nm, '{FC.HWY}')
                    AND len(list_filter(nums, f -> regexp_matches(nm, '(^|[^0-9])' || f || '([^0-9]|$)'))) = 0)::INT
                     AS x_named,
                   {xsel}unnest(ST_Dump(ST_Transform(geometry, {XY}))).geom AS g
            FROM (SELECT class, subclass, names.primary AS nm,
                         coalesce(list_transform(routes, x -> x.network), []) AS nets,
                         list_distinct(list_transform(list_filter(coalesce(routes, []), rr -> {CORE_OR_TOLL}),
                                                      q -> q.network || '/' || coalesce(q.ref, ''))) AS refs,
                         list_distinct(list_filter(list_transform(list_filter(coalesce(routes, []), rr -> {CORE_OR_TOLL}),
                                                                  q -> q.ref), v -> v IS NOT NULL AND v <> '')) AS nums,
                         geometry
                  FROM '{F.ref(r, "overture-roads")}'))
        WHERE ST_GeometryType(g) = 'LINESTRING'""")
    # boundary sample points, in ring order (k) so that neighbours give the boundary direction
    con.sql("CREATE OR REPLACE TABLE rings AS SELECT GEOID, unnest(ST_Dump(ST_Boundary(g))).geom AS ln FROM tp")
    con.sql(f"""
        CREATE OR REPLACE TABLE bp AS
        WITH r AS (SELECT GEOID, ln, ST_Length(ln) AS L, row_number() OVER () AS ringid FROM rings
                   WHERE ST_Length(ln) > 0)
        SELECT GEOID, ringid,
               unnest(ST_Dump(ST_LineInterpolatePoints(ln, least(1.0, {STEP} / L), true))).geom AS p FROM r""")
    con.sql("CREATE OR REPLACE TABLE bp2 AS SELECT *, row_number() OVER (PARTITION BY ringid ORDER BY rowid) AS k FROM bp")
    con.sql(f"""
        CREATE OR REPLACE TABLE cand AS
        SELECT b.GEOID, b.ringid, b.k, b.p, min(ST_Distance(b.p, c.g)) AS dc,
               arg_min(c.rid, ST_Distance(b.p, c.g)) AS crid
        FROM bp2 b JOIN (SELECT rid, g FROM rd WHERE core) c ON ST_DWithin(b.p, c.g, {D}) GROUP BY ALL""")
    con.sql(f"""
        CREATE OR REPLACE TABLE cand2 AS
        SELECT a.*, coalesce(o.do_, 1e9) AS do_ FROM cand a LEFT JOIN (
            SELECT b.GEOID, b.ringid, b.k, min(ST_Distance(b.p, n.g)) AS do_
            FROM cand b JOIN (SELECT rid, g FROM rd WHERE NOT core) n ON ST_DWithin(b.p, n.g, {D})
            GROUP BY ALL) o USING (GEOID, ringid, k)""")
    src = "(SELECT *, 1e9 AS dr FROM cand2)"
    if rail:
        con.sql(f"CREATE OR REPLACE TABLE rl AS SELECT unnest(ST_Dump(ST_Transform(geometry, {XY}))).geom AS g "
                f"FROM '{F.ref(r, 'overture-rail')}'")
        con.sql(f"""
            CREATE OR REPLACE TABLE cand2r AS
            SELECT a.*, coalesce(w.dr, 1e9) AS dr FROM cand2 a LEFT JOIN (
                SELECT b.GEOID, b.ringid, b.k, min(ST_Distance(b.p, l.g)) AS dr
                FROM cand b JOIN rl l ON ST_DWithin(b.p, l.g, {D}) GROUP BY ALL) w USING (GEOID, ringid, k)""")
        src = "cand2r"
    pts = con.sql(f"""
        WITH a AS (SELECT a.*, c.g AS cg, ST_LineLocatePoint(c.g, a.p) AS fr, greatest(ST_Length(c.g), 1) AS cl
                   FROM {src} a JOIN rd c ON c.rid = a.crid)
        SELECT a.GEOID, a.ringid, a.k, a.dc, a.do_, a.dr, c.x_multi, c.x_local, c.x_named,{"".join(f" c.x_{k}," for k in extra)}
               ST_X(ST_LineInterpolatePoint(a.cg, greatest(0, a.fr - 5 / a.cl))) AS rx0,
               ST_Y(ST_LineInterpolatePoint(a.cg, greatest(0, a.fr - 5 / a.cl))) AS ry0,
               ST_X(ST_LineInterpolatePoint(a.cg, least(1, a.fr + 5 / a.cl))) AS rx1,
               ST_Y(ST_LineInterpolatePoint(a.cg, least(1, a.fr + 5 / a.cl))) AS ry1,
               ST_Contains(t.g, ST_ClosestPoint(a.cg, a.p)) AS inside
        FROM a JOIN tp t USING (GEOID) JOIN rd c ON c.rid = a.crid""").df()
    nb = con.sql("SELECT ringid, k, ST_X(p) AS x, ST_Y(p) AS y FROM bp2 "
                 "WHERE ringid IN (SELECT DISTINCT ringid FROM cand)").df()
    key = nb.sort_values(["ringid", "k"]).set_index(["ringid", "k"])
    nxt, prv = key.groupby(level=0).shift(-1), key.groupby(level=0).shift(1)
    dirs = pd.DataFrame({"bdx": (nxt.x.fillna(key.x) - prv.x.fillna(key.x)).values,
                         "bdy": (nxt.y.fillna(key.y) - prv.y.fillna(key.y)).values}, index=key.index)
    pts = pts.join(dirs, on=["ringid", "k"])
    rdx, rdy = pts.rx1 - pts.rx0, pts.ry1 - pts.ry0
    cos = np.abs(rdx * pts.bdx + rdy * pts.bdy) / (np.hypot(rdx, rdy) * np.hypot(pts.bdx, pts.bdy) + 1e-12)
    follow = (pts.dc <= D) & (np.degrees(np.arccos(np.clip(cos, 0, 1))) <= ANG) & (pts.dc <= pts.do_) & ~(pts.dr < pts.dc)
    f = pts[follow].groupby(["GEOID", "inside"]).size().unstack(fill_value=0) * STEP
    f = f.rename(columns={False: "bf_out", True: "bf_in"})
    # the boundary road's extra TIGER records (one per name), counted where its copy lies OUTSIDE the tract
    o = pts[follow & ~pts.inside.astype(bool)]
    f = f.join((o.groupby("GEOID")[["x_multi", "x_local", "x_named"] + [f"x_{k}" for k in extra]].sum() * STEP)
               .rename(columns=lambda c: "bf_" + c), how="outer")
    toll = con.sql("""
        SELECT t.GEOID, sum(ST_Length(ST_Intersection(s.g, t.g))) AS toll_in
        FROM tp t JOIN (SELECT g FROM rd WHERE toll AND NOT link
                        AND class IN ('motorway','trunk','primary','secondary')) s
          ON ST_Intersects(t.g, s.g) GROUP BY 1""").df()
    df = (con.sql("SELECT GEOID FROM tracts ORDER BY GEOID").df()
          .merge(f.reset_index(), on="GEOID", how="left").merge(toll, on="GEOID", how="left").fillna(0.0))
    cols = ["bf_out", "bf_in", "toll_in", "bf_x_multi", "bf_x_local", "bf_x_named"] + [f"bf_x_{k}" for k in extra]
    for c in cols:
        if c not in df:
            df[c] = 0.0
    df = df[["GEOID"] + cols]
    if write:
        F.OUT.mkdir(parents=True, exist_ok=True)
        df.to_parquet(F.OUT / f"{r}-bndfollow.parquet", index=False)
    print(f"  {r}: {df.shape}, bf_out {df.bf_out.sum() / 1e3:,.0f} km, toll {df.toll_in.sum() / 1e3:,.0f} km, "
          f"extra records multi/local/named {df.bf_x_multi.sum() / 1e3:,.0f}/{df.bf_x_local.sum() / 1e3:,.0f}/"
          f"{df.bf_x_named.sum() / 1e3:,.0f} km in {time.time() - t0:.0f}s", flush=True)
    if samples:
        keep = ["GEOID", "dc", "inside", "x_multi", "x_local", "x_named"] + [f"x_{k}" for k in extra]
        return df, pts[follow][keep].reset_index(drop=True)
    return df


if __name__ == "__main__":
    for r in (sys.argv[1:] or F.REGIONS):
        build(r)
