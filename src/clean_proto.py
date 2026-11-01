"""DIAGNOSTIC ONLY (29 Oct, `docs/shawki_oct_29_handoff.md` 3): the clean re-solve of `docs/oct29_primer.md` 5.2,
priced on the clean pieces that exist without a rebuild. Like `diag_lineage`, it reads the whole scored span (to get
each piece's covariance with y), so it is the WEIGHTS step. A final builder that loads these weights must not import
this module.

Clean pieces (no e33/e37/e38/e42/e45 vector, no file lineage):
  const; the early probes before E33 (package builders, `verify_all` EXACT/PROVEN); the raw vehicles V4 and V8
  (`cand_round30.vehicles`); V8's four raw sub-parts A-D (`cand_round31.parts`); the R19 variant (`cand_round32.part`);
  the indicator of the two scorer-pinned water tracts.
  plus the raw post-E33 parts in `RAW` (each builder's `hypotheses`/vehicle output BEFORE projection).
Each centred piece's covariance with y is its truncated projection on the scored span, exactly as `validate.analyse`
does it (lstsq at `validate.RCOND`, span = `validate.scored`, so e33/e37 are never read): out-of-span parts count as
neutral. Weights solve (S + ridge) w = c; q = clip(m1 + Rc w), pins = 1; scored by `validate.analyse` (neutral).

**Do NOT use `diag_lineage.atoms` for this (29 Oct, measured).** Full Gram-Schmidt keeps every atom down to norm
1e-7, and the atoms of affine/blend files are clip residues whose covariances are mostly public-sampling noise. A
56-piece fit loads on them and "certifies" 0.048, which is impossible (E27's failure mode). The truncated projection
is the calibrated one (PRED_SE 0.00015 over six in-span blends).

The printed neutral has the same optimism as any affine (LEDGER 101a, 121b(2f): B22's was +7.7e-5 at 22 files), so
add +5e-5 to +1e-4 (more for more pieces) before comparing with a scored file.

    python src/clean_proto.py          # rebuild the pieces, cache them, price every block and ridge (~4 min)
    python src/clean_proto.py cached   # price from ../data/derived/clean_pieces.parquet (~20 s)
    python src/clean_proto.py build e123_clean.csv 1e-2   # write the file + params/e123_clean_weights.json
    python src/clean_proto.py rebuild e123_clean.csv      # the file again from its weights JSON alone (must be ~0)
    python src/clean_proto.py mix e124_b1.csv e123_clean.csv q2_b1.csv 0.5   # one block on the scored anchor
    python src/clean_proto.py harvest e123_clean.csv e124_b1.csv,e125_b2.csv [out.csv]   # exact read, joint harvest
"""
import contextlib
import io
import json
import sys

import numpy as np

import pandas as pd

import cand_round30 as R30
import cand_round31 as R31
import cand_round32 as R32
import published as DL
import lb
import validate

RIDGES = (3e-5, 1e-4, 3e-4, 1e-3)
SD_Y = 0.0890225                      # public SD of y, from the two constant submissions (CLAUDE.md 3)
CACHE = lb.ROOT.parent / "data" / "derived" / "clean_pieces.parquet"


def _aligned(v, geoids, g):
    return pd.Series(np.asarray(v, float), index=pd.Index(geoids).astype(str)).reindex(g.astype(str)).values


def _fire_family(g):
    """E49-E66: the misfiled/missing-station and school parts, raw h from each builder's `hypotheses` (29 Oct)."""
    import cand_fire as CF, cand_fire2 as CF2, cand_next as CN, cand_round3 as R3, cand_round4 as R4, cand_round5 as R5
    keys = [(CF, CF.table, {"blackout": "e49_fireblk", "partial": "e50_firepart"}),
            (CF2, CF.table, {"ems": "e51_ems", "bldfire": "e52_bldfire", "nfac": "e53_nfac"}),
            (CN, CF.table, {"schools": "e56_schools", "tribal": "e57_tribal"}),
            (R3, CF.table, {"school_partial": "e58_schpart", "school_clean": "e59_schclean",
                            "fire_partial_dedup": "e60_firededup"}),
            (R4, R4.table, {"settle": "e62_settle", "zipcbp": "e63_zipcbp"}),
            (R5, R5.table, {"tribal_legal": "e64_tribal_legal", "settle_loc": "e65_settle_loc", "web_gap": "e66_web"})]
    out = {}
    for mod, table, names in keys:
        H, _, d, *_ = mod.hypotheses(table())
        out.update({f: _aligned(H[k], d.GEOID, g) for k, f in names.items()})
    return out


def _transport(g):
    """E34 ring width, E36 ramp-free reference, E109 ramps restored on the spec vehicle: raw transport parts."""
    import cand_level as CL, cand_ramp as CRP, cand_ring as CG, cand_round23 as R23
    geo = CL.table_all().GEOID
    out = {"e34_ring27": _aligned(CG.hypothesis(), geo, g), "e36_ramp": _aligned(CRP.hypothesis(), geo, g)}
    d, o0, _ = R23.physics_y(1.0, quiet=True)
    _, o1, _ = R23.physics_y(0.0, quiet=True)
    out["e109_ramp_in"] = _aligned(o1["y"] - o0["y"], d.GEOID, g)
    return out


def _local(g):
    """E41 neighbourhood completeness (six readings) and E46's two POI mechanisms."""
    import cand_knn as CK, cand_poi2 as CP2
    H, _, d = CK.hypotheses()
    out = {f"e41_{k}": _aligned(v, d.GEOID, g) for k, v in H.items()}
    H, _, d = CP2.hypotheses()
    out.update({f"e46_{k}": _aligned(v, d.GEOID, g) for k, v in H.items()})
    return out


def _physics(g):
    """The fitted physics v8 vehicle (24 Oct fit on the 39-file span). Its theta is committed at
    `params/physics8_theta.json`; copy it to ../data/derived/ if that copy is missing."""
    import json
    import physics8 as P8
    src = P8.FIT if P8.FIT.exists() else lb.ROOT / "params" / "physics8_theta.json"
    d, X0 = P8.inputs()
    return {"P8": _aligned(P8.model(d, X0, json.loads(src.read_text()), draws=400), d.GEOID, g)}


def _spec(g):
    """Block B1 (29 Oct, after E123; not in E123): the spec vehicles behind E111/E115/E117/E118/E119. V0 takes the
    boundary highway from TIGER's own line, V1 adds the interior one-record-per-name records, V2 the boundary
    records, and V3 is V2 with the two smear terms at 0 (~22 s). Residual of E120 on the registry 0.01077 -> 0.01031;
    validator at ridge 1e-2 0.0554403 -> 0.0553589 (before optimism)."""
    import cand_round25 as R25, cand_round26 as R26, cand_round28 as R28
    v0 = R25.dens()["both"]
    inner, bound = R26.counting_extras()
    out = {}
    for k, den in {"V0": v0, "V1": v0 + inner, "V2": v0 + inner + bound}.items():
        d, _, o = R25.model(den)
        out[k] = _aligned(o["y"], d.GEOID, g)
    d, _, o, _ = R28.model(v0 + inner + bound, 0.0, 0.0)
    out["V3"] = _aligned(o["y"], d.GEOID, g)
    return out


def _physics_fitted(g):
    """Block B1/B2 (29 Oct, Shawki): the fitted physics v5/v6/v7 vehicles behind E90 (+0.130), E93 and E94/E95, raw
    E[y] before any projection, from the thetas committed in `params/` (physics5.model; missing keys are neutral).
    physics5.model resets physics3's module globals, so they are saved and restored around it."""
    import physics3 as P3
    import physics5 as P5
    keep = {k: getattr(P3, k) for k in ("SIGMA_T", "SIGMA_C", "P_BLD", "URBAN_BG", "P_COMM", "M_S", "W_ST", "LAM_U",
                                        "LAM_T")}
    defaults = P3.simulate.__defaults__
    try:
        d, X0 = P5.inputs()
        out = {}
        for k, f in {"P5_e90": "physics5_e90_theta.json", "P6_e93": "physics6_e93_theta.json",
                     "P7_e94": "physics7_e94_theta.json"}.items():
            th = {a: b for a, b in json.loads((lb.ROOT / "params" / f).read_text()).items() if not a.startswith("_")}
            out[k] = _aligned(P5.model(d, X0, dict(P5.THETA0, **th), draws=400), d.GEOID, g)
        return out
    finally:
        for k, v in keep.items():
            setattr(P3, k, v)
        P3.simulate.__defaults__ = defaults


class _Captured(Exception):
    pass


def _capture(call, keep=lambda i, kw: i == 0, stop_after=1):
    """Run a builder and capture the raw vector(s) it hands to `cand_mm.project` (the piece BEFORE projection, handoff
    6), aligned by the frame's GEOID. `keep(i, kwargs)` selects calls; the builder is stopped after `stop_after`
    captures (None: run to the end)."""
    import cand_mm as CMM
    orig, got, i = CMM.project, [], [0]

    def spy(h, d, L, extra=()):
        if keep(i[0], {"extra": extra}):
            got.append((np.asarray(h, float).copy(), d.GEOID.values))
            if stop_after and len(got) >= stop_after:
                raise _Captured
        i[0] += 1
        return orig(h, d, L, extra=extra)

    CMM.project = spy
    try:
        call()
    except _Captured:
        pass
    finally:
        CMM.project = orig
    return got


def _b2(g):
    """Block B2 (29 Oct, Shawki): the post-E33 parts E124's geometric fit still misses most (handoff 6 / LEDGER 122b),
    each the raw vector its own builder projects: E105 school breadth, E48 commercial CBP, physics v1 (E73) and v3b
    (E76) means, E69/E70, E81/E82, E85/E86, E95 (physics v7 tribal difference), E97, E99, E102."""
    import physics3 as P3
    import physics4 as P4
    out = {}
    one = lambda name, call, **kw: out.update({name: _aligned(*_capture(call, **kw)[0], g)})
    import cand_round19 as R19
    one("e105_schoolbreadth", R19.candidates)
    import cand_comm as CCM
    one("e48_commercial", CCM.candidate, keep=lambda i, kw: i == 1)
    import physics as P1
    one("e73_physics1", P1.evidence)
    keep = {k: getattr(P3, k) for k in ("SIGMA_T", "inputs", "BASE")}
    try:                                    # E76 ran before physics4 existed: physics3's own inputs and sigma_t
        P3.inputs, P3.SIGMA_T = P4._inputs3, 0.15
        one("e76_physics3", P3.evidence)
    finally:
        for k, v in keep.items():
            setattr(P3, k, v)
    for mod, names in [("cand_round7", ["e69_nostreet", "e70_tertiary"]), ("cand_round8", ["e81_te_import", "e82_lo_edit"]),
                       ("cand_round9", ["e85_multi", "e86_local"])]:
        m = __import__(mod)
        got = _capture(m.candidates, keep=lambda i, kw: len(kw["extra"]) > 0, stop_after=None)
        for nm, (h, geo) in zip(names, got):
            out[nm] = _aligned(h, geo, g)
    import cand_round10 as R10, cand_round12 as R12, cand_round17 as R17
    one("e97_named", R10.candidates, keep=lambda i, kw: i == 4)   # calls 0-3 re-verify E85/E86
    one("e99_joint", R12.candidates)
    one("e102_recall", lambda: R17.candidate("core"))
    out.update(_physics_tribal(g))
    return out


def _physics_tribal(g):
    """E95: physics v7 with and without its tribal-land road term (W_TRIB 0.215 on the fraction scale)."""
    import physics3 as P3
    import physics5 as P5
    keep = {k: getattr(P3, k) for k in ("SIGMA_T", "SIGMA_C", "P_BLD", "URBAN_BG", "P_COMM", "M_S", "W_ST", "LAM_U",
                                        "LAM_T")}
    defaults = P3.simulate.__defaults__
    try:
        d, X0 = P5.inputs()
        th = {a: b for a, b in json.loads((lb.ROOT / "params" / "physics7_e94_theta.json").read_text()).items()
              if not a.startswith("_")}
        th = dict(P5.THETA0, **th)
        h = P5.model(d, X0, th, draws=400) - P5.model(d, X0, dict(th, W_TRIB=0.0), draws=400)
        return {"e95_tribal": _aligned(h, d.GEOID, g)}
    finally:
        for k, v in keep.items():
            setattr(P3, k, v)
        P3.simulate.__defaults__ = defaults


# rebuilt post-E33 parts, in primer 4.3's value order where the builder is a plain `hypotheses`/vehicle call.
# E123 = every block up to "physics"; later blocks are new pieces only, so `rebuild` keeps E123 reproducible.
RAW = {"fire family": _fire_family, "transport": _transport, "local": _local, "physics": _physics, "spec": _spec,
       "physics fitted": _physics_fitted, "b2": _b2}


def extend(names):
    """Compute only the named RAW blocks and add them to the cached registry (the rest is untouched)."""
    t = pd.read_parquet(CACHE)
    g = t.GEOID.values
    for name in names:
        with contextlib.redirect_stdout(io.StringIO()):
            C = RAW[name](g)
        for k, v in C.items():
            t[f"{name}::{k}"] = np.nan_to_num(np.asarray(v, float))
        print(f"   {name}: {len(C)} pieces added to the cache")
    t.to_parquet(CACHE, index=False)


# Blocks added to the registry later with `extend`, each in a fresh process: they are built on the module state as it
# stands before the first RAW block. The original blocks (fire family .. spec) run in sequence as they did when E123-E125's
# weights were fitted: the spec block's V0-V3 are built on the physics globals the 'physics' block leaves behind (physics
# v8's theta), and building them on the fresh state gives different vectors (corr 0.55-0.95), i.e. different meanings
# for their frozen weights. b2 built in sequence loses its calibration bracket (30 Oct); built fresh it reproduces the
# cache exactly.
FRESH = ("physics fitted", "b2")
_STATE_MODS = ("physics", "physics2", "physics3", "physics4", "physics5", "physics6", "physics7", "physics8", "physics9",
               "cand_round30")


def _snapshot():
    return {m: dict(vars(sys.modules[m])) for m in _STATE_MODS if m in sys.modules}


def _restore(saved):
    for m, v in saved.items():
        mod = sys.modules[m]
        for k in [k for k in vars(mod) if k not in v]:
            delattr(mod, k)
        for k, val in v.items():
            setattr(mod, k, val)


def pieces(cached=False):
    """-> (GEOIDs in submission order, {block: {piece: vector}}); the registry is cached as a parquet."""
    if cached and CACHE.exists():
        t = pd.read_parquet(CACHE)
        blocks = {}
        for c in t.columns[1:]:
            b, k = c.split("::", 1)
            blocks.setdefault(b, {})[k] = t[c].values
        return t.GEOID.values, blocks
    L = lb.ledger()
    P = L[L.kind == "probe"].reset_index(drop=True)
    early = list(P.file[:list(P.file).index("e33_cells.csv")])
    g = lb.read_sub(early[0]).GEOID.values
    with contextlib.redirect_stdout(io.StringIO()):
        d, X4, o4, k4, X8, o8, k8, par = R30.vehicles()
        _, _, H = R31.parts()
        _, _, r19 = R32.part()
    blocks = {
        "early": {f: lb.read_sub(f).coverage_gap_score.values for f in early},
        "vehicles": {"V4": _aligned(o4["y"], d.GEOID, g), "V8": _aligned(o8["y"], d.GEOID, g)},
        "sub-parts": {k: _aligned(v, d.GEOID, g) for k, v in dict(H, R19=r19).items()},
        "pins": {"pin": np.isin(g, DL.PIN_GEOIDS).astype(float)},
    }
    fresh = _snapshot()
    for name, f in RAW.items():
        if name in FRESH:
            _restore(fresh)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                blocks[name] = f(g)
        except Exception as e:                     # a missing derived parquet must not stop the pricing
            print(f"   {name}: not rebuilt ({type(e).__name__}: {e})")
    t = pd.DataFrame({"GEOID": g})
    for b, C in blocks.items():
        for k, v in C.items():
            t[f"{b}::{k}"] = np.nan_to_num(np.asarray(v, float))
    t.to_parquet(CACHE, index=False)
    return g, {b: {k: t[f"{b}::{k}"].values for k in C} for b, C in blocks.items()}


def covariances(R, L):
    """Truncated projection of each centred column of R on the scored span: (cov with y, in-span share)."""
    m1, m2 = lb.moments(L)
    files, P, c = validate.scored(L, m1, m2)
    Pc = P - P.mean(0)
    Rc = R - R.mean(0)
    B = np.linalg.lstsq(Pc, Rc, rcond=validate.RCOND)[0]
    res = Rc - Pc @ B
    return B.T @ c, 1 - res.var(0) / np.maximum(Rc.var(0), 1e-18)


def solve(R, cov, m1, pin, ridge, L):
    Rc = R - R.mean(0)
    S = Rc.T @ Rc / len(Rc)
    w = np.linalg.solve(S + ridge * np.diag(np.diag(S)), cov)    # ridge relative to each piece's own variance
    q = np.clip(m1 + Rc @ w, 0, 1)
    q[pin == 1] = 1.0
    return float(validate.analyse(q, L)["rmse_neutral"]), w, q


def report(cached=False):
    L = lb.ledger()
    m1, m2 = lb.moments(L)
    g, blocks = pieces(cached)
    pin = blocks["pins"]["pin"]
    cols, names = [], []
    for b, C in blocks.items():
        cols += list(C.values())
        names += list(C)
        R = np.column_stack(cols)
        cov, share = covariances(R, L)
        for ridge in RIDGES:
            r, w, _ = solve(R, cov, m1, pin, ridge, L)
            wn = np.abs(w * R.std(0)).sum() / np.sqrt(m2 - m1 ** 2)     # weight norm in units of sd(y)
            print(f"+ {b:11s} k={R.shape[1]:3d} ridge {ridge:g}: neutral {r:.7f}  |w sd|_1/sd(y) {wn:6.2f}"
                  f"  min in-span {share.min():.3f}", flush=True)
    print("E120 (scored, contaminated) 0.0556538. The neutral is NOT the expected score for a fit this size: price it "
          "with `python src/clean_sim.py cov` (optimism ~+5e-4 to +8e-4 after the B22 calibration)")


def build(name, ridge=1e-2):
    """Write the clean re-solve over the cached registry and its weights (`params/<name>_weights.json`), so that a
    code-clean final builder can rebuild the file from the pieces alone: q = clip(m1 + sum w_k (R_k - mean_k)), pins 1."""
    L = lb.ledger()
    m1, m2 = lb.moments(L)
    g, blocks = pieces(cached=True)
    names = [f"{b}::{k}" for b, C in blocks.items() for k in C]
    R = np.column_stack([v for C in blocks.values() for v in C.values()])
    cov, share = covariances(R, L)
    r, w, q = solve(R, cov, m1, blocks["pins"]["pin"], ridge, L)
    out = lb.ROOT / "params" / (name.replace(".csv", "") + "_weights.json")
    out.write_text(json.dumps({"ridge": ridge, "m1": m1, "pieces": names, "means": R.mean(0).tolist(),
                               "w": w.tolist(), "validator_neutral": r}, indent=1))
    lb.write(name, g, q, f"clean re-solve over {len(names)} package-only pieces, ridge {ridge:g} "
                         f"(validator {r:.6f}; weights {out.name})", kind="model")
    print(f"{name}: {len(names)} pieces, ridge {ridge:g}, validator neutral {r:.7f}; weights -> {out}")


def rebuild(name):
    """Rebuild a written clean file from `params/<name>_weights.json` and the cached pieces alone (no scored file, no
    covariance), and return the max abs difference from the CSV. The weights step is frozen in the JSON, so the file
    stays reproducible when later blocks are added to the registry; this is the core of the final builder."""
    p = json.loads((lb.ROOT / "params" / (name.replace(".csv", "") + "_weights.json")).read_text())
    g, blocks = pieces(cached=True)
    flat = {f"{b}::{k}": v for b, C in blocks.items() for k, v in C.items()}
    R = np.column_stack([flat[k] for k in p["pieces"]])
    q = np.clip(p["m1"] + (R - np.asarray(p["means"])) @ np.asarray(p["w"]), 0, 1)
    q[flat["pins::pin"] == 1] = 1.0
    s = lb.read_sub(name)
    assert (s.GEOID.astype(str).values == g.astype(str)).all(), "row order differs from the written file"
    return float(np.abs(q - s.coverage_gap_score.values).max())


def mix(name, base, other, t=0.5):
    """Write base + t*(other - base), pins at 1: one block on a scored anchor (handoff 5). For t in [0, 1] it is a
    convex combination of two files in [0, 1], so the clip never binds."""
    a, b = lb.read_sub(base), lb.read_sub(other)
    assert (a.GEOID.values == b.GEOID.values).all(), "row order differs"
    qa = a.coverage_gap_score.values
    q = np.clip(qa + t * (b.coverage_gap_score.values - qa), 0, 1)
    q[np.isin(a.GEOID.values, DL.PIN_GEOIDS)] = 1.0
    lb.write(name, a.GEOID, q, f"{base} + {t:g} x ({other} - {base})", kind="model")
    print(f"{name}: {base} + {t:g} x ({other} - {base}); rms step {np.sqrt(((q - qa) ** 2).mean()):.5f}")


def harvest(base, files, name=None):
    """Read each SCORED file in `files` as a block d_i = file_i - base on the SCORED `base`, exactly:
    g_i = (mean(d_i^2) + MSE_base - MSE_i)/2 is the public covariance of d_i with (y - base); mean(d_i d_j) is taken
    over all tracts (its public-vs-all error is a few per cent of g at these sizes). The joint best point is
    t = M^-1 g, M = mean(d_i d_j), with public MSE_base - g'M^-1 g before clipping (clipping can only lower it).
    Writes the harvest if `name` is given."""
    L = lb.ledger()
    rm = dict(zip(L.file, L.rmse.astype(float)))
    b = lb.read_sub(base)
    qb = b.coverage_gap_score.values
    D = np.column_stack([lb.read_sub(f).coverage_gap_score.values - qb for f in files])
    msb = rm[base] ** 2
    g = np.array([((D[:, i] ** 2).mean() + msb - rm[f] ** 2) / 2 for i, f in enumerate(files)])
    M = D.T @ D / len(qb)
    t = np.linalg.solve(M, g)
    for i, f in enumerate(files):
        c = g[i] / (SD_Y * np.sqrt(M[i, i]))
        print(f"  {f}: g {g[i]:+.3e}  c {c:+.4f} (noise ~0.019)  t* alone {g[i] / M[i, i]:+.3f}  joint t {t[i]:+.3f}")
    pred = np.sqrt(msb - g @ t)
    print(f"harvest on {base} (public {rm[base]:.9f}): predicted public {pred:.7f}")
    if name:
        q = np.clip(qb + D @ t, 0, 1)
        q[np.isin(b.GEOID.values, DL.PIN_GEOIDS)] = 1.0
        lb.write(name, b.GEOID, q, f"harvest on {base} of {', '.join(files)} at t {np.round(t, 4).tolist()}",
                 kind="model")
    return t, pred


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["build"]:
        build(a[1], float(a[2]) if len(a) > 2 else 1e-2)
    elif a[:1] == ["rebuild"]:
        print(f"{a[1]}: max |rebuilt - written| {rebuild(a[1]):.1e}")
    elif a[:1] == ["mix"]:
        mix(a[1], a[2], a[3], float(a[4]) if len(a) > 4 else 0.5)
    elif a[:1] == ["harvest"]:
        harvest(a[1], a[2].split(","), a[3] if len(a) > 3 else None)
    elif a[:1] == ["extend"]:
        extend(a[1].split(","))
    else:
        report(cached="cached" in a)
