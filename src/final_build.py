"""The final builder (code review): rebuild a final submission from the challenge package and the committed parameter
files alone, check it against the uploaded CSV, and prove at runtime that no excluded module was loaded.

A final is described by `params/<name>_final.json`:

    {"base": LABEL, "components": {LABEL: SPEC, ...}, "combine": {LABEL: t, ...}}

    SPEC = {"kind": "final", "name": "<scored file>"}      another final, rebuilt recursively (a bet's scored base)
         | {"kind": "pieces", "weights": "<x>_weights.json"}
             the clean registry (rebuilt from the package by clean_proto.pieces) plus any "extras" (vehicle
             columns, `_extra`) under frozen weights; the weights were fitted to a teacher's geometry, never to a
             score (clean_geo.py)
         | {"kind": "bet", "on": LABEL, "vehicle": "physics9" | "physics10", "theta": "<file>.json", "draws": 400,
            "span_upto": "<last scored file when it was built>", "W": dose}
             on + W sd(y) x (the vehicle's part outside the scored span as it stood then), the E90 construction;
             with "theta_off": {weight: value} the vector is y(theta) - y(theta with those weights reset), one mechanism
         | {"kind": "learner", "on": LABEL, "span_upto": "<last scored file>", "W": dose, "bags": 1, "state": false}
             on + W sd(y) x the misfit learner's part outside the scored span as it stood then (misfit_gbm.column):
             LightGBM, seeded and deterministic, trained out of fold on the teacher's in-span misfit

and the final is clip(base + sum t (component - base)), with the two water tracts pinned at 1 (published.PIN_GEOIDS).
The t are the exact harvest of scored files (clean_proto.harvest), so they are the only leaderboard-derived numbers,
one per measured mechanism, each recorded in submissions/LEDGER.md.

    python src/final_build.py NAME            # rebuild from the package (~6 min) and compare with submissions/NAME
    python src/final_build.py NAME --cached   # same, from the cached registry (seconds)
"""
import json
import sys

import numpy as np

import lb
import published as PB

EXCLUDED = {"truthcard", "cand_cells", "diag_lineage"}


def _pieces(cached):
    import clean_proto as CP
    g, blocks = CP.pieces(cached=cached)
    return g, {f"{b}::{k}": v for b, C in blocks.items() for k, v in C.items()}


def _extra(e, g):
    """A package-only vehicle column used as a piece: the vehicle's y at a committed theta, raw, or (with
    "project_upto") its part outside the scored span as it stood then, unit sd."""
    import importlib
    V = importlib.import_module(e["vehicle"])
    d, X0, _ = V.inputs()
    assert (d.GEOID.astype(str).values == g.astype(str)).all(), "row order"
    y = V.model(d, X0, json.loads((lb.ROOT / "params" / e["theta"]).read_text()), draws=e["draws"])
    if e.get("project_upto"):
        import cand_mm as CMM
        L = lb.ledger()
        y = CMM.project(y, d, L.loc[:L.index[L.file == e["project_upto"]][0]])[0]
    return y


def _weighted(spec, g, flat):
    p = json.loads((lb.ROOT / "params" / spec["weights"]).read_text())
    R = np.column_stack([flat[k] for k in p["pieces"]] + [_extra(e, g) for e in p.get("extras", [])])
    q = np.clip(p["m1"] + (R - np.asarray(p["means"])) @ np.asarray(p["w"]), 0, 1)
    q[np.isin(g, PB.PIN_GEOIDS)] = 1.0
    return q


def _bet(spec, on, g):
    import cand_mm as CMM
    import importlib
    V = importlib.import_module(spec["vehicle"])
    L = lb.ledger()
    L = L.loc[:L.index[L.file == spec["span_upto"]][0]]
    d, X0, _ = V.inputs()
    assert (d.GEOID.astype(str).values == g.astype(str)).all(), "row order"
    th = json.loads((lb.ROOT / "params" / spec["theta"]).read_text())
    y = V.model(d, X0, th, draws=spec["draws"])
    if "theta_off" in spec:                       # a single mechanism: y(theta) - y(theta with it switched off)
        y = y - V.model(d, X0, dict(th, **spec["theta_off"]), draws=spec["draws"])
    col, perp = CMM.project(y, d, L)
    m1, m2 = lb.moments(L)
    q = np.clip(on + np.sqrt(m2 - m1 ** 2) * spec["W"] * col, 0, 1)
    q[np.isin(g, PB.PIN_GEOIDS)] = 1.0
    return q


def _learner(spec, on, g):
    import misfit_gbm as MG
    col, sd, geo = MG.column(spec["span_upto"], spec.get("bags", 1), spec.get("state", False))
    assert (geo == g.astype(str)).all(), "row order"
    q = np.clip(on + sd * spec["W"] * col, 0, 1)
    q[np.isin(g, PB.PIN_GEOIDS)] = 1.0
    return q


def _final(name, g, flat):
    f = json.loads((lb.ROOT / "params" / (name.replace(".csv", "") + "_final.json")).read_text())
    comp = {}
    for label, spec in f["components"].items():          # in order: a bet names a component defined before it
        if spec["kind"] == "pieces":
            comp[label] = _weighted(spec, g, flat)
        elif spec["kind"] == "final":                     # another final, rebuilt the same way (a scored base)
            comp[label] = _final(spec["name"], g, flat)[0]
        elif spec["kind"] == "bet":
            comp[label] = _bet(spec, comp[spec["on"]], g)
        elif spec["kind"] == "learner":
            comp[label] = _learner(spec, comp[spec["on"]], g)
        print(f"   {name}/{label}: {spec['kind']} rebuilt", flush=True)
    base = comp[f["base"]]
    q = base + sum(t * (comp[k] - base) for k, t in f["combine"].items())
    q = np.clip(q, 0, 1)
    q[np.isin(g, PB.PIN_GEOIDS)] = 1.0
    return q, comp


def build(name, cached=False):
    g, flat = _pieces(cached)
    q, comp = _final(name, g, flat)
    loaded = EXCLUDED & set(sys.modules)
    assert not loaded, f"excluded modules loaded: {sorted(loaded)}"
    return g, q, comp


if __name__ == "__main__":
    a = sys.argv[1:]
    name = a[0] if a[0].endswith(".csv") else a[0] + ".csv"
    g, q, comp = build(name, cached="--cached" in a)
    s = lb.read_sub(name)
    assert (s.GEOID.astype(str).values == g.astype(str)).all() and len(q) == 9794
    print(f"{name}: rebuilt from the package; max |rebuilt - uploaded| {np.abs(q - s.coverage_gap_score.values).max():.1e}; "
          f"pins {q[np.isin(g, PB.PIN_GEOIDS)].tolist()}; excluded modules loaded: none")
