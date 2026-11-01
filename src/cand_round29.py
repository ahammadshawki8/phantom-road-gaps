"""E119: Arizona's ADOT route-log highway names are not TIGER records -- the honorific-name rule re-identified on
the spec vehicle (V3), on the certified affine of the scored chain.

**What E118 left.** E118 (c +0.035, at its price) took the two smear terms out of the transport reference, and the
road level k -- fitted only to each region's own published cells -- came out at the measured TIGER/Overture tracing
ratio psi = 0.99 in four regions (OK 1.038 / CA 1.032 / TX 0.995 / WA 1.014) and at 0.601 in Arizona. A region
whose k sits 40% below the other four is a region whose reference the vehicle builds wrongly.

**Why.** E97 (+0.045) counts a third TIGER record wherever a core route's primary name is a highway name naming
none of its route numbers: TIGER keeps one record per feature name, and its names are addressing names ("Black
Canyon Freeway", "Katy Freeway"). Arizona's OSM highways also carry ADOT's route-log designations as their primary
name -- "Casa Grande-Tucson Highway" on I 10, "Benson-Steins Pass Highway", "Globe-Lordsburg Highway" -- one per
state-highway section, a place-to-place name no address uses, so TIGER holds no record under it. They are 4,400 km
of interior and 1,049 km of boundary record in Arizona, 58% of its counted honorific length, and at most 45 km in
any other region (`features_desig.py`). So the part is V3 with them taken out of the reference, minus V3.

**Unfitted identification** (k is fitted to each region's mean, nothing else): k_AZ 0.601 -> 0.733 with the other
four unmoved, Arizona's composite rural/urban 4.77 -> 4.39 against a published 3.95, the 16-cell rmse 0.0042 ->
0.0039, transport b/u 2.078 -> 1.993 toward 1.87. Every rival reading fails one of these (28 Oct dive): all of
Arizona's honorifics out (k 0.851, but rural/urban 4.62 -- the common freeway names are addressing names), its
honorifics and local names out (k 1.107, 4.90), memorial names everywhere (OK's k to 1.127), Purple Heart Trail
(k 0.612, rural/urban unmoved), future-interstate records (TX rural/urban 2.43 -> 2.39). The mask cannot move:
the records lie on core routes already in `den`.

    python src/cand_round29.py            # evidence: base, k by region, rural/urban, cells, part, traps, dose
    python src/cand_round29.py build      # e119_desig.csv
"""
import sys

import numpy as np
import pandas as pd

import strata as CC
import cand_knn as CK
import cand_mm as CMM
import cand_round25 as R
import cand_round26 as R6
import cand_round28 as R8
import features as F
import final_model as FM
import lb
import table
import validate

BASE_FILES = R8.BASE_FILES + ["e118_spec_ref.csv"]
RIDGE = 3e-6
NAME = "e119_desig.csv"
COMMENT = ("E119 Arizona's ADOT route-log highway names are not TIGER records: the honorific-name rule "
           "re-identified on the spec vehicle")


def desig():
    o = table.sample_order()[["GEOID"]]
    x = o.merge(pd.concat([pd.read_parquet(F.OUT / f"{r}-desig.parquet") for r in F.REGIONS]),
                on="GEOID", how="left").fillna(0.0)
    return (x.ds_in + x.ds_bd).values


def part():
    inner, bound = R6.counting_extras()
    den = R.dens()["both"] + inner + bound
    ds = desig()
    d, X0, o0, k0 = R8.model(den, 0.0, 0.0)
    _, X1, o1, k1 = R8.model(den - ds, 0.0, 0.0)
    assert ((X0["den"] > 0) == (X1["den"] > 0)).all(), "designation records changed the transport mask"
    return d, X0, X1, o0, o1, k0, k1


def base(L):
    """Ridge affine over the scored chain with E118 in it, pins set; (GEOID, vector, identity, validator, lambda)."""
    q, pred, err, lam = FM.affine(RIDGE, BASE_FILES, L)
    a = validate.analyse(q, L)
    g = lb.read_sub(BASE_FILES[0]).GEOID.values
    for geoid, v in R.PINNED.items():
        q[int(np.where(g == geoid)[0][0])] = v
    return g, q, pred, a["rmse_neutral"], lam


def levels(d, o, k):
    """k, composite rural/urban (model vs the published cells' own) by region, the 16-cell rmse, transport b/u."""
    rural, burn = CC.strata(d)
    reg, y, T = d.region.values, o["y"], o["T"]
    fit = reg != "eastern-wa"
    kr = R8.kreg(d, k)
    ru, err, n = {}, [], []
    for r, pub in CC.PUB16.items():
        m = reg == r
        cells = [m & rural & burn, m & rural & ~burn, m & ~rural & burn, m & ~rural & ~burn]
        nn = [c.sum() for c in cells]
        pr = (nn[0] * pub[0] + nn[1] * pub[1]) / (nn[0] + nn[1])
        pu = (nn[2] * pub[2] + nn[3] * pub[3]) / (nn[2] + nn[3])
        ru[r] = (y[m & rural].mean() / y[m & ~rural].mean(), pr / pu)
        err += [(y[c].mean() - p) ** 2 for c, p in zip(cells, pub)]
        n += nn
    return {"k": kr, "ru": ru, "cells": float(np.sqrt(np.average(err, weights=n))),
            "tbu": float(T[fit & burn].mean() / T[fit & ~burn].mean())}


def evidence(L=None):
    L = lb.ledger() if L is None else L
    m1, m2 = lb.moments(L)
    sd = float(np.sqrt(m2 - m1 ** 2))
    g, q, pred, val, lam = base(L)
    rb = 0.5 * (pred + val)
    print(f"base: {len(BASE_FILES)}-file affine ridge {RIDGE:g}, identity {pred:.6f}, validator {val:.6f}, "
          f"|lambda|_1 {np.abs(list(lam.values())).sum():.2f}, lambda_e118 {lam.get('e118_spec_ref.csv', np.nan):+.2f}; "
          f"dosing on the midpoint {rb:.6f}")
    d, X0, X1, o0, o1, k0, k1 = part()
    ds = desig()
    reg = d.region.values
    for r in sorted(set(reg)):
        print(f"   {r:17s} designation records {ds[reg == r].sum() / 1e3:7,.0f} km in {(ds[reg == r] > 0).sum():4d} tracts")
    st = {r: r.rsplit("-", 1)[1].upper() for r in set(reg)}
    for tag, o, k in [("V3", o0, k0), ("V3 - desig", o1, k1)]:
        v = levels(d, o, k)
        ks = " ".join(f"{st[r]} {x:.3f}" for r, x in v["k"].items())
        print(f"{tag:11s} k {ks} | transport b/u {v['tbu']:.3f} (1.87) | 16-cell rmse {v['cells']:.4f}")
        print("            rural/urban (model, published): "
              + ", ".join(f"{st[r]} {a:.2f}/{b:.2f}" for r, (a, b) in v["ru"].items()))
    h = o1["y"] - o0["y"]
    az = reg == "maricopa-az"
    col, perp = CMM.project(h, d, L)
    print(f"part: sd(h) {h.std():.5f}, perp {perp:.3f}, out-of-span size {h.std() * perp / sd:.4f} sd(y), reach "
          f"{(np.abs(h) > 1e-4).mean():.3f}, share of h^2 in Arizona {np.sum(h[az] ** 2) / np.sum(h ** 2):.2f}")
    print(CK.traps({"desig": col}, d, X0["den"] > 0).round(3).to_string())
    name, rt, W = R8.dose(rb, sd)
    print(f"W* to pass {name} {rt:.9f} = {W:.4f}")
    for c in [0.0, W / 2, 0.035, W, 0.05, h.std() * perp / sd]:
        print(f"   c {c:+.4f} -> {sd * np.sqrt((rb / sd) ** 2 + W * (W - 2 * c)):.6f}")
    return d, g, q, col, W, sd


def build(L=None):
    L = lb.ledger() if L is None else L
    d, g, q0, col, W, sd = evidence(L)
    assert (g == d.GEOID.values).all()
    q = np.clip(q0 + sd * W * col, 0, 1)
    for geoid, v in R.PINNED.items():
        q[int(np.where(g == geoid)[0][0])] = v
    a = validate.analyse(q, L)
    print(f"{NAME}: W {W:.4f}, neutral {a['rmse_neutral']:.6f}, clipped at 0: {(q == 0).sum()}")
    lb.write(NAME, lb.read_sub(BASE_FILES[0]).GEOID, q, COMMENT, kind="model")


if __name__ == "__main__":
    build() if sys.argv[1:] == ["build"] else evidence()
