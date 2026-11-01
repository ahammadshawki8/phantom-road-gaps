"""Assemble the modelling table: features + strata, one row per scored tract (9,794)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("BB_DATA", Path(__file__).resolve().parents[2] / "data"))
REGIONS = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
SCORED = {"eastern-ok": 1192, "eastern-wa": 415, "maricopa-az": 1593, "northern-ca": 591,
          "south-central-tx": 6003}


def sample_order():
    """GEOID order we submit in: Zindi's SampleSubmission.csv, which is every region's package sample submission
    sorted by GEOID, so it is rebuilt from the package when the Zindi file is absent (the public release)."""
    p = Path(__file__).resolve().parents[1] / "SampleSubmission.csv"
    if p.exists():
        return pd.read_csv(p, dtype={"GEOID": str}, usecols=["GEOID", "region"])
    s = pd.concat([pd.read_csv(DATA / "reference" / r / f"{r}-sample-submission.csv", dtype={"GEOID": str},
                               usecols=["GEOID"]).assign(region=r) for r in REGIONS])
    return s.sort_values("GEOID").reset_index(drop=True)


def load():
    feats, strat = [], []
    for r in REGIONS:
        f = pd.read_parquet(DATA / "derived" / f"{r}-features.parquet")
        assert len(f) == SCORED[r] and f.GEOID.is_unique, r
        feats.append(f)
        strat.append(pd.read_parquet(DATA / "strata" / r / f"{r}-strata-tract-table.parquet"))
    f = pd.concat(feats, ignore_index=True)
    ru = [c for c in f.columns if c.startswith("ru_len_")]
    f[ru] = f[ru].fillna(0)
    s = pd.concat(strat, ignore_index=True).drop_duplicates("GEOID")
    s = s.drop(columns=[c for c in s.columns if c in f.columns and c != "GEOID"])
    d = sample_order().merge(f.drop(columns="region"), on="GEOID", how="left")
    d = d.merge(s, on="GEOID", how="left")
    assert len(d) == 9794 and d.area_m2.notna().all()
    d["county"] = d.GEOID.str[:5]
    return d


if __name__ == "__main__":                      # derived/all.parquet: features + strata in submission order
    out = DATA / "derived" / "all.parquet"
    load().to_parquet(out, index=False)
    print(f"wrote {out}")
