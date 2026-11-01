"""The two stratum definitions and the derived-table loader every builder uses (moved out of `cand_cells` on 30 Oct
so that no final-path module imports the E33 builder; `cand_cells` re-exports them unchanged)."""
import pandas as pd

import lb
from published import PUB16, RURAL  # noqa: F401  (re-exported: builders read CC.PUB16)


def table_all():
    return pd.read_parquet(lb.ROOT.parent / "data" / "derived" / "all.parquet")


def strata(d):
    """(rural, burned): rural = ruca_primary >= 4, burned = MTBS ever-burned."""
    return (d.ruca_primary.fillna(1).values >= RURAL,
            d.mtbs_wildfire_ever.astype(bool).values)
