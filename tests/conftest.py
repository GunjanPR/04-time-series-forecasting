import numpy as np
import pandas as pd


def seasonal_frame(n_series: int = 3, length: int = 72, season: int = 12) -> pd.DataFrame:
    """Long-format series: a repeating seasonal pattern plus a little noise."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_series):
        pattern = rng.uniform(50, 150, season)
        y = np.tile(pattern, length // season + 1)[:length] + rng.normal(0, 1, length)
        ds = pd.date_range(end="2015-12-01", periods=length, freq="MS")
        rows.append(pd.DataFrame({"unique_id": f"S{i}", "ds": ds, "y": y}))
    return pd.concat(rows, ignore_index=True)
