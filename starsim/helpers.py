import pandas as pd
import numpy as np
import starsim as ss

__all__ = ['build_network']

def clean_data(df: pd.DataFrame, beta: float = 1.0) -> pd.DataFrame:
    df = df[df["type"] == "contact"].copy()
    df = df.dropna(subset=["user_id", "peer_id", "time"])
    df = df.drop_duplicates(subset=["user_id", "peer_id", "time"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["beta"] = beta
    return df

def remap_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, int, dict]:
    all_ids = pd.Index(pd.unique(pd.concat([df["user_id"], df["peer_id"]], ignore_index=True)))
    id_map = {old: new for new, old in enumerate(all_ids)}

    out = df.copy()
    out["p1"] = out["user_id"].map(id_map).astype("int64")
    out["p2"] = out["peer_id"].map(id_map).astype("int64")
    return out, len(all_ids), id_map

def add_subdaily_timeline(
    df: pd.DataFrame,
    time_col: str = "time",
    duration_col: str = "contact_length",
    tick: pd.Timedelta = pd.Timedelta(seconds=10),
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Convert timestamps to a fixed subdaily simulation timeline.

    The default tick is 1 ms (thousandths of a second), which preserves the
    full resolution in contact timestamps.

    Assumes `duration_col` is stored in milliseconds.
    """
    out = df.copy()
    start_date = out[time_col].min().floor(tick)
    stop_date = out[time_col].max().ceil(tick) + tick

    out["start_offset_ti"] = ((out[time_col] - start_date) / tick).round().astype("int64")
    tick_ms = tick / pd.Timedelta(milliseconds=1)
    out["dur_ti"] = np.ceil(out[duration_col].fillna(0) / tick_ms).astype("int64").clip(lower=1)

    return out, start_date, stop_date

def build_network(csv_path: str):
    df = pd.read_csv(csv_path, index_col=0)
    df = clean_data(df)
    df, n_agents, id_map = remap_ids(df)
    df, start_date, stop_date = add_subdaily_timeline(df)
    net = ss.EpigamesNet(df, label="Epigames")
    start_date = ss.date(start_date)
    stop_date = ss.date(stop_date)
    return net, n_agents, start_date, stop_date, id_map