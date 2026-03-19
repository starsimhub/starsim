import numpy as np 
import pandas as pd 
from typing import Tuple
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
import starsim as ss 


def read_data(data,):
    df = pd.read_csv(data, index_col=0)
    return df 

def clean_data(df: pd.DataFrame, beta: float = 1.0) -> pd.DataFrame:
    df = df[df["type"] == "contact"].copy()
    df = df.dropna(subset=["user_id", "peer_id", "time"])
    df = df.drop_duplicates(subset=["user_id", "peer_id", "time"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["beta"] = beta
    return df

def remap_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    all_ids = pd.Index(pd.unique(pd.concat([df["user_id"], df["peer_id"]], ignore_index=True)))
    id_map = {old: new for new, old in enumerate(all_ids)}

    out = df.copy()
    out["p1"] = out["user_id"].map(id_map).astype("int64")
    out["p2"] = out["peer_id"].map(id_map).astype("int64")
    return out, len(all_ids)

def add_daily_timeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    out = df.copy()
    start_date = out["time"].dt.floor("D").min()
    stop_date = out["time"].dt.floor("D").max() + pd.Timedelta(days=1)

    out["start_ti"] = (out["time"].dt.floor("D") - start_date).dt.days.astype("int64")
    out["dur_ti"] = np.ceil(out["contact_length"].fillna(0) / 86400000).astype("int64").clip(lower=1)

    return out, start_date, stop_date

class EpigamesNet(ss.DynamicNetwork):
    def __init__(self, df, start_col='start_ti', dur_col='dur_ti', **kwargs):
        super().__init__(**kwargs)
        self.df = df.sort_values(start_col).reset_index(drop=True)
        self.start_col = start_col
        self.dur_col = dur_col 
        self._next = 0

    def init_pre(self, sim): 
        super().init_pre(sim)
        self._starts = self.df[self.start_col].to_numpy()
        return
    
    def add_pairs(self):
        if self._next < len(self._starts):
            s = self._starts[self._next]
            print("CHECK:", s, type(s), self.ti, type(self.ti), s == self.ti)
        added = 0
        while self._next < len(self.df) and self._starts[self._next] == self.ti:
            row = self.df.iloc[self._next]
            self.append(
                p1=np.array([int(row.p1)], dtype=ss.dtypes.int),
                p2=np.array([int(row.p2)], dtype=ss.dtypes.int),
                beta=np.array([float(row.beta)], dtype=ss.dtypes.float),
                dur=np.array([float(row[self.dur_col])], dtype=ss.dtypes.float),
            )
            self._next += 1
            added += 1

        return added
    
def build_network(csv_path: str | Path, label: str = "Epigames"):
    df = read_data(csv_path)
    df = clean_data(df)
    df, n_agents = remap_ids(df)
    df, start_date, stop_date = add_daily_timeline(df)
    net = EpigamesNet(df, label=label)
    start_date = ss.date(start_date)
    stop_date = ss.date(stop_date)
    return net, n_agents, start_date, stop_date

