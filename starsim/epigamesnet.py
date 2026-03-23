import numpy as np
import starsim as ss

__all__ = ['EpigamesNet']

class EpigamesNet(ss.DynamicNetwork):
    def __init__(self, df, start_col='start_offset_ti', dur_col='dur_ti', **kwargs):
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
        added = 0
        while self._next < len(self.df) and self._starts[self._next] == self.ti:
            row = self.df.iloc[self._next]

            if added < 5:
                print(f"[ti={self.ti}] adding edge ({row.p1}, {row.p2}) for {row[self.dur_col]} ms")

            self.append(
                p1=np.array([int(row.p1)], dtype=ss.dtypes.int),
                p2=np.array([int(row.p2)], dtype=ss.dtypes.int),
                beta=np.array([float(row.beta)], dtype=ss.dtypes.float),
                dur=np.array([float(row[self.dur_col])], dtype=ss.dtypes.float),
            )
            self._next += 1
            added += 1

        return added
    