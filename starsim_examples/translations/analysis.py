"""
Analyzes results from the other languages
"""

import numpy as np
import sciris as sc
import matplotlib.pyplot as plt
sc.options(dpi=150)

# Data -- from running manually
d = sc.objdict()
d.starsim   = dict(flex=9, lines= 25, times=[2.48, 2.53, 2.50])
d.python    = dict(flex=8, lines=239, times=[2.43, 2.24, 2.31])
d.numba     = dict(flex=5, lines=371, times=[1.90, 1.87, 1.71])
d.numba_jax = dict(flex=1, lines=171, times=[1.06, 1.02, 1.02])
d.jax_cpu   = dict(flex=1, lines=153, times=[0.94, 0.98, 0.90])
d.jax_gpu   = dict(flex=1, lines=153, times=[0.06, 0.07, 0.06])
d.julia     = dict(flex=3, lines=288, times=[1.11, 1.17, 1.15])
d.rust      = dict(flex=1, lines=456, times=[0.61, 0.59, 0.59])

df = sc.dataframe.from_dict(d, orient='index')
df['mtime'] = df.times.apply(lambda x: np.array(x).mean())


# Plot
n = len(df)
y = np.arange(n, 0, -1)
fig = plt.figure(figsize=(16,9))

plots = [
    dict(col='mtime',  title='Times',              xlabel='Time (s)',                cmap='RdYlGn_r', fmt='%.2f'),
    dict(col='lines',  title='Lines of code',      xlabel='Lines of code',           cmap='RdYlGn_r'),
    dict(col='flex',   title='Flexibility (est.)', xlabel='Flexibility (out of 10)', cmap='RdYlGn'),

]
for i, p in enumerate(plots):
    p = sc.dictobj(p)
    ax = plt.subplot(1, 3, i+1)
    vals = df[p.col].values
    bars = ax.barh(y, vals, color=sc.vectocolor(vals, cmap=p.cmap))
    ax.bar_label(bars, fmt=p.get('fmt', '%g'))
    ax.set_yticks(y, df.index)
    ax.set_xlabel(p.xlabel)
    ax.set_title(p.title)
    sc.boxoff(ax)

fig.tight_layout()
plt.show()