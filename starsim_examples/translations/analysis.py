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

ax1 = plt.subplot(1,3,1)
ax1.set_title('Times')
bars = ax1.barh(y, df.mtime.values, color=sc.vectocolor(df.mtime.values, cmap='RdYlGn_r'))
ax1.bar_label(bars, fmt='%.2f')
ax1.set_yticks(y, df.index)
ax1.set_xlabel('Time (s)')

ax2 = plt.subplot(1,3,2)
ax2.set_title('Flexibility')
bars = ax2.barh(y, df.flex.values, color=sc.vectocolor(df.flex.values, cmap='RdYlGn'))
ax2.bar_label(bars)
ax2.set_yticks(y, df.index)
ax2.set_xlabel('Flexibility (out of 10)')

ax3 = plt.subplot(1,3,3)
ax3.set_title('Lines of code')
bars = ax3.barh(y, df.lines.values, color=sc.vectocolor(df.lines.values, cmap='RdYlGn_r'))
ax3.bar_label(bars)
ax3.set_yticks(y, df.index)
ax3.set_xlabel('Lines of code')

fig.tight_layout()
plt.show()