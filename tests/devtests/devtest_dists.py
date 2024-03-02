"""
Test distributions
"""

# %% Imports and settings
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np

data = np.random.normal(size=1000)
hist, bins = np.histogram(data, bins=50)
plt.subplot(121)
plt.hist(data, 50)
plt.subplot(122)
plt.hist(ss.data_dist(vals=hist, bins=bins)._sample(10_000), 50)
plt.show()
