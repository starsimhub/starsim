import numpy as np

np.random.seed(1)

dfp1 = np.array([1,   4  , 4  , 8  , 8  , 9  , 2  , 5  , 6,   5  ])
dfp2 = np.array([8,   2 ,  3  , 3  , 5  , 7  , 3  , 2  , 7,   1  ])
dfp  = np.array([0.9, 0.2, 0.5, 0.3, 0.2, 0.4, 0.8, 0.6, 0.1, 0.7])

import pandas as pd
import sciris as sc
t = sc.timer()
df = pd.DataFrame({'p1': dfp1, 'p2': dfp2, 'p': dfp})
p_acq_node = df.groupby('p2').apply(lambda x: 1 - np.prod(1 - x['p']))  # prob(inf) for each potential infectee
uids = p_acq_node.index.values
t.toc()

t = sc.timer()
p2uniq, p2idx, p2inv, p2cnt = np.unique(dfp2, return_index=True, return_inverse=True, return_counts=True)

r = np.random.rand(100) # One draw for this disease, up to max slot
q = np.random.rand(100) # One draw for this disease, up to max slot

# Now address nodes with multiple possible infectees
degrees = np.unique(p2cnt)
case_uids = []
case_srcs = []
for deg in degrees:
    if deg == 1:
        cnt1 = p2cnt == 1
        uids = p2uniq[cnt1] # UIDs that only appear once
        idx = p2idx[cnt1]
        p_acq_node = dfp[idx]
        cases = r[uids] < p_acq_node # use slots
        if cases.any():
            s = dfp1[idx][cases]
    else:
        dups = np.argwhere(p2cnt==deg).flatten()
        uids = p2uniq[dups]
        inds = [np.argwhere(np.isin(p2inv, d)).flatten() for d in dups]
        probs = dfp[inds]
        p_acq_node = 1-np.prod(1-probs, axis=1)

        cases = r[uids] < p_acq_node # use slots
        if cases.any():
            # Vectorized roulette wheel
            cumsum = probs[cases].cumsum(axis=1)
            cumsum /= cumsum[:,-1][:,np.newaxis]
            ix = np.argmax(cumsum >= q[uids[cases]], axis=1)
            s = dfp1[inds][cases][np.arange(len(cases)),ix]

    if cases.any():
        case_uids.append(uids[cases])
        case_srcs.append(s)

case_uids = np.concatenate(case_uids)
case_srcs = np.concatenate(case_srcs)
t.toc()
