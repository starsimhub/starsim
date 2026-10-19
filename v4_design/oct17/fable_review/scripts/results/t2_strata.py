"""Stratified results: select/by/arithmetic/ratios/to_df/plot"""
import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

S, I, R = ss.states(3)
N = S + I + R
infection = S >> I @ 0.3 * I
recovery  = I >> R @ ss.days(10)
prev = I / N
sir = ss.Disease(infection, recovery, prev=prev, init_prev=0.05, name='sir')
sir.stratify(risk=['low', 'high'], adjust=dict(infection=[1, 3]))
sim = ss.Sim(sir, stratify=dict(age=[0, 15, 65]), total_pop=1000, unit='day', stop=20, verbose=0)
sim.run()
res = sim.results.sir
print('keys', list(res.keys()))
print('n_I dims', res.n_I.dims, 'strata shape', res.n_I.strata.shape)

# select / by
a = res.n_I.select(age=0)
print('select(age=0):', type(a).__name__, a.label, a.values[-3:])
b = res.n_I.by('age')
print('by(age):', type(b).__name__, b.columns, b.values.shape)
print('by then select?', end=' ')
try:
    print(b.select(age=0))
except Exception as e:
    print('ERR', type(e).__name__, e)
try:
    print(res.n_I.select(age=0).by('risk').columns)
except Exception as e:
    print('select then by ERR', type(e).__name__, e)
print('by(risk) columns:', res.n_I.by('risk').columns)  # the "low/high" rename issue
print("select(sex='f'):", end=' ')
try:
    res.n_I.select(sex='female')
except Exception as e:
    print('ERR', type(e).__name__, e)

# arithmetic
tot = res.n_S + res.n_I + res.n_R
print('sum keeps strata?', getattr(tot, 'strata', None) is not None, 'select ok:', tot.select(age=0).values[-1])
ratio = res.n_I / tot
print('ratio keeps strata?', getattr(ratio, 'strata', None) is not None)
try:
    ratio.select(age=0)
except Exception as e:
    print('ratio.select ERR:', type(e).__name__, e)
print('declared prev.select(age=0)[-1]', res.prev.select(age=0).values[-1], 'vs manual', (res.n_I.select(age=0)/tot.select(age=0)).values[-1])
print('prev.by(age) cols', res.prev.by('age').columns, res.prev.by('age').values[-1])
print('prev.by(age) label', res.prev.by('age').label, 'scale', res.prev.by('age').scale)
# a ratio multiplied by 2
twice = 2*res.prev
print('2*prev keeps strata?', getattr(twice, 'strata', None) is not None, 'denom?', getattr(twice, 'denom', None) is not None)
# np.sum
print('np.sum(res.infection) =', np.sum(res.infection), ' res.infection.sum() type', type(res.infection.sum()))
# infection select totals add up
print('select low + high == total:', np.allclose(res.infection.select(risk='low') + res.infection.select(risk='high'), res.infection))

# to_df on stratified results
df = res.n_I.by('age').to_df()
print('by(age).to_df cols:', list(df.columns))
df2 = res.to_df()
print('results.to_df cols (first 12):', list(df2.columns)[:12], '... n cols', len(df2.columns))
print('does to_df include strata?', any('age' in c for c in df2.columns))
# resample a by() result
try:
    rs = res.infection.by('age').resample('week')
    print('by(age).resample(week) shape', rs.values.shape, rs.columns)
except Exception as e:
    print('resample by ERR', type(e).__name__, e)
try:
    rs = res.infection.select(age=0).resample('week')
    print('select.resample ok', rs.values.shape, 'strata kept?', getattr(rs, 'strata', None) is not None)
except Exception as e:
    print('resample select ERR', type(e).__name__, e)

# plotting
fig = res.n_I.by('age').plot()
print('by(age).plot ok, n lines', len(fig.axes[0].lines))
fig = res.n_I.select(age=0).plot()
print('select plot title', fig.axes[0].get_title())
fig = sim.plot()
print('sim.plot n axes', len(fig.axes))
# does sim.plot show strata? (no)

# a MultiSim of this stratified sim
msim = ss.parallel(ss.Sim(sir, stratify=dict(age=[0, 15, 65]), total_pop=1000, unit='day', stop=20, verbose=0, solver='binomial'), n_runs=3)
mr = msim.results.sir.n_I
print('MultiResult type', type(mr).__name__, 'values', mr.values.shape, 'strata', None if mr.strata is None else mr.strata.shape)
print('select(age=0).quantiles shape', mr.select(age=0).quantiles().shape)
print('by(age) type', type(mr.by('age')).__name__, list(mr.by('age').keys()))
print('select(run=0) type', type(mr.select(run=0)).__name__)
print('msim.results keys', list(msim.results.keys()))
print('msim.results.sir type', type(msim.results.sir).__name__)
print('prev MultiResult strata?', msim.results.sir.prev.strata is not None)
try:
    print('prev.select(age=0)', msim.results.sir.prev.select(age=0).values[:, -1])
except Exception as e:
    print('msim prev.select ERR', type(e).__name__, e)
try:
    mr.plot()
    print('MultiResult.plot ok')
except Exception as e:
    print('MultiResult.plot ERR', type(e).__name__, e)
try:
    print(mr.to_df())
except Exception as e:
    print('MultiResult.to_df ERR', type(e).__name__, e)
