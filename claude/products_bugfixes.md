# `products.py` bug audit

Audit of `starsim/products.py` (all 201 lines: `Product`, `Dx`, `Tx`, `Vx`, `simple_vx`) for real, unambiguous defects only: wrong results on in-contract input, documented arguments that crash or do nothing, silent data corruption, and logic/unit errors. Style, docstrings, performance, and contrived corner cases are out of scope. Audited against the working tree on branch `rc3.6.2` at commit `3d8dc9d5` (Starsim 3.6.1 editable install, numpy 2.4.6). Every "actual" value below was produced by running the repro script against that install; scripts are in the session scratchpad under `products/`.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | Medium | `Dx.administer()`, `Tx.administer()` | Loop over the full cross-product of `df.disease.unique()` × `df.state.unique()`, so a multi-disease dataframe crashes unless every disease lists every state | 68-77, 114-124 |

## Medium severity

### 1. Multi-disease `Dx`/`Tx` dataframes crash unless fully crossed — `products.py:68-77`, `114-124`

Both products take a dataframe with an explicit `disease` column, which implies that one product can cover several diseases. But `administer()` iterates over `for disease in self.diseases: for state in self.health_states:`, which is every disease paired with every state that appears *anywhere* in the dataframe, not the (disease, state) rows actually given. The failures are:

- In `Dx`, when a disease doesn't list some state that another disease does, `thisdf` is empty and `thisdf[...].probability.values[0]` raises `IndexError`. This happens even when nobody is in that state, because the probabilities are looked up before the UID check.
- In `Tx`, `getattr(disease, state)` raises `AttributeError` when a state named for one disease doesn't exist on another (e.g. `recovered` on SIS). When the attribute does exist but has no row, `thisdf.efficacy.values[0]` raises `IndexError`.

```python
import sciris as sc, starsim as ss
dx_df = sc.dataframe(columns=['disease','state','result','probability'], data=[
    ['sis','susceptible','positive',0.01], ['sis','susceptible','negative',0.99],
    ['sis','infected','positive',0.95],    ['sis','infected','negative',0.05],
    ['sir','infected','positive',0.90],    ['sir','infected','negative',0.10]])
tx_df = sc.dataframe(columns=['disease','state','post_state','efficacy'], data=[
    ['sis','infected','susceptible',0.9], ['sir','recovered','susceptible',0.5]])
diseases = [dict(type='sis', init_prev=0.2), dict(type='sir', init_prev=0.2)]
ss.Sim(n_agents=2000, diseases=diseases, networks='random', start=2000, stop=2010, verbose=0,
       interventions=ss.routine_screening(product=ss.Dx(df=dx_df), prob=0.5, start_year=2005)).run()
ss.Sim(n_agents=2000, diseases=diseases, networks='random', start=2000, stop=2010, verbose=0,
       interventions=ss.treat_num(product=ss.Tx(df=tx_df), eligibility=lambda sim: sim.diseases.sis.infected.uids)).run()
```

Actual:

```
Dx IndexError: index 0 is out of bounds for axis 0 with size 0
Tx AttributeError: SIS object has no attribute "recovered"
```

Expected: both run. For `Dx`, agents in unlisted (disease, state) pairs keep the default (last-in-hierarchy) result. For `Tx`, only the listed (disease, state) rows are treated.

Blast radius: anyone building a single diagnostic or treatment product that covers more than one disease with different state sets. This is the natural use of the `disease` column; single-disease dataframes (as in the tests) are unaffected.

**Fix**: iterate over the unique `(disease, state)` pairs actually present in `self.df` (e.g. `self.df[['disease','state']].drop_duplicates().itertuples()`), not the cross-product of the two unique lists.

## Verified clean

The following hypotheses were tested and found correct:

- **`Dx` accuracy**: with the test dataframe, P(pos|infected) = 0.944 (expected 0.95) and P(pos|susceptible) = 0.009 (expected 0.01). Assigning `result_dist.pars['p']` directly does take effect, calling `rvs()` several times in one step doesn't raise, and `return_format='array'` works.
- **`Tx` efficacy**: the success fraction is 0.310 (expected 0.3), and both returned outputs are `ss.uids` with correct counts. (`successful` is not sorted, which does no harm.)
- **`simple_vx` leaky**: `rel_sus` becomes exactly `1 - efficacy`.
- **`simple_vx` all-or-nothing**: it uses the global `np.random.binomial` rather than a Starsim `Dist`, but same-seed runs (serial, after unrelated use of the global RNG, and under `ss.parallel`) were identical. It is therefore reproducible. It isn't CRN-safe, but that is a design limitation rather than a bug by this audit's bar.
- **Repeated doses**: `rel_sus *= factor` compounds when `routine_vx` re-vaccinates the same agents on later timepoints. That comes from the intervention's default eligibility (everyone), and compounding leaky doses is a defensible model, so it is not recorded.
- **Vaccine protection overwritten by SIS**: SIS overwrites `rel_sus` from immunity (`diseases.py:881`), which erases a vaccine's effect in anyone who acquires immunity. This is a disease-side interaction, not a `products.py` defect.
- **`Product.init_pre()`**: it correctly skips re-initialization.
