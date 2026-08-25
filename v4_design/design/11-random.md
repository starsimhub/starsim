# Randomness and reproducibility

RNG streams, common random numbers, run identity, and what makes an analysis rather than a run reproducible.

Justified by [best-practices/stochasticity-and-reproducibility.md](../best-practices/stochasticity-and-reproducibility.md).

## Streams

**Every distribution owns its own random number generator**, seeded from the sim's `rand_seed` and a hash of the distribution's fully qualified name.

```
stream(dist) = PRNG(seed = H(rand_seed, module_name + '.' + declaration_name))
```

Consequences, and they are the reason for the design:

- Adding a module does not change any other module's draws.
- Reordering modules does not change any draws.
- Distributions may be drawn from independently, in any order, without correlating.
- A run parallelized over replicates is bit-identical to the serial run.

This is Starsim's mechanism, kept. Nothing else in the landscape does it, and a single global stream makes every scenario comparison noisier than it needs to be.

The name hash MUST be computed from the **normalized** declaration path in the IR ([14-ir.md](14-ir.md)), not from a Python object identity or a construction-order index, so that the three construction forms produce identical draws.

## Common random numbers

CRN makes paired scenario comparison difference out most of the Monte Carlo noise, so small intervention effects are detectable with far fewer replicates. It is on by default.

```python
ss.Sim(sir, crn_key='slot')                    # default
ss.Sim(sir, crn_key=['birth_date', 'sex'])     # keyed on attributes
ss.Sim(sir, crn_key=None)                      # off
```

A per-agent draw is generated as

```
u = uniform( H(stream_seed, draw_index, key(agent)) )
```

so that agent 7's uniform for a given decision is the same in the baseline and the intervention scenario, regardless of how many other agents drew from that distribution.

- **`crn_key='slot'`** uses the agent's persistent slot index. It is fast and correct whenever the compared populations are the same agents, which is the common case, and it is therefore the default.
- **`crn_key=[attributes]`** keys on declared agent attributes instead. This generalizes to comparing populations that are not the same set of agents — a 10,000-agent run against a 100,000-agent run — which the slot scheme handles awkwardly. It is slower, because it hashes several columns rather than one integer.
- **`crn_key=None`** turns CRN off, for analyses that want independent replicates or that compare structurally different models across which agent identity is meaningless.

For compartmental and stochastic-compartmental methods there are no agents, and CRN pairs at the **run** level: paired scenarios share the stream for each transition's draws. This is a weaker but still useful guarantee, and `summary()` says which form is in force.

### The contract is checked, not documented

Any operation that cannot be written as a per-agent inverse CDF — sampling without replacement, joint draws, network rewiring, shuffles — falls outside the CRN guarantee. Because transitions are declared, the compiler knows which draws are which, and MUST report it:

```text
>>> sim.summary()
common random numbers: on (key='slot')
  ok  infection, recovery, mortality — per-agent inverse CDF, CRN holds
  !   network.rewire — samples without replacement; CRN does not hold across scenarios
      paired differencing will include Monte Carlo noise from this draw
```

This is the difference between a guarantee and a habit. A user comparing two scenarios that differ only in an intervention deserves to know which parts of the difference are signal.

### The guarantee under divergent populations

Births, deaths, and migration mean two scenarios' agent sets diverge even from a common start. The statable guarantee, and the one the implementation MUST provide:

> **Agents alive and identically-keyed in both scenarios retain identical draws for every CRN-safe distribution, up to the first draw whose index differs between scenarios.** Agents created after divergence have no counterpart and no guarantee.

`sim.crn_report()` gives the fraction of agents still paired at each timestep, which is the honest measure of how much noise the pairing is actually removing.

## The array-draw rule

Restated here because it is an RNG semantic, and it is ambiguous in most frameworks and silently wrong in some:

> **A distribution placed in a per-agent slot draws once per agent. A distribution evaluated to a value draws once, and that value is shared.**

```python
ss.progression('I -> R', dur=ss.lognorm(mean=ss.days(10)))   # per agent
shared = ss.lognorm(mean=ss.days(10)).rvs()                  # once, shared
ss.by(age=ss.lognorm(mean=0.05, std=0.01))                   # once per age stratum
```

## Run identity

A seed makes a *run* reproducible. It does not make an *analysis* reproducible. A paper reporting `rand_seed=1` and nothing else has reported almost nothing.

```text
>>> sim.hash()
model=8f3a21  pars=c47e90  config=1b6d55  scenario=vaccine  fit=none  seed=1
```

Five levels, each computed from a defined slice of the run:

| Level | Covers | Changes when |
|---|---|---|
| `model` | the normalized IR: states, transitions, dimensions, routes, effects, outcome structure | the structure changes |
| `pars` | resolved parameter values, including the source and version of every fetched dataset | a number changes |
| `config` | method, dt, n, n_agents, n_reps, crn_key, output grid | the execution changes |
| `scenario` | which interventions are enabled | the scenario changes |
| `fit` | the fitting procedure: method, trials, chains, gates | the calibration setup changes |

The likelihood is part of `model`, because it is a statement about how data relate to the process. The fitting procedure is `config`-like and hashes separately, because changing the number of optimizer trials does not change the model.

```text
>>> ss.diff(run_a, run_b)
model:    same
pars:     beta 0.05 -> 0.08
config:   same
scenario: baseline -> vaccine
```

This answers the question every reviewer of a modeling analysis asks — *did the model change, or did the numbers?* — without diffing files. It is the one mechanism in the landscape that solves a problem every framework has and none of the others addresses, and it is adopted outright.

### Content addressing

- A run whose five hashes already exist in the result store is a **cache hit**, and the hit MUST be reported rather than silent.
- Writing a different result to an existing identity is an **error**, not an overwrite.
- `run(force=True)` recomputes.

Hashing the model requires a canonical form, which [14-ir.md](14-ir.md) specifies. That normalizer is also what makes prose-to-model determinism testable, so it is not an extra cost.

## Convergence is a checked claim

```python
sim.check_dt()      # halve dt, re-run, report the change in every result
sim.check_reps()    # how many replicates does the comparison being made require?
```

- `check_dt()` implements a Richardson-style convergence audit. A discrete-time model whose answer moves when `dt` halves has not converged, and the distinction between "it ran" and "it converged" is not otherwise visible.
- `check_reps()` estimates the replicate count needed for a stated comparison at a stated precision, using the observed between-replicate variance. With CRN on, the required count for a paired comparison is often an order of magnitude lower, and saying so concretely is worth more than a documentation footnote.

Neither is run automatically. Both are one call, and `summary()` notes when neither has been run for a model whose method has a convergence condition.

## Export

```python
ss.export(sim, 'analysis/')
```

Produces a directory containing the model as IR, the resolved parameters, the data references with their versions, the environment specification, the five hashes, and a runnable script. This is trivial once the model is data, and it is what makes an analysis citable rather than merely reproducible on the machine it was written on.

## Rejected

- **A single global RNG stream.** Adding a module changes everyone's draws.
- **Reproducibility by seed alone.** Necessary; wildly insufficient.
- **CRN as an implementation detail.** It changes what a comparison means; it belongs in the summary.
- **Per-particle streams as the only mechanism.** Right for particle filters, insufficient for paired scenarios.
- **Mandatory CRN.** Some analyses want independent replicates.
