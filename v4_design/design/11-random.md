# Randomness and reproducibility

Random number streams, common random numbers, run identity, and convergence checks.

Justified by [best-practices/stochasticity-and-reproducibility.md](../best-practices/stochasticity-and-reproducibility.md).

## Streams

**Every distribution has its own random number stream**, seeded from the sim's `rand_seed` and the distribution's full name.

```
stream(dist) = PRNG(seed = H(rand_seed, module_name + '.' + declaration_name))
```

Consequences:

- Adding a module does not change any other module's draws.
- Reordering modules does not change which random numbers each distribution receives. (It can still change results, because a module may act on different state; see [10-execution.md](10-execution.md).)
- Distributions can be drawn from in any order without becoming correlated.
- A run parallelized over replicates is identical to the same run in serial.

This is Starsim's mechanism, kept. One shared stream would make every scenario comparison noisier than it needs to be.

The name MUST come from the declaration's path in the model spec ([14-model-spec.md](14-model-spec.md)), not from a Python object identity or the order of construction, so that the same model written in any of the three spellings produces the same draws.

## Common random numbers

With common random numbers (CRN), the same agent making the same decision gets the same random number in the baseline and in the intervention scenario. Most of the Monte Carlo noise then cancels in the difference, so small intervention effects can be seen with far fewer replicates. CRN is on by default, as in Starsim.

A per-agent draw is generated as

```
u = uniform( H(stream_seed, draw_index, key(agent)) )
```

so agent 7's uniform for a given decision does not depend on how many other agents drew from that distribution.

```python
ss.Sim(sir, crn_key='slot')                    # default: Starsim's behavior
ss.Sim(sir, crn_key=['birth_date', 'sex'])     # keyed on attributes
ss.Sim(sir, crn_key=None)                      # off
```

- **`'slot'`** uses the agent's slot, a persistent index that Starsim already maintains for this purpose. It is fast, and correct whenever the scenarios being compared start from the same agents, which is the usual case.
- **A list of attributes** keys on those agent attributes instead. This also pairs agents across runs that do not share a population, such as a 10,000-agent run against a 100,000-agent run. It is slower. An implementation MAY omit it.
- **`None`** turns CRN off, for analyses that want independent replicates.

In compartmental and stochastic compartmental runs there are no agents, so pairing is at the level of the run: paired scenarios share the stream for each transition's draws. This is a weaker guarantee, and `describe()` says which form is in use.

### What CRN covers

CRN holds for draws that can be made one agent at a time from that agent's own uniform. It does not hold for sampling without replacement, joint draws across agents, shuffles, or network rewiring that does these things. Because transitions are declared and every distribution is named, the library knows which draws are which, and `sim.describe()` lists them:

```text
common random numbers: on (key='slot')
  ok    infection, recovery, mortality   (per-agent draws)
  note  network.rewire samples without replacement; this draw is not paired across scenarios
```

This is a note, not a warning or an error: the comparison is still valid, it just retains the noise from that draw.

### When populations diverge

Births, deaths, and migration make two scenarios' populations diverge even from a common start. The guarantee is:

> **Agents present with the same key in both scenarios get identical draws from every paired distribution.** Agents created after the scenarios diverge may have no counterpart, and are then unpaired.

`sim.crn_report()` MAY be provided to give the fraction of agents still paired at each timestep, which shows how much noise the pairing is removing.

## One draw or many

> **A distribution in a per-agent slot draws once per agent. A distribution in a model-level parameter draws once per run, and the value is shared.**

```python
recovery  = ss.progression('I -> R', dur=ss.lognorm_ex(mean=10, std=3, unit='days'))   # one draw per agent
infection = ss.transmission('S -> I', beta=ss.normal(0.05, 0.01))                      # one draw per run
```

A dwell time is a per-agent quantity, so each agent gets their own duration. A transmission rate is a property of the model, so a distribution there expresses uncertainty about one number. In a compartmental run a dwell-time distribution is not drawn from at all; its shape is represented by stages ([12-backends.md](12-backends.md)).

## Run identity

A seed makes a run repeatable. It does not say what was run. `sim.hash()` does:

```text
>>> sim.hash()
model=8f3a21  pars=c47e90  config=1b6d55  scenario=vaccine  fit=none  seed=1
```

It is one hash in five parts, so that when two runs differ you can see which part changed:

| Part | Covers | Changes when |
|---|---|---|
| `model` | the structure: states, transitions, stratifications, networks, modifiers, declared results | the model changes |
| `pars` | parameter values, including the source and version of any dataset they came from | a number changes |
| `config` | run method, `dt`, `total_pop`, `n_agents`, replicates, `crn_key` | how it is run changes |
| `scenario` | the interventions in this sim | the scenario changes |
| `fit` | fitting settings: data, weights, method, number of trials | the calibration setup changes |

A likelihood attached to a result is declared with the model and is part of `model`. Fitting settings do not change the model, so they have their own part ([13-inference.md](13-inference.md)).

```text
>>> ss.diff(sim_a, sim_b)
model:    same
pars:     beta 0.05 -> 0.08
config:   same
scenario: baseline -> vaccine
```

This answers the first question asked of any pair of results (did the model change, or did the numbers?) without comparing files. The hashes are computed from the model spec, whose canonical form is in [14-model-spec.md](14-model-spec.md). Custom Python code is hashed by its source text, which is a best effort: code that depends on something outside its own source can change behavior without changing the hash.

### Caching results (optional)

An implementation MAY cache results by run hash. Caching is off by default.

```python
sim.run(cache=True)       # reuse a stored result with the same hash, or store this one
sim.run(cache='results/') # ... in a given folder
```

When it is on, reusing a stored result is reported in one line, and a stored result is never silently overwritten by a different one with the same hash (`E901 ResultExists`; this indicates either a bug or code the hash cannot see).

## Convergence checks

```python
sim.check_dt()      # halve dt, re-run, and report the change in every result
```

`check_dt()` re-runs the sim at half the timestep and reports how much each result moves. In a discrete-time run this covers every effect that is O(dt): the rate-to-probability conversion, the delay of one step between modules, and the effect of module order ([10-execution.md](10-execution.md)). If the results of interest do not move, none of these matters for this model at this timestep. It does not run automatically; `describe()` mentions it in one line for run methods that have a timestep. A companion `check_reps()`, estimating from the variance between replicates how many a stated comparison needs, is deferred ([19-open-questions.md](19-open-questions.md#c-deferred-not-in-v1)).

## Export

A run is reproduced from its model spec (`sim.to_json()`), its data, and the run hash. A single `ss.export(sim, 'analysis/')` that writes all of these plus the package versions and a rerun script into one folder is deferred ([19-open-questions.md](19-open-questions.md#c-deferred-not-in-v1)); `ss.save` and `to_json()` cover it meanwhile.

## Rejected

- **One global random number stream.** Adding a module changes everyone's draws.
- **Reporting only the seed.** Necessary, but it does not say which model or parameters were run.
- **Mandatory CRN.** Some analyses want independent replicates.
- **A strict mode that makes unpaired draws an error.** A note in `describe()` is enough; the comparison is still valid.
- **Caching on by default.** Surprising when it serves a stale result, and the hash cannot see everything custom code depends on.
