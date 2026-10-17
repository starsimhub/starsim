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
- The runs of a MultiSim are identical whether they are run in parallel or in serial.

This is Starsim's mechanism, kept. One shared stream would make every scenario comparison noisier than it needs to be.

The name MUST come from the declaration's path in the model spec ([14-model-spec.md](14-model-spec.md)), not from a Python object identity or the order of construction, so that the same model written in any of the three spellings produces the same draws.

## Common random numbers

With common random numbers (CRN), the same agent making the same decision gets the same random number in the baseline and in the intervention scenario. Most of the Monte Carlo noise then cancels in the difference, so small intervention effects can be seen with far fewer runs. CRN is on by default, as in Starsim.

A per-agent draw is generated as

```
u = uniform( H(stream_seed, draw_index, slot) )
```

so agent 7's uniform for a given decision does not depend on how many other agents drew from that distribution.

- **The slot is the only key.** It is the persistent per-agent index that Starsim already maintains for this purpose: one integer to hash, and it identifies the same agent across scenarios that start from the same population. A newborn's slot is drawn from its parent's stream, so the same birth gets the same slot in both scenarios.
- **`ss.options.crn = False`** turns CRN off, as in Starsim today, for analyses that want independent runs or that compare structurally different models across which agent identity means nothing.

There is no keyword on the sim for this, and no way to key on agent attributes (see [Rejected](#rejected)).

Under a compartmental solver there are no agents, so pairing is at the level of the run: paired scenarios share the stream for each transition's draws. This is a weaker guarantee, and `describe()` says which form is in use.

### What CRN covers

CRN holds for draws that can be made one agent at a time from that agent's own uniform. It does not hold for sampling without replacement, joint draws across agents, shuffles, or network rewiring that does these things. Because transitions are declared and every distribution is named, the library knows which draws are which, and `sim.describe()` lists them:

```text
common random numbers: on
  ok    infection, recovery, mortality   (per-agent draws)
  note  network.rewire samples without replacement; this draw is not paired across scenarios
```

This is a note, not a warning or an error: the comparison is still valid, it just retains the noise from that draw. Draws made in custom `step()` code are reported the same way, from what the distributions saw during a run.

### When populations diverge

Births, deaths, and migration make two scenarios' populations diverge even from a common start. The guarantee is:

> **Agents present with the same slot in both scenarios get identical draws from every paired distribution.** Agents created after the scenarios diverge may have no counterpart, and are then unpaired.

`sim.crn_report()` MAY be provided to give the fraction of agents still paired at each timestep, which shows how much noise the pairing is removing.

## Run identity

A seed makes a run repeatable. It does not say what was run. The main record of that is the code: the commit, or better a tag, of the script that built the sim, together with the Starsim version and git information that Starsim already stores with a saved sim. `sim.hash()` adds to that record; it does not replace it.

```python
>>> sim.hash()
objdict(model='8f3a21', pars='c47e90', config='1b6d55', scenario='vaccine', seed=1)
```

It returns a plain `sc.objdict` of four hashes and the seed, so `sim.hash().pars` is a string that can go in a file name or a column of a data frame:

| Part | Covers | Changes when |
|---|---|---|
| `model` | the structure: states, transitions, stratifications, networks, modifiers, declared results | the model changes |
| `pars` | resolved parameter values, including the source and version of any dataset they came from | a number changes |
| `config` | solver, `dt`, `total_pop`, `n_agents`, whether CRN is on | how it is run changes |
| `scenario` | the interventions in this sim; the scenario's name where it has one (`ss.parallel(vaccine=...)`), a hash otherwise | the scenario changes |
| `seed` | `rand_seed`, as given | another run of the same thing |

The hashes are a cheap by-product of the model being data, and they are always computed. What they add to a commit hash:

- **Many runs come from one commit.** A sweep or a calibration produces hundreds of runs from the same code. The `pars` hash tells them apart, the runs of a MultiSim differ only in `seed`, and a saved result carries the hash of what produced it.
- **Not everything is committed.** Notebooks and exploratory scripts are run with uncommitted changes. The hashes are computed from the model that actually ran.
- **Two models written differently can be the same model.** The `model` hash is computed from the normalized model spec, so it is the test for "do these two scripts describe the same model" ([18-conformance.md](18-conformance.md)).

Comparing two runs:

```text
>>> ss.diff(sim_a, sim_b)
model:    same
pars:     beta 0.05 -> 0.08
config:   same
scenario: baseline -> vaccine
```

The hashes are computed from the model spec, whose canonical form is in [14-model-spec.md](14-model-spec.md). Custom Python code is hashed by its source text, which is a best effort: code that depends on something outside its own source can change behavior without changing the hash.

A likelihood attached to a result is declared with the model and is part of `model`. Fitting settings (the data, weights, fitting method, number of trials) do not change the model, so they are not in `sim.hash()`; a fit records its own hash of them ([13-inference.md](13-inference.md)), and changing the number of trials does not look like a change to the model.

Caching results by hash, so that an identical re-run is served from disk, is deferred: see [Rejected](#rejected).

## Convergence checks

```python
sim.check_dt()      # halve dt, re-run, and report the change in every result
```

`check_dt()` re-runs the sim at half the timestep and reports how much each result moves. In a discrete-time run this covers every effect that is O(dt): the rate-to-probability conversion, the delay of one step between modules, and the effect of module order ([10-execution.md](10-execution.md)). If the results of interest do not move, none of these matters for this model at this timestep. It does not run automatically; `describe()` mentions it in one line for solvers that have a timestep.

```python
msim.check_runs()   # is n_runs enough for the quantity being compared?  (deferred)
```

A companion `msim.check_runs()` on a MultiSim, estimating from the variance between runs how many a stated comparison needs, is deferred ([19-open-questions.md](19-open-questions.md#c-deferred-not-in-v1)). With CRN on, the number of runs a paired comparison needs is often an order of magnitude lower, and `check_runs()` is where that would be stated concretely.

## Rejected

- **One global random number stream.** Adding a module changes everyone's draws.
- **Reporting only the seed.** Necessary, but it does not say which model or parameters were run. The version, the commit, and the hashes do.
- **The hashes as the main record of what was run.** For an analysis that is a script in a repository, the commit or tag already says it. The hashes cover what a commit does not distinguish.
- **Common random numbers keyed on agent attributes** (Vivarium's `key_columns: ['entrance_time', 'age']`). The case made for it is comparing populations of different sizes, but a 10,000-agent run and a 100,000-agent run have 90,000 agents with no counterpart, so the runs differ whatever the key. Within one population size the slot already identifies the same agent across scenarios, including agents born during the run. No case was found where attribute keys do better; if one turns up, this can be revisited.
- **Mandatory CRN.** Some analyses want independent runs; `ss.options.crn = False`.
- **A strict mode that makes unpaired draws an error.** A note in `describe()` is enough; the comparison is still valid.
- **Caching results by hash in v1** (camdl's content-addressed runs). With the hashes in place it is easy to add: an identical re-run becomes a cache hit. It also surprises people who expect a re-run to run, the hash cannot see everything custom code depends on, and nobody has asked for it. Deferred.
- **Exporting a run as a folder or a package** (SimInf's `package_skeleton()`). A model here is a Python script, which is already the thing to share; `ss.save` and `sim.to_json()` cover the rest.
- **A rule for one draw or many** (odin distinguishes `x[] <- Normal(0, 1)`, one draw per element, from drawing once and assigning). In Python the difference is visible in the code: a distribution object in a parameter slot draws per agent, and a number is a number. What a compartmental solver does with a distribution is in [12-backends.md](12-backends.md#the-standard-approximations).
