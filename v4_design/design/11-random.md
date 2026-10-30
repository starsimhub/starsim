# Randomness and reproducibility

Random number streams, common random numbers, run identity, and convergence checks.

Justified by [best-practices/stochasticity-and-reproducibility.md](../best-practices/stochasticity-and-reproducibility.md).

## Streams

**Every distribution has its own random number stream**, seeded from the sim's `rand_seed` and the distribution's full name (a hash of the two together). In a stochastic compartmental run (`binomial`, `gillespie`), each module draws from a generator of its own, seeded from the sim's `rand_seed` and the module's name, and draws its flows in an order set by the names of their states and transitions, so the order the states are declared in does not change the draws. Each of several runs (`sim.run(n_runs=)`, `ss.parallel()`) draws everything, its outcomes included, from its own seed, as it would on its own.

Consequences:

- Adding a module does not change any other module's draws, under `abm`, `binomial`, and `gillespie`.
- A stochastic compartmental run gives the same draws whether it is taken in one call to the kernels or a step at a time, and whatever other sims run in between. Two exceptions remain: a delivery that the kernels take in a whole run is taken in Python when the run is stepped (or when the sim has several diseases), and a whole run draws its outcomes after the run rather than a step at a time; the distributions are the same, the draws are not.
- Reordering modules does not change which random numbers each distribution receives. (It can still change results, because a module may act on different state; see [10-execution.md](10-execution.md).)
- Distributions can be drawn from in any order without becoming correlated.
- The runs of a MultiSim are identical whether they are run in parallel or in serial.

This is Starsim's mechanism, kept. One shared stream would make every scenario comparison noisier than it needs to be.

**How a draw is made.** Every draw goes through one exact library (`ss.solvers.backends.draws`, compiled by Numba, hidden from users): each count or value is a function of one uniform, by inversion of its CDF where that is cheap (binomial and Poisson up to a mean of 30, negative binomial, exponential, Weibull, normal, lognormal, uniform, discrete choices), so it is monotone in the uniform, which suits common random numbers; otherwise (larger means, gamma, beta) by a rejection sampler whose own uniforms come from a stream seeded by that uniform, so the same uniform still gives the same draw. Uniforms are counter-based (SplitMix64 of a key and an index, the same hash as an agent's CRN uniform). A stochastic compartmental run draws one uniform a count from its generator, and for many groups (`ss.kernels.row_groups`, 50) a row of counts from one key, with vectorized code. A per-agent draw with CRN is the inverse CDF at the agent's uniform, compiled for the common distributions, and SciPy's for the rest (gamma, beta) and above a mean of 30, so it stays monotone. None of the common distributions is much slower than NumPy's own sampler (measured 2026-10-22; [oct25/backlog/perf_jax.md](../oct25/backlog/perf_jax.md)).

The name MUST come from the declaration's path in the model (its module and attribute names), not from a Python object identity or the order of construction, so that the same model written in any of the three spellings produces the same draws.

## Common random numbers

With common random numbers (CRN), the same agent making the same decision gets the same random number in the baseline and in the intervention scenario. Most of the Monte Carlo noise then cancels in the difference, so small intervention effects can be seen with far fewer runs. CRN is on by default, as in Starsim.

A per-agent draw is made from the agent's slot and the distribution's stream, so agent 7's uniform for a given decision does not depend on how many other agents drew from that distribution.

- **The slot is the only key.** It is the persistent per-agent index that Starsim already maintains for this purpose: one integer to hash, and it identifies the same agent across scenarios that start from the same population. A newborn's slot is drawn from its parent's stream, so the same birth gets the same slot in both scenarios.
- **`ss.options.crn = False`** turns CRN off, as in Starsim today, for analyses that want independent runs or that compare structurally different models across which agent identity means nothing.

There is no keyword on the sim for this, and no way to key on agent attributes (see [Rejected](#rejected)).

Under a compartmental solver there are no agents, so there is no pairing by agent: two scenarios with the same seed start from the same draws (each module's generator is seeded from the seed and its name), but the draws stop lining up as soon as the counts they are drawn for differ. Pairing there is weak.

### What CRN covers

CRN holds for draws that can be made one agent at a time from that agent's own uniform. It does not hold for sampling without replacement, joint draws across agents, shuffles, or network rewiring that does these things; such draws keep their noise in a comparison, which is still valid.

Each built-in module says in its docstring whether it keeps CRN. A module is CRN safe if making one agent immune, or removing one, changes the outcomes of only a handful of other agents (against hundreds with `ss.options.crn = False`):

| Module | CRN |
|---|---|
| Declared diseases (`ss.Disease`, `ss.SIR`, `ss.SEIR`, `ss.SIS`), `ss.Pregnancy`, declared births, deaths, and migration, `ss.geo.Mobility`, mixing pools, deliveries (`ss.Delivery`, `ss.vaccinate()`, `ss.screen()`, `ss.treat()`) and their products | Safe as agents (a compartmental run has no agents to pair) |
| Diseases written as code (`ss.Infection`, `ssl.AgentSIR` and the other `ssl` diseases), `ss.Births`, `ss.Deaths`, `ss.modify`, analyzers, `ssl.mnch` modules, `ssl.ART` | Safe |
| `ss.PrenatalNet`, `ss.MaternalNet`, `ss.PostnatalNet`, `ss.BreastfeedingNet`, `ss.RandomNet(method='safe')`, `ssl.ErdosRenyiNet`, `ssl.DiskNet`, `ssl.StaticNet`, `ssl.NullNet`, `ss.Network` with its edges given | Safe |
| `ssl.ClusterNet`, `ssl.HybridNet` | Safe, unless `dynamic=True` and people are born or die |
| `ss.RandomNet` | Not safe when people are born or die, unless `dynamic=False`: one death rewires almost every edge (pairing by position in a list of all agents) |
| `ss.SexualNet`, `ss.DegreeNet` | Not safe when people are born or die: people are paired in turn, so one fewer shifts the partners of those after them |
| `ssl.HouseholdNet` | Not safe unless `dynamic=False`: the partners who move out are drawn together, without replacement |

"Born or die" means differently in the two scenarios: with the same people alive in both, every network is the same in both, except `ssl.HouseholdNet` when pregnancies differ.

CRN pairs draws, not outcomes. For a small change (a few percentage points of coverage, a small change in a parameter) the two runs stay close and most of the noise cancels. For a large change, or late in an epidemic, the scenarios infect different people, so the paired draws are made for different decisions and the noise in the difference grows back towards that of independent runs. CRN reduces the number of runs a comparison needs; it does not remove the need to check it.

### When populations diverge

Births, deaths, and migration make two scenarios' populations diverge even from a common start. The guarantee is:

> **Agents present with the same slot in both scenarios get identical draws from every paired distribution.** Agents created after the scenarios diverge may have no counterpart, and are then unpaired.

## Run identity

A seed makes a run repeatable. It does not say what was run. The record of that is the code: the commit, or better a tag, of the script that built the sim, together with the Starsim version and git information that Starsim stores with a saved sim. There is no hash of the model (decided 2026-10-27: `sim.hash()` and `ss.diff()` were removed with the model spec, [14-model-spec.md](14-model-spec.md)); two models are compared by their results, by `describe()`, or by a text diff of their code, and the runs of a sweep are told apart by their parameters and seeds, which a MultiSim keeps.

## Convergence checks

```python
S, I, R = ss.states('S I R')
sir = ss.Disease(
    infection = ss.Transition(S, I, rate=0.3 * I),
    recovery  = ss.Transition(I, R, dur=4),
    init_prev = 0.01,
)
sim = ss.Sim(sir, solver='binomial', unit='day', stop=100)
sim.check_dt()      # halve dt, re-run, and report the change in every result
```

`check_dt()` re-runs the sim at half the timestep (`factor=` for another ratio) and reports how much each result moves, as a share of its largest value. In a run that takes whole steps this covers every effect that is O(dt): the rate-to-probability conversion, the delay of one step between modules, and the effect of module order ([10-execution.md](10-execution.md)). If the results of interest do not move, none of these matters for this model at this timestep. In a stochastic run the change also includes the difference between two runs. It is for sims with declared states and transitions, and does not run automatically; the note that a step is large for a transition names it.

A companion `msim.check_runs()` on a MultiSim, estimating from the variance between runs how many a stated comparison needs, is deferred ([19-open-questions.md](19-open-questions.md#c-deferred-not-in-v1)). With CRN on, a paired comparison of a small change needs fewer runs (how many fewer depends on how far the scenarios diverge), and `check_runs()` is where that would be stated concretely.

## Rejected

- **One global random number stream.** Adding a module changes everyone's draws.
- **Reporting only the seed.** Necessary, but it does not say which model or parameters were run. The version and the commit do.
- **A hash of the model as a record of what was run** (`sim.hash()`, removed 2026-10-27). For an analysis that is a script in a repository, the commit or tag already says it, and keeping a normalized form of every model to hash cost more than it told apart.
- **Common random numbers keyed on agent attributes** (Vivarium's `key_columns: ['entrance_time', 'age']`). The case made for it is comparing populations of different sizes, but a 10,000-agent run and a 100,000-agent run have 90,000 agents with no counterpart, so the runs differ whatever the key. Within one population size the slot already identifies the same agent across scenarios, including agents born during the run. No case was found where attribute keys do better; if one turns up, this can be revisited.
- **Mandatory CRN.** Some analyses want independent runs; `ss.options.crn = False`.
- **A strict mode that makes unpaired draws an error.** An unpaired draw keeps its noise, and the comparison is still valid.
- **Caching results by hash** (camdl's content-addressed runs). It surprises people who expect a re-run to run, a hash cannot see everything custom code depends on, and nobody has asked for it.
- **Exporting a run as a folder or a package** (SimInf's `package_skeleton()`). A model here is a Python script, which is already the thing to share; `ss.save` covers the rest.
- **A rule for one draw or many** (odin distinguishes `x[] <- Normal(0, 1)`, one draw per element, from drawing once and assigning). In Python the difference is visible in the code: a distribution object in a parameter slot draws per agent, and a number is a number. What a compartmental solver does with a distribution is in [12-backends.md](12-backends.md#the-standard-approximations).
