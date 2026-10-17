# Execution

The run loop: what is called in what order each step, how modules with different timesteps interleave, and what is and is not affected by order.

Justified by [best-practices/composition-and-effects.md](../best-practices/composition-and-effects.md) §Execution order, [best-practices/agent-based.md](../best-practices/agent-based.md), and [best-practices/time-and-units.md](../best-practices/time-and-units.md).

## The governing rule

> **Modules run in a defined, printable order each step, and each one sees the changes made by those before it.**

This is Starsim's loop, kept. Three consequences:

- **Within-step chains work.** If infection runs before testing and testing before treatment, a person can be infected, tested, and treated in one step. Nothing has to be split into sub-steps.
- **Mutual dependence between modules is fine.** HIV can change syphilis transmission while syphilis changes HIV transmission. Each reads the other's current value, and the order decides who goes first. There is no cycle error.
- **Module order is a modeling choice.** Its effect on results is O(dt): it shrinks as the timestep shrinks, and `sim.check_dt()` measures it ([11-random.md](11-random.md) §Convergence). The usual convention is to run last the module whose accuracy matters most, because it sees the most up-to-date state.

Order does not arise in a compartmental ODE run: all declared flows are evaluated together from the current state (see [Order under each solver](#order-under-each-solver)).

## The order of calls in one step

The loop calls these functions, in this order, at every step. This is the order in `starsim/loop.py`; nothing here is new.

| # | Called on | Function | What happens |
|---|---|---|---|
| 1 | sim | `start_step()` | Advance the sim's distributions to this step; print progress |
| 2 | every module | `start_step()` | Advance each module's distributions; per-step bookkeeping |
| 3 | plain `ss.Module`s | `step()` | Modules that are none of the types below |
| 4 | demographics | `step()` | Births and immigration add agents; background deaths are requested |
| 5 | diseases | `step_state()` | Disease state updates: every declared transition that is not contact-mediated (progression, branching, `ss.transfer`, disease deaths) |
| 6 | connectors | `step()` | Cross-module code |
| 7 | networks | `step()` | Add and remove edges; mixing pools and `ss.Mobility` update who meets whom |
| 8 | interventions | `step()` | Deliveries, products, parameter changes |
| 9 | diseases | `step()` | Transmission: every contact-mediated transition |
| 10 | people | `step_die()` | Requested deaths take effect (see [Removals](#removals-are-deferred)) |
| 11 | people, then every module | `update_results()` | Counts, flows, declared results and `ss.outcome`s for this step |
| 12 | analyzers | `step()` | Analyzers read the finished step |
| 13 | every module, people, sim | `finish_step()` | Remove dead agents; increment the time index |

The reasons for this order are Starsim's: new agents exist before diseases initialize them; states update before networks, so that a state change can alter who is eligible for contact; networks update before interventions, so that an intervention can alter the final network for the step; and interventions run before transmission, so that someone vaccinated today is protected today.

### Order within a type

**Within a type, modules run in the order given.** `ss.Sim(diseases=[syphilis, hiv])` runs syphilis before HIV in rows 5 and 9; `interventions=[screen, treat]` runs screening before treatment in row 8. There is no other ordering mechanism: no priorities, no phases, no dependency graph.

### Where declared transitions run

A disease built from declared transitions ([03-states-and-transitions.md](03-states-and-transitions.md)) needs no `step()` written by the user. The library puts each transition in one of the two disease rows by its kind, which is derived from its rate:

- **Not contact-mediated** (the rate mentions no state): row 5, `step_state()`. This covers `ss.progression`, branching, dwell times that elapse this step, `ss.transfer`, and `ss.death` declared on the disease.
- **Contact-mediated** (the rate mentions a state): row 9, `step()`, after networks and interventions have run.
- **`ss.birth` declared on a disease** adds agents at row 4, with the demographics modules, so that they are initialized with everyone else at row 5.
- **Transitions into and out of states that are not people** (a water reservoir, a mosquito population; [03-states-and-transitions.md](03-states-and-transitions.md)) are scheduled by the same rule and belong to the same disease: `shedding = ss.null >> W @ shed * I.n` and `decay = W >> ss.null @ rate` mention no bare state, so they run at row 5. See [States that are not people](#states-that-are-not-people).
- **A plain function passed to `ss.Disease(...)`** as a keyword is called at the end of that disease's row 9. A subclass that overrides `step()` or `step_state()` replaces the corresponding row, as in Starsim.

### Where `ss.modify` is evaluated

A modifier ([07-effects.md](07-effects.md)) has no row of its own. The effective value of a modified quantity is its base value combined with the modifiers currently active, and the library keeps that value up to date as follows:

1. The effective value is cached.
2. The cache is marked stale when something a modifier depends on changes: a state used in its `where=` gains or loses members, the sim passes its `start=` or `stop=`, or a base value is assigned.
3. A stale value is recomputed the next time the quantity is read, before the read returns.
4. A modifier whose amount varies in time (`multiply=ss.seasonal(...)`, a waning curve, a function) is marked stale once per step, at row 2.

So a modifier with a `where=` on a state always reflects that state as it is at the moment of the read, whatever the module order. (A `when=` function is the exception: it is called once per step, at the modifier's place among the connectors or interventions; see [07-effects.md](07-effects.md).) A modifier that is inactive or constant costs nothing per step. Because the base value is never overwritten, the original value comes back when a modifier ends.

Direct writes by custom code are a different matter: the loop does not record them unless asked. `ss.Sim(..., track_writes=True)` makes the state arrays record which module assigned them at each step; it is **off by default**, because recording the writer on every assignment is a cost inside the loop that most runs do not need. `sim.what_modifies(x)` always lists the declared modifiers on `x`, and lists the modules observed writing it only when tracking was on for the run ([07-effects.md](07-effects.md)).

Example: `ss.modify(hiv.rel_trans, multiply=2.0, where=syphilis.active)`. If syphilis is listed before HIV, a person who acquires syphilis at row 9 of this step has doubled HIV transmissibility when HIV transmission runs later in row 9 of the same step. If syphilis is listed after HIV, the doubling is first used one step later. Both are valid; the difference is O(dt).

## A within-step chain

```python
import starsim as ss

sir    = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=10),
)
test   = ss.Test(sensitivity=0.9, specificity=0.98)
drug   = ss.Treatment(efficacy=0.95)
screen = ss.screen(test, coverage=0.2, on_positive=ss.treat(drug))

sim = ss.Sim(sir, interventions=screen, unit='day', dt=1, stop=100)
```

Testing and treatment happen in the same step: the follow-up given as `on_positive=` runs immediately, inside the screening call at row 8, for the people who tested positive. Follow-ups chain to any depth this way.

Infection is at row 9, after interventions, so with the default order a person infected today is first eligible for screening tomorrow. To put screening after transmission, so that infection, testing, and treatment can all happen in one step, move it in the loop:

```python
sim.init()
sim.loop.insert(screen, label='sir.step')   # run screening right after sir's transmission
sim.run()
```

Which of the two is right depends on what a step represents in the model. With a one-day step the difference is a day; with a one-month step it may matter, and `sim.check_dt()` will show it.

## The loop is a data object

```python
sim.loop                 # the plan: every (time, module, function) entry in execution order
print(sim.loop.to_df())  # as a table: time, ti, func_order, label, module, func_name
sim.loop.plot()          # as a diagram
sim.loop.insert(fn, label='sir.step')                 # call fn(sim) after sir.step at every step
sim.loop.insert(fn, label='sir.step', before=True)    # ... or before it
sim.loop.insert(fn, match_fn=lambda plan: (plan.label == 'sir.step') & (plan.time > ss.date('2021-01-01')))
```

This is Starsim's `ss.Loop`, unchanged apart from one addition: `insert()` also accepts a module, which moves that module's `step()` to the given position (as in the screening example above). The loop exists after `sim.init()`. Labels are `'<module name>.<function name>'`; `sim.loop.to_df().label.unique()` lists them.

Whatever the order is, it can be printed, and it is the only thing that determines what runs when.

### The schedule in `describe()`

`sim.loop` and `sim.loop.to_df()` are the fully unrolled plan: every call at every time. That is exact and hard to read. `sim.describe()` MUST print the loop as a **schedule grouped by timestep** instead: one line per timestep in use, listing the modules on it in the order they run.

```text
>>> sim.describe()
...
schedule:
  every day:   sir, random network, screen
  every year:  births, deaths
```

- Groups are ordered from the shortest timestep to the longest. Within a group, modules are in loop order (the table above), and a module with two rows (a disease's state update and its transmission) is listed once, at its first.
- A module moved with `sim.loop.insert()` is shown at its new place.
- In a sim with no unit the line reads `every step:`, and a module on a coarser timestep `every 7 steps:`.
- `sim.describe(verbose=True)` adds the rows of the table above under each group; the unrolled list is `sim.loop.to_df()`, on request.

## Competing hazards are resolved jointly

Order between modules is a choice. Order between the transitions inside one module is not something a user should have to think about, so it does not affect results: **transitions out of the same state that are evaluated in the same call compete in one draw.**

```
h_total = sum_k h_k
p_exit  = 1 - exp(-h_total * dt)
P(via k | exit) = h_k / h_total
```

In a discrete-time run this MUST be implemented as a single multinomial draw per source state, not as sequential binomial draws. Sequential draws give the first-declared transition an advantage of order `h * dt`, which would make results depend on declaration order.

Details:

- Transitions out of one state that carry `p=` are a split decided on entry to the state, not a competition ([03-states-and-transitions.md](03-states-and-transitions.md) §Branching); the split is one draw as well.
- A state with both spontaneous exits (row 5) and a contact-mediated exit (row 9) resolves the spontaneous ones jointly first, and transmission then acts on whoever is still in the state. This is an ordering effect between rows, of size O(dt), and is the same in every agent-based or chain-binomial run.
- Hazards from *different* modules acting on the same people (background mortality from a demographics module, disease mortality from a disease) are not pooled; they run in module order. Because deaths are deferred (next section), neither prevents the other from being recorded.

## Removals are deferred

Nobody is removed in the middle of a step. A module that kills an agent calls `sim.people.request_death(uids)`, which only records the request; the deaths take effect at row 10 (`step_die()`), after every module has run, and the agents are removed from the active population at row 13. This is Starsim's behavior, kept. A declared `ss.death` transition, and `I >> ss.null @ rate` out of a state of people, do the same thing.

Consequences: an agent who is requested to die at row 5 can still transmit at row 9 of that step and is still counted by any module that runs before row 10; two modules requesting the same death do not conflict; and custom code never sees arrays change length partway through a step. Emigration and other non-death removals use the same mechanism (`request_removal`).

## Per-module timesteps

A module may have its own `dt`; one that gives none uses the sim's. Modules with different timesteps are interleaved by **absolute time**, which is what Starsim does:

```
plan = every (time, function) pair, for every function in the table above and every time on its module's own timeline
sort plan by (time, position in the table above)
```

So a disease at `dt=ss.days(1)` and a demographics module at `dt=ss.years(1)` share one sim: on most days only the disease's functions are in the plan, and on the days the demographics module is due it runs at its usual place in the order. When every module uses the sim's timeline, which is the common case, the plan is just the table repeated once per step.

- **Time with no unit is in timesteps.** If nothing in the model has a unit (no `unit=`, no unit on `dt`, no dates, no typed values), the sim's `dt` is one step, the times in the plan are step indices, `stop=100` is 100 steps, and a module's own `dt` is a number of steps (`dt=7` runs it every seventh step). No unit is invented, and results are indexed by step. As soon as anything has a unit the sim takes its unit from that, in the order `unit=`, the unit of `dt`, `start`/`stop`, the typed values, and prints where it came from ([02-types.md](02-types.md)).
- A module reading another module's state reads its current value, whichever timestep that module is on.
- Rates and durations are converted using the `dt` of the module that uses them ([02-types.md](02-types.md)).
- Results are reported on each module's own timeline, as in Starsim. A flow from a coarse module that is requested on a finer grid is spread evenly across the interval and noted once (`W1002 CoarseFlow`).
- `sim.loop.plot_step_order()` shows the interleaving, which is the quickest way to check a sim with mixed timesteps.

Events inside a step (a commuting population that mixes at work from 08:00 and at home from 18:00, say) use the same mechanism: give the network or `ss.Mobility` a sub-daily `dt`, and it appears in the plan at those times. See [06-routes.md](06-routes.md).

## Order under each solver

| Solver | Declared transitions | Everything else (interventions, `step()` code, results) |
|---|---|---|
| `'abm'`, `'binomial'` | Evaluated in the loop, at rows 5 and 9 | In the loop, in order |
| `'ode'`, `'sde'` | All flows are evaluated together from the current state and integrated between loop times, so there is no order among them | Called at loop times, in loop order, on groups |
| `'gillespie'`, `'tau'` | Events are drawn in continuous time between loop times; no order among flows | Called at loop times, in loop order, on groups |

In a compartmental ODE run the question "does recovery happen before or after infection?" has no meaning, which is one reason the simplest models behave identically however they are written. Things that happen at points in time (a campaign on a date, a custom function called each step) still happen in loop order under every solver. The continuous-time solvers have no `dt` of their own; the sim's `dt` is then only the spacing of the loop times and of the results. See [12-backends.md](12-backends.md).

### States that are not people

A state that is not people (bacteria in water, mosquitoes) is an ordinary `ss.State` in the same module as the disease. It holds a plain count and is never represented as agents. There is no separate module class for it and so no separate place in the loop.

| Solver | People | States that are not people |
|---|---|---|
| `'ode'`, `'sde'`, `'gillespie'`, `'tau'`, `'binomial'` | Counts per group | More counts, advanced by the same solver with every other flow |
| `'abm'` | Agents | Counts (one number, or one per patch), advanced once per step in the disease's own rows: their flows are computed from the current counts (`I.n` is the number of agents in `I` at that moment) and applied to the count |

In an agent-based run the transitions among people are draws per agent, as always. The flows into and out of a count are applied as expected values each step (with the conversions in [02-types.md](02-types.md)), because these quantities are usually large and are not individuals; `n_agents=` and `total_pop` do not apply to them, and neither does the scale factor between agents and people. A person's infection rate that uses the count (`beta * W.n / (W.n + K)`) is evaluated per agent from the count as it stands when row 5 runs. `describe()` prints which states were taken to be people and which were not, and marks the ones it guessed. Whether these flows should be random draws by default is open ([19-open-questions.md](19-open-questions.md)).

## Determinism

Requirements for reproducibility ([11-random.md](11-random.md)):

1. Given the same model, the same settings, and the same seed, a run MUST produce identical results on repeated runs on the same machine and library versions, in serial or in parallel.
2. Running the runs of a MultiSim, or scenarios, in parallel MUST NOT change any individual result.
3. Iteration over agents, edges, or groups MUST be in a defined order. Set and dict iteration order MUST NOT influence any draw.
4. Sums whose order could depend on thread count (a force of infection summed over groups, for example) SHOULD use a fixed order. Bit-identical results across different machines or numerical libraries are a goal, not a requirement.

Requirement 2 rules out one random number stream shared across workers, and is one of the reasons each distribution has its own.

## Building the sim

`sim.init()` builds the sim in this order (`sim.run()` calls it if it has not been called):

1. Resolve the declarations: states (declared or inferred, and which of them are people), the kind of each transition, the sim's unit (from `unit=`, then the unit of `dt`, then `start`/`stop`, then the typed values; none of these means time is in steps), and defaults. Every guessed default is recorded for `describe()`.
2. Expand stratifications into strata and generate the transitions between them.
3. Load named data: parameters from data sources, and observed data attached to results by name.
4. Check the dimension of every rate and the shape of every array-valued parameter.
5. Register modifiers against their targets.
6. Choose the solver if `solver='auto'`, and work out any approximations the solver needs ([12-backends.md](12-backends.md)).
7. Write the model spec ([14-model-spec.md](14-model-spec.md)) and compute `sim.hash()` ([11-random.md](11-random.md#run-identity)).
8. Build the loop.
9. Allocate state, set initial conditions, and initialize the random number streams.
10. Print the solver with its reason, any approximations, and where the sim's unit came from if it was not given as `unit=`, one line each (`verbose=0` hides them; `approx=True` hides the approximation notes only). `sim.describe()` prints the full resolved model, including the schedule, at any time.

Problems are reported at the earliest step that can detect them ([15-errors.md](15-errors.md)): a typo in a state name is an error at step 1, not at the first timestep. The exception is user functions, which are not analyzed in advance; a function that asks for something the solver cannot provide is reported when it is first called ([12-backends.md](12-backends.md) §Functions).

## Finishing

After the last step, `sim.finalize()` calls each module's `finalize()` once, in module order, and then finalizes the results (scaling, summary statistics). This is Starsim's behavior. Calling `run()` again on a completed sim raises `E1003 RunAfterFinish`; copy the sim or build a new one to run again.

## Rejected

- **Queued updates** (every module reads start-of-step state and all changes are applied together at the end). It removes order effects, but it also removes within-step chains, needs its own tie-breaking rule when two modules write the same quantity, and adds a second pass. An explicit order is simpler to explain and to debug.
- **Named phases.** A phase is a position in a list; the list is already printable and insertable-into.
- **Ordering derived from declared dependencies.** Real models have mutual dependence (HIV and syphilis), so a derived order needs a cycle-breaking rule anyway. Convention plus an explicit order does the same job.
- **Sequential draws for competing hazards.** Declaration order inside a module would bias results.
- **Removing agents mid-step.** It invalidates indices held by other modules and hides deaths from modules that run later.
