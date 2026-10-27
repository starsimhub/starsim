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
| 0 | sim | `advance()` | The time index moves to this step's (see below) |
| 1 | sim | `start_step()` | Advance the sim's distributions to this step; print progress |
| 2 | every module | `start_step()` | Advance each module's distributions; per-step bookkeeping |
| 3 | plain `ss.Module`s | `step()` | Modules that are none of the types below |
| 4 | demographics | `step()` | Births and immigration add agents; background deaths are requested |
| 5 | diseases | `step_state()` | State updates of a disease written as code (a declared disease takes all its transitions at row 9) |
| 6 | connectors | `step()` | Cross-module code |
| 7 | networks | `step()` | Add and remove edges; mixing pools update who meets whom |
| 8 | interventions | `step()` | Deliveries, products, parameter changes |
| 9 | diseases | `step()` | Every declared transition: those that are not contact-mediated (progression, branching, `ss.transfer`, disease deaths), then transmission |
| 10 | people | `step_die()` | Requested deaths take effect (see [Removals](#removals-are-deferred)) |
| 11 | people, then every module | `update_results()` | Counts, flows, declared results and `ss.outcome`s for this step |
| 12 | analyzers | `step()` | Analyzers read the finished step |
| 13 | every module, people, sim | `finish_step()` | Remove dead agents |

**One time index.** The step from time point `k-1` to `k` is step `k`, and in every row of it, in every module, `sim.ti` is `k`: the index where its results go. So `self.results.x[self.ti] = v` is this step's value whether it is written in `step()` or in `update_results()`, a time recorded as `self.ti` in one row compares equal to `self.ti` in any other, and a delivery's count and a transition's flow for the step land at the same index. `sim.now` is when the step starts, time point `k-1`: it is what the step reads values that vary in time at (rates by year, `ss.timeseries`, data by period), and what a date means (`start=2020` is the first step that starts on or after 2020). Code in a step that reads results reads the previous step's, at `ti - 1`, since this step's are recorded at row 11. At the first time point (`ti == 0`, `now` the start) there is no step: only rows 11 and 12 run (without the interventions, which have done nothing yet), so the loop itself records the start. This is Starsim 3's convention, apart from index 0 being the start of the run, so the first step is `ti == 1`. A declared engine takes people's state at the start of the step, so it times a move that is not an infection from `ti - 1`, the step's start, and an infection by contact from `ti`, its end (see `ss.solvers.Engine.ti`); but every time of entering a state reads as the step it happened in (`ti_X == sim.ti` is "entered this step", whatever the move), and a scheduled exit (`ti_exit_X`) as the step it will be taken in. Someone placed at the start reads as entering in step 1, as if they had entered at its start (a seed counted from the step before, as 0). The clocks (`X.time`, `.since`) are the time actually spent, from the engine's own times (see `ss.arrays.StepArr.internal()`). `sim.run(until=t)` takes the steps up to and including the one that starts at `t`.

The reasons for this order are Starsim's: new agents exist before diseases initialize them; states update before networks, so that a state change can alter who is eligible for contact; networks update before interventions, so that an intervention can alter the final network for the step; and interventions run before transmission, so that someone vaccinated today is protected today. A declared disease takes all of its transitions at row 9, after the interventions, so a delivery sees the same people under every solver: those at the start of the step, before any of its flows.

### Order within a type

**Within a type, modules run in the order given.** `ss.Sim(diseases=[syphilis, hiv])` runs syphilis before HIV in row 9 (and row 5, for diseases written as code); `interventions=[screen, treat]` runs screening before treatment in row 8. There is no other ordering mechanism: no priorities, no phases, no dependency graph.

### Where declared transitions run

A disease built from declared transitions ([03-states-and-transitions.md](03-states-and-transitions.md)) needs no `step()` written by the user. All of its transitions run in row 9, `step()`, after networks and interventions have run, under every solver; within the row, as agents, they are taken in two parts by their kind, which is derived from the rate:

- **Not contact-mediated** (the rate mentions no state), first. This covers `ss.progression`, branching, dwell times that elapse this step, `ss.transfer`, `ss.death` and `ss.birth` declared on the disease, and the transitions into and out of states that are not people (a water reservoir, a mosquito population; [03-states-and-transitions.md](03-states-and-transitions.md)): `shedding = ss.null >> W @ shed * I.n` and `decay = W >> ss.null @ rate` mention no bare state. See [States that are not people](#states-that-are-not-people).
- **Contact-mediated** (the rate mentions a state), then.
- **A plain function passed to `ss.Disease(...)`** as a keyword is called at the end of that disease's row 9. A subclass that overrides `step()` replaces the row, as in Starsim (and so has to call `super().step()`).

### Where `ss.modify` is evaluated

A modifier ([07-effects.md](07-effects.md)) acts where the engine uses the quantity, so a modifier with a `where=` on a state reflects that state as it is when the quantity is used, whatever the module order. A `when=` function is called once per step, at the modifier's place among the connectors. A modifier that depends only on time, and on who it applies to only through states and stratifications, is worked out before a compartmental run, so the run stays in one call to the kernels. Because the base value is never overwritten, the original value comes back when a modifier ends.

Direct writes by custom code are a different matter: the loop does not record them. `sim.what_modifies(x)` lists the declared modifiers on `x` ([07-effects.md](07-effects.md)).

Example: `ss.modify(hiv.rel_trans, multiply=2.0, where=syphilis.active)`. If syphilis is listed before HIV, a person who acquires syphilis at row 9 of this step has doubled HIV transmissibility when HIV transmission runs later in row 9 of the same step. If syphilis is listed after HIV, the doubling is first used one step later. Both are valid; the difference is O(dt).

## A within-step chain

```python
import starsim as ss

S, I, R = ss.states('S I R')
sir    = ss.Disease(
    infection = ss.Transition(S, I, rate=0.3 * I),
    recovery  = ss.Transition(I, R, dur=10),
    init_prev = 0.01,
)
test   = ss.Diagnostic(sensitivity=0.9, specificity=0.98)
drug   = ss.Treatment(cure=0.95)
screen = ss.screen(test, coverage=0.2)
treat  = ss.treat(drug, who=screen.positive)

sim = ss.Sim(sir, interventions=[screen, treat], unit='day', dt=1, stop=100)
```

Testing and treatment happen in the same step: the treatment is a step of its own, triggered by the test's outcome `screen.positive`, and it acts on the people who tested positive in that step. Within row 8, the deliveries run in the order of their triggers: each after the deliveries whose events trigger it, whatever order they are given in (otherwise in the order given), and a cycle of triggers raises, naming it. Steps chain to any depth this way ([08-interventions.md](08-interventions.md#cascades)).

Infection is at row 9, after interventions, so with the default order a person infected today is first eligible for screening tomorrow. To put screening after transmission, so that infection, testing, and treatment can all happen in one step, move it in the loop:

```python
sim.init()
sim.loop.insert(screen, label='sir.step')   # run screening right after sir's transmission
sim.loop.insert(treat, label='screen.step') # and the treatment it triggers after it
sim.run()
```

Which of the two is right depends on what a step represents in the model. With a one-day step the difference is a day; with a one-month step it may matter, and `sim.check_dt()` will show it.

## The loop is a data object

```python
sim = ss.Sim(sir, unit='day', stop=100)
sim.init()

def fn(sim):             # any function of the sim
    pass

sim.loop                 # the plan: every (time, module, function) entry in execution order
print(sim.loop.to_df())  # as a table: time, ti, func_order, label, module, func_name, cpu_time
sim.loop.plot()          # as a diagram
sim.loop.insert(fn, label='sir.step')                 # call fn(sim) after sir.step at every step
sim.loop.insert(fn, label='sir.step', before=True)    # ... or before it
sim.loop.insert(fn, match_fn=lambda plan: (plan.label == 'sir.step') & (plan.time > 50))
sim.run()
```

This is Starsim's `ss.Loop`, unchanged apart from one addition: `insert()` also accepts a module, which moves that module's `step()` to the given position (as in the screening example above). The loop exists after `sim.init()`; its plan is made when it is first needed, so a sim that the kernels run in one call never makes it. Labels are `'<module name>.<function name>'`; `sim.loop.to_df().label.unique()` lists them. In the plan, the calls of a step are listed under the time point it ends at (`ti` is 1 for the first step), while `sim.ti` during the step is the time point it starts from.

A sim with functions inserted into its loop is taken a step at a time, not in one call to the kernels; `sim.kernel_reason()` says why a run is stepped.

**What does nothing is not called.** A call that is sure to do nothing in this sim is left out of the plan, so the loop costs a module written as code only the methods it defines: a module's `start_step()` with no random numbers to move on, its `finish_step()` and `update_results()` when inherited unchanged (and, for `update_results()`, with no states to count), and the people's deaths and removals in a run without agents. The sim's own calls (`sim.advance`, `sim.start_step`, `sim.finish_step`) are always kept, as places to insert others. A method left out counts as called for `sim.check_method_calls()`. A sim of one module written as code, daily for 21 years (7,672 steps), spends about 0.01 s in the loop itself, a few percent of a run whose step is compiled (He et al. 2010, [tutorial G8](../../docs/tutorials/g8_extending.qmd)).

Whatever the order is, it can be printed, and it is the only thing that determines what runs when.

### The schedule in `describe()`

`sim.loop` and `sim.loop.to_df()` are the fully unrolled plan: every call at every time. That is exact and hard to read. `sim.describe()` MUST print the loop as a **schedule grouped by timestep** instead: one line per timestep in use, listing the modules on it in the order they run.

```text
>>> sim.describe()
...
schedule     every day: randomnet, sir
             every 365 days: births, deaths
```

- Groups are ordered from the shortest timestep to the longest. Within a group, modules are in the order of their first row in the table above, and a module with several rows is listed once.
- In a sim with no unit the line reads `every step:`, and a module on a coarser timestep `every 7 steps:`.
- The unrolled list is `sim.loop.to_df()`, on request.

## Competing hazards are resolved jointly

Order between modules is a choice. Order between the transitions inside one module is not something a user should have to think about, so it does not affect results: **transitions out of the same state that are evaluated in the same call compete in one draw.**

```
h_total = sum_k h_k
p_exit  = 1 - exp(-h_total * dt)
P(via k | exit) = h_k / h_total
```

In a discrete-time run this MUST be implemented as a single multinomial draw per source state, not as sequential binomial draws. Sequential draws give the first-declared transition an advantage of order `h * dt`, which would make results depend on declaration order.

Details:

- Separate transitions out of one state that carry `p=` are a split decided on entry to the state, not a competition, and one transition with several destinations splits when it is taken ([03-states-and-transitions.md](03-states-and-transitions.md) §Branching); the split is one draw as well.
- As agents, exits are scheduled at entry. Moves with no wait (a split with no dwell time, or a dwell time that rounds to no steps) happen in the step that caused them: the moves cascade until nobody moves (a cycle raises a `ValueError`, which says when a dwell time shorter than a step is the cause). Compartments pass such moves straight through, so this is what makes agents agree with them.
- A dwell time competing with a rate: someone who enters a state during a step, after its rates were drawn, has the rates drawn for them in the step their dwell time ends too, so the rate acts for as long as the dwell time says.
- As agents, people who arrive in a state with rates out of it get half a step of those rates in the step they arrive in, since on average they arrived halfway through it: an arrival in the state-change row has that half step drawn at once, and is in the state for that step's transmission. Someone infected (or moved at once after an infection) is put in their state at the step's end, after its transmission, so if that state is a source of infection they were infectious for half a step that no transmission counts. They have half a step of the state's rates at their first draw, in the next step (and, with a scheduled exit as well, the other half in the step it is due), so each stay counts in transmission for as many steps as it lasts, to first order in γ·dt. With a whole step at that first draw, a model in which infection leads straight to an infectious state ran as if R0 were R0·(1 − γ·dt/2) (an SIS at R0 = 1.2 with γ = 1 a year, at a quarterly step: prevalence 0.068 where the exact value is 0.167; decided 2026-10-25). A state that is not a source (a latent state) has a whole step at the first draw, since there half a step would only make the stay longer. A dwell time needs no such correction: it is rounded to whole steps from the time of entering (the end of the step, for an infection), which keeps its mean. The stock of an infectious state recorded at the step's end then reads high by about half a step of infections (+γ·dt/2: 4% at γ·dt = 1/12), since it is taken just after the step's infections, and the next step's exits balance them; its mean over the step is right to first order.
- As agents, a state with both spontaneous exits and a contact-mediated exit resolves the spontaneous ones jointly first, and transmission then acts on whoever is still in the state (both in row 9). This is an ordering effect of size O(dt); the compartmental solvers draw them together (the chain-binomial solver draws the number leaving each compartment once, then shares it between the exits).
- Hazards from *different* modules acting on the same people (background mortality from a demographics module, disease mortality from a disease) are not pooled; they run in module order. Because deaths are deferred (next section), neither prevents the other from being recorded. The exception is diseases whose states are crossed in a compartmental run ([07-effects.md](07-effects.md)): they are one model, whose rates are all worked out from the people at the start of the step.

## Removals are deferred

Nobody is removed in the middle of a step. A module that kills an agent calls `sim.people.request_death(uids, cause=self)`, which only records the request; the deaths take effect at row 10 (`step_die()`), after every module has run, and the agents are removed from the active population at row 13. This is Starsim's behavior, kept. A declared `ss.death` transition, and `I >> ss.null @ rate` out of a state of people, do the same thing, with the disease as the cause. Each person dies once, of one cause: when several modules request the death of the same person in the same step, `step_die()` chooses one of them, each equally likely whatever the order of the modules (one random number per person, so common random numbers hold), and keeps it in `people.cause_of_death` (a position in `people.causes`). Each module counts its deaths after that, from `people.died_of(module)` (a declared death flow takes out those who died of something else), so the deaths by cause add up to the sim's deaths.

Consequences: an agent who is requested to die earlier in the step can still transmit at row 9 of that step and is still counted by any module that runs before row 10; two modules requesting the same death do not conflict; and custom code never sees arrays change length partway through a step. Emigration and other non-death removals use the same mechanism (`request_removal`).

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
- Results are reported on each module's own timeline, as in Starsim.
- A declared disease with a timestep of its own is taken a step at a time (it is not run in one call to the kernels).
- `sim.loop.to_df()` shows the interleaving, which is the quickest way to check a sim with mixed timesteps.

Events inside a step use the same mechanism: a network given a sub-daily `dt` appears in the plan at those times. Commuting between places (`ss.geo.Mobility`) is not a module and has no place in the loop: it is a share of time spent in each place, applied within every step ([06-routes.md](06-routes.md)).

## Order under each solver

| Solver | Declared transitions | Everything else (interventions, `step()` code, results) |
|---|---|---|
| `'abm'` | Drawn per agent at row 9, in the two parts above | In the loop, in order |
| `'binomial'` | Drawn at row 9 from the counts at the start of the step, all flows together | Called at loop times, in loop order, on groups |
| `'ode'` | All flows are evaluated together from the current state and integrated over the step, so there is no order among them | Called at loop times, in loop order, on groups |
| `'gillespie'`, `'tau'` | Events are drawn in continuous time over the step; no order among flows | Called at loop times, in loop order, on groups |

In a compartmental run the question "does recovery happen before or after infection?" has no meaning, which is one reason the simplest models behave identically however they are written. Things that happen at points in time (a campaign on a date, a custom function called each step) still happen in loop order under every solver. A compartmental sim with nothing that needs a step of its own is run in one call to the kernels, with no loop at all: deliveries on a schedule to people chosen by states and stratifications are taken by the kernels at their place in the order (before the step's flows), and `sim.kernel_reason()` says why a run is stepped instead. The continuous-time solver has no `dt` of its own; the sim's `dt` is then only the spacing of the loop times and of the results. See [12-backends.md](12-backends.md).

### States that are not people

A state that is not people (bacteria in water, mosquitoes) is an ordinary `ss.State` in the same module as the disease. It holds a plain count and is never represented as agents. There is no separate module class for it and so no separate place in the loop.

| Solver | People | States that are not people |
|---|---|---|
| `'ode'`, `'binomial'`, `'gillespie'` | Counts per group | More counts, advanced by the same solver with every other flow |
| `'abm'` | Agents | Counts (one number, or one per patch), advanced in row 9 with the transitions that are not contact-mediated: as an ODE over the step (by RK4), with the people as they were at its start (`I.n` is the number of agents in `I` then) |

In an agent-based run the transitions among people are draws per agent, as always. The flows into and out of a count are not drawn, because these quantities are usually large and are not individuals; `n_agents=` and `total_pop` do not apply to them, and neither does the scale factor between agents and people. A person's infection rate that uses the count (`beta * W.n / (W.n + K)`) is evaluated per agent from the count as it stands when transmission is taken, after the count has been advanced. `describe()` prints which states were taken to be people and which were not, and marks the ones it guessed. Whether these flows should be random draws by default is open ([19-open-questions.md](19-open-questions.md)).

## Determinism

Requirements for reproducibility ([11-random.md](11-random.md)):

1. Given the same model, the same settings, and the same seed, a run MUST produce identical results on repeated runs on the same machine and library versions, in serial or in parallel.
2. Running the runs of a MultiSim, or scenarios, in parallel MUST NOT change any individual result.
3. Iteration over agents, edges, or groups MUST be in a defined order. Set and dict iteration order MUST NOT influence any draw.
4. Sums whose order could depend on thread count (a force of infection summed over groups, for example) SHOULD use a fixed order. Bit-identical results across different machines or numerical libraries are a goal, not a requirement.

Requirement 2 rules out one random number stream shared across workers, and is one of the reasons each distribution has its own.

## Building the sim

`sim.init()` builds the sim in this order (`sim.run()` calls it if it has not been called):

1. Resolve the declarations: states (and which of them are people), the kind of each transition, the sim's unit (from `unit=`, else the model's typed values, else dates; none of these means time is in steps; [02-types.md](02-types.md)), and defaults. Every guessed default is recorded for `describe()`.
2. Expand stratifications into strata and generate the transitions between them.
3. Load named data: parameters from data sources, and observed data attached to results by name.
4. Check the dimension of every rate, the shape of every array-valued parameter, and every parameter's value against its bounds (a `ValueError`).
5. Register modifiers against their targets.
6. Choose the solver if `solver='auto'`, and work out any approximations the solver needs ([12-backends.md](12-backends.md)).
7. Collect the loop's functions (the plan itself is made when the loop first runs).
8. Allocate state, set initial conditions, and initialize the random number streams.
9. Print the solver with its reason, any approximations, where the sim's unit came from if it was not given as `unit=`, and, for each disease with a transmission, R0 (in a population with no immunity; with the reproduction number at the start too, if immunity at the start changes it by more than 1%) and the number infectious at the start, one line each (`verbose=0` hides them; `approx=True` hides the approximation notes only). The notes are kept in `sim.notes` whatever is printed. `sim.describe()` prints the full resolved model, including the schedule, at any time.

The model spec ([14-model-spec.md](14-model-spec.md)) and `sim.hash()` ([11-random.md](11-random.md#run-identity)) are worked out when asked for, not at `init()`, since a run does not need them.

Problems are reported at the earliest step that can detect them ([15-errors.md](15-errors.md)): a typo in a state name is an error at step 1, not at the first timestep. The exception is user functions, which are not analyzed in advance; a function that asks for something the solver cannot provide is reported when it is first called ([12-backends.md](12-backends.md) §Functions).

## Finishing

After the last step, `sim.finalize()` calls each module's `finalize()` once, in module order, and then finalizes the results (scaling, summary statistics). This is Starsim's behavior. Calling `run()` again on a completed sim runs it again from the start, built again as `sim.new()` would (with any settings or parameters changed since the run); stepping or finalizing it again raises `AlreadyRunError`.

## Rejected

- **Queued updates** (every module reads start-of-step state and all changes are applied together at the end). It removes order effects, but it also removes within-step chains, needs its own tie-breaking rule when two modules write the same quantity, and adds a second pass. An explicit order is simpler to explain and to debug.
- **Named phases.** A phase is a position in a list; the list is already printable and insertable-into.
- **Ordering derived from declared dependencies.** Real models have mutual dependence (HIV and syphilis), so a derived order needs a cycle-breaking rule anyway. Convention plus an explicit order does the same job.
- **Sequential draws for competing hazards.** Declaration order inside a module would bias results.
- **Removing agents mid-step.** It invalidates indices held by other modules and hides deaths from modules that run later.
