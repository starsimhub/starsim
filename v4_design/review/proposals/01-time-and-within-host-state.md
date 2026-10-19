# Proposal 01: time and within-host state

Status: draft for Robyn and Cliff. Scope: everything in the ports that depends on elapsed time or on per-agent continuous state. Today that means undocumented `ti_<state>` arrays, `engine.events`, and fragile `step_state()` ordering (REVIEW §2a, §3). Out of scope, but coordinated with this one: branch timing (`decide='exit'`, `when=`) and programs (delivery memory). Both are covered by sibling proposals in this folder.

## 0. Summary

Four primitives. Each has one meaning as agents and one stated realization as compartments.

| Primitive | Syntax | As agents | As compartments |
|---|---|---|---|
| **Clock** | `X.time`, `(A \| B).time`, `X.progress`; longhand `ss.clock(start, within, bins)` | elapsed time since the start event, per agent | a `clock` dimension over `within`: levels of elapsed time, moved up by a conveyor; `.progress` reads the dwell's Erlang stage |
| **Attribute** | `ss.attr(value, on=event, combine=, bins=)` | a float per agent: drawn or recorded at an event, or derived (no `on=`) | recorded: a binned dimension (or the mean, with a note); derived: evaluated per group, no compartments |
| **Event** | a transition, `tr.to(B)`, `X.enter`, `X.exit`; `ss.func(fn, on=event)` | who moved, at the moment they move; usable as a condition and in `on=` | a flow; hooks taking `uids` are agent-only (E804) |
| **Spread** | `init_prev=ss.spread(x, over=dur)`, `init=dict(L=ss.spread(x))` | seeds back-dated, with the natural history run forward to the start | the linear spontaneous subsystem integrated with constant inflow, normalized |

Plus two engine fixes that the primitives rely on:

- exits are scheduled **at entry**, not at the start of the next step;
- attributes and hooks run **inside the move** (`finish_moves`, `force`), before anything reads the new state.

Together these remove every `ti_<state>`, `engine.events` and ordering workaround in the seven ports. They also answer questions-oct12 #7 (fixed delays), #8 (history before the start) and #11 (minimum stays), and spec B7 (within-host state).

## 1. Constraints taken from the spec

- **The model is data.** Clocks and attributes are declared objects, so they go into the model spec, `describe()`, `equations()` and the hash.
- **The try-it rule.** An expression of clocks and attributes is not a function, so it runs on groups. Only `fn(sim, uids)` hooks are agent-only.
- **Approximations are stated.** Clocks are the same kind of approximation as Erlang stages, so `'auto'` may make them, with a note. An attribute drawn from a distribution without `bins=` is "a distribution on a per-person quantity" (12-backends), so `'auto'` runs the model as agents. `approx=False` refuses both.
- **Few keywords.** There is one `on=` everywhere (deliveries already use it). The stratification machinery is reused, with no parallel structure.

## 2. Clocks

### Syntax

```python
EL.time                  # time in EL: since this person last entered EL
(EL | LL).time           # since they last entered the set from outside it (moves within the set don't reset it)
hiv.infected.time        # time since infection, across stages
falling.progress         # share of the scheduled dwell in `falling` that has elapsed, in [0, 1]

ss.clock(start=EL.enter, within=EL | LL, bins=ss.months(3), name='latent_time')   # longhand, for options and a name
```

- `.time` is a typed duration in the sim's unit, so `EL.time / ss.years(1)` is a number.
- A clock is a lazy expression, like `I.n`. It works in `rate=`, `dur=` (read at entry), `p=`, modifier amounts, `where=`/`eligible=`, and result expressions.
- `X.time` means `ss.clock(start=X.enter, within=X)`.
- Event clocks (`tr.since`, `delivery.since`) are `ss.clock(start=tr, within=<all people states>)`. The programs proposal owns `delivery.since` and the count of deliveries per person. They use this object, and nothing new is needed here.

### Semantics as agents

- Each clock is one `FloatArr`: the time it last started. A single state reuses the existing `ti_<state>` array (`solvers.py:3122`), and a set gets its own.
- Reading a clock gives `t - start` for agents within its domain, and NaN outside it.
- `X.progress` is `(t - ti_X) / (ti_exit_X - ti_X)`. It needs exits scheduled at entry (§4).
- A state with only hazard exits has no scheduled dwell, so `.progress` on it raises `E213 NoScheduledDwell`.

**Domain check (new, `E212 ClockOutsideDomain`).** The NaN bug class in the ports came from reading a clock or a `ti_` array where it does not exist: the CT seeds and syphilis `init=LL` got NaN `rel_trans`, and new HIV infections got NaN CD4, which silently picked the lowest mortality bin. Every use of a clock is therefore checked at build time:

- the source of a transition whose rate reads the clock must be inside its `within`;
- a modifier whose amount reads the clock must have a `where=` made of states inside `within`.

A function in a `where=` cannot be checked statically. It is checked at run time instead: a NaN amount raises an error rather than flowing into `np.digitize`.

### Realization as compartments

A clock is a dimension of kind `clock` over the states in `within`, using existing machinery:

| Piece | Existing mechanism reused |
|---|---|
| dimension over some states only | `stratify(states=...)` |
| entering `within` from outside lands at level 0 | `enter_dims` with split `[1, 0, ...]` |
| moves inside `within` keep the level | transitions between states that share a dimension keep it |
| elapsed time advances | a **conveyor**: everyone moves up one level every `w`. This is what products' `waning=` already does (`products.py:312 init_stages`) |
| last level | absorbing (the clock is held) |

- **Levels.** With `bins=`, the levels are as given. Otherwise they are chosen as products choose theirs: 10 levels covering the time until every expression reading the clock settles (to within 1%), or the whole sim. Every comparison of the clock with a constant is added as an edge, so `on_art.time >= ss.months(6)` is exact at 6 months. This is the edge rule for `age`, applied automatically because the clock's bins are generated, not chosen by the user.
- **Exactness.**
  - With levels one step wide, elapsed time is exact on the step grid. A hazard `f(X.time)` is evaluated at the same elapsed times as for agents, so the only difference is the usual ODE-versus-ABM difference.
  - With wider levels, `f` is read at the middle of each level, with a note.
  - Past the last level, `f` is read at its start, with a note if `f` has not settled.
- **`.progress`** reads the dwell's Erlang stage: stage `j` of `k` is `(j - 0.5)/k`. This is exact for the staged model the ODE runs, and an approximation of the agent-based dwell. With an exponential dwell (`k = 1`), `.progress` in a compartmental run raises `E804` and suggests `stages=`, since adding stages would change the dwell distribution.
- **Under the stochastic solvers** the conveyor is a deterministic shift at loop times, as deliveries already are. `equations()` prints it as `shift X[j] -> X[j+1] every w`.

What `describe()` prints:

```text
clocks   latent_time   (EL | LL).time   since entering EL|LL                       [read by latent_decay]
           abm: per agent
           ode: 12 levels of 3 months, conveyor, last level held      note: f read at the middle of each level
```

### Cost

- As agents: one float per agent per clock set, and none for a single state. Reads are vectorized subtractions.
- As compartments: the states in `within` multiply by the number of levels.

`describe()` reports the total number of compartments, and `W309` fires as usual.

## 3. Attributes

One constructor covers three cases. They differ only in whether `on=` is given and what `value` is.

```python
cd4_latent = ss.attr(ss.normal(500, 50), on=infection)     # DRAWN at an event: once per agent per event
cd4_preart = ss.attr('cd4', on=start_art)                   # RECORDED: an expression evaluated on the movers, before the move
rel_sev    = ss.attr(ss.normal(1, 0.2))                     # TRAIT: a distribution with no on= is drawn once, when the agent is created
cd4        = ss.attr({acute: ..., latent: ..., on_art: ...}) # DERIVED: an expression (or a dict by state) with no on=; re-evaluated when read
```

| Keyword | Meaning |
|---|---|
| `value` | a `Dist`, an expression of clocks, attributes, parameters and states, a name (`'cd4'`, so recorded attributes can be declared before what they record), or a dict keyed by state or state set. The dict is the existing per-state idiom of `ss.Test(sensitivity=dict(...))`; states not listed give NaN |
| `on=` | an event (§4), or several joined with `\|` |
| `combine=` | `'last'` (default), `'first'`, `'max'`, `'min'`, `'sum'`, or `fn(old, new)` |
| `init=` | the value before any event (default NaN, or 0 when `combine='sum'`) |
| `bins=` | the compartmental realization (below) |

**Rules:**

- Attributes on the same event are set in declaration order, and each sees the ones before it.
- Draws use a CRN slot named after the attribute.
- An attribute is a named numeric quantity, so it is an `ss.modify` target (`ss.modify(hpv.rel_sev, multiply=1.5, where=hiv.aids)`) and a valid `stratify()` key.

**As agents:**

- A drawn, recorded or trait attribute is a `FloatArr`, written inside the move (§4).
- A derived attribute is evaluated lazily and cached per step. The cache is invalidated by moves and by the clocks it reads.

**As compartments:**

| Kind | Realization | Exact? |
|---|---|---|
| derived | evaluated once per group (state × stage × clock level × attribute bins) | exact for the groups |
| drawn or trait, no `bins=` | its mean. `'auto'` runs the model as agents instead | approx (note) |
| drawn or trait, `bins=k` | a dimension over the states where it is read, with `k` equal-probability bins, each at its conditional mean | exact for the bins |
| recorded (an expression) | a dimension over the states where it is held. The level on entry is the bin of the expression in the source group. Default bins come from a stratification of the same quantity, if there is one, otherwise from `bins=`, otherwise `E804` asking for it | exact for the bins |
| recorded as a deterministic function of other recorded attributes at the same event | no new dimension: it is a function of their levels | exact |
| `combine='max'`, `'sum'`, ... | a transition between bins on each event (for `max`, the probability that the draw lands above the current bin) | exact for the bins |

**Derived dimensions.** Stratifying by a derived attribute adds **no compartments** in a compartmental run. The level is a function of each group, recomputed when the group's values change. As agents, it is recomputed each step, as `age` already is (implementation.md, `groups_changed`). This is what makes CD4 bins cheap in an ODE, and it gives deaths by CD4 as a free `.by('cd4')` (§7.4).

What `describe()` prints:

```text
attrs    cd4_latent   drawn on infection     normal(500, 50)   ode: 3 bins (exact for the bins)
         cd4_preart   recorded on start_art  cd4               ode: 5 bins from stratify(cd4=...)
         cd4          derived by state                         ode: per group
dims     cd4          [0, 50, 200, 350, 500] derived from hiv.cd4     ode: no compartments (a function of each group)
```

## 4. Events, hooks, and scheduling at entry

**Events.** These are the things that happen to someone at a moment in time:

- a transition (`infection`);
- one branch of it (`precin.to(S, L)`, the event form of `.select(to=)`);
- `X.enter` and `X.exit`, for a state or a set. Entering a set means arriving from outside it, by a transition or an inflow;
- a delivery (the programs proposal).

Every event is:

- **a condition**: true for the agents it happened to, from the moment of the move until the same point in the next step (`engine.events`, made public);
- **`event.uids`**, inside `step()` code;
- **a trigger for `on=`**, in `ss.attr`, `ss.func`, and deliveries.

**Hooks**, for anything an attribute can't express:

```python
def set_cd4(sim, uids): ...                       # any Python
cd4_hook = ss.func(set_cd4, on=hiv.start_art)     # called with the movers, at the move
```

`ss.func` already exists to attach options to a function, so `on=` is one more option. In the class form a hook is a class attribute, like transitions. A hook taking `uids` is agent-only, and `sim.solvers()` lists it as such.

**When attributes and hooks run.** They run inside `finish_moves()` (`solvers.py:3425`), after the destination is set and before the step goes on. A move forced by an intervention runs them inside `force()` (`solvers.py:3230`) too. Initial placement runs them inside `init_post()`. A hazard that reads CD4 therefore never sees a newcomer with NaN, whatever the module order. The rule that the HIV port had to discover ("infection hook before `super()`, ART hook after") disappears.

**Exits scheduled at entry.** `schedule()` (`solvers.py:3489`) is called from the same three places for the agents who just entered, not at the start of the next step. Consequences:

- `ti_exit_<state>` is never NaN for someone in the state, which `.progress` needs;
- `p=` on entry is evaluated at the time of entry. This fixes "the mother's stage is read a step late" (syphilis §9);
- branches with no dwell resolve in the same step. The branch-timing proposal owns the full fix, including the guard against cycles.

The cost is nil: the call moves earlier, and `rescale_exits()` still handles modifiers that change afterwards.

**Initial placement fires events.** People placed by `init=` or `init_prev=` count as having gone through the transitions on the shortest path from the first state to their state, at the start. So `cd4_latent` (on `infection`) is drawn for seeds in `latent`, and `ss.spread(x, over=0)` is the same as plain `init`.

## 5. Spread: starting mid-history

```python
init_prev = ss.spread(0.052, over=ss.years(10))           # infected at a constant rate over the last 10 years
init_prev = ss.spread(0.052, over=(ss.months(5), ss.years(10)))   # v3 HIV's back-dating window
init      = dict(EL=ss.spread(0.01), LL=ss.spread(0.01))  # no over=: the stationary distribution within each state
```

- **Meaning.** The given count or fraction is who is in the target at the start: the infected set for `init_prev`, the named state for `init=`. These people are back-dated: their arrival times are uniform over `over`, conditional on still being in the target at the start. Their declared natural history then runs forward to the start: stage, time in stage, every clock, and every attribute set on the way. With no `over=`, arrivals are into that state only, long enough ago to be stationary.
- **As agents.** Seed `x / s` agents with back-dated entry times, where `s` is the survival share, estimated with one probe. Run only the spontaneous transitions of this disease for those agents up to the start, at the sim's `dt`, and keep those still in the target. There is no transmission, demography or intervention before the start, and age-dependent rates use the age at the start (stated in the note). The cost is O(seeds × over/dt), negligible next to the run.
- **As compartments.** Integrate the linear spontaneous subsystem with a constant unit inflow over `over`, so that mass spreads over the Erlang stages, the clock levels and the attribute bins. Then scale to the target. With no `over=`, solve for the stationary state (`A x = -e`). This is exact for the compartmental model.
- **With only exponential dwells, constant hazards and no clocks**, `ss.spread` changes nothing (they are memoryless), and the note says so.

`describe()` prints: `init  hiv  5.2% infected, back-dated over [5 months, 10 years] (natural history only; ages as at start)`.

## 6. Interactions

- **`ss.modify`.**
  - Amounts and `where=` take clock and attribute expressions.
  - Attributes are targets.
  - `what_modifies()` prints the expression.
  - Rescaling dwells in progress (`rescale_exits`) is unchanged.
- **Stratification.**
  - A clock is a dimension of kind `clock`, so `adjust=` by clock level works. This gives a piecewise hazard in time since an event, exact in every solver.
  - `stratify(cd4=...)` on a derived attribute is a derived dimension (§3).
  - Clocks and attribute bins multiply compartments like any dimension.
- **Results.**
  - A named clock is a dimension, so `res.n_on_art.by('art_time')` gives the distribution of time on ART.
  - A flow from a state with a derived dimension is recorded at the source's level, so `res.cd4_death.by('cd4')` gives deaths by CD4.
  - A per-person mean (mean CD4 on ART) is `ss.mean(hiv.cd4, where=hiv.on_art)`. This is a small result constructor, listed as an open question (§10).
- **Conditions.**
  - `hiv.on_art & (hiv.on_art.time >= ss.months(6))` (a viral load test at 6 months) is a lazy condition.
  - In a compartmental run the 6-month edge is added for you (§2).
- **Model spec.** The spec gets two new sections:
  - `clocks: {name: {start, within, bins}}`;
  - `attrs: {name: {value, on, combine, bins}}`.

  Values are expressions, serialized as rates are now.
- **Expressions.** `np.where` should be added to the NumPy functions allowed in expressions. Comparisons cover it, but they read worse (CT, §7.3).

## 7. Worked examples

Each "before" is quoted from the ports.

### 7.1 Syphilis: latent infectiousness decays with time since latent onset

Before (`syphilis/port.py:176-180, 227`): the code relies on `ti_EL` surviving EL→LL, and gives NaN for `init=dict(LL=...)` seeds.

```python
def latent_decay(sim):
    """ ... Relies on ti_EL being kept after people leave EL for LL ... """
    syph = sim.diseases.syph
    return np.exp(-np.log(2)/latent_hl*(sim.ti - syph.ti_EL.raw)*dt_yr)
...
connectors = ss.modify(syph.rel_trans, multiply=latent_decay, where=syph.EL | syph.LL, name='latent_decay'),
syph = Syphilis(name='syph', init_prev=0.02, init=dict(EL=0.01, LL=0.01))
```

After:

```python
latent = syph.EL | syph.LL
latent_decay = ss.modify(syph.rel_trans, multiply=0.5**(latent.time/ss.years(latent_hl)), where=latent)
syph = Syphilis(name='syph', init_prev=0.02, init=dict(EL=ss.spread(0.01), LL=ss.spread(0.01)))
```

Early versus late latent, measured from **infection** as the v3 comment intends: this uses `when=` from the branch-timing proposal.

```python
to_late = ss.progression('EL -> LL', when=syph.infected.time >= ss.years(2))
```

- As agents, exact.
- As an ODE:
  - `latent.time` is 12 levels (EL and LL × 12 = 24 groups). The decay is read at the middle of each level, with a note;
  - `infected.time` has an edge at 2 years, so the conveyor moves EL to LL at exactly 2 years, which is v3's deterministic relabel.
- Before, the ODE was blocked here (a function of `ti_EL.raw`); now it isn't.

### 7.2 BV: recurrence hazard decays with time since treatment

Before (`bv_gud_connectors/port.py:21-25, 57`):

```python
def recur_rate(sim, uids):
    """ Recurrence after treatment: a hazard that decays with time since treatment (agents only) """
    bv = sim.diseases.bv
    months = (sim.ti - bv.ti_P[uids]) * sim.t.dt.months
    return ss.permonth(bv.pars.recur0 * np.exp(-months / bv.pars.recur_tau))
...
recur = ss.progression('P -> CST4p | CST4a', rate=recur_rate, p=[by_stable(0.8, 0.7, 0.1), None])
```

After (class body; `P` is entered only through `tx`, so `P.time` is time since treatment):

```python
recur0, recur_tau = ss.par(ss.permonth(0.06)), ss.par(ss.months(3))
recur = ss.progression('P -> CST4p | CST4a', rate=recur0*np.exp(-P.time/recur_tau), p=[by_stable(0.8, 0.7, 0.1), None])
```

- `recur0` and `recur_tau` are now `ss.par`s, so they can be fitted.
- The `adjust=` of `stable`, `douching` and `hc` still multiply `recur`.
- As an ODE, `P` × 10 levels of about 1.4 months (the hazard is within 1% of its floor by about 14 months). The competing `settle` dwell (6 months, exponential) is one stage per level.
- Equivalent piecewise form, exact under every solver: `bv.stratify(since_tx=ss.clock(P.enter, within=P, bins=ss.months([0, 1, 3, 6])), adjust={'recur': [1, .6, .3, .1]})`.

### 7.3 Chlamydia: bacterial load from time since infection

Before (`stisim_bacterial/port.py:450-457`): `ti_E` is NaN for seeds in `A`, so the first version made every seed non-infectious.

```python
def rel_trans(sim):
    d = sim.diseases.ct
    u = sim.people.auids
    start = np.where(np.isnan(d.ti_E.raw[u]), d.ti_A.raw[u], d.ti_E.raw[u]) # Time of infection: there is none, ...
    t = np.nan_to_num(np.maximum(sim.ti - start, 0), nan=0.0) # Months
    load = np.where(t < 2, np.exp(np.log(1e8)*t/2), 1e8*np.exp(-np.log(2)/0.58*(t - 2)))
    return 2/(1 + np.exp(-0.5*np.log10(np.maximum(load, 1)))) - 1
load = ss.modify(ct.rel_trans, multiply=rel_trans, name='load')
```

After:

```python
t = ct.infected.time/ss.months(1)                                         # E|A|P|Y|C|N|PID|T; treatment failure T->A does not reset it
log10_load = np.where(t < 2, 4*t, 8 - np.log10(2)/0.58*(t - 2))           # 1e8 at 2 months, then a half-life of 0.58 months
load = ss.modify(ct.rel_trans, multiply=2/(1 + np.exp(-0.5*np.maximum(log10_load, 0))) - 1, where=ct.infected)
ct = make_sti('ct', ..., init=dict(A=ss.spread(0.01)))                    # seeds get a time since infection
```

- `where=ct.infected` is what the domain check (`E212`) wants. Without it the build raises an error, rather than CT silently dying out.
- As an ODE: 8 states × 10 levels of the infection clock, × sex × risk = 320 groups. The load falls to `rel_trans` of about 0 by about 8 months, so the levels are about 0.8 months wide (with a note).

### 7.4 HIV: CD4 trajectory, mortality by CD4, recovery on ART

Before (`hiv/port.py:114-166`, abridged): the hooks sit on both sides of `super()`, and `ti_exit` is NaN on entry.

```python
def step_state(self):
    ev = self.engine.events
    # Entry hook for infection: must run BEFORE super(), since super() evaluates cd4_death's hazard (which reads cd4)
    new_inf = ev.get('infection', ss.uids())
    if len(new_inf):
        self.cd4[new_inf] = self.cd4_start[new_inf]
        self.cd4_latent[new_inf] = self.cd4_latent_dist.rvs(new_inf)
    super().step_state()
    ...
    new_art = ev.get('start_art', ss.uids())        # forced by the ART delivery last step
    if len(new_art):
        self.cd4_preart[new_art] = np.maximum(self.cd4[new_art], 1)
        naive = new_art[~self.ever_art[new_art]]
        pre = self.cd4_preart[naive]
        self.cd4_potential[naive] = pre + pre*(cd4_max - pre)/cd4_healthy*np.log(cd4_max/pre)
    ...
    def frac(state, uids):  # share of the dwell in this state that has elapsed
        t0, t1 = self[f'ti_{state}'][uids], self[f'ti_exit_{state}'][uids]
        return np.clip((ti - t0)/np.maximum(t1 - t0, 1), 0, 1)
    a = self.acute.uids
    a = a[~np.isnan(self.ti_exit_acute[a])]  # GOTCHA: ti_exit_<state> is drawn at the start of the NEXT step, so is NaN on the step of entry
    self.cd4[a] = self.cd4_start[a] - (self.cd4_start[a] - self.cd4_latent[a])*frac('acute', a)
    ...
        self.cd4[o] = 2*gain/(1 + np.exp(-t_art*art_cd4_growth)) - gain + self.cd4_preart[o]
```

Plus `cd4_hazard(sim, uids)` with `np.digitize` (`:31-38`), a 12-line `DeathsByCD4` analyzer that reads `engine.events` (`:181-190`), and `art_rel_trans` that reads `ti_on_art.raw` (`:194-198`).

After (the class body; the transitions are unchanged from the port):

```python
class HIV(ss.Disease):
    S, acute, latent, falling, on_art, post_art = ss.states('S acute latent falling on_art post_art')
    infected = acute | latent | falling | on_art | post_art
    # ... infection, latency, progression, aids_death, start_art, dropout: as in the port ...

    cd4_start     = ss.attr(ss.normal(800, 50), on=infection)
    cd4_latent    = ss.attr(ss.normal(500, 50), on=infection, bins=3)
    cd4_preart    = ss.attr('cd4', on=start_art)                  # CD4 just before starting
    cd4_potential = ss.attr(cd4_preart*(1 + (cd4_max - cd4_preart)/cd4_healthy*np.log(cd4_max/cd4_preart)),
                            on=start_art, combine='first')        # v3: set at the first start only
    cd4_stop      = ss.attr('cd4', on=dropout)
    gain = cd4_potential - cd4_preart
    cd4 = ss.attr({
        acute:    cd4_start - (cd4_start - cd4_latent)*acute.progress,
        latent:   cd4_latent,
        falling:  np.maximum(1, cd4_latent*(1 - falling.progress)),
        on_art:   cd4_preart + gain*(2/(1 + np.exp(-art_cd4_growth*on_art.time/ss.months(1))) - 1),
        post_art: np.maximum(1, cd4_stop*(1 - post_art.progress)),
    })
    post_art_death = ss.Transition(post_art, ss.null, dur=np.maximum(np.log(cd4_preart)*cd4_stop/cd4_potential, 0.1)*ss.years(1), stages=10)
    cd4_death      = infected >> ss.null @ ss.by(cd4=ss.peryear([0.30, 0.05, 0.01, 0.005, 0.003]))

hiv = HIV(init_prev=ss.spread(0.052, over=(ss.months(5), ss.years(10))), name='hiv')       # §7.6
hiv.stratify(cd4=[0, 50, 200, 350, 500], states=hiv.infected)                               # derived: no compartments
mods = [
    ss.modify(hiv.cd4_death, multiply=rel_art_mort, where=hiv.on_art),
    ss.modify(hiv.cd4_death, multiply=rel_death_f, where=ppl.female),
    ss.modify(hiv.rel_trans, multiply=1 - eff_art*np.minimum(hiv.on_art.time/ss.years(time_to_eff), 1), where=hiv.on_art),
    ss.modify(hiv.rel_trans, multiply=rel_trans_aids, where=(hiv.cd4 < 200) & (hiv.falling | hiv.post_art)),
]
# Results, all free: r.cd4_death.by('cd4'), r.aids_death.by('cd4'), r.n_on_art.by('cd4'), r.n_on_art.by('on_art.time')
```

What goes:

- `step_state`, `init_post`, `cd4_hazard` and `art_rel_trans`;
- the `DeathsByCD4` analyzer;
- 9 `define_states` lines;
- every `ti_`/`ti_exit_`/`engine.events` read, and every NaN guard.

The CD4 testing eligibility in `make_testing` becomes `eligible=(hiv.cd4 < 200) & ~hiv.diagnosed`, a lazy condition.

**As compartments**, in groups per stratum:

| State | Groups | Count |
|---|---|---|
| acute | 3 stages × 3 `cd4_latent` bins | 9 |
| latent | 3 bins | 3 |
| falling | 9 stages × 3 bins | 27 |
| on_art | 4 dropout stages × 10 time levels × 5 `cd4_preart` bins (the cd4 edges) | 200 |
| post_art | 10 stages × 5 `cd4_stop` × 5 `cd4_preart` | 250 |

That is about 490 groups in all, or about 7,800 with 8 age groups × 2 sexes: under `W309`.

- **Exact for the groups:**
  - CD4 per group;
  - the mortality bin of each group (the derived dimension);
  - the time-on-ART ramp, at the step grid;
  - deaths by CD4.
- **Approximated (with notes):**
  - `cd4_start` replaced by its mean (it only shapes the 2-month acute decline; give it `bins=` to keep it);
  - `cd4_latent`, `cd4_preart` and `cd4_stop` at bin conditional means;
  - `.progress` from Erlang stages;
  - the ART time levels past 50 months held.
- **Still agents-only for HIV:** the ART stock target with priority, which is the programs proposal. It runs on groups only if priority is by strata, and the derived `cd4` dimension makes "lowest CD4 first" a priority by strata.

### 7.5 HPV: progression that depends on the dwell just completed; immunity set at clearance

Before (`hpvsim_disease/port.py:94-120`): zero-dwell junctions `Pe` and `Ce` (each one step late), dwell rebuilt from `ti_` arrays, and three unchecked `p` functions per split.

```python
def pcin(sim, uids):
    hpv = sim.diseases[g]
    dur = (hpv.ti_Pe[uids] - hpv.ti_P[uids])*sim.t.dt_year
    return p_cin_of(dur, rel_sev(sim, uids), gp['k'])
...
precin        = ss.Transition(P, Pe, dur=lognorm(*gp['dur_precin']), where=is_female),
to_cin        = ss.Transition(Pe, C, p=pcin),
clear_precin  = ss.Transition(Pe, S, p=lambda sim, uids: (1 - pcin(sim, uids))*(1 - ctrl)),
latent_precin = ss.Transition(Pe, L, p=lambda sim, uids: (1 - pcin(sim, uids))*ctrl),
```

The immunity half is a 75-line `CrossImmunity` connector (`:133-206`) reading `hpv.engine.events['clear_precin']`.

After, with `decide='exit'` from the branch-timing proposal. At exit, `P.time` *is* the dwell just completed, so no `dwell` argument is needed:

```python
rel_sev = ss.attr(np.abs(ss.normal(1, 0.2)))                         # trait; an ss.modify target (HIV: x1.5 where CD4 < 200)
p_cin   = logf2(rel_sev*P.time/ss.years(1), gp['k'])                 # logf2 is pure NumPy, so this is an expression
p_canc  = 1 - (1 - gp['transform_prob'])**intlogf2(rel_sev*C.time/ss.years(1), gp['k'])**2
precin  = ss.progression('P -> C | S | L', dur=lognorm(*gp['dur_precin']), decide='exit', where=ppl.female,
                         p=[p_cin, (1 - p_cin)*(1 - ctrl), None])
cin     = ss.progression('C -> X | S | L', dur=lognorm(*gp['dur_cin']), decide='exit',
                         p=[p_canc, (1 - p_canc)*(1 - ctrl), None])
sev     = ss.attr({C: logf2(rel_sev*C.time/ss.years(1), gp['k'])}) # dysplasia severity, e.g. for test sensitivity by severity

nab = ss.attr(ss.uniform(0.5, 0.95)*ss.bernoulli(gp['sero_prob']), on=precin.to(S, L) | cin.to(S, L), combine='max', init=0)
```

Then the cross-genotype layer, at sim level (it needs bug §1.4 fixed: lazy conditions on another module's array keep their module):

```python
for i, h in enumerate(hpvs):
    ss.modify(h.rel_sus, multiply=1 - np.clip(sum(m_sus[i, j]*hpvs[j].nab for j in range(len(hpvs))), 0, 1))
```

`CrossImmunity` goes from 75 lines to 2. v3 gates seroconversion only at the first clearance. The draw above gates every clearance, and `combine='max'` makes the difference small. The exact form is `combine=fn(old, new)`.

- **As agents:**
  - the junctions and their 0.5-year bias on the cancer path go;
  - `n_Pe` and `n_Ce` go;
  - the `p` list is one object, so `E205` can check it when it is constant.
- **As compartments:**
  - `P.time` read at exit needs `P` × clock levels, with branch flows per level. Alternatively, the branch-timing proposal could convert an exit decision that depends only on the dwell into an entry decision with conditional dwells. The branch shares are `E[p(D)]` by quadrature, and each branch's dwell is fitted by Erlang to `D | branch`. That is exact in the shares, matches the mean and std of each conditional dwell, and adds no clock levels.
  - `nab` across 4 genotypes is 5⁴ bins per state. `describe()` says so, `W309` fires, and HPV stays an agent-based model in practice, honestly reported.

### 7.6 HIV: seeds that start mid-dwell

Before (`hiv/port.py:285, 91-101`): every seed starts at the beginning of its dwell (no progressions for 4 years), and CD4 is faked.

```python
hiv = HIV(init=dict(acute=0.002, latent=0.04, falling=0.01), name='hiv')
...
def init_post(self):
    super().init_post()
    # Seeded people start mid-course (v3 back-dates infections 5 mo to 10 y): give them a CD4 to match their stage
    for state in [self.acute, self.latent, self.falling]:
        uids = state.uids
        self.cd4_latent[uids] = self.cd4_latent_dist.rvs(uids)
        self.cd4[uids] = self.cd4_latent[uids]
    f = self.falling.uids
    self.cd4[f] = self.cd4_latent[f]*self.cd4_falling_dist.rvs(f)
```

After:

```python
hiv = HIV(init_prev=ss.spread(0.052, over=(ss.months(5), ss.years(10))), name='hiv')
```

- The stage mix, the time in stage (and so the progressions in the first years), `cd4_start`, `cd4_latent` and the trajectory all come out of the declared model. Nothing is hand-written.
- As an ODE, the 10-year linear solve fills the acute, latent and falling stages and bins.
- The `init_post` override goes.

## 8. Coordination with the sibling proposals

- **Branch timing.**
  - `decide='exit'` reads `X.time` (and any attribute) at exit, so it needs no `dwell` argument.
  - `when=cond` with a clock condition is a transition at a fixed elapsed time (exact with the conveyor). This covers questions-oct12 #7 (odin's fixed 14-day delay: `E >> I` with `when=E.time >= ss.days(14)`, a conveyor of 14 one-day levels, exact, linear cost) and #11 (EMULSION minimum stays: `where=A.time >= d`).
  - Scheduling at entry (§4) is shared.
- **Programs.**
  - `delivery.since` is `ss.clock(start=delivery)`, and the count received is `ss.attr(1, on=delivery, combine='sum', init=0)`. The programs proposal decides the names and the "once per episode" rule. This one provides the mechanism.
  - In a compartmental run, a clock over all people costs one dimension. Its levels come from the comparisons it is used in (`screen.since > ss.years(5)`: 20 levels at quarterly steps).
  - A priority by a derived attribute (`priority=hiv.cd4`) runs on groups.
- **Births bug (REVIEW §1.1).** Newborns must fire `S.enter` (and enter clocks within `S`). The fix for placing newborns should go through the same move path as everything else.

## 9. Rejected

| Alternative | Why not |
|---|---|
| Document `ti_<state>`, `ti_exit_<state>` and `engine.events` as public | They expose implementation details: NaN semantics, lazy scheduling, and no way to express time since entering a set. They have no compartmental reading, so the ODE stays blocked |
| A `dwell` argument to `p=` (`p=fn(sim, uids, dwell)`) | `X.time` read at exit says the same thing and also works in rates, modifiers and results |
| Special keywords (`ti_infected`, `rel_trans_decay=`, `half_life=`, `dur_since_infection=`) | One per model. A clock covers them all |
| Per-agent within-host ODEs (`d cd4/dt = ...`) | Every port's trajectory is closed-form in elapsed time and values at events. Integrating per agent brings ordering and step-size problems back, and the only compartmental form is binning with transfers. Keep `step()` for this; revisit if a model needs it |
| CD4 as hand-written band states (the HPV port's `A, B, D`) | Loses continuous CD4 for priorities, thresholds and test sensitivity. It *is* the compartmental realization, but generated, not written |
| `on_enter=` and `on_exit=` keywords on states | One `on=` everywhere (attributes, functions, deliveries) and `X.enter`/`X.exit` as events do the same with one concept |
| Erlang aging (rate `1/w`) as the default clock realization | Smooth and JAX-friendly, but elapsed time at level `j` spreads as Erlang(`j`), so `f(time)` blurs. The conveyor is exact on the grid and already implemented for `waning=`. It is kept as an option (§10) |
| `ss.spread` meaning only stationary within the state (questions-oct12 #8) | It doesn't set clocks or attributes across earlier states (time since infection, CD4 for seeds in `falling`). Back-dating with the natural history covers both |
| Clocks that are always defined (0 outside the domain) | They hide exactly the bug the ports hit. NaN plus a check at build (`E212`) is safer |

## 10. Open questions

1. **Conveyor or Erlang aging for clocks** under `ode`, `sde` and JAX? The conveyor makes the run stepped at loop times. That is linear, so differentiable, but it is not yet in the whole-run kernel. A possible spelling: `ss.clock(..., realize='conveyor' | 'erlang')`.
2. **Reuse Erlang stages as a clock.** When a state's dwell is already staged, `X.time` could read stage × `mean/k` instead of adding levels: cheaper, but it ignores the variance within a stage. Should that be the default, or an option?
3. **Default levels.** 10 levels, as products use, or a tolerance (`clock_tol=0.01`) that sets the number?
4. **`combine=` vocabulary.** Is `'first' | 'last' | 'max' | 'min' | 'sum' | fn` enough? Does `fn(old, new)` need draws inside it (HPV's seroconversion gate)?
5. **Names.** `ss.attr` versus `ss.value` or `ss.trait`; `.progress` versus `.elapsed`; `tr.to(...)` for a branch event.
6. **`ss.spread` and age.** Should back-dating age agents back (`age - elapsed`) for age-dependent rates during the backfill? The current answer is no, with a note.
7. **Per-person means as results** (`ss.mean(hiv.cd4, where=on_art)`). Is this a result constructor, or left to an analyzer? The ODE form is the group-weighted mean, which is exact for the groups.
8. **Events per person** (`tr.count`), as agents is cheap; in a compartmental run it is a capped dimension (0, 1, 2+). Is it needed beyond deliveries?
9. **Order of attributes on one event.** Declaration order is proposed. Should a cycle (an attribute that records itself) be an error at build, or allowed (`combine=` reads the old value)?
10. **Derived dimensions under the stochastic solvers.** A group's level can change when a clock moves it up. Are flows recorded at the level at the start of the step (proposed), or interpolated?

## 11. Implementation notes (pointers, not a plan)

- **Clocks:**
  - for a single state, `ti_<state>` already exists (`solvers.py:3122`);
  - for a set, add `ti_<set>`, written in `finish_moves` (`:3425`) when the source is outside the set;
  - for the compartmental side, reuse `Held.init_stages` / `step_stages` (`products.py:312`, `interventions.py:913`) as the conveyor, and `enter_dims` (`:3470`) for level 0 on entry.
- **Scheduling at entry:** call `schedule()` (`:3489`) for the people who just entered, from `finish_moves`, `force` (`:3230`) and `init_post`. Remove the NaN-guard path.
- **Events:** `engine.events` (`:3115`) becomes the store behind `tr.uids`, `X.enter` and `X.exit`. The forced-move lag (`self.forced`) stays only for `on=` deliveries' existing visibility rule; attributes and hooks fire at the move.
- **Attributes:** a declared `ss.attr` builds a `FloatArr` (drawn or recorded) or a cached lazy expression (derived). Derived dimensions reuse the per-step recomputation path used for `age`.
- **Checks:** `E212` is a set inclusion check on graph states at build; `E213` is a check on the dwells of the graph.
