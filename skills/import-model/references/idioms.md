# Starsim forms for the pieces importers leave out

What to write for the pieces that `ss.load()` lists as not translated (`W1601`), and for the same constructs when translating by hand. Prefer the first form that says what the source says; use a function or custom code only when no declared form does, and mark it with a `Workaround:` comment.

## Rates and times

| In the source | In Starsim |
|---|---|
| A rate per person | `S >> I @ beta * I` (another state over the total is exposure), `I >> R @ gamma` |
| A total flow (`gamma * I` out of `I`) | Divide by the source: `I >> R @ gamma` |
| Mass action without the total (`beta * S * I`) | Counts: `S >> I @ beta * I.n` |
| A probability per step | The rate that gives it: `E >> I @ -np.log(1 - p)/dt` (with `dt` in the sim's unit) |
| A probability per unit of time | `ss.probperday(p)`, `ss.probperyear(p)` |
| A mean duration | `I >> R @ ss.days(10)` (exponential) |
| A fixed duration | `Q >> R @ ss.constant(ss.days(30))` |
| An Erlang or gamma duration, or stages | `ss.progression('E -> I', dur=ss.erlang(k=3, mean=ss.days(10)))`, or `stages=3` on the transition |
| A lognormal or other distribution | `dur=ss.lognorm_ex(mean=ss.days(10), std=ss.days(3))` (agents sample it; compartments use Erlang stages, with a note) |
| "As soon as", probability 1 at the next step | A very high rate (`ss.perday(1e9)`), or a branch with `p=` on the transition into the state if it happens on entry |
| A delay that is exact (odin's `delay()`, a lag) | `ss.outcome(source, delay=ss.days(14))` for a reported quantity; for people, a fixed dwell (`ss.constant`) |
| Rates that change in time | `ss.timeseries({'2020-01-01': 1, '2020-04-01': 0.5})` as the value of a parameter, `ss.seasonal(mean=, amplitude=, peak=)`, or `ss.change(sir.pars.beta, at=[...], by=[...])` |
| A calendar period (on/off) | A parameter that is an `ss.timeseries` of 0 and 1, multiplying the rate (`rate * quarantine_period`); or `ss.modify(..., start=, stop=)` |

## Who it applies to

| In the source | In Starsim |
|---|---|
| A condition on another state machine or disease (`cond: is_A`, `escape: is_I`) | `where=` on the transition: `ss.Transition(NG, G, rate=..., where=lambda sim: sim.diseases.growth.A)` |
| A condition on age, sex, or another property | `where=ppl.age >= 15` (with `ppl = ss.People(n)` in scope); for a compartmental run, a bin edge at 15 in the age stratification |
| A capacity (fewer than K in a state) | A comparison of counts in the rate: `I >> Q @ 0.01 * (Q.n < K)` |
| A minimum stay before the other ways out apply | A function of the time in the state (agents only): see below |
| Groups (age, place, risk) | `dis.stratify(age=[0, 15, 65], split=..., adjust={'infection': [...]})`; places with `ss.geo(...)` and `ss.Mobility(...)` |

A minimum stay, as in EMULSION's `duration` on a state with other ways out:

```python
def stay(mod, state, dur, rate, after=True):
    """ Workaround: a rate that applies only once an agent has been in the state for dur (or, with after=False, only until then) """
    def func(sim, uids):
        stayed = sim.ti - sim.diseases[mod][f'ti_{state}'][uids] >= round(dur/dt)
        return (rate(sim, uids) if callable(rate) else rate)*(stayed == after)
    return func

onset = ss.Transition(E, I, rate=stay('health', 'E', 10, 0.3))
```

## People coming and going

| In the source | In Starsim |
|---|---|
| Births into a state | `ss.birth('-> S', rate=mu)` (per person, by the population), or `ss.null >> S @ mu * (S + I + R).n` |
| Deaths from some states, or from all | `(S \| R) >> ss.null @ mu`, `ss.death(rate=mu)` (from every state) |
| An absorbing dead state that is removed (`autoremove`) | An outflow to `ss.null` |
| Immigration and emigration | `ss.immigration('-> S', rate=)`, `ss.emigration(rate=)` |
| Aging between age groups | `use_aging=True` on the sim with an age stratification; or `ss.transfer` between levels |
| Offspring produced on a transition (EMULSION's `produce_offspring`) | Custom code on the transition's events (agents): see below |
| Scheduled moves, entries, and exits (SimInf's events) | `ss.Mobility(data=df)` with columns for the time, `from`, `to`, the number, and the states chosen from |

Actions on crossing a transition, such as offspring or a counter, as agents:

```python
class Life(ss.Disease):
    def step_state(self):
        super().step_state()
        mothers = self.engine.events.get('farrowing', ss.uids()) # Who crossed the transition this step
        if len(mothers):
            new = self.sim.people.grow(len(mothers))
            self.sim.people.age[new] = 0
            self.NG[new] = True
            self.ti_NG[new] = self.ti
```

## Variables of individuals

A numeric variable that only gates transitions (age, a counter) is either a stratification (for a compartmental run) or a state array declared in a subclass (`self.define_states(ss.FloatArr('nb_abortions', default=0.0))`), updated in `step_state()`. A variable that is a property of people (age, sex) belongs on `ss.People`.

## Contacts

| In the source | In Starsim |
|---|---|
| Homogeneous mixing | Nothing: the default; a bare state in a rate is `I.n / N` |
| A contact matrix by age | `networks=ss.contacts(matrix, age=[...])`, with the disease stratified by the same bins (or a coarser set; `W404` says how it was aggregated) |
| A matrix for each of several dimensions | `ss.contacts(age=M1, location=M2)` |
| Coupling between places | `ss.Mobility(matrix=M)` (commuting), `ss.Mobility(kind='force')` (the force of infection moves), `ss.gravity(k=, a=, b=, c=)` |
| A dynamic partnership network (ERGM-style targets) | `ss.PartnerNet(n_contacts=, concurrency=, dur=, match=dict(risk=0.9))`, `ss.MFNet(...)` for men and women |
| A random network | `ss.RandomNet(n_contacts=)`, `ss.RandomExactNet` |

## Interventions

| In the source | In Starsim |
|---|---|
| A change of a parameter at given times | `ss.change(sir.pars.beta, at=['2020-03-15', '2020-06-01'], by=[0.5, 1.0])` |
| A change for some people or under a condition | `ss.modify(sir.rel_sus, multiply=0.6, where=..., start=, stop=, when=)` |
| Vaccination as a move between strata | `sir.stratify(vaccinated=[False, True], adjust={'infection': [1, 0.15]})` and `ss.transfer(vaccinated=True, coverage=, years=)` |
| A product delivered to people | `ss.vaccinate(ss.Vaccine(efficacy=0.85), coverage=0.7, years=[...], eligible=...)`, `ss.screen(...)`, `ss.treat(...)` |
| A trigger (start when prevalence passes a threshold) | `when=lambda sim: ...` on a modifier or delivery (no memory); a rule with memory is a delivery's `deliver=` function |
| Scenarios | One sim per scenario, run together with `ss.parallel(baseline=sim, vaccine=sim2)` |

## Outputs

| In the source | In Starsim |
|---|---|
| The number in each state | Already a result: `sim.results.<disease>.n_<state>` (the number after each step) |
| New entries into a state, per step | Already a result: the transition's name (`res.infection`), and `cum_<name>` |
| An expression of counts (prevalence, a percentage) | A result on the disease: `prev = I / (S + I + R)`, `pct = 100 * I.frac` |
| A reported quantity with a delay or fraction | `cases = ss.outcome(0.3 * infection, delay=ss.days(5), every=ss.weeks(1))` |
| Data with a likelihood | `ss.outcome(0.6 * infection, every=ss.weeks(1), likelihood=ss.nbinom(n=k))` (or `ss.poisson()`), then `sim.calibrate()` with the data |

When comparing with the source, check what time each value is for: Starsim's `n_X[ti]` is the number at time point `ti` (`n_X[0]` the start) and a flow at `ti` the number in the step that ends there; a source that labels steps by their start needs shifting by one step (see the validate-import skill).
