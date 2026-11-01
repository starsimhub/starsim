# EMULSION

## What the source is

An EMULSION model is one YAML file: `time_info` (unit, `delta_t`, `total_duration`, calendars), `levels` (with `aggregation_type`: `compartment`, `IBM`, or `hybrid`), `state_machines` (each a list of states and of transitions with `rate`, `proba`, or `amount-all-but`, plus `cond`, `when`, `escape`, `on_cross`), `parameters` (numbers, or expressions using `total_X` and `total_population`), `prototypes` and `initial_conditions`, `statevars`, and `outputs`. It is run with `emulsion run model.yaml`.

## Mapping

| EMULSION | Starsim | Mechanical |
|---|---|---|
| A state machine | A disease (several machines are several diseases in one sim) | yes |
| `rate: expr` | The rate per person, classified (`beta * total_I / total_population` is exposure, `beta * I`) | yes |
| `proba: p` | `-log(1 - p)`, the rate that gives it: `p` is per time unit, not per step (EMULSION makes it `1 - (1 - p)**delta_t` for a step) | yes |
| `duration:` on a state whose only way out is `proba: 1` or `amount-all-but: 0` | A fixed dwell, `ss.constant(ss.days(d))` | yes |
| `duration:` on a state with other ways out | A minimum stay: `after=` on the ways out (and nothing on an escape), as agents or compartments | yes |
| `autoremove: yes` (a dead state) | An outflow to `ss.null` | yes |
| `escape: yes` | Nothing: once the minimum stay is not translated, every way out applies throughout | yes |
| `escape: cond`, `cond: is_X` | `where=` on the transition, against the other machine's state | no |
| `when: calendar_period` | A parameter that is an `ss.timeseries` of 0 and 1, multiplying the rate | no |
| `StrictLessThan(total_Q, K)` and the like | A comparison of counts: `(Q.n < K)` | yes |
| `on_cross`, `produce_offspring`, `set_var` | Offspring into one state: a birth driven by the transition, `ss.birth(NB, per=farrowing, n=10)`; anything else, custom code on the transition's events (agents) | no |
| `statevars` (age, counters) | A stratification, a property of people, or a state array in a subclass | no |
| `prototypes`, `initial_conditions` | `init_n=dict(...)` counts on each disease | yes |
| `outputs.extra_vars` | Results on the disease, where they are expressions of counts | yes |
| `aggregation_type` | `solver='binomial'` for `compartment`, `'abm'` for `IBM` and `hybrid` | yes |
| `productions: {from: A, to: J, rate: b}` (births) | An inflow, `ss.null >> J @ b*A.n`; with several machines, cross them into one disease first, so the newborns enter each machine's state together | no |
| A level with `aggregation_type: metapopulation` (herds) | A place dimension, `dis.stratify(pop=[...], kind='space')`, with the initial numbers by place (`ss.by(pop=...)`) and movement between places as a route | no |
| A level or process in a Python add-on (`file:`, `class_name:`) | Custom code, or a declared form if one fits (e.g. a trade register as a route) | no |
| `total_<level>` (e.g. `total_herd`) | The number of people (in each place, once stratified) | yes |

## What needs care

- EMULSION applies the processes of a step in the order of `processes:`; Starsim takes each disease's transitions together. With several machines, check whether the order matters (it did for the quickstart's gestation and health machines, where the vignette runs gestation first).
- A duration is rounded to whole steps by EMULSION. With `delta_t: 7` a 10-day stay is one step, not 10/7.
- Animals in the initial conditions start in their state before the first step, so their time in it counts from step -1, not 0 (the vignette sets `ti_NG = -1`).
- `adult_age` and other parameters can be defined twice in a file; YAML (and so EMULSION) keeps the last one, and so does the import.
- `random`/`randint` in a parameter (initial ages) are not numbers; the import leaves them out and lists anything that uses them. Give ages with `ss.People(n, age_data=ss.uniform(...))`.

## Worked example

`lingua_franca_design/vignettes/emulsion/quickstart/model.py`: three state machines (age group, gestation, health), with the minimum stays, conditions on another machine, the quarantine calendar and capacity, and births on farrowing. The import (`ss.import_from('quickstart.yaml')`) gives the health machine's 13 transitions directly, and lists the 16 pieces that the vignette writes by hand:

```python
dates = {'2021-01-01': 0, '2021-04-01': 1, '2021-09-02': 0, '2022-04-01': 1, '2022-09-02': 0}
quarantine = ss.par(ss.timeseries(dates))                     # when: quarantine_period
health = ss.Disease(
    infection = S >> E @ ss.perday(0.1)*I.n/(S + E + I + R + Q + M).n,
    onset     = ss.Transition(E, I, rate=stay('health', 'E', 10, 0.3)),   # duration: incubation_duration
    detection = I >> Q @ ss.perday(0.01)*quarantine*(Q.n < 20),           # the calendar and the capacity
    release   = Q >> R @ ss.constant(ss.days(30)),
    ...
)
life = Life(
    conception = ss.Transition(NG, G, rate=stay('life', 'NG', 90, 0.05), where=lambda sim: sim.diseases.growth.A),
    farrowing  = G >> NG @ ss.constant(ss.days(240)),
    abortion   = ss.Transition(G, NG, rate=ss.perday(abortion), where=lambda sim: sim.diseases.health.I),
)
```

## Running the original

`emulsion run model.yaml --runs 200 --seed 1 --quiet --output-dir out` writes `out/counts.csv`, one row per run (`simu_id`) and step (`step`), with a column for each state and each extra output. For the SIR, the vignette's `compare.py` found that a stock at index `k` is the number on day `k` in both, so the rows line up as they are; check this for each model, since it depends on how the steps are labelled. EMULSION's compartments and individuals are stochastic: compare the means and spreads over many runs. See `lingua_franca_design/examples/emulsion/sir/run.py`, which runs it in its environment (`_envs/emulsion`) and writes the long CSV that `validate-import` reads.
