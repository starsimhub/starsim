# The reduced compartmental support matrix (S3.1, draft for Cliff)

Written 2026-10-20 for [simplify_plan.md](../simplify_plan.md) S3.1. Each row is a declared feature; "Proposed" is whether it keeps a compartmental form. **Keep** means it runs under every solver and stays tested there. **Agents** means `auto` picks agents and an explicit compartmental solver raises one `CapabilityError` before the run. **Explicit** means it stays, but only when the user names a compartmental solver, never under `auto`, with a note. Once Cliff has settled this table, it goes into spec 12, replacing "What makes a model agent-based" and "Everything else has a compartmental form", and S3.2–S3.6 implement it.

"Whole run" is whether the kernels can take it from start to finish in one call (the fast path for ensembles and fits); "JAX" is whether `sim.gradient()` covers it. Neither is promised for every Keep row: a Keep row may run stepwise from Python.

## The core (keep)

| Feature | Today | Proposed | Whole run | JAX | Representative test |
|---|---|---|---|---|---|
| States, `ss.Transition` and operators, `ss.par`, `ss.null` | yes | Keep | yes | yes | `test_contracts.py::test_solvers_agree` |
| Branches as a list with `p=`, `ss.stay` | yes | Keep | yes | yes | `test_transitions.py::test_cascade_binomial` |
| Exponential and Erlang (`stages=`) dwells | yes | Keep | yes | yes | `test_contracts.py::test_dwell_forms` |
| Fixed delays (`ss.constant`), `after=`/`before=` as a conveyor | yes | Keep | yes | yes | `test_conveyor.py` |
| Other dwell distributions, as Erlang stages with a note | approx | Keep | yes | yes | `test_contracts.py::test_dwell_forms` |
| Homogeneous mixing, `ss.contacts()` matrices, mixing pools of everyone | yes | Keep | yes | yes | `test_mixingpools.py` |
| Explicit categorical strata (`stratify`), `ss.by` | yes | Keep | yes | yes | `test_dimensions.py` |
| `ss.transfer` (aging, counters, Erlang stages) | yes | Keep | yes | yes | `test_transitions.py::test_transfer` |
| Places with `ss.geo`, `ss.gravity`, commuting and migration | yes | Keep | yes | yes | `test_dimensions.py::test_space`, `test_commuting` |
| `ss.timeseries`, `ss.seasonal`, life tables, `ss.birth`/`ss.death` of the population | yes | Keep | yes | yes | `test_demographics.py` |
| `ss.outcome` (with `delay=`, `duration=`) | yes | Keep | yes | yes | `test_transitions.py::test_outcomes` |
| `ss.modify`/`ss.change` with conditions on strata | yes | Keep | yes | yes | `test_connectors.py` |
| Triggers within a module (`ss.on` of its own transition) | yes | Keep | stepwise | no | `test_triggers.py` |
| Deliveries by state and group on a schedule, with coverage and a capacity cap | yes | Keep | yes (tabulated) | partly | `test_deliveries.py` |
| Tests with immediate results; one-disease cures | yes | Keep | yes | partly | `test_products.py` |
| Exponential waning of a product | yes | Keep | yes | partly | `test_products.py` |
| Diseases tracked jointly (crossing), within a cell budget | yes | Keep, simplified (S4) | yes | yes | `test_treatment.py::test_syndromic` |

## Agents only (to change)

| Feature | Today | Proposed | Why |
|---|---|---|---|
| Fetal levels, `ss.mother()` in compartments, prenatal transmission as a fetal transition | yes (B7, Oct 20) | Agents | Hidden relationship expansion; the engine reads pregnancy internals by name (S3.2) |
| `ss.Lactation` as synchronized flows, `BreastfeedingNet` | partly | Agents | Needs mother–child links; the B7 attempt did not work (S3.2) |
| `ss.Pregnancy` as compartments | yes (B4/B5) | Explicit, in a small form (Cliff, 2026-10-20): its own states with Erlang or fixed gestation and births by its delivery flow, with assumptions printed; `auto` picks agents | Cliff wants an ODE approximation |
| `ss.birth(per=)` (births driven by a flow) | yes (B4) | Keep if small once fetal routing is gone, else Agents | Needed by the small pregnancy form |
| A rate or `p=` reading `X.time`/`X.dwell` (time in state) | yes (B1) | Agents | A conveyor of up to `npts` slots from 20,000 quantiles, with timing conventions that changed twice (S3.3) |
| Clocks as strata (`ss.clock`), set clocks `(A \| B).time` | partly | Agents | Same (S3.3) |
| `ss.attr` as levels (drawn, derived, recorded) | partly (B6) | Agents | 100,000 samples to make bins; the state space multiplies (S3.3) |
| `.since`/`.count` memory of an event | yes | Agents | Individual memory as hidden levels (S3.4) |
| `.ever` memory | yes | Keep (Cliff, 2026-10-20) | The simplest memory: one level |
| Queues (`queue=`), delayed test results, `delay=` as levels | yes | Agents | Identity of a selected fraction across steps (S3.4) |
| Waning given as a function (sampled at 20,001 points) | yes | Agents | (S3.4) |
| Deliveries to several diseases (`disease=[...]`), outside a crossing | yes | Agents | Shares assumed independent (S3.4) |
| Events combined as shares (`a \| b` of events) from an inferred lineage | yes | Agents (Cliff, 2026-10-20) | The lineage is inferred; its users are the rows above |
| Lasting partnerships (`MFNet`, `PartnerNet` with `dur=`) | approx, explicit only | Explicit, pending the research write-up (S3.7) | Cliff wants ODE HIV; study Optima HIV first |
| A random or static network, as its mean contacts | approx, explicit only | Explicit (as now) | Already never chosen by `auto` |
| `ss.update` in the kernels | yes | Agents (or a coded module) | (S3.6) |
| `ss.Strains` members side by side, coupled by shares | approx | Agents for coupled members, whatever `approx=` (done 2026-10-20); then S7.4 | (S3.6) |
| `ss.by` over mismatched bins, averaged | approx | Error | Silent averaging (S3.6) |
| Contact tracing, transmission trees, the infection log, partial immunity across strain levels | no | Agents (as now) | No aggregate form |
| A function of uids, a per-agent attribute that is not a stratum, a distribution across people not binned | no / mean | Agents (as now) | |

## Questions for Cliff

1. Is the core list right? In particular: triggers within a module, deliveries with capacity, and exponential waning stay in the core; the pregnancy form is decided with decision 7.
2. `.ever` memory and events as shares: keep in compartments, or agents only?
3. Should the Agents rows become refusals one at a time (S3.2 to S3.6, each with its tests), or in one sweep? (The plan prefers one sweep, since they share refusals and kernel fields.)
