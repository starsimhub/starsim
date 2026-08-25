# Starsim

|  |  |
|---|---|
| **Language** | Python (with Numba kernels and an optional Rust backend); R bindings via `rstarsim` |
| **Paradigms** | Agent-based (primary); network; metapopulation and mixing-pool; compartmental / ODE-style |
| **Specification style** | Class-based Python API, with declarative `define_pars()` / `define_states()` / `define_results()` inside imperative `step()` methods |
| **Version reviewed** | 3.6.0 (2026-08-22) |
| **Licence** | MIT |
| **Code** | <https://github.com/starsimhub/starsim> |
| **Docs** | <https://docs.starsim.org> |
| **Paper** | Kerr et al., Starsim methods paper; predecessor Covasim in *PLOS Comp Biol* [10.1371/journal.pcbi.1009149](https://doi.org/10.1371/journal.pcbi.1009149) |
| **Ecosystem** | ~12 first-party disease models (STIsim, HPVsim, TBsim, FPsim, RSVsim, Rotasim, …) all subclassing `ss.Module` and running in the same `ss.Sim` loop |

*Conflict of interest note: this project is Starsim's design successor and is authored by Starsim's maintainers. The review below is written to be usable as evidence, which means the limitations section is the part that matters most.*

## What it's for

Starsim simulates multiple co-circulating diseases and health states in one agent population on dynamic contact networks. The organizing claim is **modularity across health conditions**: diseases, demographics, networks, interventions, connectors, and analyzers are all the same kind of object (`ss.Module`), they are all added to the same `ss.Sim`, and they interact through shared per-agent state rather than through a fixed disease-model interface. Co-infection, pregnancy affecting HIV progression, malnutrition affecting measles severity, and family planning affecting the population pyramid are all expressible in the same vocabulary.

It generalizes a decade of single-disease Python models (Covasim, HPVsim, FPsim) into one framework, and the disease-specific packages are now thin layers on top of it rather than forks.

## How a model is specified

A model is a `Sim` plus a set of modules. At the simplest, everything is a dict:

```python
import starsim as ss

pars = dict(
    n_agents  = 5_000,
    networks  = dict(type='random', n_contacts=10),
    diseases  = dict(type='sir', init_prev=0.01, beta=0.05),
)
sim = ss.Sim(pars).run()
sim.plot()
```

`networks`, `diseases`, `demographics`, `interventions`, `connectors`, and `analyzers` each accept a string, a dict with a `type` key, a module instance, or a list of any of those. The dict form is a small, JSON-expressible surface — it is the closest thing Starsim has to a declarative model file, and it covers a large fraction of routine use.

Anything beyond that is a Python class. A new disease subclasses `ss.Disease` (or `ss.Infection`, which adds transmission) and uses three declaration calls in `__init__`, then implements one or more step methods:

```python
class SIR(ss.Infection):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            beta      = ss.peryear(0.1),
            init_prev = ss.bernoulli(p=0.01),
            dur_inf   = ss.lognorm_ex(mean=ss.years(6)),
            p_death   = ss.bernoulli(p=0.01),
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.BoolState('susceptible', default=True, label='Susceptible'),
            ss.BoolState('infected', label='Infectious'),
            ss.BoolState('recovered', label='Recovered'),
            ss.FloatArr('ti_infected', label='Time of infection'),
            ss.FloatArr('ti_recovered', label='Time of recovery'),
            ss.FloatArr('rel_sus', default=1.0),
            ss.FloatArr('rel_trans', default=1.0),
        )

    def step_state(self):
        recovered = (self.infected & (self.ti_recovered <= self.ti)).uids
        self.infected[recovered]  = False
        self.recovered[recovered] = True
```

The three `define_*` calls are the interesting part of the design. They are **declarations with types**: parameters carry time units (`ss.peryear`, `ss.years`) or distributions (`ss.bernoulli`, `ss.lognorm_ex`); states carry a dtype, a default, and a label; results carry a shape, a timevec, and a module. Nothing springs into existence on first use — `self.foo = 3` on a module raises unless `foo` was declared, and `define_states` locks the state list. That is a meaningful step beyond EpiModel's open `...` parameter lists, and it is the mechanism that makes `sim.to_json()`, automatic result generation, and mock initialization possible.

The step methods are ordinary imperative Python operating on vectorized agent arrays.

## Core abstractions

| Concept | Starsim name | Notes |
|---|---|---|
| Simulation | `ss.Sim` | Owns `People`, `pars`, `Timeline`, `Loop`, `Results` |
| Everything else | `ss.Module` | Diseases, networks, demographics, interventions, connectors, analyzers are all subclasses with the same lifecycle |
| Population | `ss.People` | Agents identified by persistent `uid`; `auids` are the currently-alive subset |
| State | `ss.BoolState`, `ss.FloatArr`, `ss.State` | Typed, labelled, per-agent arrays owned by the module that declares them. Boolean states automatically generate an `n_<state>` result |
| Transition | imperative code in `step_state()` / `set_prognoses()`, usually via scheduled `ti_*` times | Not a data object |
| Transmission | `ss.Route` — `ss.Network` (edge-based) or `ss.MixingPool` / `ss.MixingPools` (aggregate) | The same `infect()` machinery handles both, so switching from an ABM to a metapopulation formulation is a route swap, not a model rewrite |
| Contact | edge in `net.edges` with `p1`, `p2`, `beta`, and optional `dur` | Bidirectional by default with per-direction beta |
| Time | `ss.date`, `ss.dur` (`days`/`weeks`/`months`/`years`/`datedur`), `ss.Rate` (`per`, `prob`, `freq`) | Typed and unit-aware; see below |
| Timeline | `ss.Timeline` | Per-sim *and* per-module; modules may run at different `dt` |
| Parameter | entry in `module.pars`, typed | May be a scalar, a `TimePar`, a `Dist`, or a callable of `(module, sim, uids)` |
| Randomness | `ss.Dist` | One RNG per distribution, seeded by hashing its name; see below |
| Intervention | `ss.Intervention` subclass, with `RoutineDelivery` / `CampaignDelivery` mixins | A partial grammar, see below |
| Cross-module coupling | `ss.Connector`, or direct modification of another module's states | Deliberately unstructured |
| Output | `ss.Result` (a labelled, unit-aware array) in `ss.Results` | Declared, not created ad hoc |
| Execution order | `ss.Loop` | An explicit, inspectable, reorderable plan; see below |

## Time

Time is the part of Starsim most directly relevant to this project, because it is the one area where the framework has already made the move from `numeric` to a type system.

There is a class hierarchy under `ss.TimePar`:

- `ss.dur` — durations: `ss.days`, `ss.weeks`, `ss.months`, `ss.years`, and `ss.datedur` for calendar-aware durations
- `ss.Rate` — split three ways, which is the substantive distinction:
  - `ss.per` — a rate over time (`ss.peryear(0.1)`: a hazard, converts to a probability given `dt`)
  - `ss.prob` — a unitless probability attached to a period (`ss.probperyear(0.1)`)
  - `ss.freq` — a count of events per unit time (`ss.freqperyear(3)`)

Distinguishing *rate*, *probability*, and *frequency* at the type level is unusual and worth carrying forward: `0.1` per year could legitimately mean any of the three, they convert to a per-timestep quantity by different formulas (`1 - exp(-r·dt)`, rescaling, `r·dt`), and conflating them is one of the most common quiet errors in the field. Starsim makes the conversion the type's responsibility: `beta.to_prob(dt)`.

Durations and rates compose with dates: `ss.date` subclasses `pd.Timestamp`, so a sim can run on real calendar dates, and `ss.datedur` handles the month-length problem that pure float-year arithmetic gets wrong. A `Timeline` exposes the same time as `tvec` (ground truth), `tivec` (indices), `timevec` (human-friendly), `yearvec` (float years), `datevec` (dates), and `relvec` (relative), which is a lot of parallel representations but does mean output can be requested in whatever frame the data are in.

**Per-module time steps.** Each module has its own `Timeline`, so a within-host progression module can run at `dt = 1 day` while a demographic module runs at `dt = 1 year` in the same sim. The `Loop` interleaves them by absolute time. Nothing else in the landscape does this, and it is the strongest existing evidence that a lingua franca should not assume one global `dt`.

## Stochasticity and reproducibility

Every distribution is an `ss.Dist` object with **its own random number generator**, seeded from the sim's `rand_seed` plus a hash of the distribution's fully-qualified name. Consequences:

- Adding a module, or reordering modules, does not change the random draws of the other modules.
- Distributions can be drawn from independently and in any order without correlating.

On top of that, Starsim implements **common random numbers** at the agent level. Each agent holds a `slot`, and a per-UID draw is generated by hashing `(seed, draw index, slot)` — so agent 7's uniform for a given decision is the same in the baseline and intervention scenarios, regardless of how many other agents drew from that distribution. Paired scenario comparison therefore differences out most of the Monte Carlo noise, and small intervention effects are detectable with far fewer replicates.

This is a genuine design achievement and also the source of some of the framework's sharpest edges: sampling without replacement, joint draws, and any operation that cannot be written as a per-agent inverse-CDF fall outside the CRN guarantee, and the code has to mark those paths explicitly.

`ss.MultiSim` and `ss.parallel()` handle replicates and scenario sets; `ss.Calibration` wraps Optuna, with `CalibComponent` objects defining likelihood terms against data.

## Populations and mixing

`ss.People` holds agents with persistent UIDs; births add UIDs, deaths remove them from `auids` without invalidating references. States are stored as full-length arrays with a live-agent view, which is what makes vectorized operations over `.uids` cheap.

Transmission goes through `ss.Route`, of which there are two families:

- **`ss.Network`** — an edge list, regenerated or updated each step. Built-ins include random, static, maternal, sexual (`MFNet`, `MSMNet`), household, spatial, and disease-specific networks in `starsim.library.networks`. Transmission enumerates edges, computes per-edge probability from `beta`, `rel_trans`, and `rel_sus`, and draws.
- **`ss.MixingPool` / `ss.MixingPools`** — aggregate force of infection between groups defined by arbitrary agent predicates, with a contact matrix. This is the metapopulation and age-contact-matrix formulation, and it runs through the same `Disease.infect()` path.

That both live behind one `Route` interface is the mechanism by which the *same disease module* can be run as an individual-network ABM or as a mixing-matrix metapopulation model without editing the disease. It is a partial version of the paradigm-independence this project is after — partial because the disease module is still written as agent code, so the compartmental limit is reached by making the population homogeneous rather than by solving equations.

## Interventions

The most developed intervention vocabulary in the landscape, built as a small class lattice rather than a grammar:

- **Delivery mixins**: `RoutineDelivery` (start year, end year, coverage, interval) and `CampaignDelivery` (specific years, coverage).
- **Product base classes**: `BaseTest` → `BaseScreening`, `BaseTriage`; `BaseTreatment` → `treat_num`; `BaseVaccination`.
- **Concrete combinations**: `routine_screening`, `campaign_screening`, `routine_triage`, `campaign_triage`, `routine_vx`, `campaign_vx`, `treat_num`.
- **Products** (`ss.Product`) separate *what is delivered* — a diagnostic with sensitivity and specificity, a vaccine with efficacy and waning — from *how it is delivered*.

The product/delivery split is the right decomposition and should survive into the design. Eligibility is an arbitrary callable returning UIDs, which is fully general and completely opaque: it is Python, so it cannot be serialized, diagrammed, or checked. Anything the class lattice does not cover — a conditional cascade, a stock-constrained campaign, an adaptive trigger — is written as a bespoke `Intervention` subclass with a custom `step()`.

## Execution order

`ss.Loop` collects every module's step methods into an explicit, ordered plan, built by `collect_funcs()` in a fixed sequence: `start_step` for all modules → custom modules → demographics → disease `step_state` → connectors → networks → interventions → disease `step` (transmission) → deaths → results → analyzers → `finish_step`. The plan is a data object: it can be printed, exported with `loop.to_df()`, plotted (`plot_step_order`, `plot_cpu`), and modified with `loop.insert(func, label=..., before=True)`.

Making execution order inspectable and editable — rather than implicit in the code, as it is nearly everywhere else — is one of Starsim's better ideas, and it directly serves the "no hidden behavior" design goal.

## Results

Results are declared (`define_results`), not created on demand. An `ss.Result` is an array with a name, label, module, shape, timevec, and unit, so plots and exports are self-describing. Boolean states automatically generate an `n_<state>` result, which removes the most common piece of result boilerplate. `sim.to_df()` flattens everything; `ss.Analyzer` modules collect anything the standard results do not.

## Minimal model

```python
import starsim as ss

sim = ss.Sim(
    n_agents = 10_000,
    start    = '2020-01-01',
    stop     = '2021-01-01',
    dt       = ss.days(1),
    networks = ss.RandomNet(n_contacts=ss.poisson(lam=10)),
    diseases = ss.SIR(beta=ss.perday(0.05), dur_inf=ss.days(10), init_prev=0.01),
)
sim.run()
sim.plot()
```

## Worked example

A two-disease model with demographics, a connector, and an intervention — the shape Starsim exists for:

```python
import starsim as ss
import stisim as sti

# Population and its dynamics
demog = [ss.Births(birth_rate=20), ss.Deaths(death_rate=10)]

# Two co-transmitting diseases on the same network
hiv    = sti.HIV(beta={'structuredsexual': [0.05, 0.03]}, init_prev=0.05)
syph   = sti.Syphilis(beta={'structuredsexual': [0.10, 0.08]}, init_prev=0.02)
sexnet = sti.StructuredSexual()

# Biological interaction between them, as an explicit module
class HIVSyphConnector(ss.Connector):
    def step(self):
        # Syphilis co-infection raises HIV transmissibility
        coinf = (self.sim.diseases.syphilis.active).uids
        self.sim.diseases.hiv.rel_trans[coinf] = 2.0

# Intervention: ART scale-up as routine delivery of a product
art = sti.ART(coverage=[0.0, 0.4, 0.8], years=[2015, 2020, 2025])

sim = ss.Sim(
    n_agents      = 50_000,
    start         = 2010,
    stop          = 2030,
    dt            = ss.months(1),
    demographics  = demog,
    networks      = sexnet,
    diseases      = [hiv, syph],
    connectors    = HIVSyphConnector(),
    interventions = art,
)
sim.run()
```

Two things to notice. The connector is where cross-disease biology lives, and its content is *arbitrary Python that reaches into another module's state* — maximum flexibility, zero structure. And the disease modules themselves are unchanged from their single-disease form: the coupling is additive.

## The predecessor family

Covasim, HPVsim, and FPsim predate Starsim and are worth reading as the design's own history.

- **Covasim** (2020, COVID-19) established the pattern: typed per-agent state arrays, interventions as objects with `apply(sim)`, layered contact networks (household/school/work/community), and a results dict. Its interventions were plain callables or objects with a `days` and `changes` list — a genuinely simple time-varying-parameter grammar that Starsim's richer class lattice arguably lost some of.
- **HPVsim** added multi-strain state (genotype as a state dimension), a natural-history model with within-host progression, and the screening/triage/treatment product cascade that Starsim's `BaseTest`/`BaseTreatment` classes generalize.
- **FPsim** demonstrated that the same machinery models a non-infectious process (contraceptive use, pregnancy, birth outcomes), which is why `ss.Disease` is a subclass of a general `ss.Module` rather than the root.

The lesson recorded in the generalization: each of these hard-coded its disease into the framework, and the cost of that was three near-identical codebases. Starsim's `Module` abstraction is a direct response.

## Strengths

- **Typed time.** Durations, rates, probabilities, and frequencies are distinct types that know how to convert themselves given `dt`. This is the single most transferable idea in the framework.
- **Per-module timesteps**, interleaved by absolute time — demonstrating that a global `dt` is not necessary.
- **Common random numbers at agent level**, giving paired scenario comparison with a fraction of the replicates.
- **Per-distribution RNG streams**, so model structure changes do not perturb unrelated draws.
- **One `Module` abstraction** for diseases, networks, demographics, interventions, connectors, and analyzers, which is what makes multi-disease and non-disease health states composable.
- **Route abstraction** unifying networks and mixing pools, so the same disease runs as an ABM or a metapopulation model.
- **Declared parameters, states, and results** with labels and types — enough structure to auto-generate results, serialize, and validate.
- **The execution loop as an inspectable data object**, printable, plottable, and insertable-into.
- **A real intervention vocabulary**, with the product/delivery separation.
- **Multi-language reach**: an R interface (`rstarsim`) over the same Python engine.

## Limitations

- **Transitions are still code.** `define_states` declares the state space, but there is no object representing "S → I at rate β·I/N". Progression is scheduled imperatively by writing `ti_recovered` in `set_prognoses()`. The model is therefore not introspectable as structure, cannot be diagrammed or dimension-checked as a transition system, and cannot be automatically converted to ODEs.
- **Connectors are unconstrained.** Cross-module interaction is "arbitrary Python that mutates another module's state". This is why multi-disease works at all, and it is also the largest hole in the "no hidden behavior" goal: `rel_trans[coinf] = 2.0` in one module silently overwrites whatever another module wrote.
- **Order dependence is explicit but still load-bearing.** The loop is inspectable, which is better than implicit, but results still depend on whether the connector runs before or after the network updates, and nothing declares that dependency.
- **The compartmental limit is approximate, not exact.** Starsim can be run in an ODE-like configuration, but it gets there by simulating agents homogeneously, not by solving equations. It cannot produce the deterministic solution, and paradigm conversion is therefore one-directional.
- **Two specification languages.** The dict/`type=` form is declarative and serializable; anything real is Python classes. There is no path from the first to the second, and the boundary is where most users get stuck.
- **Python-object state is not serializable in general.** `sim.to_json()` captures parameters, not model structure; a Starsim model cannot be round-tripped as data.
- **API surface is large.** Six module categories, four state types, ~20 distributions, ~15 time classes, three rate types, and a `Loop` API. Each piece is justified; the total is a lot to hold, and the framework leans heavily on AI tooling (`starsim_ai`) to make it navigable — which is a signal about the specification burden, not a solution to it.
- **Time-type conversions have subtle corners**, particularly around calendar durations, `datedur` versus float-year arithmetic, and rate-to-probability conversion at large `dt`.
- **Performance is Python-bound** for anything not vectorized; Numba kernels and a Rust backend cover the hot paths, but a custom `step()` written naively is slow, and there is no compilation step to catch that.

## Implications for the lingua franca

1. **Carry the time type system forward, essentially as-is.** `dur` / `per` / `prob` / `freq` as distinct types, with conversion as the type's responsibility, is validated by use and directly serves the rigor goal. It also converges with camdl's dimensional types (`rate` = time⁻¹), which is strong evidence that this is the right primitive: two independent IDM designs arrived at the same place from different directions.
2. **Do not assume a global timestep.** Per-module timelines interleaved by absolute time already work in practice. The language should let each process declare its own time resolution, and the runtime should reconcile them.
3. **Make transitions first-class data — this is the gap.** Starsim's `define_states` shows what declaring the *state space* buys: automatic results, validation, serialization. Declaring the *transitions* the same way would buy diagrams, dimension checks, paradigm conversion, and prose-to-model generation. That step is the core of what this project adds.
4. **Keep one module abstraction, but give module interaction structure.** `ss.Module` as the single extension point is right. `ss.Connector` as "arbitrary mutation of another module's state" is the wrong end of the trade-off for a language whose goal is no hidden behavior. The design needs an explicit way to say *this module modifies that quantity of that module*, so that conflicts are detectable rather than last-write-wins.
5. **Keep common random numbers, and make the CRN contract explicit.** Paired comparison is too valuable to give up. But CRN in Starsim holds only for draws expressible as a per-agent inverse CDF, and that condition is documented in comments rather than enforced. In a language with declared transitions, the condition can be checked.
6. **Unify routes at the specification level.** "Transmission happens through *some* route" — edge-based, pool-based, environmental — with the route chosen separately from the disease is exactly the abstraction that lets one specification run as ABM, network, or metapopulation. Generalize it so the compartmental case is an exact route, not a homogeneous-agent approximation.
7. **Take the product/delivery split for interventions, and add an eligibility expression.** Products (what is delivered, with its efficacy characteristics) and delivery (to whom, when, at what coverage) are the right two axes. The missing third piece — eligibility as a declarative predicate over agent state rather than an opaque callable — is also what EpiModel's scenario grammar lacks, which makes it a well-evidenced requirement.
8. **Declare results; auto-generate the obvious ones.** Automatic `n_<state>` results from boolean states removes real boilerplate and guarantees consistency. With declared transitions, flows (`new_infections`, `si.flow`) can be auto-generated the same way.
9. **Treat the two-language problem as a design failure to fix.** Starsim's dict form and its class form are separate worlds. The lingua franca's premise — one text that a human, an AI, and a compiler all read — only pays off if the simple case and the complex case are the same language, differing in how much is written rather than in kind.
