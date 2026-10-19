# Malaria: what v4 should change to make the port a joy

Written 2026-10-18, at commit `1a5b7a77` on the `v4` branch of `ssv4`, the same commit as [fpsim_review.md](fpsim_review.md). Everything ran on a clean worktree of `1a5b7a77`, since the `ssv4` working tree was being edited. malariasim (`starsim-ngm`, branch `v4`, `a1351fc`) was used as a real model to test v4. It is an agent-based malaria model on Starsim 3, ported from the LASER malaria model and drawing on EMOD-Malaria, with interchangeable ODE and agent versions of both humans and mosquitoes. Five agents each took one part:
- the human disease (SIS and SEIS, biting heterogeneity, the age-binned ODE);
- the mosquitoes and the human–mosquito coupling, including the removed `malaria_hybrid` (Ross-Macdonald with four ODE/agent combinations);
- interventions (vaccine, bednets, drugs, outbreaks, campaigns, and their ODE versions);
- space and demography (the node grid, steady-state births and deaths, results by age and node);
- the framework (`ms.Sim`, the unchanged test suite, calibration, and speed against v3 and LASER).

The v3 to compare against needed choosing. malariasim passes all 284 of its tests on Starsim 3.5.2 and on `f40175b4` (the 3.6.0 performance refactor, before release), but released 3.6.0 to 3.7.2 fail 76 of them (see [Before changing more of the v3 surface](#before-changing-more-of-the-v3-surface)). So v3 here is `f40175b4`. Everything ran on one interpreter (the `ngm` env, which also has LASER). Scripts and outputs are in the `starsim-ngm` repo under `v4port/<part>/`, with malariasim's own bugs in `v4port/README.md`. I reran the main new findings myself (MA1, MA3, MA4, MA6, MA7, MA10, MA15, MA18, MA19, MA25, and the duplicates of FP1 and R8), and all of them reproduced.

Malaria is the first test model that is vector-borne. Humans and mosquitoes are two populations, transmission is by infectious bites rather than contacts, and interventions act both on people (vaccines, drugs, a net's protection) and on mosquitoes (a net's killing). v4's spec anticipates this more than it did FPsim: states that are not people (counts), a Ross-Macdonald example, `patch` stratification, species by `stratify(..., states=)`, and `ss.seasonal`. So this tests whether that vector vocabulary holds up on a real model. New findings get IDs MA1–MA33. Everything else is cross-referenced to the FPsim review (FP), the HPVsim review (HP), the STIsim review (ST), the merged findings (L, D, X, R), and [new-findings.md](new-findings.md) (N). Duplicates get one line each.

## In brief

**What works:**
- **v4's vector vocabulary fits malaria well, and the declared versions are a tenth of the code:**

  | Part | Declared v4 | v3 code it replaces | Agreement with v3 |
  |---|---|---|---|
  | Human disease: SIS or SEIS, per-node seasonal EIR, biting heterogeneity and age biting as dimensions | 36 lines | 536 | Prevalence within 0.006 as agents over 5 seeds; ODE to three decimals |
  | `malaria_hybrid`'s Ross-Macdonald, mosquitoes as counts | 24 | 330 (four interchangeable modules), plus 90 for its bus and sim | All four human × mosquito combinations within 0.5% |
  | malariasim's mosquitoes (`MosquitoesODE`, `AgentsOnly`, `NodeState`) | 16 | 442 | 916.7 mosquitoes on both; prevalence 0.475 against 0.480 |
  | Vaccine, bednet, drug, outbreak, node targeting, campaign rounds | 74 | 750 | Within 1–2% over 5 seeds where both model the same thing |
  | Nodes, steady-state demography, results by age and node | 43 | 257 (and the 109-line analyzer goes) | ODE to four decimals; agents within about 1 SE |

- **One declaration runs under every solver**, which malariasim built by hand twice (in `malaria_hybrid` and again in `malariasim`). The same text is agents, an ODE and chain-binomial. The ODE now covers the SEIS, biting heterogeneity and age biting, which v3's hand-written ODE didn't. As an ODE it is 11 to 43 times faster than v3's modules. `calibrate()` recovers the transmission probability (0.4005 against a true 0.4).
- **The coded model runs too, and v4's run tools work on it.** With a 27-line shim and no change to malariasim, 273 of 284 tests pass; with 31 more lines, all 284. Prevalence agrees with v3 within 1% in all 9 human × mosquito combinations, and ODE × ODE is bit-identical except the last point. Unchanged malariasim is 4–34% faster on v4 (the suite takes 54 s against 74 s). Because `ms.Sim` builds its modules before `super().__init__()`, `new()`, `ss.parallel(...).diff()`, name-based `pars=` into coded modules and `calibrate(gof=)` all work, unlike FPsim (FP7).
- **Coded mosquito agents plug into declared humans.** Mosquitoes as agents can't be declared (a state that is not people is always a count), but `malaria_hybrid`'s v3 `MosquitoABM` ran unchanged beside declared humans. A function rate reads the coded module, and `compartmental = True` on it lets the same coded mosquitoes drive ODE humans. The refusal that asks for it names the flag.
- **Results come by node, age, biting level and species with no analyzer**, the EIR and mosquito prevalence are one-line declared results under every solver, and `equations()` prints the Ross-Macdonald ODEs.

**What stops the port being a joy, in five groups:**

1. **v4 doesn't recognize a vector-borne infection as an infection** (MA1–MA5). It knows a transition is an infection only when its rate is exposure to a bare state (`beta * I`). A force of infection from mosquito counts (`a * b * Iv.n / N`, the spec's own form) or from an EIR series is treated as spontaneous. So, silently:
   - `rel_sus` and a vaccine's scalar efficacy do nothing: the spec's own bednet idiom on the spec's own malaria model leaves 7,387 infections at 7,387;
   - `ss.treat` reaches nobody;
   - `R0()` is 0.0;
   - and which states are people is guessed from `init=`, so `init=dict(Sv=3000)` makes the 70,000 mosquitoes the agents and the humans counts.
2. **Counts have a second integrator, and it disagrees with the ODE** (MA6–MA9). As agents and as chain-binomial, mosquito counts take one RK4 step per sim step with no substeps. At the monthly step malariasim documents, the mosquitoes go extinct (0 where 50,000 is right). A fixed 12-day incubation is exponential as agents, Erlang stages as an ODE, and 1.8e17 infectious mosquitoes as chain-binomial, so one declared model gives 0.475 human prevalence as agents and 0.256 as an ODE. Modifiers on mosquito transitions are ignored as agents.
3. **Bites need a weighted count, and the nearest spelling is silently wrong** (MA10, MA11). Mosquitoes are infected by people in proportion to how much they're bitten, and people by mosquitoes in proportion to each species' biting. `ss.by(biting=w) * Ih.n` silently takes the first level's weight (human prevalence falls from 0.77 to 0.08 where it should rise), and there is no spelling for Σ_k w_k n_k.
4. **Interventions on mosquitoes and repeated rounds have no declared form** (MA14, MA15). A net's killing depends on the waning-weighted share of people holding one, which a rate can't read (it's a function reading per-person arrays, so agents only). A product that wanes as a function is held for good, so a second round never reaches anyone who had the first: ITN redistribution, boosters and seasonal chemoprevention can't be modeled.
5. **Newborns from `ss.Births` all join the first node** (MA18), and the coded model's breakages are changes of meaning rather than removals (MA22–MA24). Over 20 years the nodes drift from 2,000/1,500/1,500 to 3,436/800/804, while prevalence by node still looks right. `ss.Disease` and `ss.SIS` now mean declared classes, so malariasim's subclasses fail late or with a suggestion that itself crashes, and analyzers now run before the first step.

**Counts:** 33 new findings (MA1–MA33), about 20 of them related to an existing finding, plus about 15 known findings that malaria hits again. Group 1 is one rule (the transitions out of the first state are infection) and about 50 lines. Group 2 is one change (counts go through the ODE kernel, with its stages and substeps, under every solver), which also removes most of the declared agents' slowness. Groups 3 and 4 are design work, which this review proposes: a declared force of infection with weights, and a product's coverage as a quantity. Group 5 is small fixes, a shim-sized compatibility layer and migration rows.

## What a joyful malaria model looks like

The target, in proposed spellings (the names marked `# proposed` don't exist today). Humans, mosquito counts, space, interventions and demography are declared. The within-host model and, where wanted, individual mosquitoes stay coded and plug in.

```python
import starsim as ss, malariasim as ms

nodes = ss.geo(names=['urban', 'peri', 'rural'], pop=[2_000, 1_500, 1_500])
a, b, c, mu = ss.perday(0.3), 0.4, 0.3, ss.perday(0.1)
Sh, Eh, Ih = ss.states(3)
Sv, Ev, Iv = ss.states(3, people=False)                                   # proposed: stated, not guessed (MA3, MA5)

itn = ss.Product(efficacy=dict(infection_in=0.8, kill=0.5), waning=ms.box_exp(ss.days(180), ss.years(1)))  # proposed: a neutral held product (MA33)
bites = ss.force(a * Iv.n / (Sh + Eh + Ih).n, weights=ss.by(biting=ms.gamma_levels(shape=1, n=10)))         # proposed: a declared force of infection (MA1, MA11)

malaria = ss.Disease(
    infection_in  = Sh >> Eh @ b * ss.by(patch=[0.8, 0.7, 0.55]) * bites,   # an acquisition: rel_sus, vaccines and ss.treat see it (MA1, MA2)
    infection_out = Sh >> Eh @ b * ss.by(patch=[0.2, 0.3, 0.45]) * bites,
    onset     = Eh >> Ih @ ss.normal(ss.days(12), ss.days(2)),
    recovery  = Ih >> Sh @ ss.lognorm_ex(ss.days(200), ss.days(200)),
    emergence = ss.null >> Sv @ ss.by(patch=ms.EMERGENCE) * ss.seasonal(amplitude=0.8, peak='July'),
    feeding   = Sv >> Ev @ a * c * bites.weighted(Ih),                       # proposed: the biting-weighted count (MA11)
    eip       = Ev >> Iv @ ss.constant(ss.days(12)),                       # the same stages under every solver (MA7)
    mortality = Sv + Ev + Iv >> ss.null @ mu * (1 + 0.5 * itn.coverage),     # proposed: coverage as a quantity (MA14)
).stratify(patch=nodes, biting=10, species=['gambiae', 'funestus'], states=[Sv, Ev, Iv])

sim = ss.Sim(diseases=malaria, connectors=ms.WithinHost(), people=ss.People(10_000, age_data=ms.AGE_DATA),
             demographics=[ss.Births(birth_rate=ms.BIRTHS), ss.Deaths(death_rate=ms.DEATHS)],   # newborns keep the node split (MA18)
             interventions=ss.vaccinate(itn, at=ss.years([1, 4, 7]), coverage=0.8, redose=True),  # proposed: re-dosing (MA15)
             start=2000, stop=2010, dt=ss.days(1))
fit = sim.calibrate(gof=ms.prevalence_by_node)
ss.parallel(agents=fit.sim, ode=fit.sim.new(solver='ode'), n_runs=5).diff('agents').plot()
```

With individual mosquitoes coded instead, the human side reads them through a named seam: `infection_in = Sh >> Eh @ b * ss.external('mosquitoes.eir_indoor')` (proposed, MA12).

Today the parts of this that run need these workarounds:
- `modify(infection, ...)` or an efficacy keyed by transition name in place of `rel_sus` or a scalar efficacy (MA1), and `where=mal.I` on every treatment (MA2);
- initial values given to the human states, never only to the mosquitoes (MA3);
- substeps by hand, a daily step, or an ODE, for anything with mosquito counts (MA6, MA7);
- two sets of mosquito states in place of a species dimension (MA11), and no biting weights on the mosquito side (13–31% too little mosquito infection);
- a function of the sim reading `has_itn` and `ti_delivered` for a net's killing, agents only (MA14);
- waning given as a duration rather than malariasim's box-then-decay function, so that rounds can repeat (MA15);
- declared `ss.birth` rather than `ss.Births` or `FastBirths` with nodes (MA18);
- for the coded model, the 27-line shim restoring v3's `ss.Disease` and `ss.SIS` (MA22, MA23), and results written at `ti + 1` (FP1).

## The port today, and after the changes

| Task | Today | After the changes below |
|---|---|---|
| malariasim unchanged | 123 of 284 tests fail on two changes of meaning; a 27-line shim leaves 11, all from the time convention (MA22–MA24, FP1) | An error at class creation naming the v3 base; a warning on v3-convention results |
| A vaccine or net's protection on an EIR or mosquito-count infection | No effect (MA1) | Applied |
| Mass drug administration | Reaches nobody unless `where=` names the infected (MA2) | Reaches the infected |
| Ross-Macdonald with initial mosquito numbers | The mosquitoes become the agents (MA3) | The humans are the agents, as stated |
| `R0()` of a vector model | 0.0 (MA4) | The next-generation value, or a refusal |
| Mosquitoes at a weekly or monthly step | Extinct as agents (MA6) | Right under every solver |
| A fixed extrinsic incubation period | Three solvers, three answers, one of them 1e17 (MA7) | The same stages everywhere |
| Biting heterogeneity on the mosquito side, several species | Silently the first level's weight (MA10); no spelling (MA11) | A weighted count |
| A net's killing of mosquitoes | Agents only, by a function (MA14) | `itn.coverage` in a rate, every solver |
| ITN redistribution every 3 years, boosters | Round 2 reaches 600 of 3,200 (MA15) | `redose=True` |
| `ss.Births` with nodes | Every newborn in node 0 (MA18) | Newborns at the node split, or the mother's node |
| The declared ODE with per-node seasonal EIR | 7–10 times slower than malariasim's hand-written ODE (MA27) | About as fast |

## What should change in v4

### 1. A vector-borne infection is an infection

v4 decides what an infection is from the shape of the rate: a transition is contact-mediated, and so an acquisition, when its rate is exposure to a bare state. In malaria the force of infection is never that. It is literal counts (`Sh >> Eh @ a*b*Iv.n/Nh`, the spec's own Ross-Macdonald form in `03-states-and-transitions.md` and `06-routes.md`) or an external series (`S >> I @ b * EIR`). Both are classed as spontaneous, and four things follow silently:

- **MA1 (silent): `rel_sus` and a vaccine's scalar efficacy have no effect on a vector-borne or EIR-driven infection.** On the spec's malaria model, human infections in a year are 7,387 as agents (7,431 as an ODE) with nothing, with the spec's own bednet idiom `ss.modify(rel_sus, multiply=0)`, and with `vaccinate(Vaccine(efficacy=1), coverage=1)`. They are 0 with `modify(infection, multiply=0)` or `efficacy=dict(infection=1)`. On an SIS with exposure, the same three calls take 113,817 to 0 (`human/repro_relsus_vector.py`; `interventions/repro_acquisition_external_foi.py`: 3,650 against 3,650 at 90% efficacy and full coverage, where `beta * I` gives 1,734 → 138). The vaccine is delivered and `what_modifies()` lists it. The same holds for `rel_trans` on the mosquito side, which counts `Ih.n`. Fix: treat every transition out of the first state as infection, whatever its rate; at least, warn when a `rel_sus` modifier or a scalar efficacy reaches no transition, naming the transitions to key it by. About 10 lines for the warning, 30 to 50 for the rule. Related to R11 (what counts as contact-mediated) and D23.
- **MA2 (silent): nothing counts as infected, so `ss.treat` reaches nobody.** `ss.treat(ss.Treatment(cure=1))` raises `UnknownEffect` ("no transition out of the infected states ... there is none"); with `transition='recovery'` it runs with 0 eligible and 0 delivered. With `where=mal.I` it treats 2,113 and n_I goes 2,366 → 270 (`interventions/probe_mda.py`). `describe()` doesn't say which states are infected. The same rule as MA1 fixes it, about 10 lines, plus "infected states" in `describe()`. Related to ST5.
- **MA3 (silent): which states are people is guessed from `init=`.** With no contact-mediated transition to anchor on, the spec's Ross-Macdonald model given `init=dict(Sv=3000)` makes the mosquitoes the 5,000 agents and the humans counts, and runs: n_Sh 0, "people alive" 70,357 at day 100 (`interventions/repro_mosquitoes_become_people.py`). `init=dict(Sv=70000)` gives `ValueError: The initial fractions of "malaria" sum to 14`, which doesn't say why. Declaring the mosquitoes `ss.State(people=False)` with only `Sv` initialized crashes with `IndexError: list index out of range` in `Graph.anchor`, and `ss.states(3)` has no `people=`. Fix: anchor on the states the infection transitions start from (MA1's rule), never on states holding more than the number of agents; guard the empty group; add `ss.states(n, people=False)`. About 15 lines.
- **MA4 (silent): `disease.R0()` is 0.0 for Ross-Macdonald**, the spec's own example and the malariasim model, where the analytic value is 154 (12.4 as the next-generation matrix's square root), and the model is endemic (`vector/repro_R0_vector.py`). Include the count states in the next-generation matrix at their disease-free equilibrium, or raise "not well defined" as the spec promises. About 30 lines.
- **MA5 (silent): `ss.State(people=True)` on the mosquito states puts them in `sim.people`.** The only spelling that makes mosquitoes agents splits 1,000 agents into 500 humans and 500 mosquitoes, gives the mosquitoes human ages and aging, and counts them in `n_alive` (`vector/probe_mosquito_agents_declared.py`). Agents of another species should stay coded (below). Warn when `people=True` states have no transition to the first state, and say so in the docs. Small.

One rule covers MA1–MA3: the transitions out of the first state are infection, contact or not. The spec should state it in `03-states-and-transitions.md`, with a conformance test that the spec's bednet idiom and a vaccine both change the Ross-Macdonald model.

### 2. Counts need the ODE's integrator under every solver

States that are not people are stepped by the ODE solver with stages and substeps. As agents and as chain-binomial they are stepped by `Agents.step_counts`: one unstaged RK4 step per sim step, in Python. The two disagree on what malaria cares most about, the extrinsic incubation period and long steps.

- **MA6 (silent): mosquito counts are unstable at long steps as agents and chain-binomial.** Emergence of 5,000 a day with a 10-day life should settle at 50,000 from 10,000. The ODE gives 50,000 at every step. As agents (`vector/repro_count_rk4.py`):

  | Step | Mosquitoes |
  |---|---|
  | 21 days | 49,961 |
  | 28 days | 16,645, a false fixed point from clipping at 0 inside the stages |
  | 30 days | 10,000 → 5,000 → 2,500 → … → 2 |
  | 365/12 days (malariasim's documented monthly step) | 0 after 2 steps |

  A fast count rate at a daily step (Ev → Iv at 50 a day) turns 1e6 exposed mosquitoes into 2.5e7 infectious ones in one step (`repro_count_order.py`). Fix: substep the counts as the ODE solver does (no count turning over more than a quarter per substep), or integrate them with `kernels.settle()`. About 15 lines. Related to X27, which proposes `settle()` for tidiness; new as a correctness bug.
- **MA7 (silent; solvers disagree): a fixed incubation period on counts means three different things.** `Ev >> Iv @ ss.constant(ss.days(12))` is an exponential as agents (with no note), 50 Erlang stages as an ODE (with a note), and the same 50 stages through MA6's unsubstepped RK4 as chain-binomial, which explodes. Infectious mosquitoes at day 300: 7,576 as agents, 5,092 as an ODE (5,020 expected for a fixed delay), and 1.8e17 as chain-binomial (`vector/repro_count_dwell.py`). On the malariasim model with a fixed EIP, human prevalence is 0.256 as an ODE and 0.475 as agents: the same declared model, 46% apart. Fix: expand count states into the same Erlang stages under every solver (they're only numbers, so stages are cheap), with MA6's substeps. About 20 lines. Related to X7 and HP28; new for counts.
- **MA8 (silent): modifiers on count transitions are ignored as agents, and `ss.change` on an inflow under every solver.** Doubling mosquito mortality with `ss.modify` gives 4,000 mosquitoes as agents and 2,000 as an ODE (`interventions/repro_modify_count_transition.py`), and `what_modifies()` lists it in both. `ss.change(mal.emergence, by=0.5)` leaves 4,000 under both solvers, while `ss.change('mal.pars.em', by=0.5)` works (2,013). A product efficacy keyed by a mosquito transition (`dict(mortality=...)`) does nothing under both. Fix: apply modifiers in the agent engine's count update and on inflows; refuse a person-held product's efficacy on a count transition until MA14 defines it. About 20 lines.
- **MA9 (spec drift, O(dt)): as agents, a count-only infection is taken before the counts move.** `10-execution.md` says such a rate is evaluated "after the count has been advanced". The code takes it before `step_counts`, so both links lag a step as agents, while the ODE has none: 0 human infections in the first step as agents against 97.5 as an ODE (`vector/repro_count_order.py`), and prevalence −2% at a daily step and −4% at a weekly one against the ODE (`vector/lag_dt_v4.py`). v3's malariasim, its own docs, v4's spec and v4's code each describe a different coupling order (malariasim bug M1 in the companion note). Fix: take count-only infections after `step_counts`, or amend the spec; print the coupling order in `describe()`. A few lines. Related to X13.

The structural fix is one path for counts: the ODE kernel, with its stages and substeps, under every solver. That fixes MA6 and MA7, and most of the declared agents' slowness (MA28).

### 3. Bites need a weighted count

The coupling between humans and mosquitoes is bites, and bites are weighted. A heavily bitten person infects more mosquitoes; a species that prefers humans infects more people. v4 has the per-person half (biting heterogeneity as a dimension with `adjust={'infection': w}` works, and agrees with v3 within 0.001) but not the count half.

- **MA10 (silent): a value by level of a dimension that the state it multiplies lacks takes level 0.** The spec says people with no level get the split-weighted mean (`05-dimensions.md`, "Values by level"). In Ross-Macdonald with biting levels on the humans, `ss.by(biting=w) * Ih.n` in the mosquitoes' infection rate makes human prevalence fall from 0.770 to 0.083 (agents) and 0.060 (ODE), where it should rise; the first level's weight is 0.05, and with the levels reversed it rises past where it should (`human/repro_weighted_count.py`). On the mosquito side, `ss.by(species=[0.15, 0.3])` in the human rate uses level 0 for everyone as an ODE and the split-weighted mean as agents: prevalence 0.135 against 0.299 (`vector/repro_species_by_ref.py`), and `adjust={'emergence': [...]}` by a plain `patch` that doesn't split the mosquitoes takes level 0's factor under both. For people, the same case already raises `InvalidAdjustment`. Fix: refuse it with that error, naming the dimension and the state. About 10 lines. Related to L11 and D4.
- **MA11 (vocabulary): there is no weighted count.** The mosquitoes need Σ_k w_k n_Ih[k], which malariasim computes in `_write_node_state`: unweighted, they get 13–31% too little infection (the biting-weighted infectious fraction is 0.61 against 0.42 at EIR 5, `human/probe_observables.py`). The humans need Σ_s a_s Iv_s / N over species. Today two sets of mosquito states are the only correct spelling (`vector/species_v4.py`); stratifying by species gives +2% in prevalence with a minor second species, more where anthropophily differs more. This answers [open question B2](../design/19-open-questions.md#b2-states-that-are-not-people) ("needs confirming on a real malaria model with several species"): stratification is not enough as it stands. Proposed: `(ss.by(biting=w) * Ih).n` meaning the weighted sum over the state's levels, under every solver, or a declared force of infection (below) whose `weighted()` gives it. About 40 lines.

**Design note: a declared force of infection.** MA1, MA10, MA11 and MA12 are faces of one gap: v4 knows an infection only by the shape of its rate. Malaria wants the force of infection as a quantity in its own right, for example `bites = ss.force(a * Iv.n / N, weights=ss.by(biting=w))`, used as `Sh >> Eh @ b * bites`. Then `rel_sus`, products and `describe()` treat the transition as an acquisition, `bites.weighted(Ih)` is the count the mosquitoes need, and its value can come from counts, a series, or a coded module (MA12). It would serve any vector-borne or environmental model, and EIR-driven models such as EMOD's.

### 4. The seams with coded modules

The within-host model (parasite and gametocyte densities, immunity, symptoms from density) should stay coded: it needs per-person continuous values (HP19). So should individual mosquitoes where wanted (EMOD-style agents, larval habitats, vector genetics). v4 makes most of this easy already. What's missing:

- **MA12 (dead end): a per-node EIR computed each step by a coded module has no spelling that runs both as agents and as an ODE.** `ss.by(patch=[fn, fn])` raises `TypeError: unsupported operand type(s) for *: 'float' and 'function'` at build. `fn(sim)` returning one value per patch works as an ODE only while `patch` is the only dimension (with biting levels: `operands could not be broadcast together with shapes (2,) (20,)`), and as agents it's indexed by UID (`IndexError: index 2 is out of bounds`). `fn(sim, uids)` works as agents and the ODE refuses it (`human/probe_external_eir.py`, `probe_fn_groups.py`). Fix: allow `ss.by(patch=fn)` with `fn(sim)` returning one value per level, evaluated per level by every solver, about 30 lines; better, a named seam such as `ss.external('mosquitoes.eir')`, so `describe()` shows where the force comes from and when it's read. Related to N20 and N21.
- **MA13 (friction, and a silent trap at init): coded code reads a declared count as `mal.Ih.count()` as agents (a `BoolState`, no `.n`) but `mal.Ih.n` as an ODE (a `CountState`, no `.count()`), and in a connector's `init_post()` nobody has been placed yet, so N = 0.** A coded mosquito module that sized its buffer from N was silently empty: prevalence 0.0004 against 0.46 (`vector/mosquito_agents_v4.py`). Fix: `.n` on `BoolState` (1 line); place declared states before other modules' `init_post()`, or document the order. Related to L31 and ST8.
- **MA14 (vocabulary): a product's coverage is not a quantity a rate can use.** A net's effect on mosquitoes (killing, repelling) depends on the waning-weighted share of people holding one. Today it's a function of the sim reading `sim.people.has_itn` and the product's `ti_delivered`, which runs as agents only (as an ODE, `AttributeError: People object has no attribute "has_itn"`), and the product in the function's closure isn't the sim's copy, so it has empty arrays (`IndexError: boolean index did not match`) (`interventions/malaria_rm_v4.py`). Proposed: `itn.coverage` (by patch, optionally weighted by an efficacy component), usable in any rate: `mortality = Sv + Ev + Iv >> ss.null @ mu * (1 + a*f*kill*itn.coverage)`. Agents compute it from holders and their time since delivery, the ODE from its product levels. ITN killing, IRS, ivermectin and attractive sugar baits are then one-line declared terms under every solver. About 60 lines. Related to D23.
- **ST11 again (duplicate):** a coded within-host model has no public way to end an infection in a declared disease.

### 5. Interventions: re-dosing and waning

The human-side interventions are a good fit: vaccine waning as a function, coverage, campaign rounds, node targeting by `where=mal.patch == 'n0'`, indoor-only protection keyed to `infection_in`, and outbreaks as `ss.Delivery(move='S -> I')`. They agree with v3 within 1–2% as agents, and each delivery reports `delivered`, `eligible` and `coverage`, which v3 didn't. v4 also avoids two malariasim bugs by construction (a bednet counted twice, and a second ODE dose that removes protection; companion note).

- **MA15 (silent change of meaning): a second round never reaches anyone holding a product whose waning is a function, even after it has faded to 0.** A holder holds it for good, and v4 never gives a product to someone who holds it. Round 2 delivers 600 against 3,237 in round 1 (v3: 3,200 each); with waning given as a duration, round 2 reaches 3,071 (`interventions/repro_no_redose.py`). Two vaccine rounds give n_I 1,827 at day 729 on v4 against 1,311 on v3. Malaria programmes re-dose: ITN campaigns every 3 years, RTS,S boosters, monthly seasonal chemoprevention. malariasim's rule, one active instance per product and a new dose overwrites, is simple and solver-neutral. Fix: a holder loses the product when its waning reaches 0; add `redose=True` on a delivery, resetting `ti_delivered` (in a compartmental run, moving people back to the first level). About 20 lines.
- **MA16 (accuracy): as an ODE, a product with waning as a function is kept in 10 levels of time since delivery**, a fixed class attribute with no keyword. That's 6.2% off agents a year after delivery for malariasim's 180-day box with τ = 365; 40 levels give 0.7% (`interventions/probe_ode_waning_stages.py`). Choose the levels from the waning's shape (breakpoints such as the box end), or add `stages=` as for Erlang dwells, and print the expected error. About 10 lines. Related to L14.
- **MA17 (speed): a function-valued parameter is evaluated about 3 times per use per step.** The net's per-node coverage function runs 9 times per step for 3 rate terms, so the Ross-Macdonald bednet scenario takes 5.8 s against 3.6 s on v3 (`interventions/probe_function_calls.py`). Evaluate each once per step. About 5 lines. Related to FP34.

### 6. Space and demography

Space and demography fit the declared vocabulary well, and nothing here should stay coded. A node is a `patch` level, per-node parameters (including time series) are `ss.by(patch=)`, emergence per node is an inflow by patch, `ss.geo(names=, pop=)` replaces `NodeGrid` and gives `total_pop`, and results by node and age come from the stratification. malariasim's `ss.Births` and `ss.Deaths(table)` and its `FastBirths`/`FastDeaths` run unchanged on v4 with the same population stability (5,009 ± 15 against 5,000 ± 12 over 30 seeds). The weak points are at the seam with coded demography:

- **MA18 (silent): newborns made by `ss.Births` (and so by `FastBirths`, `ss.Pregnancy`, anything using `people.grow()`) all join the first patch.** `place_newcomers()` (`solvers.py:4070`) puts them in the first state but leaves the dimension property at 0. A split of 0.4/0.3/0.3 becomes 0.734/0.133/0.133 after 20 years, with 5,591 of 5,591 newborns in the first patch; declared `ss.birth` gives 0.397/0.302/0.301 (`population/repro_newborn_patch.py`). In the malaria model the nodes go from 2,000/1,500/1,500 to 3,436/800/804. Prevalence by node still looks right, because the EIR is per node, so nothing flags it, but anything using node counts (mosquito feeding, cases per node) is wrong. Fix: draw each non-age dimension from its split, or from the parent's level when `parent` is set; a conformance test that every dimension's shares hold under each demography module. About 10 lines. Related to ST1: its fix put newcomers in the first state, but not "at the split of its dimensions" as ST1 proposed.
- **MA19 (silent, small): the same typed rate means different things in `ss.Births` and `ss.birth`.** `ss.Births(birth_rate=ss.peryear(25.768))` is read as an annual probability, and `ss.birth(rate=ss.peryear(0.025768))` as a rate: 16,850 ± 28 against 16,686 ± 33 after 20 years from 10,000, as each reading predicts (`population/repro_births_rate_meaning.py`). `starsim3.md` calls the mapping "Exact"; switching costs malariasim about 1.4% of its births. A row in the migration guide and `starsim3.md` (`rate=-log(1 - r)`), or make `ss.Births` read a rate as a rate with a changelog line.
- **MA20 (vocabulary): the death rate's age resolution is tied to the model's one age dimension.** malariasim has deaths by single year of age and results by 5-year group. `ss.by(age=)` with 100 values on 20 groups gives a clear `ShapeError`, and `ss.death(rate=ss.peryear(fn))` gives `TypeError: Value must be a scalar number or array, not <class 'function'>` (`population/probe9_death_resolution.py`). Let `ss.by(age=..., edges=...)` carry its own edges, used directly as agents and averaged onto the dimension for compartments, and accept `ss.peryear(fn)`. Medium. Related to FP33, FP31 and FP22.
- **MA21 (accuracy, design): in a compartmental run, aging at 1/width can't hold a non-flat pyramid at the death rates that hold it as agents.** The ODE drifts −1.7% to −1.9% in 20 years (pyramid TVD 0.015); the rule the spec already uses for named populations (the density just below the upper edge over the number in the group) gives TVD 0.005 (`population/calc_ode_aging_rule.py`). Use it for every pyramid, estimating the edge density from the split. About 15 lines.

### 7. The coded model: changes of meaning

v4 broke malariasim not by removing names, as it broke FPsim and HPVsim (FP10), but by changing what two of them mean. That is worse: the subclass still builds, and the error comes later, in the wrong place.

- **MA22 (crash, 72 tests): `ss.Disease` now means a declared disease, and the fix the error suggests crashes.** malariasim's age-binned ODE subclasses v3's bare `ss.Disease` with no states. v4 says `"MalariaHumanODE" declares no states or transitions. A disease written as code ... subclasses ss.Infection`. Doing that fails with `ValueError: Another result named "n_susceptible" already exists because a result was automatically created for the state "susceptible"`, since `ss.Infection` adds S and I. v4 has no bare coded base; the shim rebuilds one in 18 lines (`framework/malshim.py`). Fix: a bare coded base (`ss.CodedDisease`, or `ss.v3.Disease` under decision 2), named in the error, and a migration row. About 15 lines. Related to FP10.
- **MA23 (crash, 47 tests): a subclass of `ss.SIS` builds, then fails at the first step with `no attribute "ti_infected". Did you mean: 'infected'?`.** Nothing says `ss.SIS` is now declared, or that v3's class is `ssl.CodedSIS`. Fix: refuse, at class creation, a subclass of the declared `ss.SIS`/`SIR`/`SEIR` that defines `define_states()`, `set_prognoses()` or `step_state()`, naming `ssl.CodedSIS`. About 10 lines. Related to FP10.
- **MA24 (crash, or silent): v4 calls `update_results()` and every analyzer at the initial point, before any module has stepped.** malariasim's `ResultsAnalyzer` reads the agent mosquitoes' `alive`, allocated in their first `step()`: `TypeError: 'NoneType' object is not subscriptable` (6 tests after the shim). With other mosquito modules every analyzer history gains a pre-step point, whose EIR is 0 for ODE mosquitoes (`framework/diag_order.py`). Give analyzers an initial-point hook distinct from `step()`, or skip a coded analyzer's `step()` there, and say so in the migration guide. 5 lines. Related to N10 and D22 (its mirror image).
- **MA25 (crash): `ss.diff(sim1, sim2)` crashes on any sim containing a coded module with parameters**: `TypeError: string indices must be integers, not 'str'` at `spec.py:629`, because `Module.to_json()` writes `pars` as a dict and `diff` iterates it as a list (`framework/repro_diff_coded.py`, with plain `ssl.CodedSIS`). 3 lines. Related to FP9 (a different cause).
- **MA26 (cost): `calibrate()` runs 4 identical seeds per evaluation of a deterministic coded ODE**, since a coded module can't say it's deterministic: 89 runs against 23, and 24.4 s against 6.4 s, for the same answer (`framework/calib_evals.py`). A `deterministic = True` attribute read by `fit()`. About 5 lines. Related to X9 and R4.
- **FP1 again (duplicate, new instance: stocks too):** every malariasim module writes results in `step()` at `self.ti`, so the last point is 0: `eir_annual[-1]` is 0.000 on v4 and 10.000 on v3, the year-mean EIR reads 9.973 against 10.000, and ODE prevalence is NaN at the end (`framework/repro_last_point.py`). Stocks are affected as well as flows, so FP1's proposed warning should check the last point too.
- **FP1, via v3 (duplicate):** `new_infections` is 0 at every step, because malariasim's `set_prognoses()` doesn't call `super()`. This broke on v3.6 (below).
- **R8 again (duplicate):** `sim.hash()` doesn't see `NodeGrid`, held as a plain attribute of a coded connector: node EIR 10 and 100 hash the same, and give 343 against 485 infected (`framework/repro_hash.py`).
- **N20 again (duplicate):** an array parameter (12 monthly emergence values) can't be fitted: `ss.par(np.full(12, e), bounds=...)` is accepted and then crashes `calibrate()`; a list of `ss.par`s gives `NothingToFit`.
- **FP7, not hit:** `ms.Sim` builds its modules before `super().__init__()`, so `inputs` is full and `new()` rebuilds it (as a plain `ss.Sim`, harmless here). This is the shape v4's run tools handle best, and the docs should recommend it for package sims.

### 8. Speed

- **MA27 (speed): typed values inside `ss.by` and `ss.timeseries` are converted again for every group at every step** (`TimePar.to`, with `get_unit_class` and a string join per call), whenever the model can't use the whole-run kernel. A seasonal EIR per patch triggers it, and so does a constant death rate by age with births. 35–57% of the time is in `TimePar.to` (`population/repro_by_per_group_cost.py`). So the declared malaria ODE takes 3.92 s against malariasim's hand-written ODE at 0.55 s (`population`), and 1.0 s against 0.44 s for three nodes (`human`). Convert each value to the sim's unit once at init, tabulate series per step (as `series()` already does for whole runs), and evaluate once per level of the dimensions a value depends on. About 30 lines. Related to X10 and FP34; new is that the cost grows with the number of groups.
- **MA28 (speed): declared agents with counts are slower than v3's equivalent code**, because `step_counts` walks each rate's expression tree four times per step in Python, and `ss.timeseries.at()` converts every stored value on each of 8 calls per step (5.3 million conversions in 1,825 steps). About 80% of the declared disease's step goes to the counts: `malaria_hybrid` as agents takes 3.81 s against 0.77 s for v3's `HumanABM` × `MosquitoODE` (`vector/prof_hybrid_v4_abm.txt`). MA6's fix (counts through the ODE kernels) removes most of it. Related to N29 and X27.

### 9. Small things

- **MA29 (crash):** `describe()` on an ODE sim with `people=ss.People(n, age_data=...)` and an age dimension raises `AttributeError: 'NoneType' object has no attribute 'place'` (`sim.py:1535`); the run itself is right (`population/repro_describe_people_age.py`). 1 line.
- **MA30:** a result such as `c * I.frac` or `98.6 + 4 * I/(S + I)` (fever, or the human-to-mosquito infectiousness) is unstratified, so `.by('patch')` raises "is not stratified", while `c * I/(S + I)` is stratified; `ss.by(biting=w) * I/(S + I)` as a result crashes with internal errors (`'Agents' object has no attribute '_uids'`) (`human/probe_observables.py`). Carry strata through affine expressions and `.frac`. About 20 lines. Related to R18.
- **MA31:** on a `MultiSim`, `.by(dim)` returns an `objdict`, so `msim.results.malaria.people.by('age').plot()` raises; on a sim it's one result with a column per level. Small.
- **MA32:** `people_results=False`, which malariasim's own July benchmark used, crashes `ss.Births` with `'Results' object has no attribute 'n_alive'`, on v3 and v4 alike, without naming the flag. 2 lines.
- **MA33:** a bednet has to be an `ss.Vaccine`, since there's no neutral held product, and its four waning components (repelling, blocking, killing, usage) have to be folded into one efficacy and one waning ratio by hand. `ss.Product(efficacy=, waning=)` as the neutral name, with efficacy entries that each take their own waning. Small.

**Known findings malaria hits again, one line each:**
- D2: a bare-number time in `ss.timeseries` is years when `start=2000` and days when `start=0`, so a seasonal EIR given in days is flat at January's value (`human/repro_ts_bare_times.py`).
- L1: `start=2000, stop=2002, dt=ss.days(1)` runs 731 points, but 3 with `unit='day'` given, and `describe()` says "days 2000 to 2002" both times.
- X10: leaving the whole-run kernel is silent (MA27's cause).
- X8, HP28: the first ODE run builds kernels for 32 s silently, and a 480-group SEIS ODE with `normal(12, 2)` as 36 stages takes 13–17 s against 2.5 s as agents.
- FP31: `ss.peryear(ts)`, `ss.peryear(1) * ts` and `ss.peryear(1) * fn` fail, while `ts * ss.peryear(1)` works.
- FP38, L27, HP32, ST10: `describe()` prints three EIR series in full but drops `period=`, prints `0.4 × 10` with no unit, says "[guessed]" for a given `People` and its ages, drops a birth rate's `peryear`, and labels a coded sim's schedule "steps 2000 to 2002, every 0.00274 steps".
- FP3: `eir.resample('year')` on a 3-year dated sim gives 4 "years".
- D4, N4: `patch=` without `kind='space'` leaves the mosquitoes unsplit (a note is printed, which is good).
- N21: `ss.peryear(ss.by(...))` is refused, with a good message.
- N10: v3 day d is v4 time point d + 1.

## Before changing more of the v3 surface

malariasim adds a second kind of breakage to FPsim's (FP10). It broke on Starsim 3 itself, twice, between the July pre-release it was developed on and released 3.6.0:
- **A removal with a good message:** `float()` of a rate is refused (`ea12d268`), and 72 tests fail at `float(p.death_rate_per_year)`. The message says exactly what to do.
- **A silent change of meaning:** `new_infections` moved from `update_results()` (counting `ti_infected == ti`) into `Infection.set_prognoses()` (`7bae8b1e`, "final pass on infected vs infectious"), so malariasim's overrides that don't call `super()` record 0 infections. 4 tests fail, on exact zeros, and no message announced it.

v4 adds two more changes of meaning, to the names packages subclass most: `ss.Disease` and `ss.SIS` (MA22, MA23). With a 27-line shim restoring them, everything else in malariasim runs and agrees. So:
- keep the v3 coded bases reachable under stable names (`ssl.CodedSIS` exists; add a bare coded disease), and refuse, at class creation, a subclass of a declared class that defines coded hooks;
- treat a change of meaning as at least as serious as a removal: a "Regression" line and a migration row, and where possible an error at the old usage;
- add malariasim's suite, with the shim as the expected diff, to the downstream release gate proposed in FP10, beside FPsim, HPVsim and STIsim. Its 284 tests caught the time convention only through an ODE prevalence of NaN and four exact zeros, so the gate should include exact comparisons of a baseline run, not only directional tests.

## Speed

| What | v3 | v4 | LASER | Note |
|---|---|---|---|---|
| Unchanged malariasim, scenarios a–f from its July benchmark (5 years daily, `crn=False`, best of 3) | 0.42–24.8 s | 0.35–23.8 s | 0.19–11.9 s | v4 is 4–34% faster than v3, and 1.7–2.5 times slower than LASER |
| Scenario c, human SEIS + mosquito ODE, 10k | 4.02 s | 2.99 s | 1.21 s | |
| Scenario e, SEIS + 200k mosquito agents | 24.4 s | 23.4 s | 11.9 s | Dominated by malariasim's numpy mosquito code |
| Scenario f, SIS + mosquito ODE, 100k | 22.3 s | 14.7 s | 6.7 s | |
| Unchanged suite, back to back | 73.6 s | 54.0 s | | |
| Declared human SIS as agents against malariasim (10k / 100k) | 2.31 s / 8.91 s | 1.55 s / 4.20 s | | 1.4–2.1 times faster |
| SEIS with mosquito counts, single node, 10k, with demography: malariasim, against declared | 3.73 s (2.84 s on v4) | 3.36 s as agents; the ODE is refused with coded births | 1.37 s | Declared agents 1.2–1.4 times slower than coded malariasim on v4 (N29) |
| The same with no demography: malariasim, against the declared ODE | 1.56 s (1.25 s on v4) | 0.48 s | | 2.6 times faster |
| `malaria_hybrid` ODE × ODE, against the declared ODE | 0.47 s | 0.043 s | | 11 times faster |
| `malaria_hybrid` agent humans × ODE mosquitoes, against declared agents | 0.77 s | 3.81 s | | 5 times slower (MA28) |
| Declared ODE, 3 nodes × 20 age groups, seasonal EIR per node | 0.55 s (malariasim's ODE) | 3.92 s | | 7 times slower (MA27) |
| Demography only, 20,000 agents, 20 years weekly, `ss.Births` + `ss.Deaths(table)` / `FastBirths` + `FastDeaths` | 2.48 s / 1.41 s | 1.61 s / 0.55 s | | v4's built-in demography is 1.5 times faster than v3's, but `FastBirths`/`FastDeaths` are still 2.9 times faster |

The machine was shared (load 1 to 16 on 20 cores), so differences under about 20% are noise. The gain on unchanged malariasim is v3's per-call `Dist.jump()` being gone, which was the first recommendation of malariasim's July performance notes (`starsim_recommendations.md` in its repo): 22–34% in human-agent scenarios, 4% where numpy mosquito agents dominate. July's reported parity with LASER came from malariasim's own Numba kernels, which aren't in `a1351fc` (`FastBirths` and `FastDeaths` are on its branch `Interventions2-ck`), so these ratios are for the plain code. The declared versions are faster where they're ODEs and slower where they're agents with counts or per-group typed values; MA27 and MA28 account for most of that.

## Keep

- **States that are not people, in the same disease as the humans.** Seasonal emergence, an incubation period, a nonlinear relaxation (`np.maximum(target - M, 0)`), patches and species-specific mosquito parameters are each one line, and Ross-Macdonald is 24 lines where `malaria_hybrid` needed four modules and a shared dict.
- **One declaration, any solver,** including chain-binomial, which malariasim never had. The guessed counts are marked in `describe()` ("Sv a count, not people [guessed: no transition connects Sv to Sh]"), which made MA3 findable.
- **Coded modules as first-class beside declared ones.** v3 mosquito agents ran unchanged with declared humans; `compartmental = True` lets coded code drive an ODE, and the refusal names it.
- **The run tools on a package sim built before `super().__init__()`:** `new()`, `ss.parallel`, `diff('baseline')`, name-based `pars=` into coded modules, and `calibrate(gof=)` (which matches a root find to 1e-4 for the emergence scale that hits a target EIR).
- **Patches with per-patch time series, and `ss.geo()`.** They replace `NodeGrid`, the node-assignment code, the per-node EIR loop and the results analyzer, with common random numbers for node assignment where malariasim used the global `np.random`.
- **`ss.timeseries(mode='linear', period=ss.years(1))`,** which is exactly malariasim's `interpolate_monthly`.
- **Delivery results** (`delivered`, `eligible`, `coverage`) and `what_modifies()`.
- **v4's discretization,** which is closer to the continuous model at weekly steps: at a constant EIR of 6, v4 gives 0.563 as agents and 0.568 as an ODE (exact), against v3's 0.586.

## Suggested order

1. **MA1–MA3 (a vector-borne infection is an infection).** One rule, about 50 lines, with a conformance test on the spec's own bednet idiom. Until then, warn when a `rel_sus` modifier or scalar efficacy reaches nothing.
2. **MA6–MA8 (counts through the ODE kernel under every solver).** About 50 lines; also fixes MA28.
3. **MA18 (newborns keep their dimensions)** and **MA15 (re-dosing).** About 10 and 20 lines.
4. **MA22–MA24 (changes of meaning),** with the release gate above, and FP1's warning extended to the last point.
5. **MA10 (refuse level-0 fallbacks), MA4, MA9, MA16, MA17, MA19, MA25–MA27, MA29–MA33.** Small fixes and messages.
6. **Design pass on bites** (MA11, MA12, MA14): a declared force of infection with weights and an external source, and a product's coverage as a quantity. With these, all of malariasim except the within-host model and individual mosquitoes is declared, and runs under every solver.

## Index of new findings

| ID | Finding | Kind | Section | Agent's ID | Cross-reference |
|---|---|---|---|---|---|
| MA1 | `rel_sus` and scalar vaccine efficacy do nothing on a vector-borne or EIR infection | Silent | 1 | human MH1, interventions M-I1 | Related to R11, D23 |
| MA2 | Nothing counts as infected, so `ss.treat` reaches nobody | Silent | 1 | interventions M-I2 | Related to ST5 |
| MA3 | `init=` decides who the agents are; mosquitoes become people | Silent | 1 | interventions M-I6 | New |
| MA4 | `R0()` is 0.0 for Ross-Macdonald | Silent | 1 | vector VE4 | New |
| MA5 | `people=True` puts mosquitoes in `People` | Silent | 1 | vector VE7 | New |
| MA6 | Counts take one unsubstepped RK4 step as agents and binomial; extinct at a monthly step | Silent | 2 | vector VE1 | Related to X27 |
| MA7 | A fixed incubation on counts: exponential, Erlang, or 1e17 by solver | Silent | 2 | vector VE2 | Related to X7, HP28 |
| MA8 | Modifiers on count transitions ignored as agents; `ss.change` on inflows ignored | Silent | 2 | interventions M-I4 | New |
| MA9 | Count-only infections taken before the counts move, against the spec | Spec drift | 2 | vector VE5 | Related to X13 |
| MA10 | A value by level of a dimension the state lacks takes level 0 | Silent | 3 | human MH2, vector VE3 | Related to L11, D4 |
| MA11 | No weighted count (biting weights, species) | Vocabulary | 3 | human MH2, vector VE6 | Answers open question B2 |
| MA12 | A coded module's per-node EIR has no spelling for both solvers | Dead end | 4 | human MH3 | Related to N20, N21 |
| MA13 | `.count()` as agents, `.n` as an ODE; N = 0 in `init_post()` | Friction | 4 | vector VE8 | Related to L31, ST8 |
| MA14 | A product's coverage can't be used in a rate | Vocabulary | 4 | interventions M-I5 | Related to D23 |
| MA15 | No re-dosing for a product with function waning | Silent | 5 | interventions M-I3 | New |
| MA16 | ODE waning in a fixed 10 levels, 6% off | Accuracy | 5 | interventions M-I7 | Related to L14 |
| MA17 | A function parameter is evaluated about 3 times per use | Speed | 5 | interventions M-I8 | Related to FP34 |
| MA18 | Newborns from `ss.Births` all join the first patch | Silent | 6 | population P1 | Related to ST1 |
| MA19 | `ss.Births` reads a rate as a probability, `ss.birth` as a rate | Silent | 6 | population P3 | New |
| MA20 | Death-rate ages tied to the results' age dimension | Vocabulary | 6 | population P6 | Related to FP33, FP31, FP22 |
| MA21 | ODE aging at 1/width drifts a non-flat pyramid | Accuracy | 6 | population P7 | New |
| MA22 | `ss.Disease` changed meaning; the suggested `ss.Infection` crashes | Crash | 7 | framework A1 | Related to FP10 |
| MA23 | `ss.SIS` changed meaning; fails at the first step | Crash | 7 | framework A2 | Related to FP10 |
| MA24 | Analyzers and results run before the first step | Crash or silent | 7 | framework A4 | Related to N10, D22 |
| MA25 | `ss.diff` crashes on coded modules with parameters | Crash | 7 | framework A7 | Related to FP9 |
| MA26 | `calibrate()` runs 4 seeds for a deterministic coded ODE | Cost | 7 | framework A10 | Related to X9, R4 |
| MA27 | Typed values in `ss.by`/series converted per group per step | Speed | 8 | population P2, human MH6, vector VE9 | Related to X10, FP34 |
| MA28 | Counts stepped in Python, 80% of declared agents' time | Speed | 8 | vector VE9 | Related to N29, X27 |
| MA29 | `describe()` crashes on an ODE with `People(age_data=)` | Crash | 9 | population P4 | New |
| MA30 | Strata lost through affine results and `.frac` | Friction | 9 | human MH5 | Related to R18 |
| MA31 | `.by()` on a `MultiSim` can't plot | Friction | 9 | population P8 | New |
| MA32 | `people_results=False` crashes `ss.Births` without naming the flag | Friction | 9 | framework A13 | New (also v3) |
| MA33 | No neutral held product; one waning per product | Friction | 9 | interventions M-I9 | New |
