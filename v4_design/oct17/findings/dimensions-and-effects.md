# Dimensions, mixing, and effects: merged findings

This merges every finding on stratification and dimensions, routes and mixing, geography and movement, modifiers and conditions, products and deliveries, the Starsim 3 interventions and products, and demographics from the four reviews: F (`fable_review/03_dimensions_routes.md`, `04_effects_interventions.md`, the demographics part of `11_sim_people_modules.md`, and `12_user_experience.md` §4), F2 (`fable2_review/07-interventions-networks.md`, and findings 1 and 4 of `04-transitions-conditions.md`), O (`opus_review/notes/dimensions-and-mixing.md`, `effects-and-interventions.md`), and C (`codex_review/REVIEW.txt` F01, F02, F03, F06, F07, F13, F18). About 150 findings across the four collapse to the 45 below. The reviews agree closely on the shape of the area: one `stratify()` operator, `ss.modify` with a kept base value, the product/delivery split, and the `Delivery`/`Selector` mechanism that runs one declaration on agents and on groups are right and should not be cut, and the damage is at the edges, in guessed defaults that reinterpret what the user wrote, in a second (Starsim 3) intervention stack that does not interoperate with the first, and in several spellings for one idea. Where they differ is mostly on remedies, not facts. I re-ran every silent wrong answer that only one review raised (17 of them) and every factual disagreement (7); all 17 reproduced, with the same numbers where the setup was the same, while two claims in F (D10, D21) and one each in O (D37) and F2 (D4) did not hold as stated. The verdict: this area has the most single-review silent wrong answers of any part of v4, and the largest one, a modifier that does nothing when the parameter is a function of time (D1), was found by only one of the four reviews; fixing D1 to D10 is a few days of small changes and should come before any of the surface work.

All my checks are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/`, run with `/software/conda/envs/v4/bin/python` at `a20c4f7e`; each Status line names the script by its full path.

## Findings

### D1. A modifier on a parameter that is a function of time does nothing

**`ss.modify` and `ss.change` are silently ignored when the parameter is a function, an `ss.timeseries`, or an `ss.seasonal`.**

Raised by: C F01 only.

Kind: silent wrong answer.

Status: reproduced by C (`codex_review/reproductions/model.py`), and re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c1_modifiers.py`, `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c2_effects.py`). An SIR with `ss.modify(d.pars.beta, multiply=0.0)`, 10,000 people, 100 days: a scalar `beta=0.3` gives 0 infections; `beta=lambda sim: 0.3` gives 9,309 (ODE) and 9,293 (agents); `beta=ss.timeseries({0: 0.3, 100: 0.3})` gives the same 9,309 and 9,293. A seasonal beta with a lockdown, `beta=ss.seasonal(mean=0.3, amplitude=0.2, peak=0)` plus `ss.change(d.pars.beta, at=[0], by=[0.0])`, runs the full epidemic (9,570 infections) with no note. Cause: `Engine.value` returns at `solvers.py:847` for a callable before reaching `self.modified()` at `:857`.

Proposal: one return path in `Engine.value`: evaluate the raw value (scalar, callable, `by`, dict), convert it, then apply modifiers, for every form. Add the four forms times `ode`/`abm` to `tests/test_deliveries.py` as a conformance case. Seasonal transmission plus a timed policy is one of the most common models people write, which is why this is first.

Cost: a few lines; the callable result must not be cached across steps, which it already is not. Breaks: nothing.

### D2. A bare number for a time changes meaning with whether the sim has dates, and an out-of-range time is clamped silently

**In a dated sim a bare `30` is the year 30 and is clamped to the first step; in an undated sim `years=[5]` is step 5; a round after the end is dropped with no message.**

Raised by: O effects 2, F2 07 f1 and F2 03 other (`years=` means steps in an undated sim), O effects 22 (the spec says "a bare number is a year").

Kind: silent wrong answer.

Status: O and F2 reproduced; re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c3_deliveries.py`). Sim from 2020-01-01 to 2020-06-01, base 9,312 infections: `ss.change(beta, at=[30], by=[0.5])` gives 5,719, the same as `ss.modify(beta, multiply=0.5, start=30)`, because the year 30 is before the start and becomes step 0, so the halving runs from day 0; the intended `at=[ss.days(30)]` gives 8,551; `ss.vaccinate(..., years=[30])` delivers 5,000 doses at step 0; `years=[2030]` delivers nothing and says nothing. O's version (`at=[30, 60], by=[0.5, 1.0]` has no effect) is the same clamp, with the second time overriding the first at step 0. The docstring examples of `ss.modify` and `ss.change` (`start=30`, `at=[30, 60]`) are written for undated sims and become wrong when copied into a dated one.

Proposal: a time outside the sim's span is an error naming the time, the span, and the likely fix ("did you mean `ss.days(30)`?"), except `stop=` after the end, which means "never stops". Together with D34 (one schedule keyword, `at=`), the rule becomes: a bare number is a time in the sim's unit everywhere, and a calendar year is written as a date.

```python
# before, in a sim from 2020-01-01: silently halves beta from day 0
ss.change(sir.pars.beta, at=[30], by=[0.5])
# after: E5xx TimeOutsideSim: 30 is the year 30, outside 2020-01-01 to 2020-06-01; did you mean ss.days(30)?
ss.change(sir.pars.beta, at=[ss.days(30)], by=[0.5])
```

Cost: a few lines in `connectors.to_ti()`; some tests that rely on clamping need `start=None`. Breaks: models that put a round before the start on purpose (none found). The deeper rule (bare numbers never read as years) is the time area's decision; this finding is only the delivery/modifier half.

### D3. A dimension that only a transfer fills starts half full

**`stratify(vaccinated=[False, True])` splits the population equally, so the spec's "vaccination as a stratification plus a transfer" example starts half the population vaccinated and differs from `ss.vaccinate` by a factor of about four.**

Raised by: F 04 §7 and B2, F 12 §4, O effects 6, F2 07 other (the open vaccination question); F README §3.

Kind: silent wrong answer.

Status: reproduced independently by F (n_I at day 11: 355 against 1,319 for `ss.vaccinate`) and O (776 infections against 3,285); not re-run, since three reviews agree on the mechanism. `describe()` prints `[guessed: equal across levels]`; `Delivery.dims()` builds its own `has_<product>` dimension with `split=[1, 0]` (`interventions.py:672`), so the product path is right and only the user-declared dimension is wrong.

Proposal: a dimension with no `split=` and no population data starts everyone at its first level when it is boolean (`[False, True]`) or when a transfer or delivery moves people into it, printed as the guess; "equal across levels" stays for other plain dimensions. Separately, D33 removes `ss.transfer(..., coverage=)` as the documented vaccination form.

Cost: a few lines in `Sim.init_strats`. Breaks: a model that relied on 50/50 for a boolean dimension must write `split=[0.5, 0.5]`.

### D4. A dimension's name decides whether places mix

**`region=`, `patch=`, `district=`, `node=`, `county=` are isolated places; `place=`, `location=`, `herd=`, `province=` are plain dimensions with homogeneous mixing; `resistance=`, `variant=`, `genotype=` are strains with their own force of infection and a 50/50 split of initial infections. `describe()` prints the kind but does not mark it as guessed.**

Raised by: F 03 U1 and S1, O dims 3, F2 01 (cut the strain-name list); F README §6 table. F2 07 "things that are good" praises "`kind` guessed and printed".

Kind: silent wrong answer.

Status: reproduced by F and O with the same numbers; re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c8_kind.py`). Seeding 10 infections in the first of two levels, 200 days: `region` and `patch` give infections by level `[4693, 0]`; `place` and `location` give `[4693, 4702]`; `resistance` becomes a strain dimension applying to I and R, `[9396, 0]`. The `describe()` line is `dimension region (space): a, b; applies to every state; mobility: none [guessed: equal across levels]`: the split is marked guessed, the kind is not. The mechanism is the ten-name dict at `transitions.py:2430`. Every vignette and test that used a name outside the list wrote `kind='space'` by hand, and one (`atomica/hiv_dyn`) wrote `sex=[...], kind='space'` to get per-sex counts (see D17).

Proposal (F and O agree): the kind comes from the value. `age` stays by name. Space when the levels are an `ss.geo` or `mobility=` is given. Strain when `immunity=` is given or the value is a strain marker. Everything else is plain. Keep `kind=` as the explicit override (O) or drop it (F); see the disagreement below.

```python
# before: the name does the work, invisibly
sir.stratify(region=['north', 'south'])                  # isolated
sir.stratify(location=['north', 'south'])                # mixed
# after: the value does the work, visibly
sir.stratify(region=['north', 'south'])                  # plain, like any other dimension
sir.stratify(region=ss.geo(names=['north', 'south']))    # a place: isolated unless mobility= says otherwise
```

Cost: delete the name list and the two "pass kind=" errors. Breaks: `patch=[...]` meaning isolated patches (three vignettes, which already write `kind='space'` or can write `ss.geo(names=...)`).

### D5. The sim-level `stratify` is a second parser that turns options into dimensions and skips the checks

**`ss.Sim(stratify=dict(age=..., states=['S', 'I']))` creates a dimension called `states`; `immunity=` and `inplace=` likewise; a typo in a sim-level `adjust` runs silently where the disease-level call raises `E301`; a sim-level `strain` applies to `S`, the disease-level one does not.**

Raised by: F 03 U2, B1, B2, C4, O dims 4; F2 03 table calls the sim/disease split "principled" (about the concept, not the parser).

Kind: silent wrong answer.

Status: reproduced by F and O; re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c16_simstrat.py`): dimensions `[('age', ['0-14', '15-64', '65+']), ('states', ['S', 'I'])]`; `adjust={'recovry': ...}` at the sim level gives no error, at the disease level `E301 ... Its transitions are infection, recovery`. Cause: `Sim.init_strats` pops five keys (`sim.py:552-560`) and `Engine.init_adjust` has no "unknown name" branch (`solvers.py:989-1004`). O adds that stratifying one disease stratifies the whole sim anyway (`sim.strats` is global), so the sim/disease distinction is only about which module's transitions `adjust` names.

Proposal: one parser, `Strat.from_kwargs()`, used by `Graph.stratify`, `sim.stratify`, and `ss.Sim(stratify=)`, with the same validation everywhere; an unknown keyword is an error, not a dimension; a sim-level `adjust` names `'flu.infection'` or an unambiguous bare name, and an unknown one is `E301` listing every module's transitions. State the real rule in 05: a dimension belongs to the sim; where it is declared decides only the default `states=` and which transitions `adjust` names. F's alternative (sim level takes levels only) is simpler to explain but breaks `where=` at the sim level, used once in `test_dimensions.py:508`; O's one-parser version keeps everything and removes code, so I prefer it.

Cost: removes `sim.py:552-560` and a duplicated `E305`/`E1102`; low. Breaks: nothing that works today.

### D6. A routine delivery of a waning product at full coverage re-vaccinates everyone every step

**With no `coverage=`, `ss.vaccinate(ss.Vaccine(efficacy=0.9, waning=...), start=0)` gives ten times the doses of `coverage=0.99` and a hundredth of the infections.**

Raised by: O effects 3 only.

Kind: silent wrong answer.

Status: reproduced by O; re-run by me with identical numbers (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c3_deliveries.py`): ODE 104,211 doses and 73 infections against 9,900 doses and 9,196 infections at `coverage=0.99`; agents 104,042 and 39 against 9,908 and 9,160. Cause: `Delivery.remembers` (`interventions.py:635-638`) skips the `offered_<delivery>` record when coverage, capacity, and `when=` are all absent and the product is held, but a waning product stops being held, so people become eligible again the next step.

Proposal: `remembers` is also true when the product can be lost (`self.holds and self.product.loses`). Re-vaccination after waning is a schedule (`interval=`) or an explicit eligibility the user writes.

Cost: one line; one more `offered_` dimension in compartmental runs of waning products. Breaks: nothing correct.

### D7. `set=` and `add=` on `rel_sus`, `rel_trans`, or a transition act on a multiplier of 1, not on the base

**`ss.modify(sir.rel_sus, set=1.0)` on a base of 0.5 leaves 0.5; `add=0.5` gives 0.75, not 1.0.**

Raised by: O effects 8 only.

Kind: silent wrong answer.

Status: reproduced by O; re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c2_effects.py`), agents, a custom module setting `rel_sus = 0.5` for everyone: `multiply=0.5` gives effective 0.25 (correct); `set=1.0` gives 0.5 and 4,297 infections, identical to no modifier; `add=0.5` gives 0.75 and 8,292 infections (the rule says 1.0, which gives 9,293). Cause: `Engine.mult_raw` evaluates `combine(mods, 1.0, self)` (`solvers.py:963`) and the engine multiplies the result into the base. The combination rule in 07 and in `modify`'s docstring holds only for parameters.

Proposal (O): restrict `set=` and `add=` to parameters and raise for per-person states and transition rates ("`set=` on `sir.rel_sus` would replace each person's own value; use `multiply=`"). That shrinks the surface and removes the wrong answer; the alternative, passing the base into `combine`, costs an array per modified quantity per step and serves no use found in any vignette.

Cost: a check in `modify.init_pre`. Breaks: nothing that currently gives the documented answer.

### D8. `ss.change` ignores `start=`, `stop=`, and `when=`, and differs from its own long form before the first time

**`ss.change(beta, at=[0], by=[0.3], when=lambda sim: False)` and `..., stop=10)` both apply for the whole run; `ss.change` holds the base before its first time while the equivalent `ss.modify(multiply=ss.timeseries(...))` holds the first value.**

Raised by: O effects 4 and 9; F2 07 other (`ss.change` is `ss.modify(multiply=ss.timeseries(..., mode='constant'))`), F2 01 (sugar), O surface (merge into `ss.modify`).

Kind: silent wrong answer.

Status: O reproduced; re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c2_effects.py`): 492 infections with no extra keyword, with `when=False`, and with `stop=10`, against a baseline of 9,309. Cause: `change.active` is `return self.sim.ti >= self.tis[0]` (`connectors.py:384-385`) and never calls `modify.active`.

Proposal (O effects 9): keep the name `ss.change` (it is the Covasim one-liner and reads well), but make it a function returning an `ss.modify` whose amount is a step function that is 1 before the first time. The subclass, its `active`/`current` overrides, and its own `describe`/`to_json` go, and `start=`/`stop=`/`when=` work for free.

```python
# the user writes the same thing
ss.change(sir.pars.beta, at=['2020-03-15', '2020-06-01'], by=[0.5, 1.0])
# which now returns
ss.modify(sir.pars.beta, multiply=ss.timeseries({'2020-03-15': 0.5, '2020-06-01': 1.0}, mode='constant', before=1.0))
```

Cost: `ss.timeseries` gains `before=`; about 60 lines removed. Breaks: `isinstance(x, ss.change)` (internal only).

### D9. A test repeated more often than its delay never returns a result

**With `delay=ss.days(3)` and `interval=1`, every result is overwritten before it arrives: 2,000 tests, 0 diagnosed, under both solvers.**

Raised by: C F06 only.

Kind: silent wrong answer.

Status: reproduced by C (`codex_review/reproductions/interfaces.py`); re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c4_screen.py`): 100 infected people, perfect test, 20 days: interval 1 gives 2,000 delivered and 0 diagnosed (agents and ODE); interval 3 gives 700 and 600; interval 4 gives 500 and 500; no delay gives 2,000 and 2,000. At `coverage=0.5` most results get through (98 of 100 treated), so the loss scales with coverage and frequency and is easy to miss. Cause: `ti_result` is one slot per person, reset on each new test (`interventions.py:1136`), and the compartmental `pending_` dimension does the same.

Proposal: people with a result pending are not eligible for the same test again (what programs do, and the simplest rule); `describe()` says so. C's alternative, a result per specimen, is more faithful but needs an event queue per person and a multi-level pending dimension per test round.

Cost: one condition in `Delivery.offer` and the group equivalent. Breaks: nothing correct.

### D10. Networks under a compartmental solver: one is right, one ignores the graph, one ignores duration, one is refused, and the spec describes none of them

**`ss.StaticNet(graph=g)` under `solver='ode'` uses `n_contacts=10` whatever the graph; `ss.PartnerNet` runs as homogeneous mixing at `n_contacts` per day, ignoring partnership duration; `ss.MFNet` raises `E804`; 06 promises a partner-change-rate approximation for both sexual networks.**

Raised by: C F02 (StaticNet, only C), O dims 12 and F2 07 f10 (PartnerNet), C F18, F 03 S1 and S2, O dims 12, F2 03 other, F2 07 f10 (MFNet and the spec paragraph); F 03 Keep ("refusing rather than approximating sexual networks").

Kind: silent wrong answer (StaticNet, PartnerNet); spec drift (MFNet).

Status: re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c11_routes.py`, `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c12_e403.py`, `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c14_partner.py`). StaticNet on a 1,000-node ring (degree 2), `beta=0.05`, 100 days: ODE 983 infections, agents 6; on a degree-30 regular graph, ODE 983, agents 990; both ODE runs print `contacts: 10.0` (`solvers.py:149-160` reads `net.pars.n_contacts`). PartnerNet SIS prevalence, agents against ODE: `dur=5` 0.670 and 0.714; `dur=500` 0.168 and 0.714, the same numbers O reported. `solver='ode'` on PartnerNet runs without `approx=True`; `auto` picks agents. MFNet is `E804` with or without `approx=True`. F's Keep note says sexual networks are refused under an ODE; that holds for MFNet only.

Proposal: one rule for every network under a compartmental solver: it runs only if it has a stated approximation, which is printed, and is otherwise `E804`. RandomNet: mean contacts, as today. StaticNet: the supplied graph's mean degree, or `E804` (C). PartnerNet and MFNet: `E804` until O's form (formation rate `n/D` per person, per-partnership transmission `1 - exp(-beta * D)`) is implemented and tested against agents. Replace 06's sexual-network paragraph with what the code does.

Cost: a few lines in `Mixing.init`; spec text. Breaks: explicit `solver='ode'` runs of PartnerNet, which are wrong today.

### D11. The default population changes when you stratify, and named data does not bring its population

**As agents, ages are uniform 0 to 60 unstratified and uniform 0 to 100 once `stratify(age=...)` is added; `ss.contacts('Kenya')` with an age stratification models a population 35% over 65; `ss.geo('Kenya')` leaves `total_pop` at 10,000.**

Raised by: O dims 5 only; F2 README guesses table (ages 0 to 60 make `eligible=ppl.age > 65` deliver nothing).

Kind: silent wrong answer.

Status: reproduced by O; re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c9_pop.py`): unstratified agents max age 60.0, 0% over 65, 25% under 15; with `stratify(age=[0, 15, 65])` max 100.0, 35% over 65, 15% under 15. With `ss.contacts('Kenya')` the `describe()` line is `age (age): 0-14, 15-64, 65+ ... [guessed: ages uniform from 0 to 100]` (Kenya is about 40% under 15 and 3% over 65). The guess is printed, but a user who named Kenya has reason to think Kenya's people are in use.

Proposal (O): one default age distribution for every solver, stratified or not (a declining pyramid, such as the UN world average already in `starsim_data`, printed as a guess); and one named country in a sim, through any loader, means that country's people: `ss.contacts('Kenya')` brings `ss.people('Kenya')`'s age split when no people are given, and `ss.geo(...)` with populations sets `total_pop` to their sum.

Cost: small; changes default numbers in existing tests. Breaks: results that depend on the uniform default.

### D12. Aging is on with no deaths, and the compartmental aging rate assumes a flat pyramid

**With a real pyramid and no demographics, one year of aging grows the 65+ group by 40% in the ODE and 13% as agents.**

Raised by: O dims 6 only; F 11 §6 touches the aging default ("three triggers").

Kind: silent wrong answer (and an ODE/agent mismatch).

Status: reproduced by O; re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c9_pop.py`), `people=ss.people('Kenya')`, `beta=0`, 365 days: ODE 65+ 1.53M to 2.14M (+39.6%), agents 1.51M to 1.70M (+12.7%). The ODE ages people out of 15-64 at 1/50 per year (`init_movers`), right only for a uniform distribution within the band.

Proposal: (a) aging defaults on only when the sim has births or deaths (O), printed either way; (b) in a compartmental run, the rate out of band `i` comes from the split's density at its upper edge rather than `1/width` (O, about ten lines; it keeps a stationary pyramid stationary). (b) is uncontroversial; (a) reverses a recent decision (`implementation.md:101`) and is a decision for Cliff.

Cost: (b) ten lines; (a) one condition. Breaks: (a) long runs that rely on aging without demographics, which then need `use_aging=True`.

### D13. Partial cross-immunity between strains runs under `auto` as an ODE that is off by up to 2.9 times and reverses which strain dominates

Raised by: O dims 11 only.

Kind: silent wrong answer.

Status: reproduced by O; re-run by me with identical numbers (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c11_routes.py`): two strains with waning and `immunity=[[1, 0.3], [0.3, 1]]`, cumulative infections per person by strain: `auto` (which picks `ode`) `[2.066, 2.675]`, agents `[1.285, 0.927]`; without `immunity=` the two agree (`[0.159, 1.51]` and `[0.14, 1.517]`). It is a listed known issue, and a note is printed, but `auto` is supposed never to drop individual-level structure.

Proposal: (a) under `auto`, a strain dimension with partial `immunity=` runs as agents, with the reason printed; (b) later, an exact compartmental form tracking the set of strains each person has had as a hidden dimension of `2^k` levels (the history-based multistrain ODE), for `k <= 3`. (a) is one line and should come first.

Cost: (a) one line in solver choice. Breaks: nothing correct.

### D14. A contact matrix is aggregated to the model's age groups by band width, not by population, and its inputs are not validated

**`Mixing.age_matrix` weights each source band by `overlap/width` and cuts the open last band at 100, so the aggregated matrix is the same whatever the population; `ss.contacts` accepts negative entries and reversed edges.**

Raised by: C F03 and O dims 14 (rebinning); C F07 (validation, only C).

Kind: silent wrong answer.

Status: C showed the same aggregated matrix for a population 90% and 10% young (`codex_review/evidence/routes-output.txt`); O measured Kenya's adult and elderly rows 6-10% low against population weighting. Not re-run for the rebinning (two reviews, same mechanism). For validation, re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c15_misc2.py`): `ss.contacts([[-5, 0], [0, 1]], age=[0, 20])` runs under `ode` with no error; C showed `age=[20, 0]` makes a group labeled `20-0`.

Proposal: weight by the sim's age split interpolated to the bands (the same `np.interp` `init_strats` uses for named populations), falling back to width only when the split is itself the uniform guess; define the open band explicitly; reject negative or non-finite entries and unordered edges at construction, naming the offending value.

Cost: small. Breaks: default numbers for every `ss.contacts` model (they become right).

### D15. As agents, a `sex` dimension has nothing to do with the agents' sex

**`stratify(sex=['female', 'male'])` draws a new random property independent of `people.female`, so `MFNet` pairs on one and results are reported by the other.**

Raised by: O dims 2 only.

Kind: silent wrong answer.

Status: reproduced by O (with `adjust={'infection': [0, 1]}`, results say 0 women infected while the network infected 51 women and 51 men); re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c7_dims.py`): the two agree for 49.1% of agents. In a compartmental run `sim.py:279` does treat `sex` as `female`, so the solvers disagree on what `sex` means.

Proposal: a dimension called `sex` maps onto `people.female` by label, with no new attribute and no draw, and `split=` sets `p_female`. Keep the aliases of built-in people properties (`age`, `sex`) in one table both engines read.

Cost: small. Breaks: nothing correct.

### D16. The spec's gravity example sends everyone away all the time

**`ss.gravity(k=0.5)` on `ss.geo('Kenya', level=1)` makes every county's residents spend 0% of their time at home, silently.**

Raised by: O dims 9 only. F2 07 lists `sir.stratify(patch=kenya, mobility=ss.gravity(k=0.5))` as a good surface.

Kind: silent wrong answer.

Status: reproduced by O; re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c10_gravity.py`): `k=1` and `k=0.5` give mean share of time at home 0.0 with all 47 rows capped by `Mobility.time_share` (`transitions.py:2693-2703`), and a diagonal of `-0.0`; `k=1e-3` gives 0.15 at home; `k=1e-5` gives 0.976. `k` has units of `distance^c / pop^(a+b-1)`, so no default fits both a three-town test (`k=2e-4`) and a country.

Proposal (O): gravity gives the shape (`a`, `b`, `c`), and `away=` gives the amount, with a printed default; keep `k=` for calibrated users, warn whenever the cap binds (naming how many places), and clip the diagonal at 0. The surface F2 praises stays one line.

```python
# before: silently 100% away
sir.stratify(patch=kenya, mobility=ss.gravity(k=0.5))
# after: 10% of time away, split by gravity
sir.stratify(patch=kenya, mobility=ss.gravity(away=0.1))
```

Cost: the warning is two lines; `away=` changes the signature. Breaks: the spec's examples, which are wrong today.

### D17. Counts in a rate are local only within a place, the written-out `/N` form silently ignores the contact matrix, and the local count has no spelling of its own

**`I.n` in a rate is the count in the same place for a space dimension and the total for every other dimension; the spec says it is aligned on every shared dimension and that `I.n.sum('age')`/`I.n.select(...)` exist (they do not); `S >> I @ beta * I.n / N` under `ss.contacts` is literal, so the matrix does not apply, and nothing says so.**

Raised by: F 03 U3, U7, U8, S1, S3 (only F for the `/N` case); F2 README guesses table and decision 20 (a separate spelling, `W.here`); O dims 18 (`describe()` prints `I.n / N` without `[place]`).

Kind: silent wrong answer (the `/N` case); spec drift (alignment).

Status: F reproduced alignment; re-run of the `/N` case by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c13_demog.py`): `beta=0.05`, `ss.contacts('Kenya')`, three age groups, 100,000 people, 100 days: bare `I` gives 98,662 infections, `I.n / (S + I + R).n` gives 99. The vignette `atomica/hiv_dyn/model.py:71` writes `sex=[...], kind='space'` only to get a per-sex count.

Proposal: state the rule in 06 as the code has it ("a count is the total, except within a place"); add one explicit spelling for the count at the person's own level of a dimension (`I.n.by('sex')` in F's form, or F2's `W.here` for place), implemented where `local_count` already handles space; when a rate uses a people state's `.n` and a route other than homogeneous mixing is present, print one note ("`infection` uses `I.n` literally, so `contacts` does not apply to it; write `I` for exposure through the routes").

Cost: one method, one note, one paragraph. Breaks: nothing.

### D18. `ss.Treatment` is two products chosen by the type of `efficacy`, a held treatment goes to everyone by default, and `ss.Vaccine()` builds and then crashes

Raised by: O effects 13 and 14, F2 07 f4 and f2 (the `Vaccine()` crash), F 04 B4 (`ss.Dx()`).

Kind: silent wrong answer (held treatment to everyone); crash (`Vaccine()`).

Status: O and F2 reproduced; re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c3_deliveries.py`, `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c6_misc.py`): `ss.treat(ss.Treatment(efficacy=dict(transmission=0.96)), coverage=0.5, interval=5)` delivers to all 10,000 people, susceptible and recovered included, while the curing `ss.Treatment(efficacy=0.9)` reaches 2,353 (infected only); `ss.vaccinate(ss.Vaccine(), ...)` raises `TypeError: float() argument ... 'NoneType'` at run time (`products.py:264`). A user who writes `efficacy=dict(recovery=0.5)` expecting "cures half" gets "halves the recovery rate while held", with no error (F2).

Proposal (F2 and O agree): separate the two by keyword: `cure=` (a probability, one-shot) and `efficacy=` (effects while held), allowed together; a `Treatment` is for the infected by default whichever it does; `efficacy=` is required on `ss.Vaccine` at construction; `closest()` returns nothing when nothing is close, so `E505` stops listing every key as "did you mean".

```python
# before
ss.Treatment(efficacy=0.95)                                   # cures
ss.Treatment(efficacy=dict(transmission=0.96, mortality=0.9)) # held; given to anyone
# after
ss.Treatment(cure=0.95)
ss.Treatment(efficacy=dict(transmission=0.96, mortality=0.9)) # held; given to the infected by default
ss.Treatment(cure=0.3, efficacy=dict(transmission=0.9))       # both, not possible today
```

Cost: small; `Delivery.give()`'s three-way branch becomes two. Breaks: `efficacy=<number>` on a treatment (alias it to `cure=` with a note for one release).

### D19. A follow-up delivery silently ignores its own schedule

**`ss.treat(drug, years=[100], start=200)` passed as `on_positive=` delivers whenever its parent does.**

Raised by: F2 07 other only.

Kind: silent wrong answer.

Status: reproduced by F2; re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c5_sched.py`): delivered on steps 0, 7, 14, 21, ..., 1,224 in all, with no note. Cause: `step()` returns early for triggered deliveries (`interventions.py:894`).

Proposal: raise at `init_pre` when a follow-up has `start`, `stop`, `years`/`at`, `interval`, `on`, or `deliver`, saying a follow-up runs when its parent's result arrives.

Cost: a few lines. Breaks: nothing correct.

### D20. `capacity=` means per round in a campaign and per unit time in routine delivery

Raised by: F2 07 other only.

Kind: silent wrong answer.

Status: reproduced by F2; re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c5_sched.py`): `capacity=100` with `dt=0.5` gives 100 per round in a campaign and 50 per step in routine delivery (`interventions.py:1060-1061`).

Proposal (F2): a bare number is a count per round or per step; a rate is typed (`capacity=ss.perday(500)`); a bare number in routine delivery, the ambiguous case, is an error asking for one or the other.

Cost: small. Breaks: routine deliveries with a bare `capacity=` (they get an error with the fix).

### D21. Per-route `beta`: an unknown key is silent, a dict `beta` crashes every compartmental run, two `ss.contacts` layers collide as agents, and routes carry a second `beta`

**The spec's flagship layered example (home/school `ss.contacts` with `beta=dict(home=..., school=...)`) fails under both solvers; a key that names no network is ignored; `MixingPool(beta=)`/`contacts(beta=)` multiply the disease's `beta` and fold into the printed contact rate.**

Raised by: O dims 1 and 8 (last paragraph), F 03 B3 and S1, F2 07 other (route-level beta).

Kind: silent wrong answer (unknown key, route beta); crash (the rest).

Status: re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c11_routes.py`, `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c12_e403.py`): the layered example under `auto` raises `TypeError: unsupported operand type(s) for *: 'dict' and 'float'`, and so does a dict `beta` with a single `RandomNet` under `ode`; as agents with a scalar `beta`, `AttributeError: Module pool:0.0-5.0->0.0-5.0 already added`; `beta=dict(home=..., school=..., wrok=0.1)` runs with no error. F's version (a misspelled key passes silently) holds at `sim.init()` only: at the first step of `run()`, `needls` against a network `needles` raises `E403 ... there is a network called "needles" with no value`, so only a key that names no network at all is silent.

Proposal: resolve per-route `beta` in one place shared by both engines (`term.contacts * beta[term.name]`); check keys in both directions at `sim.init()`, raising `E403` for a key with no route; prefix sub-pool names with the parent's (moot after D32); for declared models drop `beta=` on routes (O) or warn on a route `beta != 1` (F2), pointing to `beta=dict(route=...)` on the transmission. Add 06's layered example to `tests/test_mixingpools.py` under `ode` and `abm`.

Cost: small. Breaks: route-level `beta` in declared models (none in the vignettes).

### D22. An intervention's `step()` is first called at `ti == 1`, so Starsim 3 code that initializes at `ti == 0` never fires

Raised by: O effects 17 only.

Kind: silent wrong answer (for users porting Starsim 3 code).

Status: reproduced by O; re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c15_misc2.py`): a recording intervention sees `ti` = 1, 2, 3, 4, 5 in a declared sim and in an `ss.SIR` sim alike. It follows from results by time point (the loop skips the first point), which is a sound choice; a comparison with 0 cannot be detected to warn about.

Proposal: say it in the migration guide and the `Intervention` docstring (`step()` first runs with `ti == 1`; one-time setup goes in `init_post()`). The loop order is the execution area's.

Cost: docs. Breaks: nothing.

### D23. The v4 effects and deliveries do not reach any Starsim 3 disease, and the errors blame the target

**`ss.modify`, `ss.change`, `ss.vaccinate`, `ss.screen`, `ss.treat` fail on `ss.SIR`, `ss.SIS`, `ss.SEIR`, and every `starsim.library` disease (HIV included), so the flagship `ss.modify(hiv.rel_trans, multiply=2.0, where=syphilis.active)` cannot be written with the library's own HIV; the v3 stack still works on them, so which intervention language works depends on how the disease was written.**

Raised by: F 04 §1, O effects 1, F2 03 f10, F2 07 f7; F README §1, F2 README, O README §1.

Kind: first-week friction.

Status: reproduced by all three; not re-run. Messages: `E504 ... not 0.1 (peryear). A plain number has no name`, `E504 ... did you mean randomnet.pars.n_contacts?`, `E506 ... the sim has no declared diseases. Use disease=.`

Proposal: two halves, and both reviews that address order agree on it. (a) Make target lookup and the `Selector` accept any module, at least as agents: a parameter of any module is `mod.pars[name]`; a v3 `Infection`'s `rel_sus`/`rel_trans` wrapped as `ss.ModifiableArr` (as `solvers.py:3176` already does for declared diseases); a cure on a v3 disease goes through its own `clear_infection()` hook or errors naming it. (b) Redeclaring `ss.SIR`/`SIS`/`SEIR` in the class form (the language area's finding) removes most of the need for (a) for the built-ins. Until either lands, the errors should say "`sir` is a class-based disease (`ss.SIR`); `ss.modify` works on diseases built from transitions", and `parse_target` should not call a typed rate "a plain number". This is the prerequisite for D31.

Cost: real work in `Selector`, `parse_target`, and the v3 engine path. Breaks: nothing.

### D24. An error inside a `when=` function is reported as "modify object has no attribute active", which hides the spec's own trigger example failing

Raised by: F 04 §9 and B1, O effects 7 and 22.

Kind: first-week friction.

Status: both reproduced; not re-run. `Module.__getattr__` (`modules.py:300-307`) re-raises any `AttributeError` from a property as a missing attribute. The `SchoolClosure` example in 08 fails because a custom `ss.Intervention` is named `schoolclosure`, not after its variable, while v4 objects are named after their variable; and without `compartmental = True` the class turns the sim into agents (D26).

Proposal: wrap user callables (`when=`, function amounts, `deliver=`, conditions) so an exception is re-raised naming the function and the module ("the when= function of 'effect' raised AttributeError: 'ndict' object has no attribute 'closure'"), and fix `__getattr__` to let a property's own error propagate. Name every module from its variable, or none. Fix 08's example (`name='closure'`, `compartmental = True` with a comment).

Cost: small. Breaks: nothing.

### D25. Lazy conditions need a throwaway `ss.People(1)`, are narrower than the spec says, and do not mix with rate expressions

**A condition on age is written with `ppl = ss.People(1)`, and passing that object to the sim gives a one-agent model scaled to the population with nothing delivered; arithmetic on an uninitialized array raises `IndexError`; `sir.rel_sus > 0.5` is a `TypeError`; `where=Q.n < 20` and `rate * (ppl.age > 5)` crash.**

Raised by: O effects 15, F 04 §6 and B3, F 11 §4, F2 07 other, F2 04 f1, C F12.

Kind: first-week friction.

Status: reproduced across the reviews; not re-run except one point: `ss.people` already exists as a function (the named-population loader, `ss.people('Kenya')`), checked in `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c6_misc.py`, so O's proposed placeholder name collides with it.

Proposal: (a) a class-level lazy form, `ss.People.age < 5` (F 11 §4), which needs no instance and no new name; document `sir.age == '15-24'` for a model stratified by age (F); a sim given `people=ss.People(1)` with a larger `total_pop` warns. (b) The spec's lazy rule lists exactly what is lazy: comparisons, `&`, `|`, `~`, `isin` on people attributes, states, transitions, and dimensions; arithmetic is not lazy and raises a clear error (F, C); `Target` raises "a float quantity cannot be a condition" on comparison. (c) At the seam between conditions and rates, at minimum a guard on each side naming the other form; F2's larger proposal (one tree: `Cond` leaves become `Expr` leaves) belongs with the language area's expression findings.

```python
# before
ppl = ss.People(1)   # used for nothing else
eligible = (ppl.age > 65) & ~sir.vaccinated
# after
eligible = (ss.People.age > 65) & ~sir.vaccinated
```

Cost: (a) class descriptors on `People`; (b) spec text and one `__gt__`; (c) two guards. Breaks: nothing.

### D26. A plain function or a custom intervention makes the whole sim agent-based

**`interventions=knock` (a two-line function) or the spec's `SchoolClosure` class switches an ODE model to 10,000 agents, decided by the class attribute `compartmental`, not by trying, and the printed reason names the class, not the function.**

Raised by: O effects 10, F 04 §9, F 11 §3 (for a declared disease with a `step()` override); C D7 argues against probing.

Kind: first-week friction.

Status: reproduced by O and F; not re-run. This overlaps the solver-choice area; recorded here because every custom intervention meets it.

Proposal: either the spec says custom modules are agent-only unless marked `compartmental = True` and the user docs show the flag (cheap and honest; C's position that probing proves little supports it), or the module is tried on groups at `sim.init()` as `12-backends.md` says (better for users, but `step()` mutates state, so it needs a throwaway copy). Either way the reason names the function or module.

Cost: docs, or the try-on-a-copy machinery. Breaks: nothing.

### D27. Follow-up deliveries lose the user's names, and products appear in the schedule and results with nothing in them

**`confirm = ss.screen(pcr, ...)` passed as `on_positive=` becomes `screening_positive` and its own follow-up `screening_positive_positive`, so `sim.results.confirm` does not exist; `rapid`, `pcr`, `drug` appear in `sim.results` with only `timevec` and in `describe()`'s schedule.**

Raised by: F 04 §8, B5, C6; O effects 12 and 16 (names read like counts of positives).

Kind: first-week friction.

Status: reproduced by F and O; re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c6_misc.py`): interventions `['screening', 'screening_positive', 'screening_positive_positive']`, results keys include `rapid`, `pcr`, `drug`.

Proposal: name follow-ups by variable too (walk `on_positive`/`on_negative` in `_name_modules`), falling back to `<parent>_positive` only for inline ones; keep products out of `sim.results` and the schedule (see D35 for making them not modules at all); an unnamed inline product takes its delivery's name rather than `test_1` with a warning.

Cost: small. Breaks: code that reads `screening_positive` where a variable name exists.

### D28. Four ways to add births and deaths, and three of them silently turn an ODE into 10,000 agents

**`ss.birth`/`ss.death` in a disease (every solver), `ss.Births`/`ss.Deaths` modules (agents only), `birth_rate=`/`death_rate=` on `ss.Sim`, and `demographics=True`; the module rates are per 1,000 (`ss.Births(birth_rate=ss.peryear(20))`) and the declared ones per person (`ss.birth(rate=ss.peryear(0.02))`); results land under different names; `ss.Sim(demographics=[ss.birth(...)])` is a `TypeError`.**

Raised by: F 11 §6, F2 07 other, F2 01 other; F README §6 table.

Kind: first-week friction.

Status: re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c13_demog.py`): a declared SIR runs as `ode`; with `birth_rate=20, death_rate=10` it runs as `abm` with 10,000 agents; `demographics=True` gives `abm`; `demographics=[ss.birth('-> S', rate=...)]` raises `TypeError: Was expecting demographics entry 0 to be ... Demographics or Module`; `ss.birth` inside the disease stays `ode`. `ss.Births(birth_rate=ss.peryear(20))` and `ss.birth(rate=ss.peryear(0.02))` give the same population after a year (20,424 and 20,425): the same model in two units.

Proposal (F and F2 combine): accept `ss.birth`/`ss.death` in `ss.Sim(demographics=[...])`, so births are declared once for the population rather than once per disease (F2); remove `birth_rate=`/`death_rate=`/`demographics=True` from `SimPars`, or make them build `ss.birth`/`ss.death` so a compartmental model stays compartmental (F); keep `ss.Births`/`Deaths`/`Pregnancy` as library modules for data-driven, agent-only demography; make the module rates per person by default with the UN loader supplying the `1e-3`.

```python
# before: silently 10,000 agents
ss.Sim(sir, birth_rate=20, death_rate=10)
# after: one declaration for the population, under every solver
ss.Sim(sir, demographics=[ss.birth('-> S', rate=ss.peryear(0.02)), ss.death(rate=ss.peryear(0.01))])
```

Cost: small for the `demographics=` change; the rate convention is breaking for anyone loading UN data by hand. Breaks: `birth_rate=` in three tutorials.

### D29. Delivery results: realized coverage averages to near zero, `deliver=` reports nobody eligible, and the spec's `a.diagnosed + b.diagnosed` does not exist

Raised by: O effects 16 and 18 only.

Kind: first-week friction.

Status: reproduced by O; not re-run (no silent change to the model, only to its reported summary). A weekly screen at `coverage=0.4` summarizes as `screening_coverage 0.059` because steps without a round count as 0; under `deliver=` `eligible` stays 0 while `delivered` is not; `symptomatic_testing.diagnosed + screening.diagnosed` from 08 raises `AttributeError`; after `ss.Sim(..., interventions=[screening])` the user's `screening.results` is empty because the sim copied it.

Proposal: `coverage` is NaN on steps with nobody eligible and its summary is total delivered over total eligible; `deliver=` counts its return as eligible; mark 08's sum as not yet available (or implement lazy delivery results); a template object's `.results` raises "the results are in sim.results.screening".

Cost: small. Breaks: plots that assume no NaN.

### D30. A network's parameter cannot be named as an object in `ss.modify`, and the error message hides the string form that works

**`ss.modify(school.pars.n_contacts, ...)`, the spec's spelling, is `E504 ... not ss.constant(v=10) ... A plain number has no name`; the string `'school.pars.n_contacts'` works, and accepts only `n_contacts` for a network although 07 says any parameter.**

Raised by: F 04 §5 and C4, F2 07 other, O effects 1 (the message).

Kind: first-week friction.

Status: reproduced by F and F2; not re-run.

Proposal: wrap network parameters in `ss.par` with a module back-reference, as declared diseases' are, so the object form works everywhere and the string form is only what the model spec writes; until then, the error names the path form. F2's `ss.modify(school, 'n_contacts', ...)` would be a third spelling; I would not add it.

Cost: a change to `Network.define_pars`, or one line for the message. Breaks: nothing.

### D31. Retire the Starsim 3 intervention and product vocabulary from `ss.`

**18 names and about 650 lines (`interventions.py:77-534`, `products.py:31-228`: `RoutineDelivery`, `CampaignDelivery`, `BaseTest`, `BaseScreening`, `BaseTriage`, `routine_/campaign_screening`, `routine_/campaign_triage`, `BaseTreatment`, `treat_num`, `BaseVaccination`, `routine_vx`, `campaign_vx`, `Dx`, `Tx`, `Vx`, `simple_vx`) sit beside v4's seven, used by no vignette, no library module, and no v4 test, and they are what the user guide teaches; `ss.Vx(efficacy=0.9)` returns a `Vaccine` that is not a `Vx`, `ss.Dx()` crashes; `ss.seasonality` overwrites `rel_trans` every step, the anti-pattern 07 exists to remove, and forces agents.**

Raised by: F 04 §1, C1, C5, B4; F2 07 f7; O effects 1, 11, 14; F 01; F README §1, F2 README §2, O README §1.

Kind: surface.

Status: reproduced by all three; re-run by me for the alias points (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c6_misc.py`): `isinstance(ss.Vx(efficacy=0.9), ss.Vx)` is `False`; `ss.Dx()` raises `AttributeError: 'NoneType' object has no attribute 'state'`. The reviews' counts differ (F "19 names, 755 lines", F2 "24 exports, about 700", O "about 530"); I counted 18 classes in the line ranges above, about 650 lines, keeping `Intervention` and `Product` as bases.

Proposal: after D23, move the 18 classes and `ss.seasonality` out of `ss.` into a compatibility module (`starsim.v3` or `starsim.legacy`) whose constructors warn with the v4 equivalent; no `Vx`/`Dx`/`Tx` aliases at `ss.`; `ss.seasonal`'s docstring shows `ss.modify(dis.rel_trans, multiply=ss.seasonal(...))`. Two v3 capabilities have no v4 form yet: `treat_num`'s queue (one `queue=True` keyword on `Delivery` if wanted) and `Dx(df=)`'s multi-outcome hierarchies (`ss.Test(results=...)` later); and the all-or-nothing vaccine (D40).

```python
# before
ss.routine_vx(product=ss.simple_vx(efficacy=0.9), prob=0.5, start_year=2005)
# after
ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.5, start=2005)
```

Cost: five doc pages and `tests/test_interventions.py` (10 tests); downstream STIsim and HPVsim, which a compatibility module keeps importable. Breaks: Starsim 3 scripts at `ss.` (one import line).

### D32. `ss.contacts` as the one way to give a contact matrix, built as a small matrix route rather than N² agent modules

**Five spellings of an age mixing matrix (`MixingPools(src={...: ss.AgeGroup}, ...)`, `LevelGroup`, `contacts('Kenya')`, `contacts(M, age=edges)`, `contacts(age=M)`, where `age=` means edges in one and a matrix in another); as agents `ss.contacts('Kenya')` builds 256 `MixingPool` modules with per-agent arrays (205 MB at 100,000 agents; 276 people states for a 4-age SIR) and `Agents.infect` has three pool paths, while the counts-based path already used with places gives the same answer three times faster.**

Raised by: F 03 U4, C1, B4; F2 07 f6; O dims 8 and 15.

Kind: surface (and codebase, performance).

Status: reproduced by F, F2, and O (O: attack by age `[0.896 0.996 0.984 0.876]` against `[0.895 0.997 0.986 0.879]`, 2.6 s against 0.8 s at 50,000 agents); not re-run. `MixingPools()` with no arguments raises `TypeError` while its docstring promises a default (F B4).

Proposal (F2 and O; F's C1 points the same way): `ss.contacts()` returns a small route holding `matrix`, the group definition (`age` edges or `dims`), `name`, and `ref`; both engines apply it through the counts-based `Mixing.term_exposure`, with or without places; `ss.MixingPool(n_contacts=c)` is its 1×1 case; `AgeGroup`/`LevelGroup` and the sub-pools leave the declared-model path, and `MixingPools(src=, dst=)` stays only for v3 class-based diseases. The edges of the matrix's own bands get their own keyword (O: `bands=`), so `age=` always means "a matrix over the model's age dimension".

```python
ss.contacts('Kenya', layer='school')
ss.contacts(M, bands=[0, 15, 65])   # the matrix's own age bands
ss.contacts(risk=M)                  # a matrix over a model dimension
```

Cost: a medium refactor of `networks.py` and `Agents.infect`; the user guide's mixing section is rewritten (it is v3's anyway). Breaks: `MixingPools(src=...)` in user code for declared models; `contacts(M, age=edges)` becomes `bands=`. It also fixes D21's name collision and makes the model spec able to rebuild the matrix (`deferred.md`).

### D33. Movement between levels has six spellings; make `ss.transfer` a transition and give the register a general home

**People change level by aging (generated), `ss.transfer(rate=)` (a `Transfer`, not a `Transition`, without the `p=` the spec gives it), `ss.transfer(coverage=)` (which returns an `ss.Delivery`, a module), `move=` on a transition, `ss.Mobility(kind='migrate')`, and `ss.Mobility(data=)`; `Mobility` itself holds four kinds and two meanings of `away=`, and `visit` is in the code and two vignettes but not in 06.**

Raised by: F 03 U6 and C3, F2 04 f4, F2 07 f8, O dims 10; F2 07 other (vaccination question), F 04 §7 (remove the delivery-keyword form).

Kind: surface.

Status: reproduced by F and F2; not re-run (structural).

Proposal, merging the parts the reviews agree on: (a) `ss.transfer` builds a `Transition` with `move=` (`ss.transfer(risk='low -> high', rate=r)` is `ss.Transition(I, I, rate=r, move=dict(risk='low -> high'))` over every state), which gives it `p=`, `stages=`, `where=`, a flow result, and modifier targeting, and deletes `Graph`'s separate transfer container (F2); (b) `ss.transfer` never returns a `Delivery`: the delivered move is `ss.vaccinate` (most uses) or `ss.Delivery(move=...)` (F); (c) keep `sir.stratify(patch=kenya, mobility=ss.gravity(...))` as the commuting surface, document `visit`, and mark `force` as a LASER-compatibility form (F and O); (d) where migration and the register live is the open part, below.

Cost: medium for (a); spec text for (c). Breaks: `ss.transfer(..., coverage=)` (one example each in 05 and 17).

### D34. One schedule keyword and one "who" keyword across `ss.change`, `ss.modify`, and deliveries

**"When" is spelled seven ways across four objects (`at=`, `years=`, `dates=`, `interval=`, `start=`/`stop=`, `when=`, `on=`); `years=` means steps in an undated sim; `on=sis.I` is exactly `eligible=sis.I, interval=1`.**

Raised by: F2 07 f1, F2 01 f5 and f6, F2 README 4; F 04 Spec (B7, "`on=` enough, leaning yes"); O effects "keep" lists `on=`.

Kind: surface.

Status: F2 verified the `on=` equivalence; re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c5_sched.py`): `on=I` and `eligible=I, interval=1` give identical `delivered` series under `ode` (10,529.18 in total) and `abm` (9,264); `on=infection` and `eligible=infection, interval=1` likewise (4,032.95 and 3,567).

Proposal: `at=` for rounds everywhere, replacing `years=`/`dates=`, matching `ss.change(at=)`; `interval=`, `start=`/`stop=` as today; `on=` kept only for transitions (events), where it reads well (`on=anc.visit`), and an error for a state or condition, pointing to `eligible=`. Renaming `eligible=` to `where=` (F2 README) is a decision below.

```python
# before
ss.vaccinate(bcg, coverage=0.7, years=[2021, 2025], eligible=ppl.age < 5)
# after
ss.vaccinate(bcg, coverage=0.7, at=[2021, 2025], eligible=ss.People.age < 5)
```

Cost: keyword renames in 08 and `tests/test_deliveries.py`. Breaks: `years=`/`dates=` (alias for one release).

### D35. Products are modifiers in fact, not by duck typing

**`products.Effect` is a second implementation of `ss.modify` (its own `op`, `active`, `apply`, `tabulated`, `describe`, and waning); products are `ss.Module`s, so they need three "initialize once" guards and a `_ti_waned` guard against waning twice; product keys (`acquisition`, `transmission`, `mortality`) differ from the keys `ss.modify` and `adjust=` use (`rel_sus`, `rel_trans`); every example needs two constructors (`ss.vaccinate(ss.Vaccine(...), ...)`).**

Raised by: F 04 C3 and C6, F2 07 f2, f3, f5, O effects 12; F 04 Keep (the `efficacy=dict(acquisition=...)` keys).

Kind: codebase (and surface for the keys and the product-less form).

Status: reproduced by F2 and O; not re-run.

Proposal: `Held.init_pre` appends real `ss.modify(target, multiply=1 - efficacy, where=<has_product>)` objects (with waning folded into the amount where waning is a function) and `Effect` goes; a product is a plain object owned by its deliveries and the sim's modifier table, not a module, with one waning step per product, so the guards and the empty results go (D27); `ss.vaccinate(efficacy=0.85, waning=..., coverage=..., at=[...])` builds its own product when none is given (F2, about 15 lines), with the explicit form kept for reuse. The keys are a disagreement (below).

Cost: medium; products' random streams and states move to the delivery or the sim. Breaks: nothing user-facing except the keys, if changed.

### D36. Base and effective values: adopt the code's rule, and drop `track_writes`

**07 says reading a modified quantity gives the effective value and `.base` gives the base; the code gives the base on read, `.effective` the modified value (agents only), and `.base` is `None`; `track_writes=` and the direct-writes half of `what_modifies()` (40 lines of frame inspection) are used by one test.**

Raised by: F 04 §3 and §4, O effects 5 and 22, C F13, F2 07 "good" (praises base/effective and `track_writes` off by default), F2 01 (settled by the departure).

Kind: spec drift.

Status: reproduced by F and O; not re-run.

Proposal: keep the code's rule (one array, no cache invalidation, the engine applies modifiers where the quantity is used, and Starsim 3 code that reads `rel_sus` keeps working), rewrite 07's three rules and `composition-and-effects.md` to one paragraph, delete `.base` from 17, and make `.effective` in a compartmental run raise a clear error. C prefers explicit `base`/`effective` access for mutable runtime arrays; the code's rule is that, minus a `.base` that would only alias the array itself. Drop `track_writes`, `ModifiableArr.writes`/`record()`, and the direct-writes branch (`sim.py:514-523`), keeping the declared-modifiers listing; print multipliers at the precision given (`0.1`, not `0.09999999999999998`).

Cost: spec text; 40 lines removed. Breaks: the `track_writes=` keyword.

### D37. A modifier in `connectors=` or `interventions=` behaves identically, even with `when=`

Raised by: F 04 Spec, O effects 11, F2 03 f8 (also accepted under `analyzers=` and `demographics=`; the spec's `modifiers:` slot does not exist).

Kind: surface.

Status: F found no difference; O said only `when=` could differ. Re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c6_misc.py`) with a state-dependent `when=` (beta times 0.2 while yesterday's `n_I` exceeds 500): 5,321.56 infections in either list under `ode`, 4,959 in either list under `abm`. F is right: a modifier is evaluated where the engine first reads the quantity in a step, whichever list holds it.

Proposal: say so (a modifier can go in any module list and it does not matter), drop 07's "differ only in where the condition is evaluated", and keep the distinction only for modules with a `step()`. Whether to add a `modifiers=` slot is a decision below.

Cost: spec text. Breaks: nothing.

### D38. Four spellings of `stratify`, and `E305` on an identical re-run

Raised by: F 03 U5, O dims 17, F2 01 f7; F README §6 table.

Kind: surface.

Status: re-run by me (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c16_simstrat.py`): `stratify(age=[0, 15])` twice raises `E305 ... this is what running the same line (or notebook cell) twice gives`.

Proposal: drop the one-line function `ss.stratify(sir, ...)`; keep `sir.stratify`, `sim.stratify`, and `ss.Sim(stratify=)` (the last two as one call after D5); an identical repeat is a no-op with a one-line note, and `E305` only when the repeat differs (O). Document or remove the undocumented list-of-dicts form of `ss.Sim(stratify=)`.

Cost: trivial. Breaks: nothing in use (no vignette or test calls `ss.stratify(`).

### D39. Prune the network catalogue

**18 network classes; `RandomSafeNet`, `MaternalNet` (an alias), `PostnatalNet`, `BreastfeedingNet`, `MSMNet`, `HybridNet`, and `ClusterNet` are each used by one test or one library module and by no vignette; three random networks with different parameter names; `MFNet`'s targeted mode duplicates `PartnerNet`; `dur=` on most networks and `duration=` on `MFNet`.**

Raised by: F 03 C2, F2 07 other (`MFNet` targeted mode), O dims 15, O surface (MFNet against PartnerNet).

Kind: surface.

Status: F counted usage; not re-run.

Proposal: move the single-use and Covasim-specific networks to `starsim/library/networks/` beside `DiskNet`/`ErdosRenyiNet`/`NullNet`; merge the three random networks into `RandomNet(method=...)` with one parameter name; cut `MFNet`'s `n_contacts`/`concurrency`/`max_contacts` mode and give `PartnerNet` `between='sex'` and `acts=` (F2); accept `dur=` on `MFNet`; drop `MaternalNet` and the abstract `Route` export (17 says it is not v4 vocabulary).

Cost: imports move. Breaks: imports of the moved classes.

### D40. There is no all-or-nothing vaccine in v4

Raised by: O effects 19 only.

Kind: surface (a missing capability that would otherwise keep `simple_vx` alive).

Status: by inspection (O); `simple_vx(leaky=False)` is tested by `tests/test_interventions.py::test_sir_vaccine_all_or_nothing`; `ss.Vaccine` is leaky only. Not re-run.

Proposal: `ss.Vaccine(efficacy=0.85, leaky=False)`: a share `efficacy` of recipients hold full protection and the rest none (two levels of `has_<product>` in a compartmental run; a draw at delivery as agents).

Cost: one keyword, one level. Breaks: nothing.

### D41. A test delay, a fading efficacy, and partial routine coverage each add a dimension to a compartmental run

**`delay=ss.days(3)` adds a `pending_<delivery>` dimension with `2m+1` levels (a toy SIS with one screen goes to 14 groups); a function-shaped waning adds ten levels; `coverage<1` or `capacity=` in routine delivery adds `offered_<delivery>`; about 150 lines.**

Raised by: F 04 C2, F2 07 f9.

Kind: codebase.

Status: F2 verified the level counts and agreement with agents to 1.5%; not re-run.

Proposal: keep exponential waning (one flow) and `offered_` (it is what makes routine coverage right); `delay=` and function-shaped waning are the candidates for agents-only (`E804` under an explicit compartmental solver, agents under `auto`), but tbsim's cascade may rely on them as ODEs, so this is a decision below, not a recommendation.

Cost: removes the pending machinery if taken. Breaks: compartmental runs with delayed tests.

### D42. `dependence='density'` and `n_ref` are specified and counted but do not exist

Raised by: F 03 S1, S2, U8; F2 07 f10; O dims 13; C F18.

Kind: spec drift.

Status: all four reproduced `ValueError: 1 unrecognized arguments for mixingpool: dependence` (C's in `codex_review/evidence/routes-output.txt`); not re-run.

Proposal: strike them from 06 and 17; say that density dependence is written `beta * I.n`, which is what the importers already do. `describe()` should stop printing "frequency-dependent" as though it were a choice.

Cost: spec text. Breaks: nothing.

### D43. Smaller spec drift and footguns in this area

Raised by, and kind, per item (spec drift unless noted):

- `sim.networks` is empty in a compartmental run (routes move to `sim.mixed`), so 06's `sim.networks.school` does not exist (F 03 S1).
- `net.check()` exists only on `DegreeTargets` and reruns the sim (F 03 S1); F2 and O praise it on `PartnerNet`; the spec should say where it is.
- `implementation.md:277` says there are no named geographies; `ss.geo('Kenya', level=1)` works offline (F 03 C5, F2 07 other, O dims 18). Delete the line.
- `metapopulation.md` writes `ss.geo('Kenya', level='county')`, which raises (O dims 18).
- `ss.geo(['north', 'south'])` is read as countries (`E605 ... did you mean Norway`), while `ss.geo(names=[...])` is labels (O dims 15). Labels when nothing matches, with a note, or `ss.geo(country=...)`.
- `describe()` prints `beta = 0.05 per day` under a contact matrix, where 06 says it is a probability per contact; and `I.n / N` without `[place]` under space (O dims 18).
- A first age edge above 0 (`age=[15, 25, 35, 50]`) puts everyone under 15 in the first bin without comment (O dims 18).
- `age=['young', 'old']` as labels becomes a plain dimension stored as `age_group` that does not age (F 03 S3, F2 04 other).
- A joint `split=ss.by(..., dims=['age', 'patch'])` without an age dimension is an uncoded `KeyError` (F 03 B5; crash).
- 07 says a `when=` is called at the modifier's place in the loop; the code calls it on first read in the step, once (F 04 Spec). Say so and close 19's question.
- `trace=` and `disease=` are delivery keywords missing from 08's table; `move=` leaks (F 04 §2).
- Routine delivery as "offered once" via an `offered_` dimension is the implemented trade; worth a line in 08 (F 04 Spec).
- The waning departure (a duration means losing protection at an exponential time, not fading) is right; update 08 (F 04 §10).
- Code that writes a property a dimension is built from must call `sim.groups_changed(uids)` or groups go stale silently (O dims 16; silent wrong answer for custom code). Mark the group cache dirty on write to those arrays.
- `ss.Test` names what it detects two ways (`sensitivity=dict(I=...)` and `detects=`) (F2 07 other; surface). Keep `detects=`.
- `ss.modify(seasonal(peak='January'))` in a sim with no dates treats steps as days with no note (O effects 22).
- As agents a delivery runs after the step's progressions and before transmission; as groups before all flows, so TPT on the first step reaches 7% fewer people as agents (F 04 Outside, O README solvers 4). The execution area owns it; it is the largest solver disagreement a delivery user sees.

Status: each reproduced by its reviewer; not re-run.

### D44. Movement between groups is a dense groups-by-groups matrix

**Default aging on 16 ages by 290 places takes 15.9 s against 0.16 s without it, with 344 MB of dense matrices, because every mover is an `ng × ng` matrix though each moves along one dimension.**

Raised by: O dims 7; F2 README second tier (one `Move` kernel for movers, deliveries, and the register).

Kind: codebase (performance).

Status: reproduced by O; not re-run.

Proposal: a mover is `(dim, R)` with `R` an `n_d × n_d` rate matrix between that dimension's levels, applied by contracting along that axis of the group grid (`O(ng · n_d)`), in the stepped and whole-run paths alike.

Cost: a contained refactor of `init_movers`/`mover_rates`/`step_movers` and the kernel equivalent. Breaks: nothing user-facing.

### D45. Smaller codebase items

Raised by, per item (all codebase):

- `Delivery` is one 730-line class doing six jobs (F 04 C2); most methods are short; it shrinks with D33(b), D35, and D41 rather than by splitting.
- `connectors.to_ti` is a linear scan over `tvec` per call and `modify.active` converts its times twice per evaluation (F 04 C7); a `Timeline.index(time)` would serve both.
- Modifiers are recomputed every step (`Engine.cache` cleared each step); a `when=` or function amount makes a 1,000-step ODE about ten times slower (137 ms against 12 ms) (O effects 21). O recommends relaxing 07's "MUST be recomputed only when something changes" rather than adding invalidation; I agree.
- `Strat.__init__` is an 80-line validator and `Sim.init_strats` re-derives much of it (F 03 C4, F2 04 other); D5's single parser removes the duplication.
- `parse_target` has two resolution schemes and `Target` exists only to name `sir.rel_sus` before the sim exists (F 04 C4); D30 shrinks both.
- `Selector.per_person` is a second parameter-ladder evaluator beside `Engine.value` (F2 04 other); they will drift, as D1 shows they already have for modifiers.

Status: by inspection; not re-run.

## Where the reviews disagree

**Whether `PartnerNet` is refused under an ODE (factual).** F 03 (S1 and Keep): sexual networks are refused with `E804`, correctly. F2 07 f10, O dims 12: `PartnerNet` is approximated by homogeneous mixing with `n_contacts` contacts per day. Check (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c14_partner.py`, `/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c11_routes.py`): `MFNet` is `E804` with or without `approx=True`; `PartnerNet` runs under `solver='ode'` without `approx=True`, and its prevalence ignores duration (0.714 at `dur=5` and `dur=500`, against 0.670 and 0.168 as agents). F2 and O are right; F's "keep refusing" applies to MFNet only. Recommendation: D10's one rule.

**Whether a misspelled `beta` key is silent (factual).** F 03 B3: `beta=dict(mf=..., needls=...)` with a network `needles` gives no error. O dims 1: only a key that matches no route is ignored. Check (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c12_e403.py`): F's script stopped at `sim.init()`, where there is no error; at the first step of `run()` the same case raises `E403 ... a network called "needles" with no value`. A key naming no network (`wrok`) runs silently. O is right. Recommendation: check both directions at `sim.init()` (D21).

**Whether `connectors=` and `interventions=` differ for a `when=` (factual).** F 04: they differ in nothing. O effects 11: they differ only for `when=` and function amounts. Check (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c6_misc.py`): identical results with a state-dependent `when=` under both solvers. F is right (D37).

**Whether `describe()` marks a dimension's kind as guessed (factual).** F2 07 "good": `kind` is guessed and printed. O dims 3: printed but not marked. Check (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c8_kind.py`): `region (space) ... [guessed: equal across levels]`: the kind is printed, only the split is marked. O is right; F2's praise stands for the surface, not for the marking (D4).

**What a bare number before the start does (factual nuance).** O effects 2: `ss.change(at=[30, 60], ...)` and `ss.modify(start=30, stop=60)` have no effect. Check (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c3_deliveries.py`): a time before the start becomes step 0, so `at=[30]` alone halves beta from day 0 (5,719 against 9,312) and O's two-time case has no effect because both times land on step 0. Both are the same bug; the symptom depends on the times (D2).

**Whether the sim/disease split of `stratify` is sound (factual).** F2 03 table: "Fine: the sim/disease split is principled". F 03 U2 and O dims 4: the sim-level parser turns options into dimensions and skips `E301`. Check (`/tmp/claude-1000/-home-cliffk-idm-ssv4/bd45ca9a-b1a0-40db-a545-71b050556d66/scratchpad/oct17/dims/c16_simstrat.py`): F and O are right about the parser; F2 is right that the concept is fine. D5 keeps the concept and replaces the parser.

**The size of the Starsim 3 stack (factual).** F 19 names and 755 lines; F2 24 exports and about 700 lines; O about 530 lines. Counted at `interventions.py:77-534` and `products.py:31-228`: 18 classes, about 650 lines, excluding the `Intervention` and `Product` bases that every proposal keeps (D31).

**How to infer a dimension's kind.** F 03 U1: from the value only; keep `age` by name; delete `kind=` and the name list. O dims 3: from the value; keep `kind=` as an override; keep the name list only as a printed suggestion ("'resistance' looks like a strain; pass kind='strain'"). F2 01: cut the strain-name list. Recommendation: O's. The override costs one keyword and covers a strain dimension with no `immunity=`; the printed suggestion catches the user who meant a strain without changing the model.

**Where migration and the movement register go.** F 03 C3: keep `Mobility`'s `commute`, `visit`, `migrate`; mark `force` as LASER-compatibility. O dims 10: movement of people is always a transfer (`ss.transfer(patch=ss.gravity(...))`, `ss.transfer(herd='movements.csv')`); `Mobility` keeps only `commute` and `visit`. F2 07 f8: lift the register out as `ss.events(table)` whose `from`/`to` name any state and level, which closes three `deferred.md` gaps (seeding on a schedule, camdl's one-off `transfer`, life-table deaths). Recommendation: keep the one-line commuting surface (`mobility=ss.gravity(...)`), take F2's `ss.events(table)` for the register because one name then covers four needs, and fold `kind='migrate'` into `ss.transfer` along the patch dimension (O) once D33(a) makes `ss.transfer` a transition. O's `moving=ss.transfer(patch=ss.gravity(away=0.01))` is longer than today's form for the common case, so commuting should not move.

**What `ss.contacts` returns.** F 03 U4 and F README: `MixingPools` stays as the class `contacts` returns, with `AgeGroup`/`LevelGroup` retired from the public surface. F2 07 f6 and O dims 8: a small matrix route, applied by the counts path under both engines; the N² sub-pools leave the declared path. F's own C1 says the pools-of-pools machinery is used only by the agent, no-places path. Recommendation: F2 and O (D32): simpler for the user (the same matrix is the same code under every solver), less code, and faster.

**What to do with the Starsim 3 classes.** F: delete. F2 07 f7: move to `starsim.library`. O effects 1: a compatibility module with deprecation warnings that print the v4 form. All three: first make v4 deliveries work on coded diseases (F, O) or redeclare the built-ins. Recommendation: O's compatibility module (`starsim.v3`), off the `ss.` namespace, with the warnings; `starsim.library` is for example models, not a legacy API, and outright deletion strands STIsim and HPVsim for a release.

**`Vx`/`Dx`/`Tx`.** F: keep out entirely. F2: plain aliases of `Vaccine`/`Test`/`Treatment`. O: plain functions or aliases once the v3 classes move. Recommendation: F's. An alias is two names for one thing, and the short names are what Starsim 3 code imports, so they belong in the compatibility module.

**`ss.seasonality`.** F 04 C5: delete. O effects 11 and F 01: keep as a thin wrapper returning the modifier, with its `plot()`. F2: replace with a docstring example. Recommendation: move it to the compatibility module and show the `ss.modify(..., multiply=ss.seasonal(...))` form in `ss.seasonal`'s docstring; two names one letter apart at `ss.` is the problem.

**`ss.change`.** O effects 9, F2 07, F2 03: keep as sugar (O: as a function returning `ss.modify`). O surface: merge into `ss.modify(target, set=..., at=...)`. F2 README: keep only if `at=` becomes the shared keyword. Recommendation: O effects 9 with `at=` shared (D8, D34): the name stays, the subclass goes, and its bug goes with it.

**Product effect keys.** F2 07 f3: products take `rel_sus`, `rel_trans`, and transition names as `ss.modify` does, with `acquisition`/`transmission` as synonyms. F 04 Keep: `efficacy=dict(acquisition=..., transmission=..., mortality=...)` is one good spelling with a good `E505`. Recommendation: keep F's keys and add no synonyms. Efficacy is a reduction against an outcome while `ss.modify` takes a multiplier on a quantity, so the clinical words say what the number means; the default (a single `efficacy=`) covers most uses; make `E505` and `E301` each name the other's keys.

**`on=`.** F2 07 f1: drop it, or restrict it to transitions. O effects "keep": one of the keywords of a good `Delivery`. F 04: `on=` is enough for event logic, leaning yes. Recommendation: restrict to transitions (D34). It reads well for an event and misleads for a state.

**`ss.transfer(coverage=)` returning a `Delivery`.** F 03 U6: remove the delivery form. F2 04 f4: make `Transfer` a `Transition(move=)`; whether the delivery form stays is Cliff's call. F2 07: a documentation decision; present it once as the general primitive. O dims 10: transfer becomes the one movement word. Recommendation: F's plus F2 04's (D33): a function that returns a module or a transition depending on its keywords is what 17's checklist warns against, and `ss.vaccinate` covers the use.

**`E305` on a repeated `stratify`.** F 03 U5 Keep: the error is right; it catches the re-run cell. O dims 17, F2 01 f7: idempotent. Recommendation: O's form, a no-op with a note when identical and `E305` when different (D38). It keeps F's protection where a wrong guess would be harmful and drops it where the safe guess is obvious.

**The aging default.** O dims 6: on only with births or deaths (or a long run). F 11 §6: "on whenever the sim has a unit and agents". Recommendation: O's (a), with the rate fix (b) either way (D12). "A closed population ages only if you give it demographics" is one sentence, and it removes a 40% bias from every short epidemic model.

**The lazy-condition placeholder.** O effects 15: `ss.people.age`. F 11 §4: class-level descriptors, `ss.People.age < 15`. F2 07: allow `ss.People()` with no size. F 04 §6: `sir.age == '15-24'` where age is a dimension. Check: `ss.people` is already the named-population function (`ss.people('Kenya')`), so O's name collides. Recommendation: F's `ss.People.age` (no instance, no new name), plus F's `sir.age == ...` documented for stratified models (D25).

**`track_writes`.** F 04 §4: drop it, by the open question's own test (no vignette uses it). F2 07 "good": off by default, fine. Recommendation: drop (D36); off by default still costs 40 lines and a keyword.

**Network approximation for `PartnerNet` under an ODE.** O dims 12: implement the partner-change form (about ten lines). F 03: refuse, since the approximation is not settled. Recommendation: `E804` now, O's form later if a user asks for it, tested against agents (D10). `auto` already picks agents, so only explicit `solver='ode'` users are affected, and they are getting a wrong answer today.

**Partial cross-immunity.** O dims 11 gives (a) agents under `auto` and (b) an exact `2^k` history dimension, preferring (b). No other review addressed it. Recommendation: (a) now (one line, correct), (b) as a later feature (D13).

**Delayed results in a compartmental run.** F 04 C2: a judgment call; agents-only would remove about 60 lines. F2 07 f9: consider `E804`. C F06: keep, and store results per specimen. Recommendation: fix D9 by excluding pending people from re-testing, and leave the compartmental form until tbsim's use is known (D41 decision).

**Route-level `beta`.** F2 07: warn for declared models. O dims 8: drop for declared models. Recommendation: O's (D21); a warning keeps two ways to say one thing.

**`modifiers=` slot.** F2 03 f8: the spec's slot does not exist; a modifier is accepted in four lists. O effects 11: accept `modifiers=` or any list and stop asking. F 04: drop one of the two lists from the explanation. Recommendation: accept a modifier in any module list, as now, document that it does not matter, and do not add a fifth list.

## Keep

- One `stratify()` operator for age, risk, place, strain, and booleans, in place, with multiplicative `adjust=` only; results read with `.select()`/`.by()` (F, O, F2).
- `where=` ragged dimensions and the grid-with-fold representation (`init_ragged`, `fold`, `grid_index`); do not turn it into a true ragged array (F).
- Strains carried together (`'A+B'`) with `move=`, checked against a hand-written ODE (F).
- Space as a dimension, not a model class; `sir.stratify(patch=kenya, mobility=ss.gravity(...))` as the whole spatial surface (F, F2, O).
- `ss.geo('Kenya', level=1)`, `ss.people('Kenya')`, `ss.contacts('Kenya')` bundled offline with provenance; `ss.contacts('Kenya')` adding the age stratification when none is given (F, F2, O).
- The movement register's named-column table with `to_states`, the clearest win over SimInf's matrices (F, F2).
- `ss.by` with the table form for sparse initial conditions (F).
- `ss.transfer`'s exact coverage on many groups (0.70, not `1 - (1 - f)^P`) and rate transfers matching `1 - e^-1` under both solvers (O).
- The model text not changing when the routes change: one `ss.Disease` under no network, `MixingPool`, `contacts`, and `RandomNet`, ODE and agents (O).
- `PartnerNet`'s degree targets and `net.check()` (F2, O).
- `ss.modify`: one target, `multiply`/`add`/`set`, `where`, `start`/`stop`, `when`; the stated combination order; the base never overwritten; `W502`/`W503` with clipping recorded (F, F2, O).
- Lazy conditions against real attributes, printed as written, with the precise `E804` when a threshold cuts a bin, and conditions crossing two diseases' states as an ODE (F, F2, O).
- The product/delivery split, one `Delivery` class driven by keywords, and the `Selector` that makes one declaration run on agents and groups; do not split it by solver (F, F2, O).
- Follow-ups as deliveries with their own results, to any depth (F, F2, O).
- `deliver=` replacing the who-keywords while the schedule still applies (F, O).
- `capacity=` separate from `coverage=` (O).
- `efficacy=dict(...)` with `E505`, and one `Treatment` object that also changes rates (F).
- Exponential waning as losing protection outright, printed unambiguously by `describe()` (F).
- The whole-run tabulation of scheduled deliveries, so a sim with a campaign is still one kernel call (F2).
- `tests/test_deliveries.py` running every scenario under `ode`, `binomial`, and `abm` (F).
- Rejecting `min_duration=` and keeping memory in a class's `step()` (F).
- `ss.Births`/`Deaths`/`Pregnancy` as library modules for data-driven, agent-only demography (F, F2).

## Decisions for Cliff

1. Fix order: D1 to D10 first, before any surface work? (All four reviews put silent wrong answers first.)
2. A time outside the sim's span: an error, except `stop=` after the end? And is a bare number ever a year? (O: error; F2: one rule, bare numbers in the sim's unit, years as dates.)
3. A transfer-fed or boolean dimension starts everyone at its first level? (F, O: yes.)
4. Dimension kind from the value: keep `kind=` as an override and the name list as a printed hint (O), or drop both (F)? (I recommend O.)
5. One `stratify` parser with sim-level options (O), or levels only at the sim level (F)? (I recommend O.)
6. `set=`/`add=` on per-person quantities and rates: an error (O), or implement against the base? (O: error.)
7. A pending test result excludes the person from re-testing (simple), or results per specimen (C)?
8. Networks under a compartmental solver: approximate only with a stated, printed approximation, `E804` otherwise, including `PartnerNet` and `StaticNet` until approximations exist? (C, F2, O point this way.)
9. One default age distribution (a declining pyramid) everywhere, and named data bringing its population? (O: yes.)
10. Aging on only with births or deaths? (O: yes; F prefers a simpler trigger.)
11. Partial cross-immunity: agents under `auto` now? (O: yes, exact `2^k` form later.)
12. `ss.gravity` as shape plus `away=` amount, with a warning when the cap binds? (O: yes.)
13. A `sex` dimension is `people.female`? (O: yes.)
14. An explicit local-count spelling (`I.n.by('sex')`, `W.here`)? (F and F2: yes; spelling open.)
15. `ss.Treatment(cure=, efficacy=)` by keyword, infected by default; `efficacy=` required on `ss.Vaccine`? (F2, O: yes.)
16. `capacity=`: a bare number per round or step, a rate typed, an error for the ambiguous routine case? (F2: yes.)
17. Route-level `beta` dropped for declared models (O), or warned (F2)? (I recommend drop.)
18. Make v4 deliveries and modifiers reach coded diseases, or rely on redeclaring `ss.SIR`/`SIS`/`SEIR` and the library diseases? (F, O: both, in that order before retiring the v3 stack.)
19. The Starsim 3 intervention and product classes: delete (F), `starsim.library` (F2), or a compatibility module with warnings (O)? `Vx`/`Dx`/`Tx` aliases or none? `ss.seasonality` delete or wrapper? (I recommend a compatibility module, no aliases at `ss.`, `seasonality` in the compatibility module.)
20. Custom modules: agent-only unless `compartmental = True`, documented, or tried on groups at `sim.init()`? (C: no probing; O: either, but name the function.)
21. `ss.contacts` as a small matrix route under both engines, `bands=` for the matrix's own edges? (F2, O: yes; F's C1 agrees in substance.)
22. `ss.transfer` always a `Transition(move=)`, never a `Delivery`? (F, F2 04: yes.)
23. The register: `ss.events(table)` general (F2), a transfer (O), or spatial only (F)? Migration as `ss.transfer` along the patch dimension (O)? (I recommend `ss.events` and migration as a transfer, commuting unchanged.)
24. Schedule keywords: `at=` everywhere replacing `years=`/`dates=`; `on=` for transitions only? Rename `eligible=` to `where=`? (F2: all three; O keeps `on=`; I recommend the first two and lean yes on the third.)
25. `ss.change` as a function returning `ss.modify`, or folded into `ss.modify`? (O effects 9 and F2: keep the name.)
26. Product keys: clinical (`acquisition`, F) or modifier (`rel_sus`, F2)? (I recommend clinical, no synonyms.)
27. Product-less `ss.vaccinate(efficacy=..., ...)`? (F2: yes.)
28. Products as plain objects realized as real `ss.modify` objects, not modules? (F, F2, O: yes.)
29. The code's base/effective rule as the spec, `.base` removed, `track_writes` dropped? (F, O: yes; F2 would keep `track_writes`.)
30. `E305` a no-op on an identical repeat? (O, F2: yes; F: keep the error.)
31. The lazy placeholder: `ss.People.age` (F), not `ss.people.age` (O, which collides)?
32. Demographics: accept `ss.birth`/`ss.death` in `demographics=`; drop or redirect `birth_rate=`/`death_rate=`/`demographics=True`; per-person rates for `ss.Births`? (F, F2: yes.)
33. Compartmental test delays and function-shaped waning: keep, or agents-only? Which tbsim or vignette models need them as ODEs? (F, F2: ask.)
34. An all-or-nothing vaccine, `ss.Vaccine(leaky=False)`? (O: yes.)
35. Strike `dependence='density'`, `n_ref`, and the sexual-network approximation paragraph from 06 and 17? (All four: strike.)
36. Prune the network catalogue to `library/networks/`, merge the random networks, cut `MFNet`'s targeted mode? (F, F2: yes.)
37. Relax 07's "recomputed only when something changes" for modifiers rather than add invalidation? (O: relax.)
