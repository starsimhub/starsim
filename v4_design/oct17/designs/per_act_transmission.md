# Per-act transmission on sexual networks

Phase H design document (implementation_oct18.md, Phase H, third bullet). Findings: ST2, ST3, N7, HP3 (the core), ST17, ST21, ST22, ST23 (the network additions), and ST9 for context. Prototype arithmetic: `wt/scratch-h3/acts_dt.py`, `wt/scratch-h3/trunc.py`, `wt/scratch-h3/describe_now.py`.

## Revision, 2026-10-19

After Cliff's `[CK:]` notes (H0 in `oct19/implementation.md`):

- **D3, directions:** the valid keys are no longer built into the engine. Each network says what its **sides** are (`ss.MFNet`: `m` and `f`, from people's sex; `between=('fsw', 'client')`: those two; mother-to-child networks: `mother` and `child`, one way), and its directions are `<side>2<side>`. `net.directions` lists them, `describe()` prints them, and an unknown key raises naming the valid ones. A network with no sides takes one value. A worked example for injecting drug use is added (a needle-sharing network beside a sexual one).
- **D5, decided:** `rel_trans`, `rel_sus`, edge weights, and modifiers multiply the per-act hazard. `beta=0.03` with `rel_trans=10` gives 0.263 per act, not 0.30. The reasons are now in the section on D5.
- **D6, condoms and microbicides are interventions:** `ss.modify(syph.infection, multiply=0.1, share=use, route='mfnet')`, not `condoms=` on the network and not `condom_efficacy=` on `ss.peract`. Under D5's hazard reading, a share `c` of acts with efficacy `ε` is exactly one multiplier, `1 − c·ε`, on the hazard, so a condom needs no special case in the engine. Two keywords on `ss.modify` are new (`route=`, `share=`); `condoms=` and `condom_efficacy=` are dropped.
- **D8:** `ss.MFNet` stays.
- **D9, the partnership approximation is in scope:** specified for the ODE, binomial, and Gillespie solvers (step 9), marked `approx`, with the source state's exit folded into the partnership's length, its known error, and when it is poor.
- New keywords: `sides=` on networks, `where=` on `ss.PartnerNet` (who takes part, for the needle network). The open decisions are at the end.

## Summary

**The problem.** A per-act transmission probability has no spelling in v4. The agent engine converts every `beta` as a rate (`-expm1(-ref*dt)`, `solvers/agents.py:1302`) and then the network raises the result to the number of acts, so the only workaround, `ss.probpermonth(b)`, is right only at monthly steps (per-act 0.01 on `ss.MFNet`: 38.1% prevalence after 10 years at monthly steps, 11.6% at weekly ones). A `beta` pair is applied after the acts, using the larger of the two for both directions and scaling by their ratio (`agents.py:1290-1300`), so female-to-male transmission per partnership-month is 0.429 against Starsim 3's 0.728 at 8 acts (0.232 against 0.278 at 2). The pair is also positional, [p1 to p2, p2 to p1], not male to female as the spec says: `beta=[b, 0]` infects 86 women and 0 men on `ss.MFNet`, and 0 women and 158 men on `hpv.SexualNetwork`, which puts women in `p1` (HP3). Two more dt dependencies sit underneath: `ss.MFNet` draws an integer count of acts per step once per partnership from `ss.poisson(lam=ss.freqperyear(80))`, so at weekly steps 21.5% of partnerships never have sex (annual chance of transmission at per-act 0.01: 0.531 monthly, 0.465 weekly, against 0.552 for 80 acts a year), and STIsim truncates its acts per step to integers (32% of acts lost at weekly steps, 7% at monthly). Around this, STIsim needs source weights that a `beta` dict can multiply (ST17, a `TypeError` today), condoms on edges (ST21; today a coded `net_beta()` reading `disease.pars.eff_condom`, which a declared disease cannot carry, ST9), a two-mode network for sex work (ST22), and a contact matrix over sex and risk together (ST23; today the `np.kron` of one matrix per dimension).

**The proposal**, as a user writes it:

```python
S, E, P, Sec, EL, LL = ss.states('S E P Sec EL LL')
syph = ss.Disease(
    infection = ss.transmission('S -> E', source=dict(P=1, Sec=1, EL=0.55, LL=0.05),        # source weights (ST17)
                                beta=dict(mfnet=ss.peract(m2f=0.3, f2m=0.15), maternalnet=0)),  # per act; mfnet's directions are m2f and f2m (ST2, ST3, HP3)
    congenital = ss.transmission('S -> CS', source='P Sec EL LL', beta=dict(mfnet=0, maternalnet=ss.permonth(0.8))),
    ...
)
mf = ss.MFNet(acts=ss.lognorm_ex(ss.freqperyear(80), ss.freqperyear(30)))            # acts per unit time, drawn once per partnership (D4)
use = ss.timeseries({2000: 0.05, 2020: 0.3})                                          # share of acts with a condom
condoms = ss.modify(syph.infection, multiply=0.5, share=use, route='mfnet')          # an intervention: a condom halves the per-act hazard, in that share of acts (D6, ST21)
sw = ss.PartnerNet(name='sw', between=('fsw', 'client'), n_contacts=dict(client=ss.permonth(1)), acts=3)   # two-mode, one-off contacts (ST22)
mix = ss.contacts(matrix, dims=['sex', 'risk'])                                         # one matrix over several dimensions (ST23)
```

(The efficacy here is for illustration: STIsim gives syphilis none, gonorrhea 0.9 (`multiply=0.1`), and chlamydia 0.4 (`multiply=0.6`).)

Under it, the probability that an edge from infectious `u` to susceptible `v` transmits in a step of length `dt` is

```
h_d   = −ln(1 − p_d)                       per-act hazard for the direction d = (side of u) 2 (side of v)
M_e   = Π_k (1 − s_k·(1 − m_k))            the modifiers on this route: each multiplies the hazard by m_k on a share s_k of acts (s_k = 1 without share=)
H_e   = w_e · a_e·dt · h_d · M_e · rel_trans[u] · rel_sus[v]
p_e   = 1 − exp(−H_e)                       (= 1 − (1 − p_d)^(a_e·dt) when nothing else applies)
```

with `a_e` the edge's acts per unit time and `w_e` its edge weight (`edges.beta`). This is the spec's existing per-edge form (`06-routes.md`, "Per-edge form") with `β` read as the per-act hazard. It does not depend on `dt`: over a year it gives `1 − (1 − p)^a`, whatever the step. A condom of efficacy `ε` used in a share `c` of acts is the modifier `m = 1 − ε`, `s = c`, so `M = 1 − c·ε`: the protected and unprotected acts need no separate terms, because hazards add.

**Decisions for Cliff** (the first round, with his notes kept; each row he questioned ends with what the revision does. The decisions still open are at the end of the document.)

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | A bare number as `beta` on a network that defines acts | (a) a probability per act, as the spec says and as Starsim 3 did; (b) raise, asking for `ss.peract(...)` or a rate | (a): it is the spec's reading of a bare number on a route ("a probability per contact"; per act where acts exist), and it makes a verbatim Starsim 3 port right (N7). `describe()` says "per act", and `to_code()` writes `ss.peract(...)` [CK: ok] |
| D2 | A typed rate or per-period probability (`ss.permonth(b)`, `ss.probpermonth(b)`, `ss.prob(b, dt)`) on a network with acts | (a) `WrongDimension`: `beta` per unit time times acts per unit time is not a hazard (the spec's existing rule for `beta` and `n_contacts`); (b) today's conversion, then raised to the acts | (a): (b) is the bug in ST2  [CK: ok] |
| D3 | Directions of a `beta` | (a) named keys on `ss.peract` (`m2f`, `f2m`, `m2m`, `f2f`), resolved per edge from the partners' sex; positional pairs removed; mother-to-child networks one-way; (b) keep pairs, documented as [p1 to p2, p2 to p1] | (a): a pair means different things on two networks today (HP3), and every package names directions by sex  [CK: ok, but this feels a bit like hard-coded magic ... would want to be really clear about how we can know what keys are valid and which aren't. like how would we capture injecting drug use here?] **Revised:** the keys come from the network's sides, which the network declares (`net.directions` lists them); nothing about sex is built into the engine. Injecting drug use is a needle-sharing network with no sides, so one value. See "Directions (D3)" and "Injecting drug use" |
| D4 | What `acts=` is | (a) acts per unit time of each partnership, drawn once per partnership (a rate, or a distribution of rates), used as the real exponent `a·dt`; on one-off contacts, acts per contact; (b) an integer count per step drawn once (today's `ss.MFNet`, dt-dependent); (c) a count redrawn every step | (a): exactly dt-invariant and cheapest. Changes `ss.MFNet`'s default to `ss.lognorm_ex(ss.freqperyear(80), ss.freqperyear(30))` (STIsim's) and its results [CK: ok] |
| D5 | What `rel_trans`, `rel_sus`, the edge weight, and modifiers of `beta` scale | (a) the per-act hazard, as in the formula above (the spec's per-edge form); (b) the per-partnership probability for the step, after the acts (Starsim 3) | (a): a per-act efficacy (PrEP, a condom, partial immunity) is per act, and (a) never exceeds 1. It changes results against Starsim 3 when `rel` differs from 1 and acts are many (syphilis, `p=0.3`, 6 acts, `rel_sus=0.5`: 0.66 against 0.44) [CK: hm, why can't it be above one? let's say beta=0.03 for HIV, but i set rel_trans for acute HIV at 10, so beta=0.3 -- that should work, no?] **Decided (2026-10-19):** (a), the hazard. 1 − 0.97^10 = 0.263 per act, close to 0.30 when `beta·rel` is small (0.0296 against 0.03 at `beta=0.003`) and never above 1 (0.70 at `rel_trans=40`, where the linear form gives 1.2). See "What rel_trans ... scale (D5)" |
| D6 | Where condom efficacy lives | (a) on the per-act value, `ss.peract(..., condom_efficacy=0.9)`; (b) a disease parameter `eff_condom` that the network reads (the ST9 path) | (a): it differs by pathogen (STIsim: gonorrhea 0.9, chlamydia 0.4, syphilis 0) and is per act; it is then an `ss.par`, fittable, and printed [CK: how would the user define custom interventions, like antimicrobial foam? seems like it should go in the intervention definition rather than either the disease or the network definition. it's a simple mechanism, just `ss.modify(rel_trans=0.1), who=sim.interventions.microbicides)` or similar, so should be easy to implement, but we should be consistent] **Revised:** an intervention, `ss.modify(syph.infection, multiply=1 - ε, share=c, route='mfnet')` (or `ss.modify(hiv.rel_sus, multiply=..., where=...)` for a microbicide its users apply). Neither the disease nor the network carries it. See "Condoms and microbicides (D6)" |
| D7 | Infectiousness weights that differ by route (ST17) | (a) `source=` weights on the transmission, one transmission per set of weights (each `beta` dict giving 0 on the other routes); (b) `source=` keyed by route | (a): in STIsim and HPVsim the mother-to-child transmission already has its own destinations (fetal states, `CS`), so it is a transmission of its own anyway [CK: ok] |
| D8 | The two-mode network (ST22) | (a) `between=(a, b)` on `ss.PartnerNet`; (b) a new class (`ss.TwoModeNet`) | (a): a keyword, no new name. `ss.MFNet` is the two-mode case by sex and could later become one [CK: um, ok, i guess ... we're keeping `ss.MFNet`, right?] **Yes:** `ss.MFNet` stays, with sides `m` and `f` |
| D9 | Partnership networks under the ODE, binomial, and Gillespie solvers | (a) keep `CapabilityError` now and build the partnership approximation of `06-routes.md` as a later step (M), with per-act values mapped as below; (b) build it in this phase | (a): the agent fix is small and urgent (silent wrong answers); the approximation needs its own contract against agents (8.2% against 10.4% in STIsim's hand-built version) [CK: if we're doing all this just for agents, seems like it should be pretty straightforward. would be inclined to build it for the ODE etc models too since it's a pretty common use case, even if approximate as it necessarily will be.] **Revised:** (b). Step 9 builds the partnership approximation for the ODE, binomial, and Gillespie solvers, marked `approx`. See "The partnership approximation (D9)" |

## How a beta reaches an edge today

Three conversions happen in turn, each assuming something different:

1. `Agents.route_beta()` returns a pair per route (a single number becomes `[b, b]`); `Agents.infect()` takes `ref = max(...)` of every direction and source and converts it as a rate: `beta = -np.expm1(-ref*dt)`, in the sim's unit.
2. `net_beta()` raises that to the number of acts: `edges.beta*(1 - (1 - beta)**edges.acts)` (`networks.py:567`, `:1123`).
3. The second direction, and each source state, gets `c[d]/ref` folded into `rel_trans`, which `_nb_transmit` multiplies into the per-partnership probability after the acts.

So a per-act number has to be pre-distorted into a rate that, after step 1 at this `dt`, gives the per-act probability back. Every port does this (`rm = -np.log(1 - beta_m2f)` then `ss.permonth(rm)` in `stisim/v4port/syphilis/v4syph.py:91`; `ss.peryear(-np.log(1 - b)/dt)` in `hpvsim/v4port/natural_history/trans_compare.py:75`), with a comment saying it is right at one step only. `describe()` shows the value with no reading (`beta = {'mfnet': [0.008, 0.004]}`, `rate = {'mfnet': [0.008, 0.004]} × I`), and `to_code()` writes it as `ss.permonth(0.008)` (rates per month), which is not what a Starsim 3 user meant.

## Use cases

### ST2: a per-act probability (STIsim, every STI)

- **Needs:** a probability per sex act that keeps its meaning at any step.
- **Starsim 3:** a bare `beta` was used unconverted as a probability per act, and acts were a count per step: `betamap['structuredsexual'][0] = self.pars.beta_m2f` (`stisim/diseases/sti.py:219-221`), then `StructuredSexual.net_beta()` raises it to the acts. Right at any step only because nothing converted it.
- **v4 today:** no spelling. `ss.probpermonth(0.01)` gives 38.1% monthly and 11.6% weekly; `ss.prob(b, dt)` 38.1% and 30.5%; the coded convention 37.5% and 28.9% (the last gap is D4's acts, below).
- **Proposed:** `beta=dict(structuredsexual=ss.peract(0.01))`, or a bare `0.01` (D1). One value, right at every step; the ports' `-log(1 - b)/dt` lines go.

### ST3: a pair applied per act in each direction (STIsim)

- **Needs:** female-to-male and male-to-female each with their own per-act probability: `1 − (1 − beta_d)^acts` for each direction.
- **Starsim 3:** `p1_to_p2` and `p2_to_p1` loops in `Infection.infect_route()`, each calling `net_beta(beta_d)`.
- **v4 today:** the larger beta for both, then the ratio after the acts: f2m 0.232, 0.346, 0.429 against v3's 0.278, 0.478, 0.728 at 2, 4, 8 acts (StructuredSexual's median is 6). HPVsim's women's prevalence is a third of v3's on this account (status `hpv.md`, ST3).
- **Proposed:** `ss.peract(m2f=0.3, f2m=0.15)`, each direction its own hazard per edge. f2m at 8 acts is 0.728, as v3.

### HP3: named directions (HPVsim)

- **Needs:** "male to female" to mean that on any network, whichever partner the network puts in `p1`.
- **Starsim 3:** HPVsim's `validate_beta()` builds `[f2m, m2f]` because its network puts women in `p1` (`hpvsim/hpv.py:309-333`), and HPVsim's own HIV builds the same order (`hpvsim/hiv.py:256`); STIsim builds `[m2f, f2m]` for `StructuredSexual`, which puts men in `p1`. Each package knows its own network's order.
- **v4 today:** a pair is [p1 to p2, p2 to p1]; spec `03:199` and `06-routes.md:89` say male to female first. `[b, 0]`: 86 women and 0 men on `ss.MFNet`, 0 women and 158 men on `hpv.SexualNetwork`.
- **Proposed:** `ss.peract(m2f=b*transm2f, f2m=b*transf2m)`. The engine picks each edge's direction from `people.female` of its two ends, so the network's order does not matter. HPVsim deletes `validate_beta()`, `_clip_beta()` (an out-of-range value now raises `OutOfBounds` instead of being clipped with a warning), and its `net_beta()` override.

### N7: a verbatim Starsim 3 beta dict (every port)

- **Needs:** a v3 `beta=dict(mf=[0.08, 0.04])` to keep its meaning, or a clear error.
- **v4 today:** fixed by a warning (`engine.py:730-743`, "rates per year, with dt = 0.0192 year ..."), since bare numbers are read as rates per unit time and the port is 12 to 52 times too small.
- **Proposed:** on a network with acts, a bare number is per act (D1), which is exactly what v3 meant, so the verbatim port is right and the warning is not needed there; a pair raises `RouteMismatch` naming the `ss.peract(m2f=, f2m=)` spelling (D3), and the migration script rewrites it for `ss.MFNet` (men in `p1`) and flags it for any other network. The warning stays for bare numbers on networks without acts.

### ST17: source weights a beta dict can multiply (STIsim syphilis)

- **Needs:** syphilis infectiousness weighted by stage (primary and secondary 1, early latent 0.55, late latent 0.05) for sex, but not for mother-to-child transmission.
- **Starsim 3:** an `infect()` override of about 45 lines (`stisim/diseases/syphilis.py:219-263`) that builds two `rel_trans` arrays and picks one by `isinstance(route, ss.MaternalNet)`.
- **v4 today:** `dict * (P + 0.5*L)` raises `TypeError: unsupported operand type(s) for *: 'dict' and 'float'`; `source=dict(...)` is not accepted. The port gives each weight its own dict of rates (`v4syph.py:92`, `sex(r_e)*EL`), which `describe()` prints as three dicts of internal rates.
- **Proposed:** `source=dict(P=1, Sec=1, EL=0.55, LL=0.05)`, the same as the operator form `beta * (P + Sec + 0.55*EL + 0.05*LL)`, which then accepts a dict or an `ss.peract` as `beta`. Mother-to-child is its own transmission (it goes to fetal states in the port anyway), unweighted (D7). The `infect()` override goes. Time-varying latent infectiousness (v3's half-life since entering latency) is out of scope (see below).

### ST21: condoms on edges (STIsim)

- **Needs:** condom use by partnership type and year (a share of acts), with an efficacy that differs by disease.
- **Starsim 3:** `edges.condoms` set each step from `condom_data` by risk-group pair and for sex work (`stisim/networks/base.py:176-207`), and a `net_beta()` override that splits acts into protected and unprotected: `(1 - b*(1-eff))**(acts*c) * (1 - b)**(acts*(1-c))` (`base.py:153-159`), reading `disease.pars.eff_condom`.
- **v4 today:** no condom attribute in any core network; a declared disease cannot carry `eff_condom` (an unused keyword raises `UnknownParameter`, and an `ss.par` in coded arithmetic gives a `BinOp`, ST9); the port sets it with `dict.__setitem__(syph.pars, 'eff_condom', ...)` (`v4syph.py:144`).
- **Proposed (revised, D6):** condoms are an intervention on each disease's transmission over a route: `ss.modify(ng.infection, multiply=0.1, share=use, route='structuredsexual')`, with `use` the share of acts with a condom (a number, an `ss.timeseries`, an `ss.by` over a dimension, or a function of the network returning one value per edge, so STIsim's use by risk-group pair and its coded `edges.condoms` both fit). Since modifiers multiply the per-act hazard (D5), a share `c` with efficacy `ε` is the multiplier `1 − c·ε`, so the engine has no condom term. STIsim deletes its `net_beta()` override and the ST9 workaround; the efficacy by disease becomes one `ss.modify` per disease. Condom promotion is a second modifier on the same route, or a larger `use` after a date.

### ST22: a two-mode network (STIsim sex work)

- **Needs:** contacts only between two pre-selected groups (female sex workers and clients), mostly one-off.
- **Starsim 3:** `SWNetwork` (`stisim/networks/fsw.py`, about 250 lines), with lifetime flags, start ages and durations for each role, a monthly seeking rate converted per step, an intensity per worker, and one-step partnerships.
- **v4 today:** nothing in core.
- **Proposed:** `ss.PartnerNet(between=('fsw', 'client'), ...)` (D8). Each mode is a condition over people (a level of a stratification such as `sw=['none', 'fsw', 'client']`, or a boolean property of people; the interface h4 provides). Edges join only one person of each mode. With `dur=` given, partnerships last and `n_contacts` is partners at a time, as for `ss.PartnerNet` now; with no `dur=`, contacts are one-off (they last the step), `n_contacts` is a rate for at least one mode (`dict(client=ss.permonth(1))`; the other mode's number follows, since each contact has one of each), and `acts=` is acts per contact. Who is a sex worker, and for how long, stays STIsim's (or h4's) business; the network only reads the condition. STIsim's network would shrink to the role assignment (about 100 lines) over a core two-mode network.

### ST23: a contact matrix over several dimensions (STIsim risk-group models)

- **Needs:** "men mix with women, and risk mixing differs by sex": a matrix over sex and risk together, which is not a product of one matrix per dimension.
- **Starsim 3:** none (STIsim's risk-group mixing is inside its network).
- **v4 today:** `ss.contacts(sex=..., risk=...)` is `np.kron` of the two (`networks.py:1983`, and `Mixing.dim_matrices()` at `solvers/mixing.py:259`); `ss.MixingPools` by group is `E804` under an ODE.
- **Proposed:** `ss.contacts(matrix, dims=['sex', 'risk'])`, with `matrix` of shape `(2, 3, 2, 3)` (or `(6, 6)` in the order of the levels' product, first dimension slowest), contacts per unit time of one person in group `(i, k)` with people in group `(j, l)`. Both engines use it where they now use the Kronecker product, so it runs under every solver. `sex` is people's sex as agents (D15, fixed). The separable form `ss.contacts(sex=..., risk=...)` stays: it says something different (independent mixing in each dimension).

### ST9 (context): parameters a network reads

ST9 was closed by its fallback (the error points to `define_pars()`). The only parameter any package's network reads from a disease is the condom efficacy. Under D6 it moves into the intervention (`ss.modify(..., multiply=1 - ε)`), so neither a declared disease nor a network carries it. Coded networks that override `net_beta()` keep the Starsim 3 contract (below), so STIsim's override keeps working until it is deleted.

### Does the code get simpler?

| Package | Starsim 3 | Under the proposal |
|---|---|---|
| STIsim | `validate_beta()` (20 lines), syphilis `infect()` override (45), `net_beta()` with condoms (7), acts truncated to integers per step | Two keywords: `beta=dict(structuredsexual=ss.peract(m2f=, f2m=))` and `source=dict(...)`; condoms as one `ss.modify` per disease; the three overrides go |
| HPVsim | `validate_beta()` (25), `_clip_beta()` (23), `net_beta()` (5) | `ss.peract(m2f=, f2m=)`; the three go |
| STIsim ports | `-log(1 - b)` rates at one step, one dict per weight, `dict.__setitem__` for `eff_condom` | All three workarounds go |
| Sex work | 250 lines | About 100 lines of role assignment over `between=` |

The one place it is not simpler: a model that relied on `rel_trans` scaling the per-partnership probability (D5(b)) changes results and has no spelling for the old behavior. That is deliberate; the old behavior is not per act.

## The proposal in detail

### `ss.peract`

`ss.peract(p)` or `ss.peract(<direction>=..., ...)` (e.g. `ss.peract(m2f=0.3, f2m=0.15)` on `ss.MFNet`): a probability per act, for a transmission over a network that defines acts.

- Each value is a number in [0, 1], an `ss.par`, or an expression of them (`ss.peract(m2f=b, f2m=b*rel)` with `b` and `rel` fitted); out of [0, 1] at run time raises `OutOfBounds`. A value may also be an `ss.timeseries` or `ss.by` over a dimension of the susceptible, as other rates are.
- With one value, every direction gets it. With keys, the keys are the network's directions (below): every direction the network lists must have one, and a key it does not list raises. Both are checked at initialization, from `net.directions`, so the run never finds a missing direction partway.
- It is not a time value: it has no unit and no conversion. It sits in `starsim/time.py` beside `ss.prob` only because the type checks (`WrongDimension`) live there.
- It is a value of `beta`, inside or outside a route dict: `beta=ss.peract(0.01)` applies to every route, which must all have acts.

**On a route without acts** (`ss.RandomNet`, mixing pools, `ss.contacts`), `ss.peract` raises `RouteMismatch`: there is no act to be per. A per-contact probability on such a route is a bare number, as now.

### Bare numbers and typed rates on a network with acts (D1, D2)

| `beta` on a network with acts | Reading |
|---|---|
| `ss.peract(0.01)` | per act |
| `0.01` | per act (D1); `describe()` says so; `to_code()` writes `ss.peract(0.01)` |
| `ss.permonth(0.01)`, `ss.probpermonth(0.01)`, `ss.prob(0.01, dt)` | `WrongDimension` (D2): "beta per unit time and acts per unit time multiply to a rate squared; give the probability per act, `ss.peract(0.01)`, or leave out acts=" |
| `[0.3, 0.15]` | `RouteMismatch` (D3): "a beta pair has no direction names; give ss.peract(m2f=0.3, f2m=0.15)" |

On a network without acts nothing changes: a bare number is a probability per contact per unit time, and a typed rate is allowed when `n_contacts` carries no unit.

### Directions (D3)

Cliff asked how a user knows which keys are valid, and how injecting drug use fits. The answer is that the engine knows no keys: a network says what its **sides** are, and its directions follow from them.

- **Sides.** A network's `sides` names the kinds of people at the two ends of its edges: a dict of a short label to a condition over people. `ss.MFNet` has `sides=dict(m=~ss.ppl.female, f=ss.ppl.female)`; `ss.PartnerNet(between=('fsw', 'client'))` has sides `fsw` and `client`; `ss.PartnerNet(sides='sex')` (partnerships between anyone, by sex) has `m` and `f`, from the `sex` dimension's levels or `people.female`; the mother-to-child networks have `mother` (`p1`) and `child` (`p2`). Any other network has none unless given `sides=`: `ss.RandomNet`, `ss.PartnerNet` without it, and a coded network that does not set it.
- **Directions.** `net.directions` lists the directions the network can carry: `<side of the infectious partner>2<side of the susceptible one>` for each pair of sides its edges can join. `ss.MFNet`: `m2f`, `f2m`. `ss.PartnerNet(sides='sex')`: `m2f`, `f2m`, `m2m`, `f2f`. `between=('fsw', 'client')`: `fsw2client`, `client2fsw`. The mother-to-child networks: `mother2child` only (they are one way, below). A network with no sides has no directions.
- **Checks, at initialization.** Each key of an `ss.peract` must be in the route's `net.directions`, and each direction there must have a key, or one value is given for all. An unknown key raises `RouteMismatch`: `beta for "mfnet" has a direction "m2m", which mfnet does not have; its directions are m2f and f2m (its sides m and f are people's sex: see mfnet.directions)`. Keys on a network with no sides raise: `"needles" has no sides, so one beta is used in every direction: give ss.peract(0.006), or give the network sides=`.
- **Per edge.** The engine labels each edge's two ends by side once a step (for `ss.MFNet` from `people.female`), and looks up the hazard for each direction of every edge (a partnership can transmit either way, except on one-way networks).
- **Printing.** `describe()` prints each network's sides and directions (`mfnet: sides m, f (people.female); directions m2f, f2m`), and `ss.peract` prints its keys against them.

So `m2f` is not magic in the engine: it is `ss.MFNet`'s pair of labels, and a coded network declares its own (`sides = dict(m=..., f=...)` as a class attribute, or `sides=` when it is made). STIsim's `StructuredSexual` declares `m` and `f`; HPVsim's `SexualNetwork`, which puts women in `p1`, declares the same two sides, so the same keys mean the same thing on both (HP3).

Positional pairs are removed. The one network that needed a pair for something other than sex, mother-to-child transmission (`[b, 0]` in Starsim 3), becomes one-way: `ss.PrenatalNet`, `ss.MaternalNet`, and `ss.PostnatalNet` transmit only from `p1` (the mother) to `p2` (the child), so a single number is enough. Today a single number on them also transmits from child to mother.

### Injecting drug use

A needle-sharing network beside a sexual one, for HIV (and hepatitis C), with a per-act probability for each sharing of a needle and no direction:

```python
pwid = ss.ppl.pwid                                                   # a boolean property of people (or a level of a stratification)
needles = ss.PartnerNet(name='needles', where=pwid, n_contacts=2, dur=ss.months(6), acts=ss.permonth(20))   # sharing partners among people who inject; no sides
mf = ss.MFNet()                                                      # sides m and f
hiv = ss.Disease(
    infection = ss.transmission('S -> I', beta=dict(mfnet=ss.peract(m2f=0.001, f2m=0.0005), needles=ss.peract(0.006))),
    ...
)
nsp = ss.modify(hiv.infection, multiply=0.2, share=0.5, route='needles', start=2005)   # needle and syringe programs: half of injections with a clean needle, 80% less risk
```

- The needle network has no sides, so its value has no keys, and keys there raise as above. If a study gave different risks by sex for sharing, `ss.PartnerNet(where=pwid, sides='sex')` would make its directions `m2f`, `f2m`, `m2m`, `f2f`.
- A person in both networks gets the hazards of both routes added (as for any two routes), so a man who injects and has a female partner can be infected by either.
- Needle and syringe programs, opioid substitution (fewer injections: `ss.modify('needles.pars.acts', multiply=0.5)`), and condoms are each an `ss.modify` on their own route; none is a parameter of the disease or of the network.
- `where=` on `ss.PartnerNet` (who takes part, a condition over people, as `ss.modify`'s `where=`) is new; today a `PartnerNet` takes in everyone. Under the compartmental solvers `pwid` must be a dimension (`sim.stratify(pwid=[False, True])`), as for any property (`behavior_vocabulary.md`, decision 5).

Under an ODE with a contact matrix over sex (ST23), there are no acts and so no `ss.peract`; a different chance by sex is `beta=ss.by(sex=...)`, which is by the susceptible's sex, as every `ss.by` in a rate is. On a heterosexual matrix that is the direction. `describe()` says so.

### Acts (D4)

`acts=` is the number of acts per unit time in each partnership: a rate (`ss.perweek(2)`, `ss.freqperyear(80)`), the same for every partnership, or a distribution of rates (`ss.lognorm_ex(ss.freqperyear(80), ss.freqperyear(30))`), drawn once when the partnership forms. The edge keeps `acts` per step as a real number (`a·dt`), and the exponent is real, never truncated. On one-off contacts (`between=` with no `dur=`), `acts=` is the number of acts in each contact, a bare number or a distribution of counts.

- Today's `ss.MFNet` default, `ss.poisson(lam=ss.freqperyear(80))`, is an integer count per step drawn once, and gives 0.531 monthly and 0.465 weekly for the annual chance of transmission at per-act 0.01 (against 0.552). The default becomes STIsim's lognormal of rates, which keeps the variation between partnerships and loses the dependence on `dt`.
- A distribution over whole numbers with a time unit in its parameters (`ss.poisson(lam=ss.freqperyear(80))`) on lasting partnerships raises, saying that a count per step drawn once depends on the step and giving the rate form.
- A bare number for `acts=` on lasting partnerships raises (`ss.PartnerNet`'s docstring today says "acts in each partnership in a step", which depends on the step), asking for a rate such as `ss.perweek(2)`.
- `describe()` prints the mean acts per unit time and the mean acts per partnership (`a·D`).

### What rel_trans, rel_sus, edge weights, and modifiers scale (D5)

**Decided by Cliff (2026-10-19): the hazard.** They multiply the per-act hazard `h_d`, as in `H_e` above, so they act on each act: `rel_sus=0.5` halves the hazard of every act, and a modifier with `share=` acts on that share of acts. `ss.modify(syph.pars.beta, multiply=0.5)` scales `h_d` too (every rate expression is a hazard; `−ln(1 − p)` is the per-act one).

Cliff's case: HIV with `beta=0.03` per act and `rel_trans=10` for acute infection gives `1 − (1 − 0.03)^10 = 0.263` per act, not 0.30. The reasons for the hazard over multiplying the probability:

- It never passes 1: `rel_trans=40` gives 0.70, where the linear form gives 1.2 and needs a clip, which would make `rel_trans` 40 and 400 the same.
- Factors compose: `rel 2` then `rel 5` is the same as `rel 10`, in any order, and with any number of modifiers. A clipped product does not.
- A share of acts is one multiplier: a condom of efficacy `ε` in a share `c` of acts is `1 − c·ε` on the hazard, exactly, so condoms, microbicides, and needle programs need no term of their own in the engine (D6).
- The two readings are close whenever `beta·rel` is small, which covers most per-act values: at `beta=0.003`, 0.0296 against 0.03.

For small `p` it is the same as scaling `p`; for large `p` and many acts it differs from Starsim 3, which scaled the per-partnership probability after the acts.

**Coded networks keep the Starsim 3 contract.** A network subclass that overrides `net_beta(disease_beta, inds, disease)` is called once per direction with the per-act probability `p_d` (unconverted, as Starsim 3 did), and its result is used as the per-partnership probability for the step, scaled by `rel_trans` and `rel_sus` after, as in Starsim 3. So STIsim's and HPVsim's unchanged networks give Starsim 3's numbers, and deleting the override moves them to the core form. With named directions, `net_beta()` is called per direction of the edge's sexes, with `inds` the edges of that direction.

### Condoms and microbicides (ST21, D6)

Cliff: a condom or a microbicide belongs in the intervention, not in the disease or the network, through `ss.modify`. Two keywords on `ss.modify` make this work, each useful beyond condoms:

- `route=` (a network name, or a list) limits a modifier on a transmission to the edges of that route. Without it, a modifier on a transmission applies on every route, as today.
- `share=` (a number, an `ss.timeseries`, an `ss.by`, or, with `route=`, a function of the network returning one value per edge) is the share of acts (or, on a route without acts, of contacts) the modifier acts on. Since modifiers multiply the hazard, `multiply=m, share=s` multiplies the hazard by `1 − s·(1 − m)`, exactly.

```python
use = ss.timeseries({2000: 0.05, 2020: 0.3})                                        # share of acts with a condom
condoms = [ss.modify(ng.infection, multiply=0.1, share=use, route='mfnet'),         # gonorrhea, efficacy 0.9
           ss.modify(ct.infection, multiply=0.6, share=use, route='mfnet')]         # chlamydia, efficacy 0.4
by_pair = ss.modify(ng.infection, multiply=0.1, share=lambda net: use_by_pair(net), route='structuredsexual')   # use by partnership type, one value per edge (agents)
gel = ss.modify(hiv.rel_sus, multiply=0.4, share=0.7, where=ss.ppl.uses_gel)        # a microbicide: 60% less risk, applied in 70% of acts by the women who use it
```

- The efficacy differs by pathogen (STIsim: gonorrhea 0.9, chlamydia 0.4, syphilis 0), so there is one modifier per disease; the share of use is shared by naming it once.
- A coded network that sets `edges.condoms` itself (STIsim's `StructuredSexual`) is read with `share=lambda net: net.edges.condoms`.
- Against STIsim's form, which multiplies the probability (`p → p(1 − ε)` for protected acts), the hazard form gives `1 − (1 − p)^(1 − ε)`: 0.035 against 0.030 at `p=0.3, ε=0.9`, and the same to three figures at `p=0.01`. Over a partnership-step with `c=0.5, ε=0.9, p=0.1`, 10 acts: 0.440 against STIsim's 0.438.
- Under the compartmental solvers, `share=` as a number, time series, or `ss.by` over a dimension is exact (it is a multiplier on the group's hazard); a function per edge is agents only (`CapabilityError` under an explicit compartmental solver, agents under `'auto'`).
- `describe()` lists each such modifier under its route (`on mfnet: × (1 − 0.9 × share 5% in 2000 to 30% in 2020), "condoms"`).

### Source weights (ST17, D7)

`source=` on `ss.transmission` takes, besides a state or a list, a dict of weights: `source=dict(P=1, Sec=1, EL=0.55, LL=0.05)`, which is the operator form `beta * (P + Sec + 0.55*EL + 0.05*LL)`. A weight is a number, an `ss.par`, or an expression. The operator form is fixed so that a dict or `ss.peract` `beta` can multiply a weighted sum (`graph.py:277-295` builds `as_expr(beta) * source`; the `TypeError` is in the expression's multiplication of a dict-valued parameter by a sum). Each route's beta multiplies the whole weighted sum. Weights that differ by route are written as separate transmissions (D7).

### Two-mode networks (ST22, D8)

`ss.PartnerNet(between=(a, b), n_contacts=..., dur=..., acts=...)`:

- `a` and `b` are conditions over people, or names of a stratification's levels or of boolean properties of people; everyone in neither mode is never in this network. A person in both raises.
- Each edge joins one person of each mode; `p1` is in `a`. The modes are the network's sides, so its directions are `a2b` and `b2a`.
- With `dur=`: lasting partnerships; `n_contacts` is partners at a time, a number or `dict(fsw=..., client=...)`; the targets for the two modes must balance (`N_a·k_a = N_b·k_b`) only in the mean: the side with fewer available partners waits, as men and women do in `ss.MFNet`, and `net.check()` reports both.
- With no `dur=`: one-off contacts, redrawn every step; `n_contacts` is a rate per person of one mode (`dict(client=ss.permonth(1))`), and the other mode's contacts follow. Contacts are spread over the other mode at random (a weight per person, such as STIsim's `sw_intensity`, is a later addition).
- `acts=` as above (acts per contact for one-off contacts). Condoms in sex work are an `ss.modify(..., route='sw', share=...)` as on any route.

### Contact matrix over several dimensions (ST23)

`ss.contacts(matrix, dims=['sex', 'risk'])`. `matrix[i, k, j, l]` is contacts per unit time of one person at levels `(i, k)` with people at `(j, l)`, or the same flattened to `(6, 6)`. Shapes are checked against the dimensions' levels when the sim is initialized (`ShapeError`, naming each dimension's levels in order), and `ss.contacts()` keeps no `beta` of its own. The matrix is not balanced for the user: an unbalanced matrix over sex means men and women report different numbers of contacts, which the run takes as given (as any contact matrix is taken today).

## Semantics under every solver

| Feature | ODE | Binomial | Gillespie | Agents |
|---|---|---|---|---|
| `ss.peract` on `ss.MFNet`/`ss.PartnerNet` (lasting partnerships) | the partnership approximation below (step 9), marked `approx` | as the ODE, binomial draws | as the ODE, as events | per edge, `H_e` |
| Directions | by the groups of each side (each side must be a dimension or a level of one), or `ss.by(sex=...)` on a contact matrix | as the ODE | as the ODE | per edge, from the sides |
| `acts=` | the mean acts per unit time `a` in the approximation | as the ODE | as the ODE | `a_e·dt` per edge per step |
| `ss.modify(..., route=, share=)` | a multiplier `1 − s(1 − m)` on that route's hazard, by group when `share=` is by group; a function per edge is agents only | as the ODE | as the ODE | per edge |
| `source=` weights | exact | exact | exact | exact |
| Two-mode, one-off contacts | exact: a contact matrix between the two modes, with `1 − exp(−k·h)` per contact | exact | exact | per contact |
| Two-mode, lasting | as `ss.PartnerNet` (`CapabilityError` until step 9) | as the ODE | as the ODE | per edge |
| `ss.contacts(matrix, dims=)` | exact | exact | exact | exact (mixing pools by group) |

### The partnership approximation (D9)

Cliff: build it for the ODE and the other compartmental solvers too, since partnership networks are a common use case, even if the result is approximate. It is the form `06-routes.md` names for networks of partnerships ("Networks of partnerships"), and it is used for `ss.MFNet` and `ss.PartnerNet` (lasting partnerships, with or without `between=`) under the ODE, binomial, and Gillespie solvers. One-off contacts need no approximation (they are a contact matrix between groups).

**What it computes, per step, for each susceptible group `i` and each group `j` of partners:**

- `φ_i = n_i/D̄`: the rate at which a person in `i` forms partnerships (partners at a time over the mean duration). With `match=` or two modes, `m_ij` is the share of `i`'s new partners in `j`, from the network's mixing, worked out once at build (and again when `n_contacts` is modified).
- `H = a·h_d·M·r`: the hazard of transmission per unit time in a partnership with an infectious partner, with `a` the mean acts per unit time, `h_d` the per-act hazard for the direction from `j`'s side to `i`'s, `M` the route's modifiers (`share=` included), and `r` the group means of `rel_trans` and `rel_sus`.
- `k_j = 1/D̄ + γ_j + μ`: the rate at which such a partnership stops being able to transmit, because it ends (`1/D̄`), the infectious partner leaves the source state (`γ_j`, the total of its exit rates, or one over its dwell's mean), or either partner dies (`μ`).
- `P_{ij} = H/(H + k_j)`: the chance the partnership transmits before that, for exponential durations. For other duration distributions (lognormal, the default of `ss.MFNet`), `P = 1 − E[exp(−H·T)]` over the time `T` the partnership can transmit, by a fixed 32-point quadrature over the duration distribution, made at build.
- `λ_i = φ_i · Σ_j m_ij · Σ_s w_s·P_{ij,s}·I_{j,s}/N_j`: the force of infection, frequency-dependent, with `w_s` the source weights (`source=`). Source states with different exits have their own `P`.

`P` is worked out again each step from the current values, so time series, `ss.change`, and modifiers act as they do elsewhere; it is a closed form (or 32 terms) per group pair, so it costs little. Under binomial and Gillespie, `λ_i` is used as any force of infection is.

**Its error.** It treats each partnership as formed with someone whose state is drawn from the population of `j`, and as transmitting once. It leaves out what the network remembers: a person's current partners are more likely to share their state (an infected person's partners were exposed to them, and a susceptible's partner who is still susceptible is less likely to be infected later), and people with several partners at once link chains of infection. So:

- It is close when partners change often relative to the infectious period, and when partnerships are many and short (sex work, high-risk groups with many partners): the population each new partner comes from is then the right reference.
- It is poor for long, mostly monogamous partnerships with a long infectious period (HIV in a stable, low-risk population): infection stays within pairs, which the approximation spreads into the population, so it overstates spread early and misplaces it between groups. With concurrency (`concurrency=`), the error goes the other way in part.
- Measured once, in STIsim's hand-built risk-group ODE (`stisim/v4port/networks/riskgroup_ode.py`): 8.2% prevalence against agents' 10.4%. The "mean current contacts" form gave 54%, with the risk gradient reversed, so it is not offered.

Step 9 measures it against agents on three cases (STIsim's risk groups, a short infection like gonorrhea, a long one like HIV) and writes the numbers into the docs and the spec's table of solver differences (D-T3).

**How it is reported.** It is approximate, so it follows the `approx` rules: `describe()` prints, under the route, `partnership approximation (approx): partnerships formed at 0.5 a year; chance over a partnership m2f 0.42, f2m 0.25`; `approx=False` raises `CapabilityError` naming the solver and `solver='abm'`; and `sim.solvers()` lists it among what an ODE run approximates. Each side must be a dimension (`ss.MFNet` needs `sim.stratify(sex=['female', 'male'])`): without one it raises, naming the `stratify()` call, as for any property that is not a dimension.

**Per-act against per-partnership.** Three quantities are easy to confuse, and `describe()` prints the first and derives the others:

- *Per act* (`ss.peract(p)`): a property of the pathogen and the act; independent of the step and of how long partnerships last.
- *Per contact per unit time* (a bare number or rate on a network without acts): a hazard per edge; independent of the step, but it folds acts into `beta`, so a change in acts (an intervention, a by-age pattern) cannot be expressed.
- *Per partnership* (`P_d` above): what the compartmental form uses; it depends on acts and on duration, so it is derived, never given. A user who has only a per-partnership number (from a study of couples) gives it per act by solving `1 − (1 − p)^(a·D) = P`, which `describe()` can show the other way round.

**`describe()`** prints, per route, the reading and the derived numbers:

```
infection    S -> E   contact-mediated, source P + Sec + 0.55 EL + 0.05 LL
             beta on mfnet: per act, m2f 0.3, f2m 0.15 (mfnet's sides m, f: people's sex)
                    acts 80 a year per partnership (lognormal, sd 30); over a mean partnership of 2.0 years (exponential), m2f 0.98, f2m 0.96
                    × (1 − 0.5 × share 5% in 2000 to 30% in 2020), "condoms"
             beta on maternalnet: 0
```

and for a bare number, `beta on mfnet: per act (a bare number on a network with acts), 0.01`.

**The model spec** (`to_spec()`) gets one new quantity type, alongside `per`, `prob`, `freq`, and `dur` (spec 14, "Quantities"):

```json
{"type": "peract", "m2f": 0.3, "f2m": 0.15}
{"type": "peract", "value": 0.01}
```

A bare number on a network with acts is written as `{"type": "peract", "value": 0.01}` with `"guessed": ["transitions.infection.beta"]`, so the spec never depends on knowing the network to be read. Source weights are written in the rate text (`"rate": "beta * (P + Sec + 0.55*EL + 0.05*LL)"`), as the operator form already is. `acts`, `sides`, `between`, `where`, and `dims` belong to the networks, which the spec does not yet carry (`to_code()` lists them under `LossyExport`); they are written with the networks when networks enter the spec, which is out of scope here.

**`to_code()`** writes `ss.peract(m2f=0.3, f2m=0.15)` (each value as its `ss.par` if it has one) and `source=dict(...)` as the operator form. Round trip: `to_code(to_spec(sim))` reproduces the same `beta`.

## Fit with what exists

**New public names: one.** `ss.peract`. Lowercase, as the typed values are, and in the `ss.perday` family by name. The alternative, a keyword on the network that says "beta is per act" (`ss.MFNet(beta_per='act')`), puts a fact about the pathogen on the network; extending `ss.prob` with a period of "one act" (`ss.prob(0.01, 'act')`) makes an act look like a time unit, which `ss.prob`'s conversions would then mishandle (`ss.prob(0.01, 'act') * ss.days(3)`). A typed value carries the direction names where they belong, on the transmission.

**New keywords:** `sides=` (on networks, and a class attribute for coded ones), `between=` and `where=` (on `ss.PartnerNet`), `route=` and `share=` (on `ss.modify`), `dims=` (on `ss.contacts`), and a dict for `source=` (on `ss.transmission`). Each extends an existing object instead of adding one. New attribute: `net.directions`. Not added (revision): `condoms=` on networks and `condom_efficacy=` on `ss.peract`, which the intervention form replaces.

**Reused:** `ss.by` and `ss.timeseries` for values that vary; `ss.par` for fitting; `ss.modify` for condoms, microbicides, and needle programs; `ss.PartnerNet`'s targets and `net.check()` for the two-mode network; the existing `RouteMismatch`, `WrongDimension`, `OutOfBounds`, `CapabilityError`, and `ShapeError`; stratifications (and h4's dimensions over people's properties) for the modes.

**Removed or replaced:**

- Positional `beta` pairs (`[b1, b2]`), everywhere: replaced by `ss.peract(m2f=, f2m=)` and one-way mother-to-child networks. No alias; the migration script rewrites `mf`/`mfnet` pairs and flags others.
- The rate conversion and ratio scaling of a route's betas in `Agents.infect()` (`agents.py:1289-1302`), replaced by per-direction hazards.
- The N7 warning on networks with acts (kept for networks without).
- `ss.MFNet`'s integer acts per step; `ss.PartnerNet`'s bare `acts=` per step.
- `compartmental.py:643`'s "a beta for each direction of a route needs a directed network" (replaced by the partnership approximation, step 9).
- STIsim's `edges.condoms` split and its `eff_condom` disease parameter, replaced by `ss.modify(..., share=, route=)`.
- Spec text: `03-states-and-transitions.md` around line 191 to 199 ("A pair is the two directions of a directed network"), and `06-routes.md` lines 76 to 94 ("Beta per route and direction", "Units"), the per-edge form (lines 64 to 74, `β` as the per-act hazard and `acts` per unit time), the table row for partnership networks (line 179), and the `ss.PartnerNet` paragraph (line 237, acts as a rate). `02-types.md` gains `ss.peract` in its table (not a time value), `14-model-spec.md` the `peract` quantity, `15-errors.md` the new messages, `17-vocabulary.md` a row for `ss.peract` and the new keywords; `06-routes.md` the sides and directions of a network, and the partnership approximation in full; `08` (or wherever `ss.modify` is specified) `route=` and `share=`; `12-backends.md` the approximation's entry in the table of what each solver approximates.

## Implementation plan

| Step | What | Size | Files |
|---|---|---|---|
| 1 | `ss.peract`: construction, bounds, expressions of `ss.par`, repr, `ss.by`/`ss.timeseries` values | S | `time.py`, `__init__.py` |
| 2 | Sides and directions: `sides=` and `net.directions` on networks (`ss.MFNet` `m`/`f`, mother-to-child one way); the agent engine's per-direction per-act hazards, resolved per edge from the sides; a kernel computing `1 − exp(−H_e)` with `rel_trans`, `rel_sus`, and modifiers inside the exponent (D5, the hazard); the Starsim 3 contract for coded `net_beta()` overrides | S to M (about 80 lines) | `solvers/agents.py`, `networks.py` (`Network.sides`, `directions`, a hazard method), the Numba kernel |
| 3 | Acts as a rate: `ss.MFNet` and `ss.PartnerNet` store real acts per step; the new `ss.MFNet` default; the errors for count distributions and bare `acts=` on lasting partnerships | S | `networks.py` |
| 4 | The reading rules: bare number per act on acts networks, `WrongDimension` for typed rates there, `RouteMismatch` for pairs, for `ss.peract` without acts, and for keys that are not the route's directions (listing them); the N7 warning narrowed; migration rows and the script's pair rewrite | S | `solvers/engine.py`, `solvers/choose.py`, `networks.py`, `migrate.py`, `skills/migrate-starsim3/SKILL.md` |
| 5 | Interventions on a route: `route=` and `share=` on `ss.modify` (a multiplier `1 − s(1 − m)` on the hazard; per edge as agents, per group under the compartmental solvers); `where=` on `ss.PartnerNet`; the injecting drug use example as a test | S to M | `connectors.py`, `solvers/agents.py`, `solvers/engine.py`, `networks.py` |
| 6 | Source weights: `source=dict(...)`, and a dict or `ss.peract` `beta` times a weighted sum | S to M | `graph.py`, `expressions.py`, `solvers/agents.py`, `solvers/compartmental.py` |
| 7 | `ss.contacts(matrix, dims=)`: the full matrix in `ss.contacts()` and in `Mixing.dim_matrices()` instead of `np.kron` | S | `networks.py`, `solvers/mixing.py` |
| 8 | Two-mode `between=` on `ss.PartnerNet`, lasting and one-off, and its compartmental form for one-off contacts (a contact matrix between the modes) | M | `networks.py`, `solvers/mixing.py`, `solvers/choose.py` |
| 9 | (D9, now in scope) The partnership approximation for `ss.PartnerNet`/`ss.MFNet` under the ODE, binomial, and Gillespie solvers, per act and per direction, with the source state's exits and deaths in `k`, the quadrature for non-exponential durations, the `approx` reporting, and its measured error on three cases against agents | M to L | `solvers/mixing.py`, `solvers/compartmental.py`, `solvers/choose.py`, `presentation.py` |
| 10 | `describe()`, `to_spec()` (`peract` quantity), `to_code()`, and the spec and docs edits listed above; a CHANGELOG line for the result changes (D4, D5, the `ss.MFNet` default) | S | `presentation.py`, `spec.py`, `docs/user_guide/modules_networks.qmd`, `v4_design/design/*.md` |

Steps 1 to 4 close the silent wrong answers (ST2, ST3, HP3) and are about two days of agent work together; they need D3's sides (decided here, pending Cliff's look at the open decisions below) and D5 (decided). 5 to 7 are about two more; 8 is a few days; 9 is about a week, and can start once step 2's hazards exist.

**Tests**, on numbers a user sees:

- Per act, one direction: per-act 0.01 on `ss.MFNet` with 80 acts a year, the ST2 probe's SI over 10 years, gives the same prevalence at monthly and weekly steps within Monte Carlo error (today 0.37 against 0.29 for the coded form, 0.39 against 0.12 for `ss.probpermonth`); and a single partnership's annual chance of transmission is 0.552 at both steps.
- Directions: f2m per partnership-step at 2, 4, 8 acts is 0.278, 0.478, 0.728 (the v3 numbers); `ss.peract(m2f=b, f2m=0)` infects only women on `ss.MFNet` and on a test network that puts women in `p1` (HP3).
- Readings: a bare per-act number gives the same results as `ss.peract`; `ss.permonth(b)` on `ss.MFNet` raises `WrongDimension`; a pair raises `RouteMismatch` with the `ss.peract` spelling in its message; `ss.peract` on `ss.RandomNet` raises `RouteMismatch`; a missing `f2m` on `ss.MFNet` raises at initialization; `m2m` on `ss.MFNet` raises listing `m2f, f2m`; keys on the needle network raise saying it has no sides.
- D5: `beta=0.03` with `rel_trans=10` gives 0.263 per act (one edge, one act); `rel_trans=2` with a modifier `multiply=5` equals `rel_trans=10`.
- Coded overrides: an unchanged network subclass with STIsim's `net_beta()` gives Starsim 3's per-partnership probabilities (`tests/switching/` style).
- Condoms: `ss.modify(..., multiply=0, share=1, route=...)` transmits nothing on that route and leaves other routes alone; `multiply=0.1, share=0.5`, `p=0.1`, 10 acts gives `1 − exp(−10·0.1054·0.55) = 0.440` per partnership-step; the same modifier gives the same prevalence as agents and as an ODE over a mixing pool (share by group).
- Injecting drug use: the example above runs; with `beta` 0 on `mfnet`, nobody outside `pwid` is infected.
- Partnership approximation: on the three cases of step 9, prevalence at 10 and 20 years under the ODE against agents (20 runs), with the gaps recorded rather than asserted tight (the test checks they stay within what was measured); `approx=False` raises; an exponential-duration case with no recovery matches `H/(H + 1/D̄)` per partnership exactly in a two-person check.
- Source weights: `source=dict(P=1, EL=0.55)` and the operator form give identical results, as agents and as an ODE over a mixing pool.
- `ss.contacts(dims=)`: a separable matrix given whole gives identical results to `ss.contacts(sex=..., risk=...)`; a non-separable sex-by-risk matrix gives ODE and agents prevalence within the contract's tolerance (`tests/test_contracts.py`).
- Two-mode: one-off contacts at `ss.permonth(1)` per client give 12 contacts a year at monthly and weekly steps; ODE and agents agree for one-off contacts.
- `describe()` golden output for a sexual network model; `to_code(to_spec())` round trip with `ss.peract`.

**What the downstream ports then delete:** STIsim: `BaseSTI.validate_beta()`'s structuredsexual lines (the `beta_m2f`/`rel_beta_f2m` parameters can stay as STIsim's user-facing names, building one `ss.peract`), `SexualNetwork.net_beta()`, syphilis's `infect()` override, the integer truncation of acts; the port's `-log(1 - b)` rates, one dict per weight, and `dict.__setitem__` for `eff_condom` (STIsim's condom data become a `share=` function or time series, one `ss.modify` per disease). HPVsim: `validate_beta()`, `_clip_beta()`, `SexualNetwork.net_beta()`, HIV's `validate_beta()`; the port's `rate = lambda b: ss.peryear(-np.log(1 - b)/dt)` and its `[f2m, m2f]` comment. Starsim's own `ssl.HIV` loses its unused `eff_condoms` parameter.

**Downstream note, independent of v4:** STIsim's acts are drawn per partnership in per-step units and truncated with `.astype(int)` (`stisim/networks/mf.py:297`, `fsw.py:220`), which loses 7% of acts at monthly steps and 32% at weekly ones (16% of partnerships with none). HPVsim keeps acts real (`network.py:414`). Under D4 the core network does what HPVsim does.

## Out of scope

- Time-varying infectiousness within a state (syphilis's latent infectiousness decaying with time since entering latency; HIV's by viral load): a per-person `rel_trans` that depends on time since an event, from h4's behavior vocabulary (time since an event). Erlang stages with weights per stage are a workaround now.
- Declared versions of STIsim's structured network (`ss.SexualNet`, N15: degree by sex, age preference, durations by type under one concurrency cap, MSM subsets). The two-mode network here is one of N15's five pieces.
- Weights per person within a mode (sex work intensity), partner notification and contact tracing over these networks, and transmission trees by direction (R17's per-route flows).
- Networks in the model spec (acts, sides, modes, and contact matrices would be written with them). A modifier's `route=` and `share=` are written with the modifier.
- Balancing a contact matrix over sex for the user.

## Dependencies and overlaps with other Phase H documents

- **h4, behavior vocabulary and dimensions over people's properties.** Assumed interface: a condition or a level over a property of people (`sw` level `fsw`, a boolean `fsw`, `debuted`), usable as `between=(a, b)` and by `ss.by(...)`, evaluated each step. Condom use by a property of both partners (STIsim's risk-group pairs) would use the same thing; until then it is a `share=` function of the network or a coded network's `edges.condoms`. `ss.ppl` (the lazy reference to people's properties, renamed from `ss.People` there) is used here for sides and `where=`.
- **h1, care cascade.** PrEP and condom promotion as deliveries: PrEP is a product that modifies `rel_sus`, which under D5 scales the per-act hazard; a condom distribution is the `ss.modify(..., share=)` above, given as an intervention, or a delivery that sets a per-person condom propensity read by `share=` (out of scope here). Partner notification needs network conditions (deferred).
- **h2, family of diseases.** Each genotype of HPVsim has its own `ss.peract` (`m2f=b_g*transm2f`); a family default (`beta` shared, scaled per member) is h2's to define. Nothing here needs the crossing.
- **D10 (fixed as `CapabilityError`).** Step 9 is the approximation D10's text proposes, with per-act values.
- **D15 (fixed).** Directions and `dims=['sex', ...]` rely on the `sex` dimension being `people.female` as agents.

## Decisions still open

For Cliff's `[CK:]`:

| # | Decision | Options | Recommendation |
|---|---|---|---|
| R1 | How a network names its directions (D3) | (a) `sides=`, labels to conditions, and keys `<side>2<side>` (`m2f`, `fsw2client`, `mother2child`); (b) the same sides, but keys as pairs (`{('m', 'f'): 0.3}`); (c) keys only from sex, as before, with other networks taking one value | (a). The engine knows no names; each network declares its own, `net.directions` lists them, and errors list them. Keys stay short for the common case. (b) is unambiguous for long labels but clumsy to type; (c) leaves out sex work and anything not by sex [CK: i think A and B should both work] **Decided:** both: `'m2f'` and `('m', 'f')` are the same key |
| R2 | Condoms and other per-act protection (D6) | (a) `ss.modify(<disease>.infection, multiply=1 − ε, share=c, route=...)`, one per disease; (b) as (a), plus a small `ss.condoms(use=, efficacy=dict(ng=0.9, ct=0.4))` that makes those modifiers; (c) as (a), with `share=` left out (the user writes `multiply=1 - c*ε`) | (a). Two keywords on `ss.modify`, no new name, and the same form for microbicides, PrEP adherence, and needle programs. (b) saves a line per disease but adds a name for one use; (c) works only for a number, not for a time series or a function per edge, unless arithmetic on time series is added [CK: hm ... i think i like efficacy, for some interventions] **Decided:** (a), and `ss.modify` also takes `efficacy=`, the natural spelling for a product that protects: `ss.modify(ng.infection, efficacy=0.9, share=use, route='mfnet')` multiplies the hazard by `1 − share·efficacy` (the same as `multiply=1 − efficacy`; giving both raises) |
| R3 | Where `route=` lives | (a) a keyword on `ss.modify`; (b) in the target, `'ng.infection@mfnet'` or `ng.infection.on('mfnet')` | (a). A path suffix would be a new mini-syntax; `ng.infection.<x>` already means the destination `<x>` (`destination_results.md`), so `.mfnet` there would clash [CK: ok] **Decided:** (a) |
| R4 | When the partnership approximation is used (D9) | (a) only when a compartmental solver is asked for; `'auto'` keeps choosing agents for partnership networks; (b) `'auto'` may choose it for a large population, with a note | (a). Agents are exact for partnerships and `'auto'` should not trade exactness away without being asked; a user who asks for `solver='ode'` gets the approximation, printed as `approx` [CK: ok] **Decided:** (a) |
| R5 | A modifier that multiplies probability rather than hazard, for a user who wants Starsim 3's form | (a) none: the hazard everywhere (D5); (b) a keyword on `ss.modify` (`scale='prob'`) | (a). One reading; the difference is small where it matters (below 0.01 per act) and documented [CK: ok] **Decided:** (a) |
| R6 | `where=` on `ss.PartnerNet` | (a) a condition over people, as on `ss.modify`; (b) a name of a boolean property only | (a), the same spelling everywhere [CK: ok] **Decided:** (a) |
