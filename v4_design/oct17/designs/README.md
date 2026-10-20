# Phase H designs

The four design documents for Phase H of [implementation_oct18.md](../implementation_oct18.md) (decision D-F: one design each before code). Each starts with a one-page summary and ends with an implementation plan. Code waits on Cliff's answers below; write them as `[CK: ...]` after each line.

| Document | Covers | New public names | Size | What the ports delete |
|---|---|---|---|---|
| [care_cascade.md](care_cascade.md) | Events, remembered results, `on=`, stock targets, tests with several results, products over several diseases, deliveries in the kernels | `ss.Event` | 11 steps; the kernels part L | About 2,200 lines of STIsim, most of HPVsim's wrapper classes |
| [disease_family.md](disease_family.md) | Genotypes and strains: union results, joint seeding, cross-immunity, one cancer per person, the crossing's size | `ss.Diseases`, `ss.immunity` | About two weeks; the compartmental part M to L | About 300 of HPVsim's 402 lines of family code |
| [per_act_transmission.md](per_act_transmission.md) | Per-act probabilities with named directions, condoms, route weights, two-mode networks, contact matrices over several dimensions | `ss.peract` | ST2, ST3, HP3 in about two days; the rest about two more, plus M for the two-mode network | STIsim's per-act conversions |
| [behavior_vocabulary.md](behavior_vocabulary.md) | Regressions, tables, counters, time since an event, dimensions over people's properties, units of bare numbers | none | Steps 1 and 2 S; contraception declarable in about two weeks | About 60 of 134 lines of FPsim's `contra_v4.py` |
| [destination_results.md](destination_results.md) | HP20: the flow per destination of one transition with several (new cancers out of `C -> [X, R]`) | none (a result `cin.X`, the HP6 and cascade name) | About two days | HPVsim's cancer counters and by-age cancer analyzer |
| [given_dwell.md](given_dwell.md) | HP18: a branch probability that depends on the time spent in the state (CIN and cancer given the duration) | none (`P.dwell`, an attribute of a state; `ss.given_dwell` not added) | About four days; the compartmental conveyor M | About 60 lines of the port's `hpv_declared.py`, then most of `set_prognoses()` |
| [people_as_data.md](people_as_data.md) | TLO review group 1: people from a table (TL1), `init=` per person, time already spent at the start (TL2), splits given other properties (TL5), a module of properties (TL10), only occupied groups as agents (TL7) | `ss.Properties`, `ss.People.from_df()` (and `since=`) | 9 steps, about two weeks; occupied groups L | TLO's post-`init()` writes and engine arrays, the Lifestyle port's loops and fake disease; FPsim's coded starting draws |
| [burden_of_disease.md](burden_of_disease.md) | TLO review group 6 (TL50, TL51) and TL23's `net_of=`: disability weights, YLD, YLL, DALYs by cause, person-time, combining weights per person, discounting | `ss.Burden`, `ss.integral` (and `disability=`, `cause=`, `net_of=`) | 7 steps, about a week | TLO's burden analyzer (65 lines), HPVsim's `dalys` (about 100), STIsim's syphilis DALY analyzers, MIGHTI's multimorbidity and YLL code |

## Where the documents disagree (my recommendation)

1. **Values by disease on a product.** The cascade writes `efficacy=dict(hpv16=..., hpv18=...)`; the family writes `ss.by(hpv=...)`. Recommend `ss.by(<family>=...)`: a family is a dimension, and values by a dimension's level are already written with `ss.by`, so it is one spelling. A product over unrelated diseases (not a family) takes `ss.by(disease=...)` in the same way. [CK: ok]
2. **A family under the compartmental solvers.** The cascade assumes the crossing; the family document runs members side by side with couplings from shares, marked `approx`, since the exact crossing grows about fivefold per member (impossible at 9 or more genotypes). Recommend the family document's choice, which it owns. [CK: ok]
3. **Remembered results.** The cascade gives every event `.ever`, `.count`, and `.since` per person (kept only when read); the behavior document writes counters as `move=dict(parity='+1')` and time since an event as a `kind='clock'` dimension. Recommend one spelling for users, the cascade's (`tx.since < ss.years(1)`, `preg.conception.count`), with the behavior document's clock and counter dimensions as how a compartmental run carries them. A relative `move=` stays for a property that is not the count of one event. [CK: ok]

## Decisions, by document

**Care cascade** ([care_cascade.md](care_cascade.md)):
1. Follow-ups as peers (`on=screen.positive`); `on_positive=`/`on_negative=` removed.
2. No `ss.cascade` container for now; a cascade is a list of deliveries.
3. Several events in `on=` with `|` only; a list raises and points to `|`.
4. `.ever`/`.count`/`.since` on every event, kept only when read (the shared primitive; see disagreement 3).
5. Products have events too (the union over their deliveries, e.g. `lbc.since`).
6. `target=` only tops up; dropout comes from the product's `waning=` (STIsim's removal above the target is not reproduced).
7. Deliveries in the kernels as a short program of delivery steps (L), Numba first and then JAX. Today a treatment makes a ten-year daily ODE take 0.60 s against 0.065 s without one.
8. The `diagnosed` result is renamed `positive`, so every outcome is an event and a result of the same name.
9. Values by disease on a product: see disagreement 1.

**Family of diseases** ([disease_family.md](disease_family.md)):
1. A family is `ss.Diseases`, a container of ordinary declared diseases (not a replicated dimension).
2. Cross-immunity combined by `combine='max'|'sum'|'multiply'`, default `'max'`, the same object as the strain dimension's `immunity=`.
3. Compartmental solvers run members side by side, marked `approx` (within 5% on cancers and 10% on prevalence at protection 0.3; 25% to 50% off on prevalence where a member is nearly excluded); the exact crossing stays available explicitly and prints its size first.
4. Joint seeding with the family's `init_prev=` and `split=`; without them each member seeds as now; both raises.
5. One cancer per person as `exclusive='Ca'` on the family.
6. `of=` in `ss.immunity` takes states or sums of states now; continuous immunity per person after the behavior vocabulary.
7. (Fixed this session, Phase A style) the frozen expression multiplier: under the compartmental solvers `multiply=1 - 0.7*b.R.frac` was evaluated once at the start (1,605 against the agents' 414).

**Per-act transmission** ([per_act_transmission.md](per_act_transmission.md)):
1. A bare number as `beta` on a network with acts is a probability per act (as the spec and Starsim 3 have it); `to_code()` writes `ss.peract`.
2. A typed rate or per-period probability on a network with acts raises `WrongDimension`.
3. Directions named on `ss.peract` (`m2f`, `f2m`, `m2m`, `f2f`), worked out per edge from the partners' sex; positional pairs removed; mother-to-child networks transmit one way only.
4. `acts=` is acts per unit time, drawn once per partnership, used as a real exponent; `ss.MFNet`'s default becomes STIsim's lognormal (changes its results: at a weekly step today 21.5% of partnerships never have sex, so the yearly chance at 0.01 per act is 0.465 against 0.552).
5. `rel_trans`, `rel_sus`, edge weights, and modifiers scale the per-act hazard; networks that override `net_beta()` in code keep Starsim 3's behavior.
6. Condom efficacy on `ss.peract(condom_efficacy=)`, not a disease parameter the network reads.
7. Route-specific infectiousness as one transmission per set of `source=` weights.
8. A two-mode network as `between=(a, b)` on `ss.PartnerNet`.
9. Partnership networks keep raising `CapabilityError` under the compartmental solvers for now; the partnership approximation is specified and built later (M).

Also found: STIsim rounds acts per step down to whole numbers (`mf.py:297`, `fsw.py:220`), losing 32% of acts at a weekly step; that is STIsim's to fix.

**Behavior vocabulary** ([behavior_vocabulary.md](behavior_vocabulary.md)):
1. Regressions as expressions over people's properties, with `expit` and `ss.by` tables, not `ss.logistic`.
2. Tables as an `ss.by` table with a `to` column, normalized over the destinations it names, as one entry of `p=`, not `ss.table`.
3. Counters: see disagreement 3.
4. Time since an event: see disagreement 3.
5. A value by a property that is not a dimension runs as agents under `'auto'` and raises `CapabilityError` under an explicit compartmental solver, saying to stratify.
6. Numeric levels are always lower bin edges, and people hold the value, never the level's index.
7. A property people already have is kept and binned; `split=` for one raises (as spec 05 says).
8. Units: a bare `dt` takes the unit of `start`/`stop`; a number returned by a function takes its distribution's `unit=`; `ss.choice` and `ss.histogram` accept `unit=` and typed values.
9. FPsim declares contraceptive choice (and education, later); pregnancy and conception stay code, since `ss.Pregnancy` holds the mother-fetus structure STIsim and HPVsim share and runs only as agents either way.

**Results by destination** ([destination_results.md](destination_results.md)):
1. A result per destination named `'<transition>.<state>'` (`res.cin.X`, `res['cin.X']`), the name it has as a modifier target (HP6) and an event (cascade); no `to` axis.
2. Kept for every transition with several destinations (`results=` can leave them out).
3. `select(to=)` raises, saying to write `res['cin.X']`.
4. The destinations are stored and the total is their sum.

**Branch probability given the time in the state** ([given_dwell.md](given_dwell.md)):
1. Spelled `P.dwell`, an expression of the time in the state (`p=[logf2(P.dwell), None]`), not `ss.given_dwell(fn)`, a clock dimension, or stages declared by the user.
2. Allowed only in `p=` of a transition with several destinations out of that state for now (a rate by time in state later); on separate transitions it raises.
3. Under compartments, a conveyor of whole steps in the state with hazards from the dwell distribution and `p` per slot (exact to the step; about 400 slots per state for HPV at a quarterly step), not phase-type stages.
4. As agents, the time actually spent (to the middle of the exit's step), not the dwell drawn.

## What can start without waiting

- The behavior document's steps 1 and 2 (dimension and unit fixes) need only its decisions 5 to 8.
- The per-act document's steps 1 to 4 (ST2, ST3, HP3, about two days) need only its decisions 1 to 5.
- The family's agents-exact steps (1 to 4) need its decisions 1, 2, 4, and 5.
