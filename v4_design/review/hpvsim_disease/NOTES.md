# HPVsim v3 natural history on Starsim v4: review

`port.py` runs: 5,000 agents, 1980–2030, quarterly steps, 4 genotypes + HIV, about 2 s. As agents (`solver='auto'` picks `abm`: the networks and the uids functions). Output (seed 1): any-genotype HPV prevalence in women 15–65 goes 12–13% → 7%; CIN prevalence rises with age (0.2% at 15–25 to 4.1% at 55–65); 35 cancers, ASR about 34/100k (17 cancers in the last 20 y, so noisy); 110 reactivations; HPV prevalence at the end 13% in HIV+ women with CD4<500 against 5% in HIV− women. The parameters are uncalibrated and the network is generic `ss.MFNet`, so these numbers only show that the model runs.

Line counts: v3 `hpv.py` 916 + `cross_genotype.py` 503 + `hiv.py` 366 + genotype pars/severity utils ~250 ≈ 2,000 (this includes multiscale, HIV_incidence and par routing, which are not ported). Port: 447 lines, of which ~300 are the model and ~120 are the alternatives and checks.

## Feature by feature

### 1. Genotypes: one disease per genotype (v3 does the same)
- v3: `sim.py:199` `HPV(genotype=k)` per key. Each instance has its own states, and `CrossImmunity` (`cross_genotype.py:55`) ties them together.
- v4: `make_hpv(g)` (`port.py:80`) builds one declared `ss.Disease` per genotype from a dict of pars. The closures are bound to `g`.
- **Declarative. Clean.** A factory function is all it takes, and every genotype gets its own `n_*`, flows and `by('age')`.
- **The strain dimension cannot express HPV**, and it is not because of size:
  - Each person holds **one natural-history state** for all their strains. With `strain=['16','18','16+18',...]`, a woman with 16 in CIN and 18 in precin cannot be represented. Carriage is "several strains, one course". HPV has independent courses per genotype at the same time.
  - With 4 genotypes, carriage takes 2⁴−1 = 15 levels (`'hpv16+hpv18+hi5+ohr'` etc.). That runs (`strain_alternatives()`), but it needs an explicit superinfection transition `P >> P @ beta*(P+C)` or nobody ever gets a second strain. With HPVsim's historical 9+ genotypes it would be 511+ levels.
  - `immunity=` over a network: `E804 ... immunity= between strains, which cannot run over networks yet` (the implementation notes say so). HPV is network-only, so this is a hard stop.
  - `immunity=` is binary (the `strains_had` bits times a matrix). HPV immunity is a continuous per-agent level, drawn on each clearance, kept as a running max, gated by seroconversion, with separate susceptibility and severity matrices. That cannot be expressed as a binary history.
- Suggestion: spec 05 says co-infection "with a different natural history" needs a second disease. It should also say that **independent progression of the same natural history** (HPV, and also multi-strain malaria or pneumococcus) needs one disease per strain. Then the strain dimension should not be offered for this case. A helper `ss.Disease.replicate(names=[...], pars=by_name)` plus a "sibling group" that a connector can loop over would replace the factory and `isinstance` scans.

### 2. Progression whose probability depends on the sampled duration (logf2 severity curves)
- v3: `hpv.py:527–600`. Sample `dur_precin`; P(CIN) = `logf2(dur*rel_sev)`. If CIN, CIN onset is at `ti+dur`, otherwise clearance is at `ti+dur`. Then the same again for `dur_cin` and `cin_integral` → cancer.
- v4 (`port.py:112–120`): `P >> Pe @ dur`. Then **zero-dwell junction states** `Pe`/`Ce` with immediate `p=` branches, whose functions recompute the duration from the engine's entry times: `(hpv.ti_Pe[uids] - hpv.ti_P[uids])*dt_year`.
- **Partially declarative. Ugly.**
  - `p=` is decided **on entry**, before the dwell is drawn (`solvers.py:3506–3518`). So "branch at exit, as a function of the time spent" needs a junction state, which spec 03 says "is not needed and is not part of the language". It is needed here.
  - **A junction costs one full timestep**: people entering `Pe` are only scheduled on the next step (`check_junction_delay()`: mean 1.00 steps; mean `n_Pe` of 115 people sitting in a fake state). With two junctions on the cancer path and dt=0.25 y, that adds 0.5 y to every cancer trajectory. The junction states also show up in results (`n_Pe`, `n_Ce`) and must be added by hand to "infected" (they are not infectious for that step).
  - Duration is not exposed: the user reads `ti_<state>` arrays that the engine made up, by naming convention.
  - A three-way split (`C | S | L`) is three functions, each recomputing `pcin`. Nothing checks that they sum to 1 (they are functions, so `E205` cannot fire).
- Suggestion: let a branch see the dwell it is ending, e.g. `ss.progression('P -> C | S', dur=dur_precin, p=lambda sim, uids, dwell: [logf2(dwell), None], decide='exit')`. Or expose `hpv.P.time_in[uids]` / `tr.dwell[uids]` as public API. With `decide='exit'` the junction and the one-step delay both go away.

### 3. Sex-specific natural history
- v3: `hpv.py:527–531`. Males get `dur_inf_male` and never reach CIN.
- v4: two competing dwell times out of `P`, each with `where=` (`precin` `where=female`, `clear_male` `where=male`). The rule is that the earliest wins, and the excluded one is infinite.
- **Declarative. Awkward.** It works, but the reader has to know that "where= on a dwell time" means rate 0, so duration inf, so it never wins. Every step it emits `RuntimeWarning: divide by zero encountered in divide` (`solvers.py:3570`).
- `where=ppl.female` (lazy) needs the same `ss.People` object as the sim. A disease factory does not have it, so I used `lambda sim: sim.people.female`.
- **Bug: `stratify(sex=['female','male'])` as agents creates a new random `sex` attribute that is independent of `people.female`.** It agreed with `female` for 48% of agents (`check_sex_dimension()`), so `res.n_C.select(sex='female')` returned "male" CIN. `stratify(female=[False, True])` raises `ValueError: Cannot add object "female" since already present in ndict`. The sim already treats `sex` as `female` for compartmental runs (`sim.py:279`), but not for agents. Female denominators therefore need an analyzer.
- Suggestion: map `sex` to `people.female` for agents. Also accept a distribution by level, `dur=ss.by(sex=[lognorm(3, 9), lognorm(1, 1)])`. A branch that is impossible for one sex could be `where=` on the branch, without the divide-by-zero.

### 4. Age-specific progression (`age_risk` on dur_cin)
- v3: `hpv.py:573–575`. `dur_cin *= ramp(age at CIN onset)`.
- v4: `ss.modify('g.cin', multiply=lambda sim: 1/age_ramp(sim.people.age.raw))` (`port.py:217`).
- **Declarative. Clean.** The semantics differ: v4 rescales the remaining dwell every step as the woman ages (`rescale_exits`), while v3 fixes it at onset. The v4 version is arguably better, but it is not the same, and each step it rescales every CIN woman's exit (a cost). A modifier on a duration that is a multiplier on a *rate* (so `1/ramp`) is easy to get backwards.

### 5. Latency and reactivation
- v3: `hpv.py:456–472` (a roll flag `to_latent`) and `830–855` (a hazard; `set_prognoses` again; backing the reactivation out of `new_infections`).
- v4: the clearance branches go to `L` with `p=(1-p)*ctrl`; `reactivation = L >> P @ ss.peryear(0.025)`; `ss.modify(g.reactivation, multiply=1-sev_imm)`.
- **Declarative. Clean.** Reactivation is its own flow, so nothing has to be backed out of incidence. Re-entering `P` gives a fresh trajectory because the dwell is redrawn on entry. This is the nicest part of the port.

### 6. Natural immunity: continuous per-agent levels plus a cross-genotype matrix
- v3: `hpv.py:792–828` (on clearance: a uniform draw, seroconversion gating the first draw, an HIV factor, a running max) and `cross_genotype.py:172–202` (`nab @ M_sus.T` → rel_sus, `cell @ M_sev.T` → sev_imm).
- v4: `CrossImmunity(ss.Connector)` (`port.py:133`) holds `nab_g`, `cell_g`, `sus_imm_g`, `sev_imm_g`, `rel_sev`. It applies them through `ss.modify(g.rel_sus, multiply=fn)`, `ss.modify(g.precin, multiply=1/(1-sev))` and `ss.modify(g.reactivation, ...)`.
- **Only via `step()`. Awkward.**
  - Finding who cleared this step means reading **`hpv.engine.events['clear_precin']`**, which is an engine internal. A transition works as a condition (`conditions.py:317`), but there is no public "uids that went through transition X this step" for module code.
  - Applying the levels through `ss.modify` with a per-agent function works, and the base `rel_sus` stays clean, which is good. But it means 4 modifiers × 4 genotypes = 16 connector modules. Each one also creates an empty result group at `sim.results.<name>`, and `where=` on transitions adds hidden `where_<g>_<tr>` connectors. In total there are 33 connectors, and `sim.results` is full of empty `timevec`-only entries.
  - `sev_imm` shortening `dur_precin` becomes a rate multiplier that rescales dwells already in progress whenever another genotype clears. v3 fixed it at infection.
- Suggestion: a public event accessor (`hpv.clear_precin.uids`, or `on=hpv.clear_precin` for connector methods). Modifier modules should not make result groups unless they have results. Also consider `ss.modify(..., targets=[...])` to cover several genotypes with one modifier.

### 7. Cancer, detection and cancer death; cancelling other genotypes
- v3: `hpv.py:596–615`, `868–903` (ti_cancerous, ti_cancer_detection, ti_dead_cancer, `request_death`) and `728–753` (clearing all other genotypes).
- v4: `cancer_death = X >> ss.null @ lognorm(8, 3)`; `detected = ss.outcome('to_cancer', delay=lognorm(1, 1))`; `cancelled = ss.Transition([S,P,C,L], Z, rate=ss.peryear(1e3), where=other_cancer)`.
- Death and detection are **declarative and clean**. The death clock runs from onset whether or not the cancer has been detected, because detection is an outcome and not a state. A screening module could not target "undetected" cancers, but that is out of scope.
- Cancelling the other genotypes is **declarative but ugly**: an "instant" move is faked with a huge hazard and a `where=` function that scans the other modules. There is no `rate=ss.immediately` (or `p=1` on a condition) for a move triggered by a condition.
- `ss.Transition(S | P | C | L, Z, ...)` (a StateSet as src in the explicit form) crashes **inside the error reporter**: `TypeError: argument 'a': 'StateSet' object cannot be converted to 'PyString'` (`transitions.py:78` `closest()`). A list works.

### 8. HIV coinfection
- v3: `hiv.py:76–101`. Per-step factor arrays from CD4 (`hiv_rel_sus/sev/imm/reactivation`), written into each HPV's `rel_sus`, and read in `set_prognoses`/`step_state`.
- v4: HIV with CD4 bands as states (`A>500`, `B 200–500`, `D<200`, `T` ART). `ss.modify(g.rel_sus, multiply=2.2, where=hiv.B | hiv.D)`. `rel_sev` and `rel_imm` are read in the `p=` functions and in the connector.
- **rel_sus is clean; rel_sev and rel_imm need code (awkward).** They are not multipliers on a named quantity (rel_sev scales time inside logf2), so there is no `ss.modify` target. Once CD4 is a continuous float, as in stisim's HIV, every one of these becomes code.
- **Bug:** `beta * (A + B + D + 0.04*T)` with a per-network dict beta raises `TypeError: unsupported operand type(s) for *: 'dict' and 'float'` at the first step (`solvers.py:4081`). I worked around it with `T = ss.State('T', rel_trans=0.04)`. A dict beta under homogeneous mixing gives the same raw TypeError instead of a coded error.

### 9. Demographics
- **Bug, silent:** with `ss.Births()`/`ss.Deaths()` (the Starsim-3 modules), newborns are added by `people.grow()` and **enter no state of any declared disease**. After 50 years, 7,288 of 10,108 agents were in no HPV or HIV state and could never be infected; prevalence fell to zero. Only `ss.birth()` inflows go through `Agents.add_people()` (`solvers.py:3881`), which places people in each declared module's first state.
- Workaround: a declared population module, `ss.Disease(births=ss.birth('-> N', ...), deaths=ss.death('N ->', ...), name='pop')` (`port.py:242`). It works, and its newborns land in `S` of every disease.
- Suggestion: newborns from any source should enter each declared disease's first state (hook `people.grow`). At minimum, raise an error when `ss.Births` is used with declared diseases.

### 10. Results
- **Automatic and good:** per genotype `n_S/P/C/X/L`, every flow (`to_cin`, `to_cancer`, `cancer_death`, `reactivation`, `cancelled`), the `detected` outcome with its delay, and `by('age')` for all of them once the sim is stratified by age. CIN is female-only by construction, so `res.hpv16.n_C.by('age')` is CIN counts by age with no extra code.
- **Still code (as in v3):** any-genotype unions (HPV+ / CIN+ women), female denominators by age (because of the sex-dimension bug), and ASR cancer incidence. This is `HPVTotal`, 45 lines against v3's 300.
- **Gotcha:** a transition named `clear` is shadowed by `dict.clear` (`res.clear` is a method), with no warning. Builds should reject result names that collide with dict methods.
- No per-agent `people.scale`, so v3's multiscale (fine cancer agents) cannot be ported without custom result code. I did not attempt it.

### 11. Spec/implementation mismatches found
| Spec says | Implementation |
|---|---|
| One branch may omit `p=` and gets the remainder (03 §Branching) | `E209 UndrivenTransition` for `ss.Transition(Pd, S)` alongside `p=` siblings |
| A branch with `p=` and no rate/dur happens immediately on entry | It happens on the **next** step (1 dt late) |
| List-form branch `ss.Transition(Pd, [C, S], p=[f, None])` with no rate | Runtime `TypeError: Cannot evaluate None` (treated as a hazard) |
| `stratify(sex=...)` is a property of people | As agents, a new random attribute that is not `people.female` |
| "A junction is not needed and is not part of the language" | HPV needs one wherever progression depends on dwell |

## Ranked: top problems for HPVsim
1. **Newborns from `ss.Births` enter no declared state.** This is silent and fatal for any multi-decade model.
2. **Progression that depends on duration needs junction states**, which add a 1-dt bias per junction, use engine-internal `ti_*` arrays, and give three unchecked p functions per split. This is the core of HPVsim's natural history.
3. **The strain dimension cannot express co-circulating genotypes** (one state per person, 2ⁿ−1 levels, binary immunity, `immunity=` not over networks). One disease per genotype works, but the cross-genotype layer (immunity, cancelling, unions) is all hand-written connector code.
4. **The `sex` dimension is broken as agents**, so female-only results need an analyzer.
5. **No public "who went through transition X this step"** for module code, so immunity-on-clearance reads `engine.events`.
6. Modifier sprawl: 16+ `ss.modify` modules plus hidden `where_*` ones, each with an empty result group.
7. Smaller: dict beta × weighted state raises TypeError; the StateSet src crashes the error reporter; the `clear` name collision; the divide-by-zero warnings from `where=` on dwells.

## Surprisingly nice
- Latency/reactivation is two lines, with its own flow; v3 needed a flag, a second `set_prognoses` and backing reactivations out of incidence.
- Cancer death as `X >> ss.null @ dur`, and detection as `ss.outcome(to_cancer, delay=dist)`, replace about 60 lines of `ti_*` bookkeeping, rounding and `request_death`.
- `ss.modify` with a per-agent function plus `where=` on another disease's state covers HIV's effect on susceptibility and the immunity application in one line each, and keeps the base `rel_sus` untouched (v3 resets `rel_sus[:] = 1` every step so other modules can multiply into it).
- Age-stratified results for every state and flow come from one `stratify=dict(age=...)` on the sim.
- It is fast: 5k agents, 200 steps, 6 declared modules in about 2 s.
