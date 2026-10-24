# Strains: the implementation plan (S7.4)

Written 2026-10-21. This plan combines [strain_proposal_claude.md](strain_proposal_claude.md) (ownership of each state, one spelling) and [strain_proposal_codex.md](strain_proposal_codex.md) (one strain system, an explicit contract for each event, no promised equivalence between representations). It also takes in Cliff's answers to the Claude proposal's questions and what Covasim's variants need ([covasim_review.md](covasim_review.md) CV12–CV13; `covasim/immunity.py`, `connectors.py`, `covid.py` in the Covasim repo). It does not replace [strains.md](strains.md): that document's inventory of today's code (file and line references, solvers, tests) still holds, and this plan cites it. Where this plan and its option A differ, this plan wins: Cliff chose the bigger change.

**Updated 2026-10-22** with Cliff's comments and answers ([oct22/plan.md](../oct22/plan.md), section C and decisions 3–5). His original comments are in this file's history; what they settled is written in below.

The [constitution](../constitution.md) governs. Each item below is checked against it, mainly against simplicity for epidemiologists, which counts for more than keeping the code small, and against avoiding several ways of doing one thing.

Sizes: S is under a day of agent work, M a few days, L more. **SWA** marks a silent wrong answer.

## Definition of done

This plan is done when all of these hold, and not before or beyond:

1. **Decisions D1–D8 below are settled**: Cliff has answered each, or it has been moved to [admin/deferred.md](../admin/deferred.md). (Done 2026-10-22.)
2. **Phases P1–P3 are built**, with the old public surface removed: `kind='strain'`, the bare-matrix `immunity=`, the "dimension called immunity" special case, hand-written `'A+B'` levels, arrow strings in `move=`, `protected=`, and `exclusive=`. Separately declared members stay, as a convenience for long definitions (D2).
3. **The three test builds of P4 run and meet their checks**: a tbsim-shaped resistance model, an HPV-shaped genotype model, and a Covasim-shaped variant model. These are minimal builds, as Cliff asked ("do minimal test builds now, and also include covasim"), not the downstream ports.
4. **Checks pass**: the semantic tests of P4 (at most about six new tests), the full suite, `tests/switching` (apart from known timing flakes), and the perf guard, with no S0.3 benchmark slower than its baseline.
5. **Records are updated (P5)**: the spec (`design/05-dimensions.md`, `07-effects.md`, `12-backends.md`) is changed in `ssv4` and LFD alike; deferred.md has the items set aside here; the [Stale](plan.md#stale) list has what the docs and migration need; [plan.md](plan.md)'s S7.4 points here and is checked off.

Out of scope: the tbsim, HPVsim and Covasim ports themselves (S8.2); the docs; CV13 (a clock of time since an event, which Covasim's waning NAbs need), except as a dependency noted under P4; an exact compartmental form for interacting own courses (KI15, `2^k` histories); within-host abundance, and several lineages of one modeled type. A new idea found along the way goes to deferred.md. If one is added here instead, the estimate goes down and the update says why.

**Estimate**: about 95% done (2026-10-22, 04:10: P1–P5 built and merged; the three test builds meet their checks except the Covasim build's check 3, which needs CV13; the LFD copy of the spec is synced with the oct22 plan's F2). Details: [oct22/worklogs/c_strains.md](../oct22/worklogs/c_strains.md). Update this line with every change to the plan. It should only go up.

## The design in one page

**One concept, "strains of a disease"; one structural statement, ownership.** Cliff: "yes, i like ownership". Every state of a disease with strains has an owner. Either it is the person's (**shared**: the person is in it once, carrying a set of strains), or it is each strain's (**own**: each strain the person carries has its own copy of the state, with its own clock). This is the Claude proposal's rule. It covers the Codex proposal's two course scopes as special cases: `course='host'` is every state shared, and `course='strain'` with host states is shared only for those states. It also covers the cases that fall between the two, such as HPV's cancer and malaria's fever. The Codex proposal's three kinds of information map onto it. What the host carries is the carried set, plus the own states. Host condition and care are the shared states. Immune memory is `ss.immunity` with `due_to=`.

**The spelling** (decision D1, settled): strains are part of the disease, `ss.Disease(..., strains=ss.Strains(...))`, which gives `tb.strains`. Internally a strain set is still a kind of dimension.

```python
# TB: one course for the person, several strains in it
strains = ss.Strains(
    dict(DS  = dict(rel_trans=1,     cure=1),     # per-strain values by strain; lists by parameter also work
         RIF = dict(rel_trans=0.575, cure=1/3)),
    immunity    = ss.immunity(due_to=L, rel_sus=0.79), # latent infection protects partly against a new strain; the rest is superinfection
    max_strains = 2,
    split       = [0.9, 0.1],
)
DS, RIF = strains
treatment = ss.Transition(Tx, [R, I], p='cure', dur=ss.months(6), per='strain') # treatment ends after 6 months; each strain carried is cured with its own chance; anyone still carrying one goes back to I
failure   = ss.Transition(Tx, I, rate=ss.peryear(0.1), move=DS >> (DS, RIF))   # treatment fails, and DS acquires resistance (emergence; DS >> RIF would be replacement)
tb = ss.Disease(..., treatment, failure, strains=strains) # then tb.strains is the strain set

# HPV: each genotype its own course, cancer the person's; each genotype is long, so it is defined on its own (D2)
hpv = ss.Disease(...,
    strains = ss.Strains([hpv16, hpv18, hi5, ohr],
        shared   = ['Ca'],
        immunity = [ss.immunity(due_to='nab', rel_sus=m_sus, combine='sum'),
                    ss.immunity(due_to=Cleared, dur=dict(In=m_sev, Ic=m_sev))]),
)

# COVID: one infection at a time (the default), protection by the last variant's antibodies, in Covasim's layout
covid = ss.Disease(..., strains=ss.Strains(variant_pars, cross_immunity=M)) # as covasim.parameters.get_variant_pars() and get_cross_immunity()
covid.strains.immunity = ss.immunity(due_to=covid.strains.last, scale='nab', p=dict(to_presym=M, to_sympsev=M), curve=ve)
```

Per-strain values can be given three ways, which are one container read three ways: by strain as dicts, as lists by parameter, or in Covasim's layout (variant parameters plus a cross-immunity matrix). Strains are objects, so `>>` replaces the arrow strings, as everywhere else (`ss.Delivery(move=)` and `ss.transfer()` lose theirs too).

The levels being an `ss.Strains` is what makes a dimension a strain dimension. That removes `kind='strain'` and with it the silent tbsim trap in strains.md: with no `kind=`, there is nothing to forget. All of a strain set's keywords are on `ss.Strains`, including the ones every dimension has (`split=`, and per-strain values in place of `adjust=`). Cliff asked to keep the name `ss.Strains` ("yes, i think so"), and here it is the container.

**The defaults give the simplest model** (Cliff: "complete protection is fine"). With every state shared and no `immunity=`, a person carries one strain at a time. Being infected or recovered protects against every strain, and a new case takes its source's strain. This is today's strain dimension, and Covasim's variants with no cross-immunity. A shared state protects against all strains, because it is the person's. An own state protects only against its own strain, unless `immunity=` says otherwise.

**The engine chooses the representation.** It does not tell the modeler, except in `describe()`. The three mechanisms of strains.md become three internal representations behind one front end:

| What is declared | Representation (today's code) | Solvers |
|---|---|---|
| All shared, complete protection from current infection | One strain per person: levels × states (strain dimension) | All, with gradients |
| All shared, partial protection from current infection | Carried sets × states (co-carriage), sets generated up to `max_strains` | All, with gradients |
| All own, no coupling | Side-by-side courses (uncoupled family) | All. Independent courses are k copies of the states side by side, not 2^k: the combinations a person carries matter only when courses interact (the next row). Cliff asked about this; see oct22/plan.md, section C |
| Own with coupling (`immunity=`), or mixed ownership | Agents: a carried-set bitmask, and a per-strain state array for the own states (coupled family, members built internally) | Agents. A compartmental solver raises `CapabilityError`, and `auto` picks agents |

The Codex proposal adds the guard: representations need not be equivalent, and the plan does not promise they are. A model that asks for something a solver cannot do gets a `CapabilityError`. It never gets a silent change in coinfection, memory, or counts. A shared-course model and an own-course model of the same disease are different models. They are compared only where their assumptions match.

**Each event has one meaning in every representation** (the Codex proposal's contract). Here are those meanings, with what this plan uses for each:

| Event | Meaning | Spelling |
|---|---|---|
| Acquisition of A | Exposure to A × residual susceptibility to A × eligibility. When strains compete, the hazards decide which is acquired, never the strains' order | Transmission, `ss.immunity(rel_sus=)`, `max_strains=` |
| Acquisition while carrying B | Add A (superinfection), or none (protected). With `max_strains` reached, A replaces the oldest strain, with a fresh clock, and this is counted as a flow (D4) | Protection from current infection: `ss.immunity(due_to=<shared state>)` |
| Transmission from a carrier of A and B | One strain per transmission, chosen in proportion to relative transmissibility (`'one'`, default), the fittest only (`'fittest'`), or each strain at its own rate (`'each'`) | `ss.Strains(transmit=)` (D5) |
| Clearance of A | Removes A only. B's stage and clock are kept, and memory of A is recorded if declared | An own state's exit; or `per='strain'` in a shared state |
| Strain event in a shared course | One host event with an outcome for each strain (cure per strain); the person's outcome is worked out from what is left | `per='strain'` (D7 for "one progresses alone") |
| Replacement, `{DS} → {RIF}`, and emergence, `{DS} → {DS, RIF}` | Not a transmission, and not counted as an infection. Immune history is unchanged | `move=DS >> RIF`, `move=DS >> (DS, RIF)`, on the transition that causes it (e.g. treatment failure), applied to every set that holds the source strain |
| Entering a shared state from own courses | The other own courses end (today's `exclusive=` rule), counted once | `shared=['Ca']` |
| Host death | Ends every course, counted as one death | Automatic |
| Results | `res.n_I` counts each person once (a union); `res.n_I.by(<dim>)` overlaps; acquisitions, emergences and new episodes are separate flows | `.by()`, as now |

The table keeps three Codex points that the Claude proposal left implicit. First, protection is evaluated against the incoming strain, on networks too. Second, emergence is not a new infection. Third, `ss.immunity` names the endpoint it acts on: acquisition (`rel_sus`), infectiousness (`rel_trans`), duration (`dur`), or a branch probability (`p`, new, which Covasim needs). Shortening a stay is not the same as preventing severe disease, so the endpoints stay separate.

## What Covasim needs, and where it lands

Covasim's variants are the plainest case: one infection at a time, a shared course, and per-variant traits. They also exercise the memory semantics more than tbsim or HPV do. Each feature of Covasim 4.0.1 (Covasim repo, files in the second column) maps onto this plan as follows:

| Covasim feature | Where | v4 spelling | Phase |
|---|---|---|---|
| One infection at a time, shared natural history | `covid.py:200–204` (`exposed_variant`, `infectious_variant`) | The default: every state shared, complete protection | P1 (exists today as the strain dimension) |
| Per-variant `rel_beta` | `covid.py:573` | `adjust=dict(infection=[...])` | Exists |
| Per-variant `rel_symp_prob`, `rel_severe_prob`, `rel_crit_prob`, `rel_death_prob` | `covid.py:483–513` | `adjust=` on the branch transitions (`to_presym`, `to_sympsev`, `to_crit`, `death`), on their `p=` | Exists; checked in P4 |
| Cross-immunity matrix by the variant *last* recovered from | `connectors.py:351–380`, `immunity.py:133` (`[challenge, source]`, so transposed for `ss.immunity`'s rows-from layout) | `ss.immunity(due_to=R, ...)` if people stay in R, or `due_to=covid.strains.last` (with `scale=` for an attribute) once they return to S (D3) | P3 |
| Protection on three endpoints: infection, symptoms, severity | `connectors.py:406–408` (`sus_imm`, `symp_imm`, `sev_imm`) | `rel_sus=` and the new `p=dict(<branch>=M)` | P3 |
| Over networks | Everything | HP14: per-strain protection in `Agents.infect()` | P3 |
| NAb level × matrix, then a logistic curve per endpoint | `connectors.py:382–408`, `immunity.py:163` (`calc_VE`) | `ss.immunity(due_to=covid.strains.last, scale='nab', ..., curve=f)`, with protection `curve(M[last, g] × nab)` (D8) | P3 |
| NAbs that peak, boost and wane | `covid.py` NAb update, `immunity.py:193–290` | Needs CV13 (`event.time`), which is outside this plan; the build uses a NAb recorded at recovery | Dependency |
| Vaccine efficacy by variant, combined with natural immunity by `max` | `connectors.py:391–404` | `ss.Vaccine(efficacy=ss.by(variant=...))` against the acquisition of each variant (today `check_dims()` refuses it, `products.py:401–410`), plus `combine=` across sources | P3; anything not covered goes to deferred |
| Introduce a variant on a date, `n_imports` (rescaled) | `immunity.py:21–130` (`cv.variant`) | Seeding by strain at a date. v4 has no declared spelling yet. The minimum is `dis.infect(uids, strain='alpha')` for code, so `cv.variant` stays a short helper | P3 (the code call); a declared form goes to deferred |
| A tie between variants goes to the lowest index | `covid.py:559, 599` | Competing hazards decide; parity is checked within noise, not exactly | P3 |
| Results by variant | `flows_by_variant`, `by_variant` stocks | `res.<x>.by('variant')` | Exists |

So the Covasim build checks most of what the other two builds don't: last-strain memory, protection on branches, protection over networks, attribute-scaled protection on a dimension, and products by strain. CV12's four items are each assigned above: (1) HP14, (2) attribute-weighted protection, (3) protection on branches, and (4) the last strain.

## Decisions for Cliff

All eight are settled (2026-10-22). Settled earlier, from the Claude proposal's questions: ownership is the structural idea; the default is complete protection from shared states; `ss.Strains` keeps its name; and the work is minimal test builds now, including Covasim.

1. **D1: spelling. Settled:** `ss.Disease(strains=ss.Strains(...))`, giving `tb.strains`, as in the design above. Was recommended: `stratify(<name>=ss.Strains([...], shared=, immunity=, max_strains=, transmit=), split=, adjust=)`. The `ss.Strains` levels mark the dimension as strains, so `kind=` goes. The alternative puts the strain keywords on `stratify()` itself (`stratify(strain=[...], shared=...)`). That is one fewer object, but `kind` would then be inferred from the name or from strain-only keywords, which brings the trap back. [CK: i don't like it ... i think it should be either `hpv = ss.Disease(name='hpv', strains=dict(shared=, immunity=, ...))`, or `hpv.strains(shared=, immunity=, ...)`, with strains a subtype of stratify].
2. **D2: families of separately declared members. Settled:** kept as a convenience, `ss.Strains([hpv16, hpv18])`, sharing one natural history; letting a member's structure differ goes to deferred.md unless it is no harder. Was recommended: drop them. HPV's genotypes share one natural history. Pathogens whose natural histories differ are different pathogens: separate diseases and connectors, as for HIV and TB. Removing them saves the member checks and spec rows. The Codex proposal would keep them as an escape. The cost of dropping them: a strain whose structure differs (say, one genotype with an extra state) has to be modeled through `adjust=` (a rate of 0) or as a separate disease. [CK: mixed ... i think the main reason we'll want `hpv.strains([hpv16, hpv18])` is because real-world definitions are going to be quite long, definitely not the sort of thing you'd want to inline. but agree about this being a convenience thing, not necessarily an architectural thing. but if it's not HARDER to let them differ, i suppose they can, though i suppose this would mess with results aggregation etc.]
3. **D3: memory. Settled:** `due_to=`, with objects rather than strings: a state (`due_to=R`), an attribute (`due_to='nab'`, scaling each strain's row), the last strain (`due_to=covid.strains.last`), or every strain ever had (`due_to=flu.strains.ever`). Was recommended: `protected=` is required on a strain dimension, with three readings:
   - a state: status, meaning in it now, at the strain or strains that state holds;
   - an attribute (`ss.attr`): its value scales the matrix row of each own strain, or, with `last=True`, the row of the person's last strain;
   - `'ever'`: every strain ever had, which is today's `strains_had` and the classic history-based model.

   Covasim needs `last=True`, Gog–Grenfell-style flu models need `'ever'`, and HPV needs the attribute. The engine records the last strain for free (it is the level the person last left an infected state at). This is the one place where the Codex proposal's "one interpretation of memory everywhere" costs a keyword. [CK: hm, i'm ok with the disease logic, but maybe not the spelling. `ss.immunity(protected='ever')` does not make sense to me, if that's what you're proposing.]
4. **D4: acquisition when full. Settled:** a new strain replaces the oldest, and its clock starts fresh; counted as a flow. Was recommended: below `max_strains`, partial protection means superinfection adds a strain. At `max_strains`, the new strain is not established, and this is counted as a flow (`blocked`), so the assumption shows up in the results. There is no `'replace'` option, because replacement is `move=` on a transition. The Codex proposal asks that `max_strains=1` require an explicit rule. Under this design, one strain at a time is the default complete protection, not a capacity limit, so that case does not arise. [CK: i'd think with max_strains, then adding a new strain would bump the oldest strain and reset the clocks.]
5. **D5: transmission from mixed carriers. Settled:** `'one'` (default: one strain per transmission, in proportion to relative transmissibility), `'fittest'`, and `'each'`. Was recommended: `transmit='each'` (the default, as co-carriage does now) or `'fittest'` (tbsim's `transmission_independent=False`). [CK: is this just the default? we need both options. am not sure about 'each' though, does this mean someone with DS+RIF would pass BOTH along? surely not?]
6. **D6 (accepted): per-strain `adjust=` on a set, for a transition that is not `per='strain'`.** Examples are the death rate of a person carrying DS and RIF, or their activation rate. Recommended: the set takes the largest of its strains' values (the worst strain sets the course), with no adjustments by set label. A model that needs another rule writes `per='strain'`. To be confirmed by the tbsim build, since tbsim's `[1, 1, 0.4]` activation by set may need a different rule. [CK: i'm skeptical, but ok]
7. **D7 (accepted): one strain progresses alone** (tbsim's `p_multi`). Cliff: "not sure, would need to try it out". The tbsim build tries both `per='strain', alone=True` and the same event written with `move=`, and the simpler is kept. [CK: sounds good]
8. **D8 (accepted): Covasim's curve.** Covasim's protection is `calc_VE(nab × M[last, g])`: a nonlinear curve applied after the matrix, with its own parameters on each endpoint. Recommended: `curve=` on `ss.immunity`, a function from the effective level to protection (one `ss.immunity` per endpoint when the curves differ). This works on agents and has gradients if the function is JAX-friendly. The alternative is no `curve=`, with Covasim keeping the curve in code. That is about 20 lines of `immunity.py`, and it keeps the immunity connector alive. [CK: ok]

## Phases

### P1. One front end over the existing representations (M)

- `ss.Disease(strains=ss.Strains(levels, shared=True, immunity=None, max_strains=None, transmit='one', split=None))`, with per-strain values given by strain (dicts), by parameter (lists), or in Covasim's layout. Strain objects support `>>`. `stratify()` drops `kind='strain'`, the bare-matrix `immunity=`, and the "dimension called immunity" case (`dimensions.py:46, 107–124, 495–497`). A level containing `+` in a plain dimension raises an error that points to `ss.Strains`.
- Generate the carried sets from the strains and `max_strains` (today's `init_carriage()` tables, `dimensions.py:186–212`, from a list of sets instead of parsed labels; the labels are for display only). Map per-strain `adjust=` onto sets (D6).
- `shared=` decides the representation (the table above). For own states, build the members internally from the one natural history, using today's family code (`strains.py`: protection, seeding, unions, `res.by()`). `shared=[...]` takes over from `exclusive=`.
- Port the existing tests to the new spelling: `test_dimensions.py` `test_strains` and the carriage test (`269–303`), `test_diseases.py:559–815`, and `test_products.py:257`. **Exit**: each gives the same numbers as before with the same seed. That checks that P1 is a front end and not a new model.
- Remove what P1 replaces: the public members API (if D2 drops it), `exclusive=`, and the unreachable compartmental branch of `Protection.apply()` (`strains.py:166–176`).

### P2. Strain events in a shared course (M)

- `per='strain'` on a transition in a shared state: each carried strain draws its own outcome, with its `adjust=`. The person moves on when the rule is met (none left, for cure). The engine generates the set-to-set outcomes, as tbsim does by hand (`tbsim/resistance.py:186–231`), in the ODE as well as agents.
- `move=` written per strain (`DS >> RIF`, `DS >> (DS, RIF)`), applied to every set that holds the source strain. Neither is counted as an infection. Remove the arrow strings from `ss.Delivery(move=)` and `ss.transfer()` too.
- D4: at `max_strains`, a new strain replaces the oldest, with a fresh clock, counted as a flow.
- D7: try both spellings for "one progresses alone".
- The person's results count each person once; the strain results count each carried strain.

### P3. One protection semantics, on every representation (M)

- `due_to=` as in D3, including `strains.last` and `strains.ever`, for shared and own states alike. Today's dimension branch (`engine.py:338–370`, `strains_had`) is the code for `'ever'`. The family's `Protection` modifier is the code for states and attributes. The last strain is new (about one array).
- Endpoints: `rel_sus`, `rel_trans`, `dur`, and the new `p=dict(<transition>=M)` on branch probabilities, through the same path `adjust=` uses for `p=`.
- HP14: per-strain protection over networks in `Agents.infect()`, where each network contact's source strain is known (strains.md, section 5, about 20 lines, with its own random stream for CRN). When several strains reach one person in a step, they compete by hazard, not by index.
- Products by strain: an efficacy `ss.by(<strain dim>=...)` acts on acquiring each strain (the same per-strain hazard as `rel_sus` protection), instead of the error in `check_dims()`.
- `curve=` (D8). `dis.infect(uids, strain=)` for code that seeds a strain.
- Performance: per-strain protection is evaluated only for those exposed, not for every agent and strain each step (the cost Covasim's connector pays, `connectors.py:382–408`). The perf guard and the S0.3 benchmarks must not slow down.

### P4. Test builds and semantic tests (M)

The scripts go in `v4_design/oct21/strain_builds/`. They read the downstream repos but do not change them.

- **tbsim-shaped**: DS and RIF, latent superinfection by partial protection, cure per strain, emergence on treatment, `max_strains=2`. **Checks**: (a) agents and the ODE agree, as the carriage test does now; (b) against the same model written the old way (hand-written `'DS+RIF'` levels and `move=` copies, on the commit before P1), the results are the same with the same seed, or within noise where the draws reorder, with the reason given; (c) the line count, compared with `tbsim/resistance.py`'s 90 lines of set bookkeeping.
- **HPV-shaped**: four genotypes, own courses, shared `Ca`, cross-immunity scaled by NAbs, severity immunity. **Checks**: the same results as the same model written as today's `ss.Strains` family; clearing hpv18 leaves hpv16's stage and clock alone; entering `Ca` ends the other courses and counts one cancer.
- **Covasim-shaped**: `declared_covid.py` (Covasim repo, `v4port/disease/`) with wild, alpha and delta given in Covasim's layout, introduced on dates through `dis.infect(strain=)`, over a network. **Checks**:
  1. With static cross-immunity by last variant on infection, symptoms and severity, the build matches Covasim 4.0.1 on Starsim 3.7.2 with `use_waning=False` and `cv.CrossImmunity` within noise: 10 seeds, |z| < 3 on cumulative infections by variant, severe cases and deaths (using `features.py`'s comparison style).
  2. With a NAb recorded at recovery and `curve=` (or the D8 alternative), it runs and gives protection between the static and no-immunity cases.
  3. Waning NAbs are checked only if CV13 has been built; otherwise they are listed as the remaining gap.
- **Semantic tests** (about six, short, in `test_dimensions.py`). Clearing A keeps B's clock. One regimen clears only DS. Emergence creates no infection. Immunity survives clearance. A person carrying two strains counts once in `n_I` and once in each strain. A shared-course model agrees between the ODE and agents.

### P5. Records (S)

- Spec: `05-dimensions.md` (strains section rewritten around ownership; the table of four rows becomes "shared, own, mixed; other pathogens use connectors"), `07-effects.md` (`ss.immunity` endpoints, `due_to=`, `curve=`), `12-backends.md` (the representation table and capability errors). Make the same changes in LFD's `design/`.
- deferred.md: a declared form for introducing a strain on a date; waning cross-immunity beyond what CV13 gives; enhancement (protection above 1, dengue); an exact compartmental form for own courses (KI15); within-host abundance; correlated transmission of several strains in one contact.
- Stale: the migration skill rows (HPVsim's genotype connectors → `ss.Strains` with `shared=`; tbsim's `add_resistance()` → per-strain biology; Covasim's variants and `CrossImmunity` → `ss.Disease(strains=ss.Strains(variant_pars, cross_immunity=M))`); the docs. Suggest that tbsim rename `tbsim.Strains` (strains.md).
- Mark strains.md and the two proposals as superseded by this plan, and check off S7.4 in plan.md.

## Risks

- **P1 changes more than it appears to.** Building members internally from one natural history moves code between `strains.py` and the engines. The P1 exit check (same numbers, same seed, on the ported tests) is the guard. If it fails because draws reorder, say so rather than loosening the check silently.
- **D6 may be wrong for tbsim.** Its set-level activation `[1, 1, 0.4]` (a mixed infection activates more slowly) is not the largest of its strains' values. If tbsim's biology needs it, the choice is `per='strain'` with a rule, or an adjustment by set as a documented escape. The tbsim build decides.
- **Covasim parity has two known offsets** that are unrelated to strains: the one-day dating difference of declared infections (CV5; `latent_shift=1` in `declared_covid.py`), and the tie rule above. Check 1 of the Covasim build is within noise, not exact, for these reasons.
- **Scope creep from Covasim's NAbs.** The peak, boost and wane kinetics belong to CV13, not here. P3 gives the protection side (attribute × last-strain row × curve). It does not give the antibody kinetics.
