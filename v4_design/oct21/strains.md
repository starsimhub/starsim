# Strains: families and strain dimensions (S7.4)

**Superseded (2026-10-22)** by [strain_plan.md](strain_plan.md), now built (see [oct22/worklogs/c_strains.md](../oct22/worklogs/c_strains.md)). Kept as the record.

Written 2026-10-21 on `v4-simplify` (uncommitted tree after `9cf7766f`), for S7.4 of [plan.md](plan.md) and decision 16. Research only: no code was changed. Scratch checks are in `/tmp/claude-1000/-home-cliffk-idm-ssv4/95a2919b-da64-4203-8e56-0999a64fe595/scratchpad/s74/` (`checks.py`, `checks2.py`, `checks3.py`, `check4.py`).

## Summary

- There are three strain mechanisms, not two: a **family** (`ss.Strains`, several diseases a person can have at once, each with its own natural history), a **strain dimension** (one strain at a time, one natural history), and **co-carriage** (a strain dimension whose levels are sets of strains, `'DS+RIF'`, sharing one natural history). They answer different questions, and the two downstream users each need a different one: HPVsim needs the family, tbsim needs co-carriage. Neither uses the strain dimension's `immunity=`.
- They already share one vocabulary where it matters: `ss.immunity` (one class, one matrix layout, one `combine=` rule), `split=` for the shares of strains among those seeded, `res.x.by(<name>)` for per-strain results, and `move=` for changing strain. There is no redundant co-carriage syntax left to remove: `'A+B'` levels are its only spelling, and superinfection and `move=dict(strain='A+B -> A')` are ordinary transitions.
- **Found: a silent wrong answer for tbsim v1.0.0.** tbsim's `add_resistance()` returns `stratify(resistance=[...], split=, adjust=, states=)` with no `kind=` (`tbsim/resistance.py:240`). It relied on the name `resistance` meaning a strain, which S7.1 removed with the name hints. Now the dimension is `'plain'`, co-carriage is off, and new infections take a level from the split instead of from their source: with a split of 50% DS and 50% RIF, `DS+RIF` never appears (agents and ODE alike), with no note (`check4.py`, `checks3.py`). Either tbsim passes `kind='strain'`, or Starsim infers the kind from `'+'` levels (recommended, 3 lines).
- **Recommendation**: keep all three mechanisms and all their solvers; make the user story one table and one vocabulary; fix the tbsim trap by inferring `kind='strain'` from co-carriage levels; do HP14 (about 20 lines); delete one dead branch. Net change about −10 to +10 package lines. Do not merge the representations (option D), and keep the family's small compartmental form (option B is offered, not recommended).

## 1. What each representation is today

### The family: `ss.Strains`

Syntax (`starsim/strains.py:193–247`):

```python
hpv = ss.Strains('hpv', [hpv16, hpv18],                       # members: declared diseases, each with its own name
    immunity  = [ss.immunity(protected='R', rel_sus=m_sus, combine='sum'),            # states of each member that protect
                 ss.immunity(protected='nab', rel_sus=m_sus, combine='sum'),          # D9: an attribute of each member scales the protection (agents only)
                 ss.immunity(protected='cleared', dur=dict(In=m_sev, Ic=m_sev))],     # shorter stays: severity immunity
    init_prev = 0.1,          # share seeded with any member (number, ss.by, or function); each seeded person gets one member
    split     = [0.6, 0.4],   # which member they get
    exclusive = 'Ca',         # one course at a time in this state: entering it in one member moves the person to the dead-end state (e.g. X) in the others
)
sim.results.hpv.n_I                 # union over members (each person once); flows are sums
sim.results.hpv.n_I.by('hpv')       # each member's own
where = hpv.infected                # a condition: in it in any member; hpv.infected.count() >= 2 for the co-infected
ss.Delivery(..., disease=hpv)       # a product over every member
```

`ss.immunity` (`strains.py:36–125`) takes the quantity lowered as the keyword (`rel_sus=`, `rel_trans=`, `dur=dict(state=m)`), a matrix (rows from, columns against) or a scalar, `protected=` (family only), and `combine='max'|'sum'|'multiply'|function`. `combine_protection()` is `strains.py:13–33`.

Solvers: as agents, everything. Under the compartmental solvers, uncoupled members run side by side, with unions and seeding from each member's shares (`'yes'`, or `'approx'` with `init_prev=`); members coupled by `immunity=` or `exclusive=` are refused (`choose.py:521–527`, since S3.6), and `auto` picks agents. Attribute-scaled protection (`protected='nab'`) is agents only (`strains.py:399–400`). No gradients (`gradients.py:50–57`).

Code (713 lines in `strains.py`):

| Part | Lines | What |
|---|---|---|
| `combine_protection`, `ss.immunity` | `strains.py:10–125` (116) | Shared with the dimension |
| `Union` | `128–144` (17) | `where=hpv.infected`, `.count()` |
| `Protection` | `147–190` (44) | A modifier on each member, through the generic `sim.modifiers` table; lines 166–176 (the compartmental branch) are now unreachable, since coupled families are refused there |
| `Strains` | `193–713` (521, of which 49 docstring) | Members, checks, protection set-up (`init_pre` 374–451), seeding (`init_post` 499–532), protection by step (534–577), `exclusive=` (579–596), unions (620–641), `res.by()` (643–655), spec (663–682), `describe()` (684–713) |
| Compartmental side by side | `shares` 458–476, `held` 478–490, `step_shares` 598–618, `update_results` 625–632, note 447–451, `engine.py:1074–1077` | About 70 lines |
| Elsewhere | `choose.py:341, 521–527`; `interventions.py:475–478, 662`; `gradients.py:50–57`; `spec.py:391–401, 659–665, 808–815`; `results.py:429–435`; `parameters.py:466`; `sim.py:903`; `agents.py:67` | About 60 lines |

Tests: `test_diseases.py:559–815` (8 tests: results, immunity, attribute immunity, seeding, compartmental, spec, branch flows, errors), `test_products.py:257`, `test_attrs.py`.

### The strain dimension: `stratify(strain=..., immunity=...)`

Syntax (`dimensions.py:22–124`, spec `05-dimensions.md:220–280`):

```python
flu.stratify(strain=['A', 'B'], immunity=[[1, 0.3], [0.3, 1]])                                   # kind='strain' inferred from immunity=
flu.stratify(strain=['A', 'B'], immunity=ss.immunity(rel_sus=[[1, 0.3], [0.3, 1]], combine='multiply'))  # the same, another rule (no protected=, rel_sus= only)
tb.stratify(resistance=['DS', 'MDR'], kind='strain', split=[1, 0], adjust={'infection': [1, 0.7]},    # no immunity=: kind= must be given
            acquired=ss.transfer(resistance='DS -> MDR', rate=ss.peryear(0.02), states=['on_treatment']))
res.infection.by('strain')     # incidence by strain acquired
```

Each strain has its own force of infection, a new case takes its source's strain, `adjust=` on a transmission is the strain's transmissibility, and `immunity=` protects people in non-infected, non-source states (e.g. `R`) by every strain they have had (`strains_had` bits). By default the dimension covers the states reached only through infection (`dimensions.py:536–539`). One strain dimension per disease (`engine.py:227–231`). A corner case: `stratify(immunity=[...])` with a 1-D list is a dimension *named* immunity (`dimensions.py:495–497`).

Solvers: without `immunity=` (or with complete immunity), every solver, with gradients (`gradients.reason()` is None for an ODE strain model, `checks2.py`). With partial immunity, agents only (KI15: a compartment keeps only the last strain; `engine.py:363–365`, `choose.py:666–669`; the exact `2^k` form is deferred). As agents, `immunity=` refuses networks (`agents.py:2546–2547`; HP14).

Code, about 200 lines in the engines: `engine.py` 227–231 (which dimension), 297–301 (`dim_map`: a new case takes the strain), 338–370 (`init_immunity`, `protection`), 378–383 (`strain_levels`), 391–411 (`init_strain_weights`), 662/685–686 (`strain_adjust`), 725–732 (`strain_rate`); `agents.py` 106–107 (`strains_had`), 241–243, 671–672, 1155–1158, 1564, 2382, 2537–2629 (per-strain hazards and pairs in `infect()`), 2695–2728 (`draw_infections`, `set_strain`, `protection`); `compartmental.py` 262–277, 344–349, 747–748, 1052–1053, 1139–1144, 1249–1252, 1334–1402 (a flow and an exposure per strain). Of this, `immunity=` itself is about 75 lines (`dimensions.py:107–124, 495–497`, `engine.py:338–370`, `agents.py:106–107, 2382, 2546–2547, 2717–2728`, `compartmental.py:277, 1251–1252`, `choose.py:666–669`). Tests: `test_dimensions.py:201–268` (`test_strains`), plus the TB example at 62–80.

### Co-carriage: `'A+B'` levels

Syntax (spec `05-dimensions.md:267–280`, test `test_dimensions.py:269–303`):

```python
dis = ss.Disease(
    infection      = S >> L @ ss.peryear(6)*I,
    superinfection = L >> L @ ss.peryear(3)*I,                                    # a transmission that leaves people in their state: adds a strain
    to_A           = ss.Transition(L, I, rate=ss.peryear(0.3), move=dict(strain='A+B -> A')),  # one of two strains progresses
    ...)
dis.stratify(strain=['A', 'B', 'A+B'], kind='strain', split=[0.5, 0.5, 0], adjust={'activation': [1, 1, 0.4], 'rel_trans': [...]})
```

A level that joins other levels with `+` carries those strains. A carrier passes each strain on in proportion to its own `rel_trans`; a carrier infected with a strain they lack moves to the level that adds it, if there is one; a self-transmission (`L >> L`) that changes nothing is not counted.

Solvers: all of them, including the ODE (the test checks agents and every compartmental solver against an exact ODE, `test_dimensions.py:285–301`) and gradients.

Code, about 75 lines: `dimensions.py:93–95, 186–212` (`init_carriage`: the `carries` and `adds` tables, 30); `engine.py:298–299, 383, 385–389, 403–411, 414–421` (about 24); `agents.py:157, 171–172, 2544–2545, 2620–2628, 2697–2699, 2714–2715` (about 19); `compartmental.py:1253–1254` (2). It is table-driven already: the engines read `strain_weights` (levels × strains; the identity without carriage) and `adds`, not the labels.

## 2. What HPVsim and tbsim need

### HPVsim: a family

HPVsim 3.2.1 (`/home/cliffk/idm/hpvsim`, branch `v4`) has one coded `HPV` disease per genotype, tied by `CrossImmunity` and `HPVTotal` in `hpvsim/cross_genotype.py` (503 lines). The October 17 port ([../oct17/hpvsim_review.md](../oct17/hpvsim_review.md), section 3 and HP14/15/22/23) found:

- **Co-infection with separate courses is the core need.** A woman can have hpv16 in CIN and have just acquired hpv18. As co-carriage levels that is 15 levels at 4 genotypes (511 at 9), and one clearance clears all of them (`v4port/genotypes/probe_strain.py`, item 3). So a strain dimension, with or without carriage, cannot hold HPV, and the item forbids forcing it.
- **What the family must provide**, all of which `ss.Strains` has as agents: cross-immunity with HPVsim's rule, `(1 − clip(Σ))·(1 − vaccine)`, which plain modifiers get wrong (0.2 against 0.35; HP22): `ss.immunity(..., combine='sum')`, scaled by a per-person antibody level (`protected='nab'`, D9) and on severity (`dur=`); joint exclusive seeding by age and sex (HP15): `init_prev=` (a function) and `split=`; one cancer per person (`exclusive='Ca'`); any-genotype results counting the co-infected once (HP23): unions; products over every genotype (HP7): `disease=hpv`.
- **Agents only in practice.** HPVsim runs on a sexual network with per-person antibody levels (`ss.attr`), both agents only. It does not need the family's compartmental side-by-side form; the ODE the review praised (0.43 s) is of one genotype's natural history, a single disease.
- **Uses neither the strain dimension nor co-carriage.**

The port in the HPVsim repo (`v4port/genotypes/genotypes_v4.py`, 270 lines) predates `ss.Strains`: it uses 88 `ss.modify` and a seeding function. The family version measured on October 19 (cross-immunity effect 0.687 at 2 genotypes against HPVsim's 0.659, 0.483 at 4 against 0.428, about 2.5 times faster than the port; [../oct19/implementation.md](../oct19/implementation.md) line 180) is not in the repo. Neither can run now: `import hpvsim` fails on `ss.Calibration` (now `ss.fit`).

### tbsim: co-carriage

tbsim v1.0.0 (`/home/cliffk/idm/tbsim`, `d3996bf`) builds drug resistance in `tbsim/resistance.py` (268 lines) as one strain dimension of `tbsim.TB`, `resistance`, whose levels are the sets of strains people can carry (`DS`, `RIF`, `DS+RIF`: 3 levels for one drug, 10 for two drugs with `max_strains=2`). It generalizes the `two_strain_ode.R` model and is meant to run as an ODE, a stochastic compartmental model, or agents (`resistance.py:1–25`). It uses:

- `'+'` levels, with `rel_trans` by level (the fittest strain, or all of them; `resistance.py:172–173`);
- superinfection as `S[state] >> S[state]` self-transmissions (`resistance.py:176–181`);
- `move=dict(resistance='DS+RIF -> RIF')` on generated copies of transitions, for one strain progressing alone, resistance acquired on progression or treatment, and partial cure (`resistance.py:163–170`, 183–231); and `adjust=` for everything that scales by level.

So co-carriage is really used, centrally: it is how one TB course (one latent period, one active disease, one treatment) holds several strains. One disease per strain would give each strain its own course and treatment, which is wrong for TB, and the item forbids it. tbsim uses neither `immunity=` nor `ss.Strains`. Note that tbsim exports its own `tbsim.Strains` (the drug-resistance sets, `resistance.py:55`), a different thing from `ss.Strains`.

**The trap**: `add_resistance()` returns no `kind=` (`resistance.py:240`, applied at `tb.py:295`), so under today's Starsim the dimension is plain. Checked with a tbsim-shaped model (`checks3.py`, split 50% DS, 50% RIF, 10 years): with `kind='strain'`, agents end with `DS 1987, RIF 148, DS+RIF 642` latent; without it, `1318, 1326, 0` (ODE alike), silently. tbsim cannot run under v4 today anyway (`ss.State(infectious=)` is gone, `tb.py:170`), so this would surface only after S8.2's fixes, as wrong numbers. Spec `05-dimensions.md:175` still promises a note for names like `resistance`; none is printed (`check4.py`).

## 3. Overlap and difference

| | Family (`ss.Strains`) | Strain dimension | Co-carriage |
|---|---|---|---|
| A person has | any subset of members, each in its own state | one strain | one set of strains, one state |
| Natural history | one per member | shared | shared |
| Size | members × their states | levels × states | sets × states (combinatorial; tbsim caps it with `max_strains`) |
| Protection between strains | `ss.immunity(protected=, rel_sus=/rel_trans=/dur=)`, as modifiers | `immunity=` (matrix or `ss.immunity(rel_sus=)`), an engine branch, by strains had | none (tbsim needs none) |
| Seeding | `init_prev=` and `split=` on the family | the disease's `init`/`init_prev`, the dimension's `split=` | same as the dimension |
| Results | unions, sums, `.by('hpv')` | `.by('strain')` | `.by('strain')` |
| Solvers | agents; uncoupled side by side under compartments; no gradients | all; partial immunity agents only and not over networks | all, with gradients |

**Shared already**: `ss.immunity` and `combine_protection()` (one class and one rule for both); `split=` meaning the shares of strains among the seeded; `.by(<name>)`; `move=` (generic for every dimension); products by family or by `ss.by(strain=...)`.

**Duplicated in mechanics, but for different reasons**: cross-immunity is implemented twice. The family's goes through the generic modifier table (`Protection`, `sim.modifiers`), which is the "ordinary interface" the item asks for. The dimension's is an engine branch (`init_immunity`, `protection`, `strains_had`, `immune_states`), because protection there depends on which strain is doing the infecting, so it has to sit inside the per-strain force of infection; a per-person `rel_sus` modifier cannot know the strain. No generic mechanic replaces it today.

**Duplicated in syntax**: the dimension takes `immunity=` as a bare matrix or as `ss.immunity(rel_sus=...)` (`dimensions.py:109–124`), and the bare form forces the "dimension called immunity" special case (`dimensions.py:495–497`). `kind='strain'` is inferred from `immunity=` but not from co-carriage levels, though both say "strain" unambiguously.

**Genuinely different**: concurrent independent courses (family) against one course over a set of strains (co-carriage) against one strain at a time (dimension). Each has a model that needs it, and none expresses another simply: HPV as carriage needs `2^k − 1` levels and loses per-genotype clearance; TB as a family loses the single course and treatment; a family for flu A/B would need a cross-member "one infection at a time" rule the family does not have.

## 4. Options

From least to most change. Line counts are package lines, approximate.

**A. One story, small fixes (recommended).** Keep the three mechanisms and their solvers. Change:
1. Infer `kind='strain'` when a level joins other levels with `+` (`dimensions.py:46`, about 3 lines), so tbsim's dimension is a strain again with no change there; also have tbsim pass `kind='strain'` (one line, in the S8.2 handover). Alternatively raise an error for `'+'` levels on a non-strain dimension.
2. One spelling of a dimension's `immunity=`: `ss.immunity(rel_sus=...)` only, as in the family; drop the bare matrix and the "dimension called immunity" special case (about −5 lines; tests and spec 05/07 change their examples). Optional: the bare matrix is short, and Cliff may prefer to keep it.
3. Delete `Protection.apply()`'s compartmental branch (`strains.py:166–176`, −11), unreachable since S3.6.
4. HP14 (below): about +20.
5. Spec: the table in `05-dimensions.md:222–228` gains the co-carriage row ("several at once, sharing one course: drug resistance"), and line 175 loses the promised name note.
- Removes about 15–20 lines and adds about 20–25. Risk: low; the kind inference could catch a plain dimension with labels like `'A+B'` that mean something else, which is why it applies only when every part is a level of its own. Cost to users: none for HPVsim; for tbsim, none (or the one-line `kind=`).

**B. A, and the family agents only.** Drop the side-by-side compartmental form: `shares`, `held`, `step_shares`, the compartmental branch of `update_results`, the note, `engine.py:1074–1077` (`seeded_by`), the family rows in `choose.py` (to one `'no'`), and the family branch of `gradients.reason()`. About −80 package lines and `test_strains_compartmental` (28 lines). Risk: reverses Strains step 5 and S3.6's "keep exact supported independent members"; an independent family under the ODE then needs its members given as separate diseases with unions worked out by hand. Cost: nobody downstream (HPVsim is agents only). Not recommended: the code is self-contained in `strains.py`, touches the engines in 4 lines, and is the only compartmental route for a family, so its maintenance cost is low.

**C. A, and one code path for strain dimensions.** Give every strain dimension `carries` (the identity) and `adds` tables, so the `carries is None` checks go (`engine.py:298–301, 383, 403–405`; `agents.py:2544–2545, 2620, 2714`; `compartmental.py:1253`). About −15 lines. Risk: `adds[l, j] = j` for a plain strain dimension changes what a self-transmission does there (today nothing; it would replace the strain), so the meaning of reinfection would need care and new tests in every solver. Small gain; not recommended now.

**D. One representation.** Express families as a dimension, or co-carriage as a family, or both as a new joint store. Removes a few hundred lines from one side and adds as many to the other, and breaks a constraint of the item either way (HPV co-infection as exclusive labels, or tbsim's compact dimension as one disease per strain). Not recommended.

**Recommendation**: A, with A.2 at Cliff's choice, and HP14 in it. This answers decision 16 for co-carriage: keep it (tbsim needs it in every solver, it is about 75 lines, table-driven, and has one spelling). The user story is one table of four rows (one at a time; several sharing one course; several each with its own course; unrelated pathogens), with one `ss.immunity`, one `split=`, and `.by(<name>)` throughout. Suggest tbsim rename its `tbsim.Strains` (e.g. `tbsim.DrugSets`) so `Strains` means one thing.

## 5. The carried-over sub-items

| Sub-item | Status | Detail |
|---|---|---|
| `exclusive=` in a crossing | **Obsolete** | Since S3.6 a family with `exclusive=` (or `immunity=`) needs agents whatever `approx=`, crossed or not (`choose.py:521–523`; `checks2.py`: `solver='ode'` raises, `auto` runs agents). Building it would rebuild a compartmental form S3 chose not to have. |
| Gradients | **Narrowed; move to deferred** | A strain dimension (and co-carriage) already has gradients under the ODE. A family has none (`gradients.py:56–57`), but a coupled family is agents only, so has none by design; only an uncoupled family under the ODE could gain one, which nobody needs. Function `combine=` has none by design (`gradients.py:51–53`). |
| Migration skill rows | **Still applies (S8.2)** | `skills/migrate-starsim3/SKILL.md` has no row for multi-strain models. Starsim 3 had no strain mechanism, so the rows are for HPVsim's pattern (one disease per genotype with `CrossImmunity`/`HPVTotal` connectors → `ss.Strains` with `ss.immunity`) and for tbsim's `kind='strain'`. `interop/migrate.py` has none either. Per "work nimbly", add to Stale rather than now. |
| Re-running the HPV port | **Still applies; blocked (S8.2)** | The repo's port (`v4port/genotypes/genotypes_v4.py`) predates `ss.Strains`, the family version measured on October 19 is not in the repo, and `import hpvsim` fails under v4 (`ss.Calibration`). It needs HPVsim's own v4 fixes first. |
| HP14: a strain dimension with `immunity=` over a network, as agents | **Still applies** | Reproduced (`checks.py`): `CapabilityError: "flu" has immunity= between strains, which cannot run over networks yet ... or run as compartments` (`agents.py:2546–2547`). The suggestion "run as compartments" is a dead end, since compartments refuse partial immunity (KI15). The fix: in `Agents.infect()`, after the network pairs give each new case's source strain (`agents.py:2615–2629`), keep each case with probability `1 − protection` for that strain from the target's `strains_had`, with a new random stream for CRN (about 20 lines). Exact up to the order of several infectious contacts of different strains in one step. Matters for single-course strains on a network (gonorrhea resistance), not HPV or tbsim. A family with `immunity=` over a network already runs as agents. |
| D9: `ss.immunity(protected='nab', rel_sus=m_sus)` | **Built; Cliff to confirm** | The code matches the revised D9 exactly (`strains.py:36–101`; `test_strains_immunity_attr`). Nothing to build. If confirmed, it is also the one spelling for a dimension (A.2), without `protected=`. |

Found on the way, outside S7.4: `str(ss.MFNet())` before `sim.init()` raises `KeyNotFoundError: Key p1 not found` (`networks.py:352` → `to_df()`), so an error message that prints a network can itself crash (it hid the real `WrongDimension` in `checks.py`'s first run).
