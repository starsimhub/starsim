# Strains from the biology up: a proposal (S7.4)

Written 2026-10-21, as a counterpart to [strains.md](strains.md). That document compares the three mechanisms Starsim has today (the family `ss.Strains`, the strain dimension, and co-carriage) and argues for keeping all three. This one starts from what strains do in a host and a population, asks whether HPV and TB really differ in kind, and proposes one spelling built on that. Research and design only: no code was changed.

## Summary

- **The biology does not cleave by disease; it cleaves by state.** The real question is, for each stage of the natural history, *whose* stage it is: the strain's infection (an HPV16 lesion clears or progresses on its own, whatever HPV18 is doing), or the person's (TB disease, treatment, and death happen to the person, whatever strains they carry). HPV is mostly strain-owned stages with one person-owned stage (cancer: one per person). TB is mostly person-owned stages with a few strain-owned events inside them (each strain is cured or not by the drugs; one strain can progress alone). Every multi-strain disease I checked mixes the two. So "family vs dimension vs co-carriage" is not a biological distinction between diseases; it is three corners of one space, and which corner a model lands in is today decided by which API it picks.
- **Everything else strains do is the same machinery for every disease**: per-strain traits (transmissibility, progression, clearance, drug response), protection by what a person has had *or has now* (cross-immunity, and also superinfection, coinfection, and "one strain at a time", which are the same thing: protection from current infection, at 1, partial, or 0), change of strain within a host (mutation, acquired resistance), seeding by strain, products by strain, and results by strain and for any strain.
- **The cost of today's split is real and falls on modelers.** tbsim's `add_resistance()` spends about 90 lines (`tbsim/resistance.py:150–240`) turning per-strain biology (a fitness per strain, a cure probability per strain, resistance acquired on failure) into per-set bookkeeping (`DS+RIF` levels, generated copies of transitions with `move=`, combinatorial outcomes of treatment). HPVsim, for its part, has to declare one disease per genotype and tie them together with a family. Both are computational representations leaking into the model. The silent `kind=` trap in strains.md is a symptom of the same thing: the modeler has to say which representation they mean.
- **Proposal**: one spelling, `dis.stratify(strain=[...])`, with one new idea, `shared=` (which states are the person's rather than each strain's), on top of vocabulary that exists already (`ss.immunity` with `protected=`, `adjust=`, `split=`, `move=`/`ss.transfer`, `.by('strain')`), plus one transition keyword for strain-level events inside a shared course (`per='strain'`). The engine then picks the representation (single-strain levels, carried sets, side-by-side courses, or per-person-per-strain arrays as agents), the way it already picks between ODE and agents. The three mechanisms become three internal representations, not three user concepts; nobody writes `'DS+RIF'` by hand.
- **Where two mechanisms are justified**: strains of one pathogen (this proposal) versus different pathogens that interact (HIV and TB; connectors, as now). That line is biological: strains share a natural history up to parameters; different pathogens don't.

## 1. What strains do, biologically

A strain (genotype, serotype, variant, resistance profile) is a variant of one pathogen. Across the diseases Starsim's users model, strains do these things:

| Process | What it is | Examples |
|---|---|---|
| Own traits | A strain has its own transmissibility, progression, clearance, severity, response to drugs and vaccines; fitness costs | HPV16 progresses faster than other types; RIF-resistant TB transmits less (fitness 0.575 in tbsim) and is cured less often; COVID variants differ in severity |
| Protection from past infection | Having had strain *i* lowers susceptibility to, transmission of, or severity of strain *j*; may scale with an antibody level and wane; may be above 1 (enhancement) | Flu and COVID cross-immunity; HPV type-specific antibodies; dengue's short heterotypic protection then antibody-dependent enhancement |
| Protection from current infection | Carrying strain *i* now lowers the chance of acquiring *j* now. Complete: one strain at a time. Partial: superinfection or competitive coinfection. None: independent coinfection | Flu (effectively one at a time); TB latent infection protects partly against reinfection; pneumococcal serotypes compete in co-colonization; HPV types are acquired roughly independently |
| Whose course it is | Whether a stage belongs to the strain's infection (each strain progresses and clears on its own) or to the person (one disease episode, whatever is carried) | HPV lesions are per type, cancer is the person's; TB disease and treatment are the person's; malaria clones clear on their own but fever is the person's |
| Strain events inside a person's course | Within one episode, something happens to one strain only: cured by a drug, outcompeted, or the one that progresses | TB: DS cured, RIF survives treatment (partial cure); "one of two latent strains reactivates" |
| Change of strain | A strain becomes another within a host: mutation, selection under treatment; added alongside (a resistant subpopulation emerges) or replacing | TB and gonorrhea acquired resistance on treatment failure; HIV drug-resistance mutations; within-host emergence of new variants (rare per host, important in a population) |
| Transmission from a mixed infection | Someone carrying several strains passes on each at its own rate, or one (the fittest), or a mixture | tbsim offers both (`transmission_independent`); HPV types are passed independently |
| Introduction | A strain is seeded at a time and place (a new variant arrives) | COVID variants, imported MDR-TB |
| Products and results by strain | Vaccines and drugs that act by strain; counts by strain and for any strain (a person counted once) | HPV vaccines cover 2, 4 or 9 types; prevalence of any HPV; MDR share of TB notifications |

How the example diseases use these:

| | Own course? | Protection from current infection | Protection from past | Strain events in a course | Change of strain |
|---|---|---|---|---|---|
| HPV (genotypes) | Yes, except cancer (one per person) | None (independent coinfection) | Partial, antibody-scaled, on susceptibility and severity | (none: courses are already per strain) | No |
| TB (resistance) | No: one episode | Partial while latent (superinfection), complete while diseased in most models | Small, mostly strain-agnostic | Yes: cure per strain, one strain progressing alone | Yes, on treatment and progression |
| Flu, COVID (variants) | No | Complete (one at a time) | Partial, waning | No | Rare per host; introduction matters more |
| Gonorrhea (resistance) | No | Complete or partial | None | Cure per strain | Yes, on treatment |
| Malaria (clones) | Partly: clones clear on their own, fever and treatment are the person's | Partial (multiplicity of infection) | Strain-specific and strain-transcending | Drug clears sensitive clones | Yes (resistance) |
| Pneumococcus (serotypes) | Yes (colonization per serotype) | Partial (competition) | Serotype-specific plus non-specific | No | No (but vaccine-driven replacement) |
| Dengue (serotypes) | No | Complete | Short cross-protection, then enhancement | No | No |

The point of the second table: HPV and TB are not different *kinds* of model. They are different values in the same columns. Even the column that looks like a split ("own course?") has mixed answers within single diseases (HPV's cancer, malaria's fever).

### Why the representation must not decide the biology

The choice of one-at-a-time versus coinfection, and of shared versus own courses, changes model outcomes, not just bookkeeping: whether strains coexist, how fast resistance spreads, how much a vaccine against some types lets others replace them. Lipsitch and colleagues made this point for multi-strain models: model structures that look like neutral bookkeeping choices can create or prevent coexistence by themselves ("no coexistence for free"). So a modeler should state each of these as a biological assumption, with a parameter, and not get one implicitly by choosing `ss.Strains` over `stratify(strain=)`.

## 2. Why HPV and TB look different today

Both differ from the classic single-strain-at-a-time model in one column each, and in opposite directions:

- **HPV** needs own courses: a woman can have HPV16 in CIN2 and an HPV18 infection acquired last month, each clearing on its own. Forcing that into one shared course loses per-type clearance and needs `2^k − 1` levels (strains.md, section 2). So HPV took the family: one declared disease per genotype, tied by `ss.immunity` and `exclusive='Ca'`. The family *is* "own courses", with the one person-owned state written as `exclusive=`.
- **TB** needs one shared course with several strains in it: one latent period, one disease episode, one treatment. One disease per strain would give each strain its own treatment, which is wrong. So tbsim took co-carriage, and then had to hand-build what happens to each strain inside the shared course (cure per strain, one progressing alone, resistance acquired) as transitions between sets.

Seen from the biology, each mechanism covers one corner and leaves the modeler to build the rest by hand: the family has no strain events inside a shared state (it has only `exclusive=`, a special case), and co-carriage has no per-strain anything except through set levels and `adjust=` by level. The strain dimension is co-carriage with complete protection from current infection.

## 3. Proposal: one spelling

### The rule

A disease with strains has, for each state, an owner: **the person** (shared: the person is in it, carrying some set of strains) or **each strain** (own: each strain the person has is in its own copy of the state). Everything strains do is then said with existing vocabulary, written per strain:

| To say | Spelling | Status |
|---|---|---|
| The strains, and the shares among those seeded | `stratify(strain=['DS', 'RIF'], split=[0.9, 0.1])` | exists |
| A strain's own traits | `adjust=dict(rel_trans=[1, 0.575], cure=[1, 1/3])` | exists (by level today; by strain here, with sets worked out by the engine) |
| Which states are the person's | `shared=True` (default: every state), `shared=['Ca']`, or `shared=False` | **new** |
| Protection from past or current infection | `ss.immunity(protected='R', rel_sus=M)`, `ss.immunity(protected='L', rel_sus=0.79)`, `ss.immunity(protected='nab', rel_sus=M, combine='sum')` | exists in the family (`protected=`); here it applies to both |
| A strain event inside a shared course | `ss.Transition(Tx, R, prob=0.9, per='strain')` | **new** |
| Change of strain | `move=dict(strain='DS -> RIF')` on a transition, or `ss.transfer(strain='DS -> RIF', ...)`; written per strain, applied to every set that holds it | exists for single levels; per strain is new |
| A solver limit on how many strains one person carries | `max_strains=2` | exists in tbsim; moves into Starsim as a solver option |
| Results, products | `res.n_I.by('strain')`; `res.n_I` counts each person once; `ss.Delivery(..., where=ss.by(strain=...))` | exists |

**Defaults reproduce the simplest model**, and each keyword adds one biological statement. With `shared=True` and no `immunity=`, the disease is one disease with labeled strains: a person carries one strain at a time, anyone infected or recovered is protected against every strain, and a new case takes its source's strain. That is today's strain dimension, and the classic multi-strain model. The default protection follows from ownership, not from a magic rule: a shared state is the person's, so being in it protects against the disease (all strains) as it would without strains; an own state is one strain's, so it protects only against that strain unless `immunity=` says more.

### TB

```python
tb.stratify(strain=['DS', 'RIF'], split=[0.9, 0.1],
    adjust   = dict(rel_trans=[1, 0.575], cure=[1, 1/3]),             # fitness cost, and resistance to treatment
    immunity = ss.immunity(protected='L', rel_sus=0.79),              # latent infection protects partly against a new strain; the rest is superinfection
    max_strains = 2,
)
cure    = ss.Transition(Tx, R, prob=0.9, per='strain'),                  # each strain carried is cured on its own (times its adjust); the person moves on once none is left
failure = ss.Transition(Tx, A, ..., move=dict(strain='DS -> RIF'), prob_move=0.05)  # resistance acquired on failure (spelling to settle)
```

What the engine does with this: builds the carried sets (`DS`, `RIF`, `DS+RIF`, up to `max_strains`), the `carries` and `adds` tables it already has, the per-set rates of each transition from the per-strain `adjust=`, and the outcomes of a `per='strain'` transition over each set (for `DS+RIF` in `Tx`: both cured, DS cured and RIF left, and so on). That is the combinatorics of `tbsim/resistance.py:186–231`, done once in Starsim instead of in every model that has resistance (tbsim, a gonorrhea AMR model in STIsim, a malaria resistance model). Superinfection is no longer a self-transmission `L >> L` the modeler writes: it is incomplete protection from current infection.

### HPV

```python
hpv.stratify(strain=['hpv16', 'hpv18', 'hi5', 'ohr'], split=[...],
    shared   = ['Ca'],                                                                     # one cancer per person; every stage before it is each genotype's own
    adjust   = dict(infection=[...], progression=[...]),
    immunity = [ss.immunity(protected='nab', rel_sus=m_sus, combine='sum'),                # antibody-scaled cross-immunity (D9)
                ss.immunity(protected='cleared', dur=dict(In=m_sev, Ic=m_sev))],           # severity immunity
)
```

One declared natural history, four genotypes; coinfection is independent by default because the stages are each genotype's own. `exclusive='Ca'` becomes what it biologically is: cancer is the person's state. What happens to a genotype's own course when the person enters a shared state (the others stop, as `exclusive=` sends them to `X` today) is the one rule `shared=` needs, and it is the same rule for death or hysterectomy.

### What the engine chooses

| Model | Representation | Compartmental? |
|---|---|---|
| All shared, complete protection from current infection | One strain per person: levels × states (today's strain dimension) | Yes, all solvers, gradients |
| All shared, partial protection from current infection | Carried sets × states (today's co-carriage), capped by `max_strains` | Yes, all solvers, gradients |
| All own, no coupling | Side-by-side courses (today's uncoupled family) | Yes |
| All own with cross-immunity, or mixed ownership | Agents: a carried-set bitmask per person, and a per-strain state array for the own states (today's family members, made internally) | Agents (exact compartmental form is `2^k` histories; KI15, deferred) |

So the code of the three mechanisms mostly survives, as the three representations behind one front end; what goes is the user-facing choice between them, the hand-written set levels, `kind='strain'`, `exclusive=`, and the separate member declarations.

## 4. What changes

- **Users**: one concept ("strains of a disease"), with ownership as the one structural statement. tbsim's `add_resistance()` should shrink to the biology: per-strain fitness, cure and acquisition (perhaps 30 lines instead of 130; not measured). HPVsim declares one natural history instead of members and a family. The `kind=` trap goes, since there is no `kind=`.
- **Starsim**: new code for `shared=` with mixed ownership (agents), `per='strain'` transitions over carried sets, and per-strain `adjust=`/`move=` mapped to sets; removed: `ss.Strains` as a public class, `exclusive=`, the bare-matrix `immunity=` (only `ss.immunity`), and the "dimension called immunity" corner. I have not estimated the net lines carefully: the representations stay, so I'd guess the change is a few hundred lines rewritten with a small net, not a large saving. It is a representation rewrite in the sense of S7.4's "keep the comparison as a decision before a major representation rewrite", so this is that decision.
- **S7.4's wording**: it says "`ss.Strains` stays in Starsim". This proposal drops the class (or keeps the name only as the container passed to `stratify`, if you prefer `strain=ss.Strains([...], shared=..., immunity=...)`). That is your call.
- **Gaps the biology table shows that neither mechanism covers today**, for deferred.md rather than this item: waning cross-immunity; enhancement (protection above 1, dengue; check `ss.immunity` allows it); introducing a strain at a date (variant arrival), which may just be seeding by strain with `ss.by`; choosing which strain a mixed carrier transmits (fittest vs each).

## 5. Open questions for you

1. **Is ownership the right single structural idea?** I think it is the biological cleave, and it is per state, not per disease. The alternative is two user concepts (own courses vs shared course), which matches HPV and TB but not malaria or HPV's cancer. [CK: yes, i like ownership]
2. **Strain events inside a shared course**: `per='strain'` covers "each strain cured on its own". TB also has "one of the carried strains progresses alone, the others are dropped" (tbsim's `p_multi`). Is that a second flavor (`per='strain', alone=True`), or a model detail tbsim can write with `move=`? This is the least settled part of the spelling. [CK: not sure, would need to try it out]
3. **Defaults**: complete protection from shared states (one at a time) and none across own states, as above. Alternatively, no default protection at all, so that every model says it; that is more explicit but makes a plain shared-course strain model explode into `2^k` sets unless the modeler adds `immunity=`. [CK: um, i guess complete protection is fine...]
4. **Keep `ss.Strains` as a name?** As the container, or not at all. [CK: yes, i think so]
5. **Scope now**: build this before S8.2's downstream ports, or settle the spelling now and build it with the tbsim and HPVsim ports, which are its two acceptance tests? [CK: do minimal test builds now, and also include covasim]

## References (from memory; not checked against the papers)

- Lipsitch M, Colijn C, Cohen T, Hanage WP, Fraser C (2009). No coexistence for free: neutral null models for multistrain pathogens. *Epidemics* 1:2–13.
- Gog JR, Grenfell BT (2002). Dynamics and selection of many-strain pathogens. *PNAS* 99:17209–17214 (status-based vs history-based immunity).
- Andreasen V, Lin J, Levin SA (1997). The dynamics of cocirculating influenza strains conferring partial cross-immunity. *J Math Biol* 35:825–842.
- Cobey S, Lipsitch M (2012). Niche and neutral effects of acquired immunity permit coexistence of pneumococcal serotypes. *Science* 335:1376–1380.
- Cohen T, van Helden PD, Wilson D, et al. (2012). Mixed-strain *Mycobacterium tuberculosis* infections and the implications for tuberculosis treatment and control. *Clin Microbiol Rev* 25:708–719.
- Andrews JR, Noubary F, Walensky RP, et al. (2012). Risk of progression to active tuberculosis following reinfection with *Mycobacterium tuberculosis*. *Clin Infect Dis* 54:784–791 (latent infection about 79% protective against progression after reinfection; the 0.79 above is illustrative).
- Ferguson N, Anderson R, Gupta S (1999). The effect of antibody-dependent enhancement on the transmission dynamics and persistence of multiple-strain pathogens. *PNAS* 96:790–794.
