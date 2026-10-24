# Strains, starting from the biology

**Superseded (2026-10-22)** by [strain_plan.md](strain_plan.md), now built (see [oct22/worklogs/c_strains.md](../oct22/worklogs/c_strains.md)). Kept as the record.

Proposal for discussion, 2026-10-21. Based on [strains.md](strains.md), the current Starsim strain implementations, tbsim's resistance model, and the biological studies linked below. API sketches are proposals, not working Starsim syntax.

**TB and HPV do not biologically require unrelated strain machinery.** Both involve distinguishable pathogen populations occupying the same host, potentially interacting through immunity, competition, disease, and treatment. They differ in which of those processes matter, and in how much detail a particular model retains. My recommendation is one public strain concept, with shared operations for transmission, carriage, immunity, and changes of strain. Within it, support two explicit ways of describing disease history: one course for the host, or a course for each strain. These are choices about model resolution, available to either disease.

The strongest objection to the existing document is its claim that one disease course per strain is biologically wrong for TB. Separate strain histories do not require separate patients, separate treatment prescriptions, or independent clinical outcomes. Conversely, putting several strains into one host-level disease state is a useful simplification, but is not the only biologically valid account of TB. The existing implementations are valuable; their boundaries should not define the biological concepts.

**What is actually different about these pathogens?**

First, “strain” is doing different classificatory work. HPV16 and HPV18 are distinct viral types. Papillomavirus classification is sequence based; these labels are not successive stages of one infection. For a transmission model, I would treat the types as stable identities and provide no HPV16-to-HPV18 mutation transition. This is a modeling implication of their distinct identities, not a claim that papillomaviruses never evolve. [ICTV's Papillomaviridae report](https://ictv.global/report/chapter/papillomaviridae/papillomaviridae) supplies the taxonomic context.

“Drug susceptible” and “rifampicin resistant,” by comparison, describe bacterial phenotypes. Resistance can emerge within an existing infection, and treatment can change which bacterial subpopulations dominate. Longitudinal sequencing directly observes within-host diversity and its relationship to treatment efficacy. An effective DS-to-resistant transition is therefore useful, although its rate may summarize mutation, establishment, and selection rather than a literal mutation rate. [Trauner et al., 2017](https://doi.org/10.1186/s13059-017-1196-0).

This distinction matters to parameterization, not to whether both belong in a common strain system. A strain identity can carry several properties: antigenic identity, resistance profile, transmissibility, and progression parameters. Two distinct bacterial lineages can share a resistance profile. Grouping all DS bacteria into one modeled type is an intentional aggregation. The framework should not infer antigenic difference from resistance difference, or immune protection from equality of resistance labels.

Second, both pathogens can occur as mixed infections. Human TB lung samples have revealed unrelated bacterial genotypes coexisting, sometimes with different resistance profiles and distributions across lesions. A patient's sampled resistance profile can change as different populations become detectable. That is not necessarily a new mutation or a new acquisition. The surgical cohort in this study is selected; its mixed-infection frequency should not be treated as a general-population estimate. [Moreno-Molina et al., 2021](https://www.nature.com/articles/s41467-021-22705-z).

For HPV, concurrent types can be followed separately over time. A cohort study found no significant difference in type-specific persistence between single and multiple infections. That supports retaining separate type-specific histories; it does not establish complete independence of infections under every condition. [Campos et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3071138/).

Third, one person having TB does not mean all their bacterial populations have one biological clock. In macaques, individual granulomas within a host had markedly different bacterial burdens and killing, including sterile lesions in animals with active disease. This does not prescribe a particular human population model, but it contradicts the idea that one synchronized within-host trajectory is biologically mandatory. [Lin et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3947310/).

Finally, immunity belongs in both systems. Experimental TB reinfection studies in macaques found protection during an existing infection; that does not establish a universal human cross-strain protection matrix. HPV cohort data associate naturally acquired type-specific antibody levels with reduced subsequent detection of the corresponding type; these estimates do not establish all off-diagonal cross-protection terms. The machinery should accommodate these hypotheses while keeping protection against acquisition distinct from protection against progression. [Cadena et al., 2018](https://doi.org/10.1371/journal.ppat.1007305), [Safaeian et al., 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2970577/).

The practical comparison is therefore:

| Modeling question | Typical TB resistance model | Typical HPV genotype model | Shared capability |
|---|---|---|---|
| What distinguishes types? | Drug response, sometimes fitness and lineage | Genotype, antigenicity, progression parameters | Named pathogen types with properties |
| Can several be present? | Yes, if mixed infection is represented | Yes | Concurrent carriage |
| Must their stages and clocks be separate? | Often omitted in compact models; useful in more detailed models | Usually needed for genotype-specific persistence and lesions | Optional strain-specific histories |
| Can one give rise to another? | Acquired resistance can be represented | No conversion between HPV16 and HPV18 in the proposed model | Explicit allowed conversion or emergence events |
| Does one exposure affect another? | Potential protection, competition, or altered progression | Type-specific and potentially cross-type protection | Target-specific interactions |
| What is shared by the person? | Symptoms, diagnosis, regimen, adherence, death | Vaccination, screening, clinical management, death | Host-level state and events |

**Separate what the host carries from what happens to the host.**

Conceptually, keep three kinds of information:

- **Pathogens currently present:** the set of modeled types carried, with a stage and clock for each if the model needs them. Presence need not imply infectiousness.
- **Host condition and care:** symptoms, clinical disease, treatment regimen, adherence, vaccination, and survival. Their effects may depend on which types are present.
- **Immune memory:** protection generated by past or current infections and vaccination, potentially with boosting and waning. It can remain after clearance.

These need not become three new framework classes. They are a way to assign existing states, attributes, and transitions to the right biological subject. In particular, “separate courses” should mean separately tracked courses, not statistically independent courses. They can influence each other and share host events.

Consider one person carrying susceptible and resistant TB. They begin one regimen. Its efficacy differs between the populations; the susceptible population clears while the resistant population persists. There is one patient and one treatment episode throughout. A compact model can represent this as:

```text
(on treatment, {DS, RIF}) → (treatment failure, {RIF})
```

A detailed model can retain a separate infection history for DS and RIF while reading the same regimen and adherence variables. Both are coherent. In fact, tbsim's current `resistance.py` already computes cure separately for each carried strain and then determines the host's treatment outcome. The distinction the original document presents as a barrier is already being crossed in the disease model.

For HPV, consider:

```text
HPV16: persistent infection with a precancerous lesion
HPV18: recently acquired infection
host:  eligible for one screening visit
```

Clearing HPV18 should not reset HPV16's lesion or elapsed time. A screening or treatment event can nevertheless act on the person and affect multiple type-specific processes. The same separation of host and pathogen information handles this case.

**Two course descriptions are justified; three unrelated kinds of strain are not.**

I would expose these two descriptions:

| Choice | Information retained | Example | Information deliberately omitted |
|---|---|---|---|
| Host course | One disease stage and its clocks, plus the carried set | `latent, {DS, RIF}` | Separate stages and elapsed times for DS and RIF |
| Course per strain | Stage and clocks for each type, plus shared host information | `DS: latent; RIF: active; regimen: first-line` | Finer detail such as individual lesions unless added explicitly |

Allowing only one carried type is a concurrency restriction on either description. It does not require a third foundational mechanism. Likewise, using the same natural-history template for every type says nothing about whether the clocks are shared. Two HPV genotypes can use identical transition diagrams and still occupy different stages at the same time.

A host-course model is sometimes an exact reduction of a more detailed model, but often it is a separate approximation. Suppose two hosts both carry `{A, B}`. In one, A is newly acquired and B long established; in the other, their ages are reversed. If those histories affect subsequent clearance, one set label cannot determine both hosts' transition rates correctly. A compiler cannot recover missing information. The modeler must either retain it or explicitly choose effective aggregate rates.

This is where two representations earn their keep: they preserve different information. The distinction should be explained in those terms, rather than assigned permanently to TB and HPV.

**What the common machinery should do.**

The shared contract should describe biological events, even when different execution paths implement them:

| Event | Required meaning |
|---|---|
| Exposure and acquisition of A | Identify the transmitted type, evaluate the recipient's protection and eligibility against A, then establish A according to the model's acquisition rule |
| Acquisition while B is present | Explicitly add A, replace B, or reject establishment; do not derive the rule from a label or from solver choice |
| Clearance of A | Remove A alone; update its immune memory if specified; determine host recovery separately |
| Progression | Advance a type-specific course or the shared host course, with an explicit rule for what happens to other carried types |
| Emergence of B from A | Either retain A and establish B, or replace A with B; preserve or initialize stages and clocks explicitly |
| Treatment | Apply one host event, with effects conditional on each type and the regimen; aggregate its clinical outcome as the disease model specifies |
| Host death | End every infection and pending course for that person, counting one death |

For acquisition, the contract can be expressed simply as:

```text
acquisition hazard for type j
    = exposure hazard for j
      × residual susceptibility to j
      × eligibility to establish j
```

Eligibility includes the chosen concurrency rule. Residual susceptibility can depend on infection history, vaccination, and current carriage. Protection must be evaluated against the actual incoming type, including on networks. With mutually exclusive acquisition, competing hazards determine which type establishes; iteration order should not choose the winner. With coinfection allowed, the event scheme can permit multiple acquisitions. Correlated transmission of several types in one exposure is an additional explicit rule, not automatically equivalent to independent exposures.

For a source carrying A and B, the disease model must specify how much infectiousness each contributes. Summing their contributions, capping total infectiousness, and allowing only the dominant population to transmit are different assumptions. tbsim already exposes a choice between summed and maximum fitness-based infectiousness. Preserve that choice in the common machinery; do not hide it inside the storage format.

Cross-immunity should use one interpretation everywhere: matrix row = immunizing type, column = target type, value = protection, with 0 meaning none and 1 complete. The memory supplying that matrix can be a record of prior infections, current carriage, or antibody attributes. In the existing `ss.immunity(rel_sus=...)` spelling, entries are protection, despite the keyword naming the quantity reduced; documentation must make that convention clear.

Keep the existing configurable combination rules, but make their scope explicit. For example, combining infection-derived protection by a clipped sum and then multiplying residual susceptibility by a vaccine effect is a particular model assumption. It should not be embedded in the definition of “HPV.” No matrix symmetry, full homologous protection, or particular waning law follows automatically from calling something a strain.

Also distinguish the endpoint affected. Protection against acquisition, infectiousness, progression to disease, and duration are different effects. Shortening every exit time from a state is not generally equivalent to reducing severe disease: it can accelerate progression as well as recovery. Existing `dur=` protection should be described as a duration effect; a model wanting reduced progression should target that transition or probability explicitly. Interaction mechanisms outside immunity, including facilitation, should use ordinary rate modifiers and may increase rates.

For mutation, distinguish these two events:

```text
replacement: {DS} → {RIF}
emergence:   {DS} → {DS, RIF}
```

They have different consequences for mixed infection, diagnosis, and treatment. Neither is a new transmission to the person. In a detailed course model, emergence also needs a rule for the new population's stage and clock; blindly starting a fresh exposure clock is not generally appropriate. If RIF is already present, the model must define whether populations merge or remain separate. Immune history should not be rewritten merely because the current resistance phenotype changes.

A finite catalogue of types is enough for the first implementation. Multiple simultaneous populations with the same modeled type, within-host abundance, and open-ended sequence evolution should remain possible extensions. A Boolean carried set cannot distinguish two DS lineages or represent selection among their abundances. An effective replacement event can summarize that process, but should be labeled as such.

**A concrete public interface to aim for.**

I would extend the meaning of `ss.Strains` to be the common declaration, accepting a natural-history template or explicit members. The important part of this sketch is the two modeling choices, not the exact argument names:

```python
# PROPOSED API — these calls are not currently supported.

tb = ss.Strains(
    'tb', ['DS', 'RIF'],
    natural_history=tb_course,
    course='host',
    max_strains=2,
)

hpv = ss.Strains(
    'hpv', ['hpv16', 'hpv18'],
    natural_history=hpv_course,
    course='strain',
    max_strains=2,
)

# A more detailed TB model could instead choose:
tb_detailed = ss.Strains(
    'tb', ['DS', 'RIF'],
    natural_history=tb_infection_course,
    course='strain',
    max_strains=2,
)
```

The detailed TB model would additionally declare shared clinical and treatment states. Those do not spring automatically from `course='strain'`; the disease author must specify how infection states affect clinical disease and how host care affects infections. Type-specific parameter overrides, or explicit member models where their structures differ, remain available. Existing immunity objects attach through the same interface, with consistent memory semantics.

The declaration supplies the named types and course scope. Ordinary transitions and modifiers supply acquisition, clearance, progression, treatment, and allowed strain changes through the shared operations above. I would avoid a large new collection of special-purpose classes until small examples demonstrate that ordinary transitions cannot express a required operation cleanly.

`max_strains=1` should require an explicit answer about exposure during infection: exclusion, replacement, or another declared response. At a larger limit, the behavior at capacity also needs to be explicit. Otherwise a memory-saving limit quietly becomes a biological assumption. Initial conditions should similarly allow an explicit distribution of carried sets or joint courses; seeding exactly one strain per infected person is a useful shortcut, not the definition of multistrain seeding.

Generated compartment labels may still look like `DS+RIF`. Users should be able to declare `{DS, RIF}` as data, with the string merely a display label. A plus sign in a name should not be the long-term switch that activates biological inheritance. During migration, tbsim should explicitly declare its existing dimension as `kind='strain'` to fix the documented silent error. That urgent correction is separate from the eventual interface choice.

**Keep efficient execution without promising impossible equivalence.**

The proposed common interface does not imply storing every model as a huge joint state space. The current compact dimension can execute host-course models. The existing family machinery can execute courses per strain, with shared host variables and events. Common strain identity, event semantics, immunity evaluation, and result definitions should sit above those choices.

Agent models can store one state per type per person, without enumerating every combination. Exact compartmental models generally need the joint distribution when histories interact: with K types and m possible states per type, the unconstrained joint space can have m^K combinations. Even presence alone has 2^K subsets including the empty set. Summed marginal prevalences cannot identify overlap, and history-dependent immunity can require further state.

Consequently, retain exact compact ODE models; retain detailed agent models; offer exact joint compartmental models only where tractable and implemented. Independent marginal models are useful under stated independence assumptions. Any closure that discards correlations must be an explicit approximation. Unsupported requested behavior should produce a capability error, not silently change coinfection, memory, or person-level counts. A shared interface need not promise immediate support for every combination of features and solvers.

Results deserve the same clarity. People carrying any type are a union; type-specific carriage counts overlap. New acquisitions, within-host emergence, and new clinical disease episodes are different events. Report each separately. A person carrying both DS and RIF contributes once to the number with TB, once to each appropriate strain count, and once to a treatment initiation. A diagnostic result is an observation of this state, potentially imperfect, rather than a change of biological identity.

This also limits the role of `exclusive='Ca'`. It can preserve HPVsim's chosen rule for cancer handling, but should not define what strains are. Whether cancer in one course suppresses other courses, how attribution works, and which person-level outcomes are counted are disease-model decisions. Shared events should implement those decisions explicitly.

**What I would change in the existing recommendation.**

Keep its inventory of working code and its evidence about the tbsim failure. Replace the claim that TB inherently requires one course while HPV inherently requires several with the distinction between retained host history and retained strain history. Treat single carriage as a restriction, and mixed carriage as a capability. Expand the commonality beyond spelling: matching `ss.immunity` syntax is insufficient when one representation implicitly remembers every past strain while another takes an explicit state or antibody attribute.

I would implement this incrementally:

1. **Agree on the biological contract using individual patient examples.** Work through mixed TB with partial cure, HPV with asynchronous clearance, sequential infections with persistent immune memory, and resistance emergence during treatment. Record the state and clock changes each event must cause.
2. **Make the common declaration select the existing representations.** Preserve the current compact TB model and the current HPV genotype model. Fix the explicit strain-kind bug independently. Do not require a new universal storage engine.
3. **Share actual operations.** Give both implementations access to target-specific protection, acquisition policies, selective clearance, and strain changes. Keep disease-specific rates and outcome rules in disease models. Avoid a facade that merely dispatches to two permanently incompatible meanings of immunity and mutation.
4. **Validate semantics before broadening solver support.** Check that clearing A preserves B's stage and clock; one regimen can clear only DS; emergence creates no transmission case; immunity survives clearance; and person-level events count once. Compare compact ODE and agent versions of the same declared model. Compare detailed and aggregate models only under conditions where their assumptions actually match.

This is a more substantive change than the original document's small cleanup, and it deserves a prototype before a broad rewrite. The first prototype should demonstrate both existing disease models through one declaration and then add a detailed TB example with shared care. If that requires users to manually coordinate duplicated treatment courses or hand-code two incompatible immunity systems, the abstraction has not succeeded.

My preferred decision is therefore **one strain system with two explicit course scopes**. Use shared host courses when the scientific question permits aggregation; retain strain-specific histories when their timing and outcomes matter. TB and HPV motivate different defaults, but neither should be confined to a different conceptual language.
