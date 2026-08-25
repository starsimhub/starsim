# Interoperation with other frameworks

Import from and export to the rest of the landscape. Distinct from paradigm conversion ([12-backends.md](12-backends.md)), which moves one model between execution modes; this moves a model between languages.

Justified by [best-practices/paradigm-conversion.md](../best-practices/paradigm-conversion.md) §Interconversion and the reviews in [approaches](../approaches).

## The governing rule

> **The authoring language is the artifact; the serialization is derived.**

An interchange format that nobody authors in becomes a lossy export target, and partial conformance becomes the normal state — this is the documented history of the closest large-scale precedent. The consequence for this project is a requirement, not an aspiration: this has to be a genuinely pleasant authoring language first. Interoperation is a feature of a language people use, never a substitute for being one.

## The pivot

All conversion goes through the IR ([14-ir.md](14-ir.md)). There is no direct framework-to-framework path and there MUST NOT be one: `n` frameworks need `2n` adapters, not `n²`.

```
EpiModel ─┐                              ┌─▶ SBML
summer2  ─┤                              ├─▶ Petri-net AMR
epipack  ─┼─▶  importer ─▶ IR ─▶ exporter┼─▶ odin
epydemix ─┤                              ├─▶ mermaid / graphviz
SBML     ─┘                              └─▶ a databook form
```

## Import

```python
ss.load('model.xml',  format='sbml')
ss.load('model.yaml', format='emulsion')
ss.load('config.yml', format='flepimop')
ss.load(epimodel_object)          # format inferred from the object
ss.load('framework.xlsx', format='atomica')
```

An importer MUST:

1. Produce a **complete, normalized IR document** or fail. A partial import is not an import.
2. Report every construct it could not translate, by name and source location, with `W1601 ImportIncomplete`.
3. Mark every value it *inferred* rather than read, so that the imported model's `summary()` is as honest as a hand-written one's.
4. Preserve names. A transition called `infection` in the source is called `infection` here.
5. Never silently choose a route convention. If the source model's denominator convention is ambiguous, the importer asks or refuses; `beta * S * I / N` and `beta * S * I` are different models.

### Fidelity by source

| Source | Structure | Time | Interventions | Observation | Expected fidelity |
|---|---|---|---|---|---|
| **Starsim v3** | states declared, transitions in code | typed | full | calibration components | Structure requires lifting `set_prognoses` into transitions; interventions map directly. **Assisted, not automatic.** |
| **epipack** | reaction tuples | untyped | none | none | Near-complete: 3-tuples → `progression`, 5-tuples → `transmission`. Rate expressions need the route convention resolved. |
| **summer2** | named flow kinds, stratifications | untyped | parameter functions | requested outputs | High: the flow-kind vocabulary maps one-to-one, and `stratify_with` maps to `stratify`. |
| **SimInf** | `mparse` transition strings | untyped | events | none | High for `u`; `E`/`N` select-and-shift matrices must be re-expressed as `ss.Mobility` and are the hard part. |
| **odin** | `deriv`/`update`/`initial` | `dt` in scope | none | `data()` + likelihood | Medium: array-indexed models must be re-expressed as dimensions or imported as opaque. |
| **EMULSION** | YAML state machine | untyped | none | none | High, and valuable: its shipped feature models are the paradigm-switch test corpus. |
| **flepiMoP** | compartment product + `proportional_to` | dates | modifiers | **outcomes cascade** | Medium for structure, high for outcomes, which map directly to `ss.Outcome`. |
| **EpiModel** | derivative function or `status` strings | steps | scenario table | `transmat` | Low for `dcm`; medium for `netsim`, where `inf.prob`/`act.rate` map exactly onto `beta`/`acts`. |
| **Atomica** | transition matrix + characteristics | typed columns | programs | databook | High and worth doing: the framework workbook is nearly the IR already. |
| **SBML / AMR / MIRA** | species and reactions | annotations | none | none | Medium: general reactions must be classified into the closed transition vocabulary, which is exactly what a Template taxonomy does. |
| **EMOD / EpiHiper** | JSON campaigns and rules | typed | **full** | reporters | Low for structure, high for interventions: `Targeting_Config` and typed set algebra map onto `ss.Expr` cleanly. |
| **epydemix / MetaCast / epymorph** | code | mixed | contact reductions | none | Medium; their contact and mobility data layers are more valuable as data providers than as model imports. |

The two entries worth the most effort first are **Starsim v3** (the migration path for existing users, and the one that has to work) and **EMULSION** (the paradigm-switch test corpus already exists there).

### Classification on import

The hardest import problem is the same in every source: **a general rate expression must be classified into the closed transition vocabulary**. `beta * S * I / N` is a transmission; `gamma * I` is a progression; `mu * N` is a birth. The rules:

1. A rate expression mentioning a state other than the source is a `transmission`, with that state as `source`.
2. Dividing by a population total indicates frequency dependence; not dividing indicates density dependence. If it is unclear, refuse (`E1601 UnclassifiableRate`) and show the expression.
3. An expression mentioning only the source state is a `progression`.
4. An expression mentioning no state is a `birth` if it has no source and a per-capita `death` otherwise.
5. Anything else is imported as `ss.process` with its expression recorded verbatim in `meta`, and the resulting capability loss is reported.

Rule 5 is what keeps imports honest: an import that cannot classify a rate produces a model that runs agent-based and refuses ODE, which is exactly what should happen, rather than a guess.

## Export

```python
ss.save(model, 'model.json')                    # IR: complete, round-trips
ss.save(model, 'model.sbml',  format='sbml')
ss.save(model, 'model.json',  format='petrinet')
ss.save(model, 'model.R',     format='odin')
ss.save(model, 'model.md',    format='mermaid')
ss.save(model, 'form.csv',    format='databook')
```

Only the IR format is lossless. Every other export MUST report what it dropped, at declaration granularity:

```text
ss.save(model, 'model.sbml')
  ! dropped: ss.SexualNet route (no SBML representation)
  ! dropped: 2 effects with agent-rank conditions
  ! approximated: lognormal dwell time on 'recovery' -> Erlang(k=2), KS distance 0.04
  kept: 3 states, 2 transitions, 1 dimension
```

A silent lossy export is the same category error as a silent paradigm approximation, and is treated the same way.

## Ontology annotation

Nobody except a dedicated ontology tool can tell whether two models' compartments mean the same thing. That is why `ontology=` on a state ([03-states-and-transitions.md](03-states-and-transitions.md)) earns its place as an **optional** field: it is exactly what automated model merging needs, and requiring it on every compartment is real friction for every user who is not merging models.

Importers SHOULD populate it when the source has it. Exporters MUST emit it when present. Nothing in the language may require it.

## Model comparison

Once two models are IR, comparing them is mechanical:

```python
ss.diff(model_a, model_b)          # structural diff, declaration by declaration
ss.diff(run_a, run_b)              # the five-level hash diff (11-random.md)
```

Structural diff reports added, removed, and changed declarations in the user's vocabulary — "transition `hospitalization` added", "`recovery` dwell time Exp(10 d) → Erlang(k=3, 10 d)" — not a JSON patch. This is what makes model review possible in the way code review is possible, and it is a capability none of the frameworks in the landscape has.

## The vignette obligation

The claim that this language can express the landscape is empirical. [vignettes](../vignettes) translates the [examples](../examples) corpus into this specification, and each translation is a test:

1. The translation runs.
2. Its results match the original framework's to a stated tolerance ([18-conformance.md](18-conformance.md)).
3. Where the translation is not faithful, the vignette says so and names the construct that did not survive.

Point 3 is not a failure condition. A catalogue of what does not translate, with reasons, is a more useful deliverable than a claim that everything does.
