# Atomica

|  |  |
|---|---|
| **Language** | Python engine; **models defined in Excel workbooks** |
| **Paradigms** | Compartmental, deterministic, with cascade analysis and budget optimisation |
| **Specification style** | Two spreadsheets — a *framework* workbook defining structure, a *databook* supplying data |
| **Version reviewed** | current `master` |
| **Licence** | MIT |
| **Code** | <https://github.com/atomicateam/atomica> |
| **Docs** | <https://atomica.tools/docs> |
| **Paper** | — |
| **Ecosystem** | Atomica / Optima (Burnet Institute; Optima Consortium for Decision Science) |

## What it's for

Atomica is a compartmental engine for health-system and disease-programme modelling, coupled to allocative-efficiency optimisation: given a budget and a set of programmes, what allocation minimises burden? Its models are used for national HIV, TB, and health-system planning.

Its inclusion here is entirely about **specification medium**. Atomica is the only framework in the review whose models are authored in a spreadsheet, and it is by a wide margin the most accessible path from "domain expert who is not a programmer" to "running model" that the landscape offers.

## How a model is specified

A framework workbook with named sheets. Here is the shipped SIR framework, in full:

**Compartments**

| Code Name | Display Name | Is Source | Is Sink | Is Junction | Databook Page |
|---|---|---|---|---|---|
| sus | Susceptible | n | n | n | state_variables |
| inf | Infected | n | n | n | |
| rec | Recovered | n | n | n | |
| dead | Dead | n | y | n | |

**Transitions** — a *matrix*, rows = source, columns = destination, cells = the parameter driving that flow:

|  | sus | inf | rec | dead |
|---|---|---|---|---|
| **sus** | | foi | | susdeath |
| **inf** | | | recrate | susdeath,infdeath |
| **rec** | | | | susdeath |
| **dead** | | | | |

**Characteristics** — named aggregates and ratios over compartments:

| Code Name | Display Name | Components | Denominator |
|---|---|---|---|
| ch_all | Total number of entities | sus, inf, rec | |
| ch_prev | Prevalence | inf | ch_all |
| ch_infrec | Number ever infected | inf, rec | |
| ch_propinfrec | Proportion ever infected | ch_infrec | ch_all |

**Parameters** — with a **Format** column that is a de facto type:

| Code Name | Display Name | Format | Targetable | Default | Function |
|---|---|---|---|---|---|
| transpercontact | Transmission probability per contact | Probability | y | 0.0008 | |
| contacts | Number of contacts annually | | y | 80 | |
| recrate | Average duration of infections (years) | Duration | y | 0.5 | |
| infdeath | Death rate for infected people | Rate | y | 0.016 | |
| foi | Force of infection | Rate | | | `(1 - (1-ch_prev*transpercontact)**floor(contacts)*(1-ch_prev*transpercontact*(contacts-floor(contacts))))*(1-susdeath)` |

**Cascades** — named sequences of characteristics for cascade (care-continuum) analysis.

That is the entire model. No code.

## Core abstractions

| Concept | Atomica name | Notes |
|---|---|---|
| Compartment | row on the Compartments sheet | With `Is Source` / `Is Sink` / `Is Junction` flags |
| **Junction** | a compartment flagged `Is Junction` | Zero-residence-time: individuals pass straight through, splitting by outflow proportions — the mechanism for branching |
| Transition | a cell in the transition matrix | Contains the *name of the parameter* driving the flow |
| Characteristic | named combination of compartments, optional denominator | Aggregates and prevalences as first-class named objects |
| Parameter | row on the Parameters sheet | With `Format` (Probability / Duration / Rate / Number / Proportion), min/max, default, `Targetable`, and an optional `Function` |
| Function | an expression over parameters and characteristics | Parsed by `function_parser.py` |
| Databook | a second workbook | Time series of data per population, generated *from* the framework |
| Programme | `progbook` | Interventions with unit costs, coverage, and outcome effects |
| Cascade | named sequence of characteristics | For care-continuum analysis |
| Optimisation | `optimization.py` | Budget allocation across programmes |

## What the spreadsheet form gets right

**The transition matrix is the right visual form for a compartmental model.** A square grid of compartments with parameter names in the cells is, at a glance, the flow diagram. It also makes structural errors visible: an empty row is an absorbing state, an empty column is unreachable.

**Parameters are typed by `Format`.** Probability, Duration, Rate, Number, Proportion. This is a units system in a spreadsheet column — coarser than camdl's dimensional types but doing the same job, and notably it is the *only* place in the review where a non-programmer-facing tool makes this distinction.

**Characteristics separate what is modelled from what is reported.** Prevalence is not a compartment; it is a named ratio of a compartment set to a denominator set, declared once and available to functions, data entry, and output. This is a cleaner treatment of derived outputs than most code-based frameworks manage.

**The databook is generated from the framework.** Define the structure, and Atomica produces a spreadsheet with exactly the data fields the model needs, per population, with the right units and labels. That is the data-requirement contract epymorph declares in code, delivered as a data-entry form.

**Junctions.** A zero-residence-time compartment that splits inflow by proportion is the compartmental equivalent of camdl's branching transitions, and it is expressible as one flag on a row.

## Time

Annual or sub-annual steps set in the framework; parameters carry `Format` (Duration in years, Rate per year) so units are declared even though they are not checked arithmetically. Data is entered by calendar year in the databook.

## Programmes and optimisation

The `progbook` is a third workbook where interventions are entered as **programmes** with unit costs, coverage functions, and effects on targetable parameters. Optimisation then reallocates a fixed budget across programmes to minimise an objective.

This is a real intervention vocabulary of an unusual kind: an intervention is a *funded programme with a cost function and a parameter effect*, which is the representation a health-ministry planner needs and which almost nothing else in the review provides. The `Targetable` column on the Parameters sheet is what connects the two — it declares which model parameters a programme is allowed to move.

## Strengths

- **A genuinely non-programmer specification path.** A domain expert can define a compartmental model, its parameters, its derived indicators, and its intervention programmes without writing code. Nothing else in the review comes close.
- **The transition matrix as the specification form** — visual, complete, and error-revealing.
- **Parameter `Format` as a lightweight type system**: Probability / Duration / Rate / Number / Proportion.
- **Characteristics** as declared derived quantities with denominators.
- **Databook generated from the framework**, so the data requirements are a form rather than a document.
- **Junctions** for branching without residence time.
- **Programmes with costs, coverage, and targetable parameters**, plus budget optimisation over them.
- **Cascades** as a first-class analysis object.
- **The model *is* data.** An `.xlsx` file, versionable (awkwardly), inspectable, and shareable with people who will never open a terminal.

## Limitations

- **Excel.** Binary format, poor diffs, no comments in a useful sense, formula cells that can silently break, and a template that relies on cross-sheet Excel formulas to stay in sync. Every argument against JSON and YAML as a model language applies here with a version-control penalty on top.
- **Deterministic compartmental only.** No stochasticity, no agents, no networks.
- **No stratification operator.** Populations are a separate axis handled by the databook; anything else means more compartments.
- **The `Function` column is a string expression** in a bespoke mini-language parsed by `function_parser.py`, with the `foi` example above showing how quickly that becomes unreadable. It is the escape hatch, and it is where a spreadsheet model stops being legible.
- **`Format` is documentation, not enforcement.** Nothing checks that a Duration is used where a duration is expected.
- **No inference beyond calibration** to time series; no observation model.
- **Small development community**, and documentation aimed at Atomica's own user base rather than at general modellers.
- **No public paper.**

## Implications for the lingua franca

1. **Atomica is the accessibility benchmark.** If the lingua franca is only writable by people comfortable with a text-based DSL, it will not reach the audience Atomica reaches. That does not mean adopting spreadsheets; it means the design should have a **canonical tabular projection** — the model's compartments, transitions, parameters, and derived quantities as tables that could be rendered as a form, a spreadsheet, or a web UI, and read back. Since the model is data underneath, this is a serialisation question rather than a language question.
2. **The transition matrix is a projection worth supporting.** A square compartment × compartment grid with parameter names in cells is a complete, legible view of a compartmental model. Any lingua franca whose IR contains named transitions can render this for free, and it should.
3. **Take `Format` and make it a real type.** Probability / Duration / Rate / Number / Proportion is exactly Starsim's `prob` / `dur` / `per` / `freq` and camdl's `probability` / `time` / `rate` / `count`, in a spreadsheet column. **Three frameworks with completely different specification media independently converged on the same small set of parameter kinds.** That is about as strong an argument as this review can produce for it being the right vocabulary.
4. **Characteristics belong in the language.** A named combination of compartments with an optional denominator covers prevalence, coverage, cascade steps, and most reported indicators. It is declarative, it is data, and it unifies "what to fit to", "what to plot", and "what to report". summer2's `request_output_for_compartments` and camdl's `prevalence(...)` are partial versions; Atomica's has the denominator built in.
5. **Generate the data-entry contract from the model.** Atomica's databook and epymorph's `AttributeDef` requirements are the same idea in different media: the model declares what data it needs, and the tooling produces either a form or a fetch. This should be a standard output of the lingua franca's compiler.
6. **Programmes with costs are a missing intervention dimension.** Across the whole review, only Atomica represents an intervention as something with a **unit cost and a coverage function** that a budget constrains. For the policy audience this project ultimately serves, that is not an optional extra, and no other framework here supplies it. The `Targetable` flag — declaring which parameters a programme may move — is the mechanism, and it is also a partial answer to Starsim's unconstrained-connector problem.
