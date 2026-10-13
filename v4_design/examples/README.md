# Examples

Models written in other frameworks, to be expressed by the lingua franca.

## Purpose

This folder is the test corpus. It collects real models — as originally written, in their original frameworks — that the lingua franca must be able to represent. If a model here cannot be expressed cleanly, that is an issue for the lingua franca.

## Contents

Models are kept as close to their source form as possible, with each example accompanied by:

- A short description of what the model does and which features it exercises
- The original source code, with sufficient information to ensure reproducibility (`uv.lock` or equivalent)
- Provenance: framework and version, origin (paper, repository, tutorial), and license
- Expected outputs or reference results, where available, so translations can be checked
- Two numbers the translation will be held to: the lines of model code (excluding imports, comments, blank lines, and plotting) and the wall-clock time for a reference run, with the machine it was measured on

## Selection criteria

Aim for coverage rather than volume: span the supported paradigms, span frameworks, and include models that stress specific features (multi-strain, age structure, spatial coupling, complex interventions, time-varying parameters).

Prefer each framework's *own* examples (its tutorials, vignettes, and published models) over examples written for this project. The test is that a framework's showcase models are no harder to write in the lingua franca than in the framework itself, and a showcase model is the one its authors chose to make their framework look good.

## Build set and held-out set

The implementation is built against a small set of models and then checked against models it was not built against. The two lists are fixed before any code is written, so that the held-out models are not chosen to suit what the implementation already handles.

### Build set

Nine steps, covering ten of the eighteen models in the corpus in [design/18-conformance.md](../design/18-conformance.md#the-corpus), in the order they are implemented. Each adds one layer to the one before.

| Order | Corpus # | Model | What it adds |
|---|---|---|---|
| 1 | 1 | SIR, in all spellings | States, transitions, every solver |
| 2 | 2, 3 | SEIR with an Erlang dwell time | The source of infection, sub-stages |
| 3 | 5 | A branching model with a competing death hazard | Split on entry versus competing hazards |
| 4 | 11 | Cholera, with a water reservoir | States that are not people, counts in rates, flows from and to `ss.null` |
| 5 | 6 | Age-stratified SIR with a contact matrix | Stratification and mixing |
| 6 | 8 | Vaccination as a stratification plus a transfer | Interventions |
| 7 | 15 | A model with outcomes, data, and bounds | Results and calibration |
| 8 | 14 | A model with custom code | The try-it rule, `step()` |
| 9 | 10 | A sexual-network model | Agents and networks |

### Held-out set

These are brought in as waves, not all at the end, because a surprise in the design is cheapest to fix early.

| Wave | When | Models |
|---|---|---|
| 1 | Once build steps 1 to 5 run | One simple example from each framework: MEmilio, EpiModel, epydemix, summer2, SimInf, odin, EMULSION, flepiMoP, Atomica, LASER, camdl |
| 2 | Once the build set runs | The other eight conformance corpus models (4, 7, 9, 12, 13, 16, 17, 18) |
| 3 | Once the build set runs | One complex example from each of the same frameworks, plus the Ghana measles model and a TB model |

The wave 1 frameworks were all reviewed before the design was written, and the wave 2 models were written by the same authors as the design, so neither is fully unseen. Wave 3 is the real test.

Starsim's own tests and documentation examples are tracked separately. They are a migration requirement, not a validation set: each one either runs in the new version or has a stated reason why not.

## Relationship to other folders

- The frameworks these models come from are reviewed in `approaches`.
- Each example should have a corresponding translation in `vignettes`.
