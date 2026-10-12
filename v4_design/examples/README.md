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

## Relationship to other folders

- The frameworks these models come from are reviewed in `approaches`.
- Each example should have a corresponding translation in `vignettes`.
