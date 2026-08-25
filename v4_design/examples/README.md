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

## Selection criteria

Aim for coverage rather than volume: span the supported paradigms, span frameworks, and include models that stress specific features (multi-strain, age structure, spatial coupling, complex interventions, time-varying parameters).

## Relationship to other folders

- The frameworks these models come from are reviewed in `approaches`.
- Each example should have a corresponding translation in `vignettes`.
