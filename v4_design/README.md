# Design of the Epi Modeling Lingua Franca

## Introduction

This project contains ideas for implementing a _lingua franca_ for epi modeling (and also known as Starsim v4). Principles of the lingua franca are:

- AI-native, serving as a rigorous-yet-simple deterministic representation of modeling concepts (expressed as text or other modeling frameworks) into code;
- Support for different modeling paradigms, including at least:
    - Compartmental/ODE models
    - Metapopulation models
    - Stochastic compartmental/SDE models
    - Agent-based models
- Interconversion between modeling frameworks (e.g. EpiModel and epydemix) and paradigms (e.g. compartmental to agent-based)

## Structure

- `approaches` contains a review of the current approaches used in the epi modeling landscape, based on https://starsim.org/disease_modeling_landscape.
- `best-practice` distills the modeling landscape into best practices for each type of modeling.
- `design` contains design specifications for the language, based on the best practices described.
- `examples` contains examples of models written in other frameworks to be expressed by the lingua franca.
- `vignettes` contains translations of the model examples into the lingua franca.