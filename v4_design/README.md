# Design proposal: a lingua franca for epi modeling

_**Version:** 0.1 | **Date:** 2026.08.25 | **Status:** Under development, not yet human-reviewed_

## Introduction

This project contains ideas for implementing a _lingua franca_ for epi modeling. Principles of the lingua franca include:

- It should capture the best aspects of existing epi models (in terms of flexibility, simplicity, and elegance), and improve upon them when possible.
- Be AI-native, serving as a rigorous-yet-simple and deterministic representation of modeling concepts (as expressed as prose or using other modeling frameworks) into code.
- Provide a unified representation across different modeling paradigms, including at least:
    - Compartmental/ODE models
    - Stochastic compartmental/SDE models
    - Metapopulation models
    - Agent-based models
- Allow interconversion between modeling frameworks (e.g. EpiModel to epydemix) and paradigms (e.g. compartmental to agent-based).

## Motivation

Why do we need a lingua franca? Why can't LLMs just write disease models directly? Two main reasons:

- AI is most efficient and accurate as a thin wrapper for code that already provides most desired functionality; imagine an LLM trying to correctly do a dataframe merge without access to R or `pandas`.
- A lingua franca serves as a rigorous source of truth for comparing other models to, and provides a robust endpoint (MCP) to develop AI skills, guardrails, and evals against. If a compartmental model written in R with EpiModel gives a different result than an agent-based model written in Python in Vivarium, currently it's extremely difficult to understand why. A lingua franca allows the differences between models to be clarified and explored.

## Project structure

- `approaches` contains a review of the current approaches used in the epi modeling landscape, based on https://starsim.org/disease_modeling_landscape. *[first pass done and reviewed by humans]*
- `best-practices` distills the modeling landscape into best practices for each type of modeling. *[first pass done, and in the process of being reviewed by humans]*
- `design` contains design specifications for the language, based on the best practices described. *[first pass done, NOT yet reviewed by humans]*
- `examples` contains examples of models written in other frameworks to be expressed by the lingua franca. *[not done]*
- `vignettes` contains translations of the model examples into the lingua franca. *[not done]*
- `webapp` is a tool for viewing `examples` models side by side with their `vignettes` translations, with diff-style markup. *[not done]*

## AI disclaimer

AI tools (mostly Claude) were used to summarize information, compare and assess frameworks, extract code examples, and create vignettes. Assessments and design decisions are owned by the humans.