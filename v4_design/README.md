# Design proposal: a lingua franca for epi modeling

## Start here

This folder is the design record of Starsim v4. What governs, and what is current:

1. [constitution.md](constitution.md): the principles every choice is checked against.
2. [targets.md](targets.md): what v4 has to achieve, model by model.
3. [design/](design/): the spec, kept in line with the code by `tests/devtests/test_spec.py`.
4. The working plan, now [oct22/plan_b.md](oct22/plan_b.md), with its worklogs; the latest review against the constitution is [oct22/claude_assessment.md](oct22/claude_assessment.md).
5. [admin/deferred.md](admin/deferred.md): what was set aside, and "Won't do", what was decided against. [admin/implementation.md](admin/implementation.md) records what is done.

The dated folders (`oct17/` to `oct22/`) are history: each plan in them says what superseded it. The rest of this page is the original proposal of 2026-08-25.

_**Version:** 0.1 | **Date:** 2026.08.25 | **Status:** Under development, not yet human-reviewed_

## Introduction

This project contains ideas for implementing a _lingua franca_ for epi modeling. Principles of the lingua franca include:

- It should capture the best aspects of existing epi models (in terms of flexibility, simplicity, and elegance), and improve upon them when possible. In other words, **a model should never be easier to write without the lingua franca than with it**.
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
- `best-practices` distills the modeling landscape into best practices for each type of modeling. *[second pass done after human review]*
- `design` contains design specifications for the language, based on the best practices described. *[second pass done after human review; brought in line with the code after the review of October 2026, and its code blocks, error catalog, and vocabulary counts are checked by `tests/devtests/test_spec.py`]*
- `examples` contains examples of models written in other frameworks to be expressed by the lingua franca. *[done in the `lingua_franca_design` repository, which holds the originals; this folder has only the README. The smaller fixtures the tests need are in `tests/`, and the build-set models are in `tests/corpus.py`]*
- `vignettes` contains translations of the model examples into the lingua franca. *[23 done, in the `lingua_franca_design` repository (this folder has only the README); `tests/north_star.py` measures them, and `tests/switching/` runs a scaled-down demo for each of 13 frameworks]*
- `webapp` is a tool for viewing `examples` models side by side with their `vignettes` translations, with diff-style markup. *[not done]*

## AI disclaimer

AI tools (mostly Claude) were used to summarize information, compare and assess frameworks, extract code examples, and create vignettes. Assessments and design decisions are owned by the humans.