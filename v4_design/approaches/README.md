# Approaches

A review of the current approaches used in the epi modeling landscape, based on the [disease modeling landscape](https://starsim.org/disease_modeling_landscape).

## Purpose

This folder catalogs what already exists: the frameworks, libraries, and conventions that epi modelers use today, and how each one represents the core modeling concepts (states, transitions, populations, networks, interventions, time). It is descriptive rather than prescriptive — the goal is an honest survey of the landscape, which the `best-practices` folder then distills into recommendations.

## Contents

One document per framework or family of frameworks, covering:

- What the framework is for, and which modeling paradigms it supports (compartmental/ODE, metapopulation, stochastic compartmental/SDE, agent-based)
- How a model is specified (API, DSL, config file, GUI)
- The core abstractions and their names
- Notable strengths, limitations, and idioms
- A minimal representative model, ideally the same reference model across frameworks for comparison

## Relationship to other folders

- Feeds into `best-practices`, which synthesizes these reviews into best practices per paradigm.
- Concrete code from these frameworks lives in `examples`; this folder holds the analysis, not the corpus.
