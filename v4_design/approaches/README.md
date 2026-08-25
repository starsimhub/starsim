# Approaches

A review of the current approaches used in the epi modeling landscape, based on the [disease modeling landscape](https://starsim.org/disease_modeling_landscape).

## Purpose

This folder catalogs what already exists: the frameworks, libraries, and conventions that epi modelers use today, and how each one represents the core modeling concepts (states, transitions, populations, networks, interventions, time).

## Contents

One document per framework or family of frameworks, covering:

- What the framework is for, and which modeling paradigms it supports (compartmental/ODE, metapopulation, stochastic compartmental/SDE, agent-based)
- How a model is specified (API, DSL, config file, GUI)
- The core abstractions and their names
- Notable strengths, limitations, and idioms
- A minimal representative model (SIR if possible), and a more complex fully worked example

## Relationship to other folders

- Feeds into `best-practices`, which synthesizes these reviews into best practices per paradigm.
- Concrete code from these frameworks lives in `examples`.
