# Design

Design specifications for the epi modeling lingua franca, based on the best practices described in `best-practices`.

## Purpose

This is the core of the project: the specification of the language itself. It defines the deterministic, AI-native representation that modeling concepts — expressed as text or as other modeling frameworks — are translated into.

## Contents

Specification documents covering:

- The class structure, core vocabulary, and syntax (states, transitions, populations, networks, interventions, results)
- Semantics: execution order, time handling, stochasticity, reproducibility
- Paradigm support and how a single specification maps onto compartmental, stochastic, metapopulation, and agent-based execution
- Interconversion: import from and export to other frameworks, and conversion between paradigms

## Design goals

- **AI-native**: unambiguous enough that a model description in prose maps deterministically to a specification
- **Rigorous yet simple**: minimal vocabulary, no hidden behavior
- **Paradigm-agnostic**: the same model expressible across paradigms where that is meaningful

## Relationship to other folders

- Justified by `best-practices`; each specification decision should cite the practice it follows.
- Exercised by `vignettes`, which translate the `examples` corpus into this specification.
