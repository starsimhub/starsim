# Targets

## Introduction

This document describes what the targets of the v4 port are: what to check to see how the current implementation is doing.

Note: "compartmental" refers to both ODE and stochastic compartmental, unless otherwise noted.

## Starsim-family models (against v3)

Reviewed with `/v4.review <model>`, one session and one report per model.

For all:

- [ ] Fewer lines of code
- [ ] At least slightly faster when running as ABM
- [ ] No ugly hacks required to make each model work, but also no code in v4 to support a single random feature in a single model
- [ ] Not expected to work out of the box (some migration ok)

Specific goals:

- [ ] Covasim: Compartmental version works except with networks, including with strains
- [ ] FPsim: Compartmental version does NOT need to work, but great if it can
- [ ] HPVsim: Compartmental version works except with networks, including with strains
- [ ] Malariasim: Unified compartmental/ABM representations for both mosquitoes and agents
- [ ] MIGHTI: Either compartmental or agent-based, but much simplified
- [ ] STIsim: Compartmental works for individual diseases; networks and some disease combinations might not be expressible
- [ ] TBsim: Unified compartmental/ABM representation, except for households etc
- [ ] Typhoidsim: Unified compartmental/ABM, except for genetics, networks, etc

R-Starsim wraps Starsim rather than modeling a disease, so the targets above don't apply, but it can still break when Starsim changes: its tests are run by `.github/workflows/test_r.yml`, and the others' by `test_downstream.yml`. Rotasim is not tested yet, since it has no `v4` branch.

## Other models

The originals are in `lingua_franca_design/examples/<framework>/<model>/` (with their line counts and reference timings in `meta.yaml` and `README.md`), and the Starsim versions, with `compare.py` and a comparison table, are in `lingua_franca_design/vignettes/<framework>/<model>/`. Reviewed with `/v4.others`.

- [ ] measles metapop (`measles_metapop/seir_districts`: SEIR in 260 districts, on laser-measles)
  - [ ] Works as-is (more or less) when supplied as a custom module
  - [ ] Written in Starsim v4, is faster (compiled) with fewer lines of code
- [ ] LASER
  - [ ] If England-Wales performance can't be beaten with standard v4, then the ultrafast LASER module should be usable more or less as-is, ideally with roughly as many lines and roughly as fast
    - `laser/england_wales`: measles in England and Wales (the LASER tutorial)
  - [ ] `laser/hub_spoke_seir`: spatial SEIR on a hub and spokes (held to the same targets as the survey models below)
- [ ] Other models that were part of the "Approaches" survey
  - [ ] Faster or within 2x for already very fast models
  - [ ] Fewer lines of code except for very specialized things
  - The models, two from each framework:
    - [ ] Atomica: `sir` (SIR with deaths, defined in spreadsheets); `hiv_dyn` (the HIV care cascade, with programs and budgets)
    - [ ] camdl: `sir` (SIR with weekly reported cases); `stratified` (SEIR by age and patch)
    - [ ] EMULSION: `quickstart` (the quickstart model); `sir` (SIR as a compartmental, individual-based, and hybrid model)
    - [ ] EpiModel: `sis-network` (SIS on a dynamic network); `hiv` (HIV with a care cascade and PrEP)
    - [ ] epydemix: `sir_kenya` (SIR in Kenya with age-structured contacts); `covid19` (the first wave of COVID-19 in Massachusetts)
    - [ ] flepiMoP: `seir_2pop` (SEIR in two subpopulations); `vaccine_scenarios` (vaccination scenarios in two subpopulations)
    - [ ] MEmilio: `secir` (ODE SECIR with severity branches); `secir_ageres` (age-resolved ODE SECIR, with seasonality and testing capacity)
    - [ ] odin: `stochastic_sir_arrays` (discrete-time stochastic SIR, written with arrays); `seir_array` (age-structured SEIR with a delay)
    - [ ] SimInf: `sir_herds` (SIR in 1,600 cattle herds with births, deaths, and movements); `sise3` (SIS by age in 1,600 herds, with an environmental infectious pressure)
    - [ ] summer2: `sir-age-location` (SIR stratified by age and location, with mixing matrices); `seir-age-clinical` (SEIR by age and clinical status, with derived outputs)
- [ ] Canon
  - [ ] Nothing should be ugly
  - [ ] Nothing should be slow
  - The models: the textbook canon of [oct17/canon_review](oct17/canon_review/README.md), 42 models from Keeling and Rohani (2008, KR) and Vynnycky and White (2010, VW). Reviewed with `/v4.canon`.
    - [ ] KR chapter 2, simple models: 2.1 SIR; 2.2 SIR with demography; 2.3 SIR with disease mortality, frequency-dependent; 2.4 the same, density-dependent; 2.5 SIS; 2.6 SEIR with demography; 2.7 SIR with carriers
    - [ ] KR chapter 3, risk and age structure: 3.1 SIS with high and low risk; 3.2 SIS with five risk groups; 3.3 SIR with two age groups and aging
    - [ ] KR chapter 4, multiple strains and hosts: 4.1 two strains with partial cross-immunity; 4.x three strains, the exact 2^k history model; 4.3 two host species; 4.4 host and vector (Ross-Macdonald)
    - [ ] KR chapter 5, seasonal forcing: 5.1 SIR with sinusoidal forcing; 5.1 SEIR with strong forcing (chaos); 5.2 SEIR with term-time forcing; 5.3 SIR with seasonal births
    - [ ] KR chapter 6, stochastic models: 6.1 and 6.2 SIR with noise; 6.3 event-driven SIR; 6.4 tau-leap SIR; 6.5 persistence and the critical community size; 6.6 SIR with imports
    - [ ] KR chapter 7, spatial structure: 7.1 two patches with a coupled force of infection; 7.2 three patches with migration; 7.3 commuters; 7.4 a 5 by 5 lattice; 7.6 a random 4-regular network
    - [ ] KR chapter 8, control: 8.1 vaccination of newborns; 8.2 constant vaccination; 8.3 pulse vaccination; 8.4 isolation (SIQR); 8.5 contact tracing and ring vaccination
    - [ ] VW: 2.1 closed SEIR; 3 SEIR with prior immunity (herd immunity); 4.1 measles SEIR with births and deaths; 5.1 MSEIR (maternal immunity); 5.2 MSEIR with infant vaccination; 6.1 Reed-Frost; 6.2 Greenwood; 7.1 WAIFW with two age groups; 8.1 STI core group with assortativity

## Naive user

The benchmark of [oct17/naive_review](oct17/naive_review/README.md): simulated newcomers (a careful novice with the docs, an assistant with `llms.txt`, a Starsim 3 user with the migration pages, and a habitual user who guesses) doing 16 tasks with hidden reference answers. Reviewed with `/v4.naive`.

- [ ] Every final answer right
- [ ] No silent wrong first attempts (code that runs cleanly and gives the wrong number)

## Reviews

The skills that check these targets are in [ai/](ai/README.md). Each writes its report to `<date>/reviews/` (e.g. `oct22/reviews/canon.md`):

- `/v4.review <model>`: one Starsim-family model per session (`covasim`, `fpsim`, `hpvsim`, `malariasim`, `stisim`, `tbsim`), one report each
- `/v4.others`: Measles metapop, LASER, and the survey models, one report
- `/v4.canon`: the textbook canon, one report
- `/v4.naive`: the naive-user benchmark, one report
