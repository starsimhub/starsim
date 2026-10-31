# Port plan: STIsim, HPVsim, FPsim on Starsim v4

Worktrees on branch `v4-port`: `../stisim-v4`, `../hpvsim-v4`, `../fpsim-v4`. Starsim: `v4-rs`.

## Principles

- **One spec, two solvers.** Each model is a declared `ss.Disease` (or module) that runs with `solver='abm'` and with `solver='ode'`. When a feature has no compartmental form, the model keeps it, and `sim.solvers()` says why that version runs as agents. A separate compartmental variant is written only where the full model can't be crossed (FPsim's faithful form, HPV genotypes); it lives beside the full model as `<model>_compartmental`.
- **Agreement tests.** For each model, a small test checks that ABM and ODE agree on a simplified configuration (homogeneous mixing or contact rates by risk group), within a stated tolerance.
- **The disease layer stays thin.** Parameters, data and model structure live in the disease repo. Anything engine-shaped (stock targets, queues, clocks, pregnancy) goes into starsim, as the proposals specify. No workarounds in the ports: if something is missing, it's added to starsim or listed in `v4_review/proposals/`.
- **Keep the current API where it costs nothing.** `sti.Sim(diseases=['hiv', 'syph'], ...)`, `hivsim.demo('zimbabwe')`, `hpv.Sim(...)` and `fp.Sim(location=...)` should still work. Registries and par routing stay unless v4 makes them redundant.
- **Baselines.** Each port keeps its v3 test suite where it applies. The target is agreement with v3 on summary outputs, within noise, at fixed parameters, not the same random draws.

## Dependencies on starsim v4-rs (from `proposals/00-README.md`)

| Step | Feature | Blocks |
|---|---|---|
| done | bug fixes (births, beta, sex, lazy conditions, branches, capacity) | everything |
| 1 (in progress) | exits scheduled at entry, same-step cascade, events, `X.time` (agents) | HIV CD4, syphilis latent, BV recurrence, CT load |
| 2 (in progress) | undriven edges, `ss.Treatment(moves=, on_fail=)` | all treatment |
| 4 | conveyor; compartmental clocks, `since`, `queue`, fixed delays | ODE versions of everything time-dependent |
| 5 | `ss.on`, `ss.stay`, `ss.people`, Dists with `ss.by` | BV gating, syphilis tertiary, sex-specific durations (all bacterial STIs), FPsim postpartum |
| 6 | `priority=`, `queue=`, `target=`/`remove=`, `ss.Capacity` | HIV ART/PrEP/VMMC targets, HPV treat_num, clinics |
| 7 | `ss.attr`, `ss.spread` | HIV CD4 (declarative), HPV immunity, seeding |
| 8 | multi-disease products, multi-outcome tests | HPV vaccines, screening, syndromic management |
| 9 / 04 | declared pregnancy, births from deliveries, `ss.mother` | FPsim, congenital syphilis, HIV vertical transmission |
| — | condoms in `ss.SexualNetwork`; `match=` and per-group targets on `MFNet` | the STIsim network; the ODE version of sexual networks |

## Order

1. **STIsim bacterial STIs** (CT, NG, TV) plus testing and treatment. These are the smallest models and use steps 1, 2 and 5. They're also the first ABM/ODE agreement test.
2. **GUD, BV, connectors.** The connectors are mostly done in `v4_review/bv_gud_connectors`.
3. **Syphilis** (steps 1, 2, 5, 9).
4. **HIV and HIVsim** (steps 1, 2, 6, 7). Also `hivsim.demo('zimbabwe')` and the baseline test.
5. **The STIsim network.** Keep StructuredSexual as a subclass; move condoms into starsim. Its ODE form needs contact rates by risk and sex group, which is a new mixing approximation (spec B5).
6. **HPVsim natural history.** One disease per genotype. The full model runs as agents; the compartmental version is single-genotype or has no cross-immunity (decision 8).
7. **HPVsim interventions** (step 8).
8. **FPsim.** The declared life course, plus a compartmental rewrite, once pregnancy (04) lands.

## Decisions needed before ports go far

- `proposals/00-README.md`, "Decisions for Robyn and Cliff", 1–8.
- STIsim: keep `sti.Sim`'s string construction and par routing, or replace them with v4's declared-model construction plus thin helpers?
- Where ABM/ODE agreement tests live: in each disease repo, or in starsim's conformance corpus (spec 18)?
