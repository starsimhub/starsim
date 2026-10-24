# Invariance fuzzing: what was run

Counts per transform over all runs in `runs/` (from `fuzz/summarize.py runs/*.jsonl`). "models" is the number of generated models the transform was tried on; `skip` means it does not apply (e.g. `rename_dim_space` needs a place dimension without mobility). `invalid` means the original model itself was rejected by v4: apart from a harness bug in an early `to_code_core` (records removed) there was one (`permute_dims`, a timeout), so no generated model that the spec calls valid was rejected.

| Run file | Models (seeds) | Transforms | Workers |
|---|---|---|---|
| `runs/cheap.jsonl` | 231 (0-230) | the 32 cheap transforms except `to_code_core` | 4 |
| `runs/roundtrip.jsonl` | 181 (0-180) | `add_analyzer_ode`, `save_load`, `init_save_load`, `init_parallel`, `parallel_seeds` | 1 |
| `runs/supp.jsonl` | 400 (0-399) | `to_code_core`, `run_save_load`, `rename_dim_space`, `permute_dims` | 4 |
| `runs/cross.jsonl`, `runs/cross2.jsonl` | 52 (0-16, 80-114) | `cross_binomial`, `cross_gillespie`, `cross_abm` | 2 + 3 |

The plan was a few thousand models per cheap transform. The machine ran at a load of 40 to 55 on 20 cores throughout (other agents' test suites), so with at most 6 worker processes the cheap transforms reached about 230 models, the round trips about 180, and the cross-solver comparisons 52 (each runs 24 stochastic simulations, and agents at `dt=0.1`). Every run is resumable: rerun the same command with a larger `--n` (see `fuzz/README.md`).

| transform | models | pass | fail | error | invalid | skip | harness |
|---|---|---|---|---|---|---|---|
| add_analyzer | 231 | 231 | 0 | 0 | 0 | 0 | 0 |
| add_analyzer_ode | 181 | 15 | 0 | 166 | 0 | 0 | 0 |
| add_disease | 231 | 216 | 15 | 0 | 0 | 0 | 0 |
| cross_abm | 52 | 48 | 2 | 2 | 0 | 0 | 0 |
| cross_binomial | 52 | 52 | 0 | 0 | 0 | 0 | 0 |
| cross_gillespie | 52 | 45 | 0 | 7 | 0 | 0 | 0 |
| dt_half | 231 | 166 | 4 | 0 | 0 | 61 | 0 |
| dur_rate | 231 | 187 | 3 | 0 | 0 | 41 | 0 |
| init_literal | 231 | 64 | 0 | 0 | 0 | 167 | 0 |
| init_new | 231 | 192 | 14 | 25 | 0 | 0 | 0 |
| init_parallel | 181 | 1 | 180 | 0 | 0 | 0 | 0 |
| init_run | 231 | 231 | 0 | 0 | 0 | 0 | 0 |
| init_save_load | 181 | 181 | 0 | 0 | 0 | 0 | 0 |
| literals | 231 | 231 | 0 | 0 | 0 | 0 | 0 |
| new | 231 | 187 | 12 | 32 | 0 | 0 | 0 |
| new_simstrat | 231 | 0 | 31 | 54 | 0 | 146 | 0 |
| one_level | 231 | 197 | 0 | 0 | 0 | 34 | 0 |
| parallel_seeds | 181 | 124 | 35 | 22 | 0 | 0 | 0 |
| permute_dims | 631 | 94 | 0 | 0 | 1 | 536 | 0 |
| permute_modules | 231 | 208 | 23 | 0 | 0 | 0 | 0 |
| permute_states | 231 | 152 | 35 | 0 | 0 | 44 | 0 |
| permute_transitions | 231 | 231 | 0 | 0 | 0 | 0 | 0 |
| rename_dim | 231 | 73 | 11 | 0 | 0 | 147 | 0 |
| rename_dim_space | 631 | 44 | 19 | 0 | 0 | 568 | 0 |
| rename_module | 231 | 231 | 0 | 0 | 0 | 0 | 0 |
| rename_states | 231 | 218 | 13 | 0 | 0 | 0 | 0 |
| rename_transitions | 231 | 231 | 0 | 0 | 0 | 0 | 0 |
| run_new | 231 | 199 | 11 | 21 | 0 | 0 | 0 |
| run_save_load | 400 | 0 | 0 | 400 | 0 | 0 | 0 |
| save_load | 181 | 181 | 0 | 0 | 0 | 0 | 0 |
| spell_class | 231 | 150 | 0 | 81 | 0 | 0 | 0 |
| spell_explicit | 231 | 231 | 0 | 0 | 0 | 0 | 0 |
| spell_lookup | 231 | 231 | 0 | 0 | 0 | 0 | 0 |
| spell_positional | 231 | 231 | 0 | 0 | 0 | 0 | 0 |
| spell_string | 231 | 231 | 0 | 0 | 0 | 0 | 0 |
| strat_kwarg | 231 | 85 | 0 | 0 | 0 | 146 | 0 |
| strat_sim | 231 | 85 | 0 | 0 | 0 | 146 | 0 |
| to_code | 231 | 36 | 29 | 166 | 0 | 0 | 0 |
| to_code_core | 400 | 161 | 202 | 37 | 0 | 0 | 0 |
| units_sim | 231 | 223 | 8 | 0 | 0 | 0 | 0 |
| units_typed | 231 | 231 | 0 | 0 | 0 | 0 | 0 |

Timeouts (240 s per case): 7 `cross_gillespie`, 2 `cross_abm`, 1 `permute_dims`, all under load; counted as `error` above and as cluster `timeout` by `fuzz/clusters.py`.
