Calibration workflows

- Layered parallelization
- Common workflow
- Multi-machine parallelization
- Controlling Optuna database/storage

Seeding philosophy
- In general, *either* let the seeds be managed automatically, *or* use n_reps=1 and reseed=False to do your own thing

Alternate workflows
- build_fn returning Sim vs MultiSim
  - How to set n_cpus in that case
  - If build_fn returns a MultiSim then the MultiSim ind is set rather than the seed
- Random seed control and reseeding


Note
- If build_fn returns a MultiSim, `n_reps` should be 1 (cannot reseed/repeat the multisim), `reseed`




# TODO
x - Check that resuming works properly
- Check that plotting with multisim seeding works properly
x - Develop distributed workflow