# Notes 2026.10.12

## Todo

- [ ] Make list of examples -- include:
  - [ ] Ghana measles
  - [ ] TB
  - [ ] camdl examples
  - [ ] Everything in the Starsim tests and docs
  - [ ] 1 simple and 1 complex example from each framework
- [ ] sim.run(n_runs=100) as a shortcut for ss.parallel(sim, n_runs=100); both return a MultiSim
- [ ] Actually implement in code -- start with Starsim, or new folder?
- [ ] Yes I think we should add Erlang

## Thoughts

- I do think there needs to be separation between e.g. SIR and a water reservoir, but `ss.Compartments` isn't it ...
- I.n is nice, I.prev is bad. I.frac maybe?