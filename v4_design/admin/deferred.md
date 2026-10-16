# Deferred

Work that has been looked at and set aside for now: worth doing, but not yet. Each item says what it would take and why it waits. Items that are done move to `implementation.md`.

## Performance

- **Whole runs in the kernels with deliveries.** A compartmental run with deliveries, modifiers with `when=` or a function as their amount, or transfers whose rate varies is stepped from Python, with one kernel call a step. A step costs 120 µs (with a `when=` modifier) to 300 µs (with a routine vaccine), against 5 µs for a step of a whole run; the time is spread over the Python loop, with no one large cost. A `when=` function of the sim cannot be compiled, but deliveries whose eligibility is on states and stratifications only (with coverage, and no capacity or test results) could be: as shifts of a fraction of each compartment and group to another group, worked out in the kernel, with the delivery's results (delivered, eligible, coverage) recorded there too. This would also let models with a care cascade (tbsim) be restarted and differentiated as a whole run, and so fitted by NUTS. It needs the shifts in the program, in the stepped kernels, and in JAX. Deferred by Cliff (2026-10-15).
- **Dropping the empty groups of a ragged stratification.** With `where=` on `stratify()`, a compartmental run keeps the full grid of groups: the people outside `where=` are counted at the first level, and the groups at the other levels stay empty but are still worked out (four age groups of women and one of men is eight groups, not five). Keeping only the groups that can hold people would need the group index to go through a lookup wherever it is now worked out by arithmetic (`group_mults`), in the solvers, kernels, movers, and fitting. Worth it only if a real model has many empty groups.
- **The rest of an agent-based step.** After this round, what is left in a large agent-based step is spread out: the infection pass, exposure by place, and finding the members of each state (one `flatnonzero` a state), each 1 to 3 ms a step for 668,000 agents. Finding the members of every state in one pass gains little, since each state is its own array.

## Accuracy

- **Thinning for time-varying rates under Gillespie.** Gillespie and tau-leaping hold a value that varies smoothly in time for a quarter of a step at a time (with a note), rather than thinning as the spec asks for. Thinning needs a bound on every rate over the step, which a rate written as any expression of the parameters does not give; a bound could be worked out by interval arithmetic over the kernels' operations, if exactness is needed.

## Features

- **Triangles as a target statistic.** `ss.PartnerNet` and `ss.MFNet` target the number of partners people have at a time and mixing by a property (`match=`). Triangles (two of a person's partners who are partners of each other), which an ERGM can also target, need the network fitted as a whole, not partner by partner.
- **WorldMove (synthetic movement within cities).** A collaborator suggested it ([Scientific Data, 2026](https://www.nature.com/articles/s41597-026-06555-2); https://github.com/tsinghua-fib-lab/WorldMove): one day of synthetic trajectories for each of about 1,600 cities in 179 countries, on a 1 km grid in half-hour slots, from a diffusion model, with commuting origin-destination matrices between the cells of each city. It is within cities, while `ss.geo` and `ss.gravity` are between administrative units, and a v4 run at a step of a day has no use for half-hourly paths. Its commuting matrices could give `ss.Mobility(matrix=...)` for a model of one city on a grid. It waits for a vignette that needs one; the repository has no license, and the paper is CC BY-NC-ND, so `starsim_data` could not redistribute it as things stand.
- **Gaps found while writing the importers and the import guides.** Each was worked around or listed as not translated; none is fixed yet:
    - A branch into `ss.null` (`ss.Transition(I, [R, ss.null], p=...)`) is not allowed, and the error message itself crashes (`closest()` is given the `Null` object). A v3 SIR's `p_death` has to be two transitions with `p=`.
    - `ss.PartnerNet` has no `acts=`, and there is no declared condition on a network (a person's degree, or being the partner of someone infected).
    - A transition's current flow cannot be used in a rate (MEmilio's test-and-trace capacity).
    - Deaths from a life table, and seeding a fixed number of people on a schedule, have no declared form.
    - `ss.Mobility` cannot scale gravity to a mean share away with a cap for each place (LASER).
    - The binomial solver substeps the whole model for one Erlang stage, so `substeps=1` is needed to match a source in discrete time.
    - `ss.gamma`'s second positional argument is `loc`, which nothing warns about.
    - `MultiResults` (from `sim.run(n_runs=)`) can be read by attribute (`msim.results.sir`), but `keys()` and `[name]` come back empty; and after `init()`, `sim.pars.diseases` is no longer the list of diseases.
    - The network of an `ss.contacts()` given as a matrix is written in the model spec as the mixing pools it makes, not as its matrices, so neither `ss.load()` nor `ss.to_code()` can rebuild it from the spec (an importer gives `to_code()` the code instead). A named one (`ss.contacts('Kenya')`) is written as where it came from, and is rebuilt.
    - A one-off move between states at given times (camdl's `transfer`) has no declared form; a product that protects people without moving them is the nearest.
    - `ss.PartnerNet` takes edges, concurrency, a cap on degree, and matching on one attribute; other ERGM terms (triangles, `nodematch(diff=TRUE)`, `nodefactor`) have no equivalent (EpiModel).
    - EpiModel lets people recover in the step they are infected, and counts after that, so its prevalence is 3 to 6% lower for the same epidemic; Starsim has no option for that order within a step.
    - An initial number below 1 is read as a fraction of the population, which a model with a fractional count (camdl's test models) cannot say otherwise.
    - Priors and likelihoods are not in the model spec, so `to_code()` lists them for adding by hand.
