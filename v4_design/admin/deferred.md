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
