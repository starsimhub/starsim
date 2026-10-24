# Stochastic compartmental

Gillespie (exact, in continuous time, with tau-leaping for a run of too many events) and chain-binomial (in discrete time). One model, several ways of being random.

## Recommendation

1. **Stochasticity is a run-time choice with several named solvers**, not a property of the model.
2. **Prefer continuous time where the model allows it**, because it eliminates rate-to-probability conversion entirely.
3. **Name the approximation.** Gillespie, tau-leaping, and chain-binomial are different models with different validity conditions, and the result should say which one produced it.
4. **One sim is one trajectory; the distribution comes from many runs**, and running many is one call.
5. **Quantities that are not people are states too.** An environmental reservoir or a vector population is a state holding a count, with its own transitions, and disease rates may use that count directly.
6. **Extinction is the reason this paradigm exists**; make it visible (`msim.extinction()` says how often each disease died out over many runs).

## Why

### One model, several stochastic methods, already demonstrated

[camdl](../approaches/camdl.md) runs the same model text on Gillespie SSA (exact continuous-time), chain-binomial (Euler-multinomial discrete-time), and ODE. [epipack](../approaches/epipack.md) runs the same process tuples deterministically and stochastically, including "a correct inhomogeneous-Poisson Gillespie implementation" for time-varying rates — "a real capability that most frameworks skip or approximate". [odin](../approaches/odin.md) shares `initial()` between `deriv()` and `update()`.

[PyRoss](../approaches/notes.md#pyross) is the sharpest evidence for the *ladder* rather than the binary: it implements Gillespie, tau-leaping, and Gaussian/van Kampen approximations of the same model side by side. [MEmilio](../approaches/memilio.md) has a whole `sde_*` family alongside its `ode_*` family, separately implemented.

### Continuous time removes an error class

[SimInf](../approaches/siminf.md) is continuous-time Gillespie throughout, and its review makes the point directly: "Being continuous-time is a real advantage for correctness — the exponential-conversion errors that afflict discrete-time frameworks simply do not arise — and a real limitation for scale, since Gillespie cost grows with event count." [EoN](../approaches/eon.md) is continuous-time for the same reason.

The implication drawn in the SimInf review: "The lingua franca should treat discrete-time versus continuous-time as a backend choice with a declared cost, not as a property of the model."

### Disease state coupled to an environment

SimInf's `u`/`v` split (integer compartment counts governed by the Markov chain, alongside continuous per-node variables updated by a post-timestep function) is used by its `SISe` family for a SIS disease coupled to an environmental compartment. Its review: "Environmental reservoirs, accumulated exposure, and vector abundance are all naturally continuous and coupled to a discrete disease process." The need is real. The lingua franca meets it without a second kind of state: the reservoir is an ordinary state that holds a count instead of people, and rates are written in terms of that count.

### Each stochastic solver has conditions under which it is valid

Gillespie's method is exact, and its cost grows with the number of events. Tau-leaping is valid when the rates are roughly constant over a leap. A chain-binomial run is valid when `rate × dt` is small. The diffusion approximation is "valid only at moderate-to-large populations" ([MEmilio](../approaches/memilio.md)), so it fails exactly where stochasticity was the reason to use it; it is not offered as a solver. An ODE is valid when counts are large and extinction is not the question. Because the model is data, these conditions can be checked rather than documented, and the run says when one is in doubt. The exact method is called `'gillespie'` because that is the name the field uses for it (EoN, epipack, SimInf, camdl, and PyRoss all say Gillespie).

## The specification

[12-backends.md](../design/12-backends.md) specifies the solvers (`'auto'`, `'ode'`, `'binomial'`, `'gillespie'`, and `'abm'`), how `'gillespie'` switches to tau-leaping for a run of too many events (with a note; `approx=False` keeps it exact), how a value that varies in time is followed within a step, Erlang stages for dwell times, and the notes on small counts, possible extinction, and large steps. States that hold a count (a reservoir, a vector population) are in [03-states-and-transitions.md](../design/03-states-and-transitions.md), and many runs, with `msim.extinction()`, in [09-results.md](../design/09-results.md).

## Trade-offs

- **Exact Gillespie simulation does not scale**, and the crossover to tau-leaping is population- and rate-dependent. Automatic selection with a printed reason is better than a documentation footnote, and worse than the user understanding it. Both should be possible.
- **Automatic method selection can be actively misleading** if the user is asking an extinction question and gets tau-leaping. `solver='gillespie'` switches only when an ODE run gives more than about a million events, which only large populations reach (extinction questions arise when counts are small), and it says so; `approx=False` keeps the exact method however long it takes.
- **A reservoir with very large counts would make exact Gillespie simulation expensive.** Bacteria in water are shed and decay in numbers far larger than the number of infections, and simulating each one as an event is wasted effort. A state that is not people is advanced as its expected value by the stochastic solvers, between the events of the people, and its flows are not counted as events when `solver='gillespie'` decides whether to take tau leaps.
- **Count-based rates are literal.** `beta * W.n / (W.n + K)` is used exactly as written, so the library cannot check that it is a sensible force of infection or adjust it for mixing. That is the intent, and it means the user owns the formula.
- **Continuous-time methods make "results at time t" a resampling question.** [MEmilio](../approaches/memilio.md)'s `interpolate_simulation_result()` and SimInf's `tspan` both handle this and most frameworks leave it to the user. Here every solver reports its results at the sim's time points.

## Rejected

- **Stochasticity as a model property** (MEmilio's `sde_*` classes, EpiModel's `icm` versus `dcm`). It is a choice of solver.
- **A single stochastic solver** (SimInf: Gillespie only; Epydemix: chain-binomial only). Each is right for a different population size.
- **Substituting an approximation without saying so.** If tau-leaping was used, the result says so.
- **A separate kind of continuous state for environmental quantities** (SimInf's `v`). Ordinary states holding counts, with count-based rates, cover the same cases with no new concept.
- **A stochastic sim that is many trajectories.** A sim is one run; `ss.parallel(sim, n_runs=500)` is many.
- **The diffusion approximation as the default stochastic solver.** It fails at small counts, which is where stochasticity matters. (The `'sde'` solver has since been removed altogether.)
- **Requiring the user to choose.** Choose, print, allow override — the same policy as [calibration](calibration.md).

## Open questions

The open questions from this document are in [19-open-questions.md](../design/19-open-questions.md).
