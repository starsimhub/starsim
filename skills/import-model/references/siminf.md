# SimInf

## What the source is

A SimInf model is an R script that calls `mparse()` (transitions as text, `"S -> beta*S*I/(S+I+R) -> I"`, with `compartments`, `gdata`, `ldata`, `u0`, `v0`, `tspan`, `events`, `E`, `N`, and maybe `pts_fun` in C) or one of SimInf's predefined models (`SIR`, `SIS`, `SEIR`, `SISe`, `SISe3`). Each row of `u0` is a node (a herd), and scheduled events (`enter`, `exit`, `intTrans`, `extTrans`) move animals in, out, and between nodes. It runs in continuous time (Gillespie) within each node, with the events applied each day.

## Mapping

The import reads the R script without running R: literal numbers, strings, `c()`, `seq()`, `rep()`, `a:b`, `data.frame()`, `list()`, `nrow()`, arithmetic, earlier variables, and assignments such as `u0$I[1] <- 1`.

| SimInf | Starsim | Mechanical |
|---|---|---|
| A transition `"S -> rate -> I"` | A transition, the rate a total flow, classified (`beta*S*I/N` is exposure) | yes |
| `@` | `ss.null` (births and deaths) | yes |
| `N <- S + I + R` | A named expression | yes |
| `pow()`, `fmin()`, `fmax()`, `(int)` | The same in the rate | yes |
| Stoichiometry with net change one person moving (`S + I -> ... -> 2*I`) | `S -> I` | yes |
| Other stoichiometry | Listed | no |
| Rows of `u0` | A stratification `node` (`kind='space'`), split by node size; initial numbers by node as `ss.by(node=...)` | yes |
| `gdata`; `ldata` that differs by node | Parameters; `ss.par(ss.by(node=...))` | yes |
| `enter`, `exit`, `intTrans`, `extTrans` events, with `E` and `N` | `ss.Mobility(data=register)` (`to_states` for `intTrans`; entering animals go to the first compartment selected) | yes |
| `SISe`, `SISe3` (`phi` in the environment, by quarter) | `phi` as `ss.State(people=False)`, with a seasonal decay `phi >> ss.null` and shedding `ss.null >> phi` from `I` | yes |
| `tspan` | `solver='gillespie'`, in days, from `tspan[0] - 1` to `tspan[-1]` | yes |
| `u0`, `events` computed in R (`u0_SIR()`, `events_SIR()`, `read.csv()`, loops) | Listed as "computed in R"; pass them in: `ss.load('model.R', u0='u0.csv', events='events.csv.gz')` | no |
| `pts_fun` (C code updating `v`) | Transitions of a state that is not people (below) | no |
| Events with `proportion` and `n = 0` | Numbers, or an intervention | no |

## What needs care

- **Data computed in R.** Have R write it after sourcing the script: `write.csv(cbind(node = seq_len(ncol(model@u0)), as.data.frame(t(model@u0))), "u0.csv", row.names = FALSE)` and `write.csv(as.data.frame(events(model)), gzfile("events.csv.gz"), row.names = FALSE)`. The script's own assignments to `u0` are applied to what is passed. Everything can also be given as a dict: `ss.interop.siminf.load_siminf(dict(transitions=[...], compartments=[...], u0=df, gdata=dict(...), tspan=..., events=df, E=..., N=...))`.
- **A daily update in `pts_fun`**, `v' = v*(1 - b) + g`, is a decay at rate `-log(1 - b)` plus an inflow at `g*(-log(1 - b))/b`, which give the same `v` at the end of each day:

```python
phi = ss.State(people=False)
decay = phi >> ss.null @ ss.perday(-np.log(1 - b))
shed  = ss.null >> phi @ ss.perday(alpha*(-np.log(1 - b))/b) * I.n/(S + I).n
infection = S >> I @ upsilon * phi.n # phi.n, not phi: phi is a count, not people (E211)
```

- **Compartments joined only by events** (age categories joined by `intTrans`) are declared as people by the import. By hand, write `ss.State(people=True)`, or v4 treats them as counts.
- **Outbreaks die out early in some runs** (about 7% in the herds example). Compare with 100 runs or more, or compare the share that died out separately from the mean of the rest.

## Worked example

`lingua_franca_design/examples/siminf/sir_herds` (SIR in 1,600 herds with births, deaths, and movements) and `examples/siminf/sise3` (SISe3, three age categories with the environment). With `u0` and the events written by R, both import with nothing listed. Over 96 runs of the SIR, the mean of every compartment and of the number of infected herds is within about 1.4 standard errors of SimInf's at days 100, 365, 700, 1,000, and 1,450 (`I` on day 700: 47,303 ± 1,888 against 49,853); over 24 runs of SISe3, `I` in each age category is within about 1 standard error (all `I` on day 1,450: 3,273 ± 537 against 3,692).

```python
sim = ss.load('events_SIR.R', u0='reference/u0.csv', events='reference/events.csv.gz')
```

## Running the original

In SimInf's environment (`lingua_franca_design/_envs/siminf`; `bin/Rscript` sets `R_LIBS_USER`): source the script (with `pdf(NULL)` first, since the examples plot), run `run(model)` n times, and write `trajectory(result, format = "matrix")` (one row per node and compartment, one column per `tspan` time), summed over nodes, as one row per run and time. `examples/siminf/sir_herds/run.py` is a complete driver, which also writes `u0.csv` and `events.csv.gz`. SimInf's record at `tspan` time t matches Starsim's `n_<state>[t]`, with no shift. Compare statistically.
