# SBML

## What the source is

An SBML model is an XML file (from BioModels, COPASI, libRoadRunner, and many others): species (with initial amounts or concentrations), compartments, parameters (global, or local to a reaction), reactions (reactants, products, stoichiometries, and a kinetic law in MathML), rules (assignment, rate, algebraic), function definitions, initial assignments, and events. SBML rarely says its time unit.

## Mapping

| SBML | Starsim | Mechanical |
|---|---|---|
| A species | A state (`initialAmount`, or `initialConcentration` × the compartment's size) | yes |
| A species with `boundaryCondition` or `constant` true | A parameter | yes |
| Global, compartment, and local parameters | Parameters (a local one renamed `<reaction>_<id>` if it clashes) | yes |
| A reaction with net change one species down and one up | A flow; the kinetic law is the total, classified (`S + I -> 2 I` is `S -> I`) | yes |
| A reaction with only a product or only a reactant | An inflow or an outflow | yes |
| An assignment rule | A named expression (`N = S + I + R` makes `beta * S * I / N` exposure) | yes |
| A function definition | Written out where it is called | yes |
| MathML arithmetic, power, exp, ln, log, root, abs, min, max, comparisons | The same in the rate | yes |
| An event at a time | `ss.timeseries` as a parameter's value, or `ss.modify(start=...)` | no |
| An event on a condition | A rate multiplied by a comparison (`beta * (I.n < K)`), or an intervention with `where=` | no |
| A rate rule for a parameter | A state that is not people, `ss.State(people=False)`, with flows against `ss.null` | no |
| An algebraic rule | By hand, solved for the variable | no |
| Other stoichiometries (`2 R -> ...`) | Split into flows, or a literal rate with counts | no |
| `piecewise`, `csymbol time`, `delay` | A product of comparisons, an `ss.timeseries`, or a fixed dwell | no |

## What needs care

- Give `unit=` (default `'day'`) and `stop=`, and check them in `describe()`.
- A reversible reaction is translated as its net flow forward (and listed); split it into two transitions if the net flow can be negative.
- A law written in concentrations, in a compartment whose size is not 1, is not a number of people per unit of time until multiplied by the size (listed).
- Many biochemical SBML models are not epidemic models at all; the rules for reading rates as exposure assume that species are people.

## Worked example

There is no SBML vignette yet. A minimal SIR in SBML (species `S`, `I`, `R`; parameters `beta`, `gamma`; an assignment rule `N = S + I + R`; reactions `S + I -> 2 I` with law `beta * S * I / N` and `I -> R` with `gamma * I`) imports as:

```python
infection = S >> I @ beta * I     # exposure: the law over N, with S taken out
recovery  = I >> R @ gamma
```

which equals a hand-written model to 1e-6 (`tests/test_io_odin_sbml.py`).

## Running the original

There is no SBML environment in `lingua_franca_design/_envs`. A deterministic model runs with libRoadRunner (`pip install libroadrunner`, in its own environment: `rr = roadrunner.RoadRunner(path); out = rr.simulate(0, stop, stop + 1)`, with columns `time` and `[S]`, ...) or COPASI's `basico`. Compare each species with `res.n_<species>` at the same times to a relative tolerance of about 1e-6 (both are ODEs). A stochastic run (`rr.gillespie(...)`) compares as distributions with Starsim's `solver='gillespie'`.
