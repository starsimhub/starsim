# MetaCast

|  |  |
|---|---|
| **Language** | Python |
| **Paradigms** | Metapopulation ODE — a user-supplied subpopulation model broadcast across arbitrary dimensions |
| **Specification style** | A Python function defining the subpopulation model, plus a declarative dimension specification |
| **Version reviewed** | 0.x |
| **Licence** | (see repository) |
| **Code** | <https://github.com/m-d-grunnill/MetaCast> |
| **Docs** | <https://metacast.readthedocs.io/> |
| **Paper** | Grunnill et al. (2024), *JOSS* 9(99): 6851, [10.21105/joss.06851](https://doi.org/10.21105/joss.06851) |

## What it's for

MetaCast takes one subpopulation model and "broadcasts" it across a metapopulation whose dimensions the user declares. Its premise is that stratification, metapopulation structure, and any other partition of the population are **the same operation** — a dimension the model is replicated over — and that the model author should write the within-group dynamics once.

It is a small package, and its contribution to this review is conceptual: it treats structure as a **dimension algebra** rather than as model content.

## How a model is specified

The subpopulation model is a function with a fixed signature:

```python
def subpop_model(y, y_deltas, parameters, states_index, subpop_suffix, foi):
    infections               = foi * y[states_index['S']]
    progression_from_exposed = parameters['sigma'] * y[states_index['E']]
    p_hosp                   = parameters['p' + subpop_suffix]   # subpopulation-specific
    progression_from_inf     = y[states_index['I']] * parameters['gamma']
    recovery                 = progression_from_inf * (1 - p_hosp)
    hospitalisation          = progression_from_inf * p_hosp
    hospital_recovery        = y[states_index['H']] * parameters['eta']

    y_deltas[states_index['S']] += -infections
    y_deltas[states_index['E']] += infections - progression_from_exposed
    y_deltas[states_index['I']] += progression_from_exposed - progression_from_inf
    y_deltas[states_index['H']] += hospitalisation - hospital_recovery
    y_deltas[states_index['R']] += recovery + hospital_recovery
    return y_deltas
```

Then the dimensions:

```python
metapop = MetaCaster(
    dimensions        = [['high', 'low'], ['unvaccinated', 'vaccination_lag', 'vaccinated']],
    subpop_model      = subpop_model,
    states            = ['S', 'E', 'I', 'H', 'R'],
    infected_states   = ['E', 'I', 'H'],
    infectious_states = ['I'],
    subpop_params     = {...},
    universal_params  = [...],
)
```

`dimensions` accepts an int, a list of labels, a list of ints, a list of label lists (multidimensional), or a list of **transfer dictionaries** describing flows between subpopulations:

```python
{
  "from_coordinates": ["high", "unvaccinated"],
  "to_coordinates":   ["high", "vaccination_lag"],
  "states":           "all",
  "parameter":        "nu_unvaccinated",
}
```

So "vaccination" is not a process in the model; it is a **transfer between coordinates of the vaccination dimension**, declared in the dimension specification.

## Core abstractions

| Concept | MetaCast name | Notes |
|---|---|---|
| Subpopulation model | a function `(y, y_deltas, parameters, states_index, ...)` | Written once, run per coordinate |
| Dimension | entry in `dimensions` | Ints, labels, nested lists, or transfer dicts |
| Coordinate | a tuple across dimensions, e.g. `[high, vaccinated]` | Suffixes parameter names: `beta_[high,vaccinated]` |
| Transfer | a transfer dictionary | Movement of hosts between coordinates, for named states, at a named rate |
| Force of infection | computed by the framework | From `transmission_term_prefix` (`beta`), `subpop_interaction_prefix` (`rho`), `population_term_prefix` (`N`), and `foi_population_focus` |
| Parameter class | `universal_params` vs `subpop_params` | Explicitly separates "same everywhere" from "varies by dimension" |
| Observed state | `observed_states` | Accumulators appended to the state vector |
| Event | `event_handling.event_queue` | Discrete events: moving populations, changing parameters |
| Sensitivity | LHS + PRCC, parallelised via dask | Built in |

## The force-of-infection convention

MetaCast's most opinionated design choice is that it **computes the force of infection for you** from a naming convention. Given prefixes `beta`, `rho`, and `N`, and a `foi_population_focus` setting (`None`, `'i'`, or `'j'`), it assembles

λᵢ = Σⱼ ρᵢⱼ · βⱼ · Iⱼ / Nⱼ

with the choice of denominator controlled by `foi_population_focus`. The subpopulation model receives `foi` as an argument and never computes it.

This is the right instinct — force of infection between groups is exactly the part that gets written wrong, and factoring it out of the model is correct — implemented in the most fragile possible way, through string concatenation of parameter names (`'beta_[high,vaccinated]'`). The lesson is in the split, not the mechanism.

Also worth noting: `asymptomatic_transmission_modifier` applies automatically to any state in `infectious_states` but not in `symptomatic_states`. That is real epidemiological semantics encoded in a set relationship, which is the kind of thing a language should be able to say.

## Strengths

- **Structure as a dimension algebra.** Age, space, risk, and vaccination status are all just dimensions; the model author writes within-group dynamics once. Conceptually the cleanest statement of the stratification problem in the review.
- **Transfers between coordinates as a declarative form.** `{from_coordinates, to_coordinates, states, parameter}` expresses vaccination, ageing, and migration in one vocabulary — and expresses them as *dimension movement*, not as model transitions.
- **Force of infection factored out** of the subpopulation model and computed by the framework, with an explicit `foi_population_focus` choice for the denominator convention.
- **Parameters explicitly classified** as universal or subpopulation-specific.
- **Epidemiological set semantics**: `infected_states`, `infectious_states`, `symptomatic_states` as distinct declared sets, with the asymptomatic transmission modifier following from the difference.
- **Discrete event queue** for scheduled population movements and parameter changes.
- **Sensitivity analysis built in** (LHS + PRCC, parallelised).

## Limitations

- **The subpopulation model is imperative NumPy index arithmetic.** `y_deltas[states_index['S']] += -infections` is not a specification; it is code, and it is code that indexes into a shared array. There is no transition object, nothing to introspect, and easy scope for sign errors.
- **Positional magic in the state vector.** `y_deltas[-2]` and `y_deltas[-1]` are the observed-state accumulators, addressed by negative index with a comment explaining what they are.
- **String-concatenated parameter names.** `parameters['p' + subpop_suffix]`, `'beta_[high,vaccinated]'`. Typos are runtime `KeyError`s, and nothing checks that the set of expected names was supplied.
- **ODE only.** No stochastic or agent-based execution.
- **Small and lightly adopted.** One demonstration notebook; a JOSS paper; limited independent use.
- **No dimensional types**, no calendar dates, no observation model, no inference beyond sensitivity analysis.
- **The force-of-infection convention is inflexible.** It handles the frequency-dependent multi-group case it was designed for; anything else means bypassing it.

## Implications for the lingua franca

1. **Take the framing: stratification, metapopulation, and multi-group structure are one operation.** MetaCast, summer2 (`stratify_with`), camdl (`dimensions` + `stratify`), and epymorph (`MultiStrataRUME`) are four attempts at the same thing, and MetaCast states the general case most plainly. The lingua franca should have **one dimension concept** covering age, space, risk, strain, and vaccination status — with, as summer2 and camdl both show, a way to mark which dimensions carry extra structure (ageing flows, movement, residence times).
2. **Transfers between coordinates deserve to be a declared form.** Vaccination as `{from: [high, unvaccinated], to: [high, vaccination_lag], states: all, rate: nu}` is a genuinely different and useful way of saying it: the individual does not change disease state, they change *dimension coordinate*. camdl's `transfer(from = S, to = V)` is close, but MetaCast's version generalises across all states at once and across arbitrary dimensions. This is a strong candidate primitive for the intervention vocabulary.
3. **Force of infection should be framework-computed and explicitly configured.** Factoring λ out of the model is right. Doing it through string-concatenated parameter names is not. The lingua franca needs an explicit, checked form: which states are infectious, which mixing structure applies, which denominator convention (`None` / `i` / `j`) is in force — with the convention named rather than assumed, because the three choices are genuinely different models and the difference is invisible in most frameworks.
4. **Declare the epidemiological state sets.** `infected_states`, `infectious_states`, `symptomatic_states` as separate declared sets, with behaviour following from their differences, is a small idea worth keeping. It is how a framework knows that "infected" and "infectious" are not the same claim — a distinction Starsim also makes (`infectious` defaults to an alias for `infected` and can differ) and most others elide.
5. **This is the strongest argument against imperative subpopulation models.** MetaCast has the right decomposition and the wrong medium: the dimension algebra is declarative and checkable, while the model it broadcasts is index arithmetic. Put camdl's or epipack's transition objects inside MetaCast's dimension algebra and you have most of what this project needs.
