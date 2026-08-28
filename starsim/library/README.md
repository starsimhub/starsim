# Starsim library

The Starsim library is a collection of example and reference modules that build on the core Starsim package but are not part of it. Core Starsim (`import starsim as ss`) provides the framework and generic building blocks — `ss.SIR`, `ss.RandomNet`, `ss.Pregnancy`, and so on. The library provides concrete, opinionated modules built on top of them: specific diseases, specialized networks, and worked examples of harder modeling patterns.

Library modules are held to a lower bar than core Starsim. They are illustrative rather than validated: parameter values are indicative, some are placeholders, and the APIs may change between versions. Use them as starting points to copy and adapt, not as off-the-shelf models for a specific setting.

## Usage

```python
import starsim as ss
import starsim.library as ssl

sim = ss.Sim(diseases=ssl.Measles(), networks=ssl.DiskNet())
sim.run()
sim.plot()
```

Classes can be accessed either at the top level (`ssl.Measles`) or via their submodule (`ssl.diseases.Measles`); the two are identical. The submodule form is more explicit and is preferred when it aids readability.

## Contents

### Diseases (`diseases/`)

| Class | Base | Description |
| --- | --- | --- |
| `Cholera` | `ss.Infection` | SEIR-type cholera with symptomatic/asymptomatic infection plus indirect transmission through a decaying environmental (waterborne) reservoir |
| `Ebola` | `ss.SIR` | Ebola with severe disease and continued transmission from unburied bodies |
| `HIV` | `ss.Infection` | HIV with CD4 count dynamics, CD4-dependent mortality, and vertical transmission |
| `ART` | `ss.Intervention` | Scales up antiretroviral therapy over time; requires `HIV` |
| `CD4_analyzer` | `ss.Analyzer` | Records every agent's CD4 count at every timestep |
| `Measles` | `ss.SIR` | Measles as an SEIR model, with CDC natural history parameters |

### Networks (`networks/`)

| Class | Base | Description |
| --- | --- | --- |
| `HouseholdNet` | `ss.Network` | Households built from DHS-style survey data, optionally evolving over time as women move out and give birth |
| `DiskNet` | `ss.Network` | Agents move within a unit square and connect to others within a given radius |
| `ErdosRenyiNet` | `ss.DynamicNetwork` | Every possible edge is created with probability `p` each timestep |
| `NullNet` | `ss.Network` | Self-connections only, with zero transmission; useful as a placeholder or for debugging |

### MNCH (`mnch/`)

Maternal, newborn, and child health examples, demonstrating patterns that are more involved than a standard disease module. See [`mnch/README.md`](mnch/README.md) for details.

| Class | Base | Description |
| --- | --- | --- |
| `CongenitalDisease` | `ss.SIR` | Congenital outcomes (stillbirth, congenital infection, normal) via the generic `set_congenital()`/`step_congenital()` framework |
| `NeonatalSepsis` | `ss.SIR` | Infects newborns at birth and kills some within days, exercising the Pregnancy module's passive neonatal death detection |
| `FetalHealth` | `ss.Module` | Tracks fetal growth and birth weight outcomes (LBW, VLBW, SGA); disease-agnostic, modified by other modules via callbacks |
| `fetal_infection` | `ss.Connector` | Links a disease to fetal health outcomes: preterm risk via delivery timing shifts, low birth weight via growth restriction |
| `treat_pregnant` | `ss.Intervention` | Treats infected pregnant women and partially reverses accumulated fetal damage |

## Documentation

- User guide: [Library modules](https://docs.starsim.org/user_guide/modules_library.html)
- API reference: [starsim.library](https://docs.starsim.org/api/)

## Contributing

New library modules are welcome, especially ones that demonstrate a modeling pattern that is otherwise hard to discover. Guidelines:

1. Put the module in the most specific existing subfolder, or create a new one with an `__init__.py` that has a module docstring.
2. Define `__all__` in each file, and re-export its contents explicitly from the subpackage `__init__.py` and from `library/__init__.py` — no star imports, so that the public API is unambiguous.
3. Follow the [Starsim style guide](https://github.com/starsimhub/styleguide).
4. Document each class with a short description, a `Pars:` section, a `States:` section, and a runnable `Examples:` block.
5. Add a test to `tests/test_library.py`, and a row to the tables above.
6. Add the class to the relevant `library.*` page in the `quartodoc` section of `docs/_quarto.yml` so it appears in the API reference. (Members are listed explicitly there, since quartodoc's static analysis doesn't follow re-exports.)
