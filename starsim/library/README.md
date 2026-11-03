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

About twenty diseases: outbreak diseases written as code (measles, Ebola, cholera), HIV with treatment and CD4 counts, a non-communicable disease, Starsim 3's SIR, SIS, and SEIR written as code, and declared models of malaria, TB, COVID, influenza, RSV, pertussis, dengue, typhoid, polio, syphilis, and gonorrhea. Each says in its docstring where its structure and its default parameters came from. The user guide page [Library: diseases](https://starsim.org/v4/user_guide/library_diseases.html) describes them, with their sources, and its appendix lists every one, generated from the classes: whether it is declared or written as code, its default parameters (flagging placeholders), and its results.

### Networks (`networks/`)

Households (`HouseholdNet`, `ClusterNet`, and Covasim's `HybridNet`), space (`DiskNet`, `GridNet`, and `GeoNet` for real places), and theoretical networks (`StaticNet` from any NetworkX graph, `SmallWorldNet`, `ScaleFreeNet`, `ErdosRenyiNet`, and `NullNet`). See [Library: networks](https://starsim.org/v4/user_guide/library_networks.html), whose appendix lists every one with its parameters and whether it keeps common random numbers.

### MNCH (`mnch/`)

Maternal, newborn, and child health examples, demonstrating patterns that are more involved than a standard disease module. See [`mnch/README.md`](mnch/README.md) for details.

| Class | Base | Description |
| --- | --- | --- |
| `CongenitalDisease` | `AgentSIR` | Congenital outcomes (stillbirth, congenital infection, normal), scheduled around delivery by its `set_congenital()` and `step_congenital()` |
| `NeonatalSepsis` | `AgentSIR` | Infects newborns at birth and kills some within days, exercising the Pregnancy module's passive neonatal death detection |
| `FetalHealth` | `ss.Module` | Tracks fetal growth and birth weight outcomes (LBW, VLBW, SGA); disease-agnostic, modified by other modules via its methods |
| `fetal_infection` | `ss.Connector` | Links a disease to fetal health outcomes: preterm risk via delivery timing shifts, low birth weight via growth restriction |
| `treat_pregnant` | `ss.Intervention` | Treats infected pregnant women and partially reverses accumulated fetal damage |

### Economics (`econ/`)

Health economics, which most sims don't need, so it is here rather than in core Starsim. See the user guide page on [Costs](https://docs.starsim.org/user_guide/interventions_costs.html).

| Name | Base | Description |
| --- | --- | --- |
| `Costs` | `ss.Analyzer` | Adds up the costs of a sim as results: the `cost` of each delivery, a cost for each event of any other result (e.g. care for each infection), and a fixed cost per year, with their `total` |
| `discount()` | | Discounts a result (or any values over time) at a yearly rate from a start year |
| `DALYs` | `ss.Analyzer` | Adds up disability-adjusted life years as results: years lived with disability, and years of life lost |
| `ICER` | `ss.Analyzer` | Compares the costs and effects of scenarios, after they have run or as a sim runs, with incremental cost-effectiveness ratios, plots, and the best scenario at a willingness to pay |

## Documentation

- User guide: [Library: diseases](https://starsim.org/v4/user_guide/library_diseases.html), [Library: networks](https://starsim.org/v4/user_guide/library_networks.html), and [Library: maternal-child health](https://starsim.org/v4/user_guide/library_mnch.html)
- API reference: [starsim.library](https://docs.starsim.org/api/)

## Contributing

New library modules are welcome, especially ones that demonstrate a modeling pattern that is otherwise hard to discover. Guidelines:

1. Put the module in the most specific existing subfolder, or create a new one with an `__init__.py` that has a module docstring.
2. Define `__all__` in each file, and re-export its contents explicitly from the subpackage `__init__.py` (no star imports, so that the public API is unambiguous); `library/__init__.py` and the API docs then pick it up automatically.
3. Follow the [Starsim style guide](https://github.com/starsimhub/styleguide).
4. Document each class with a short description, an `Args:` section, an `Attributes:` section for its states, and a runnable `Examples:` block.
5. Add a test to `tests/test_library.py`. The appendices of the user guide pages are generated from the subpackages, so a new disease or network appears there by itself; add it to the descriptions on the page, and to the tables above for MNCH and economics.
