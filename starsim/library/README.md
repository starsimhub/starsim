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

Each disease says in its docstring where its structure and its default parameters came from.

| Class | Base | Description | Source of the model |
| --- | --- | --- | --- |
| `Cholera` | `AgentSEIR` | SEIR-type cholera with symptomatic/asymptomatic infection plus indirect transmission through a decaying environmental (waterborne) reservoir | |
| `AgentSIR`, `AgentSEIR`, `AgentSIS` | `ss.Infection` | Starsim 3's SIR, SEIR, and SIS (with waning immunity), written as code: examples of a disease written as code, and base classes for the diseases here (the core `ss.SIR`, `ss.SEIR`, and `ss.SIS` are declared) | |
| `COVID` | `ss.Disease` | COVID-19 as a SECIR model: a chain of increasingly severe states, some of whom die | The simple SECIR example of MEmilio (Kühn et al. 2021) |
| `Dengue` | `ss.Disease` | Several dengue serotypes (`ss.Strains`), each with mosquitoes of its own, with partial cross-protection between them | The Ross-Macdonald structure of Keeling and Rohani (2008) 4.4, with indicative dengue values |
| `Ebola` | `AgentSEIR` | Ebola with severe disease and continued transmission from unburied bodies | |
| `Gonorrhea` | `ss.Disease` | Gonorrhea as an SIS model on a sexual network | STIsim's gonorrhea module |
| `HIV` | `ss.Infection` | HIV with CD4 count dynamics, CD4-dependent mortality, and vertical transmission | |
| `ART` | `ss.Intervention` | Scales up antiretroviral therapy over time; requires `HIV` | |
| `CD4_analyzer` | `ss.Analyzer` | Records every agent's CD4 count at every timestep | |
| `Influenza` | `ss.Disease` | Seasonal influenza as a SEIRS model with a yearly cycle in transmission | Carrat et al. (2008) and Biggerstaff et al. (2014) |
| `Malaria` | `ss.Disease` | Malaria as humans and mosquitoes, the mosquitoes as counts | The Ross-Macdonald model, as Keeling and Rohani (2008), Program 4.4 |
| `MalariaImmunity` | `ss.Disease` | Malaria with immunity: saturating inoculation, superinfection, and a fast recovery rate once immune | The Garki model: Dietz, Molineaux and Thomas (1974); Molineaux and Gramiccia (1980) |
| `Measles` | `AgentSEIR` | Measles as an SEIR model, with CDC natural history parameters | |
| `NCD` | `ss.Disease` | A non-communicable disease: a risk factor, the condition, and death from it; not transmitted | |
| `Pertussis` | `ss.Disease` | Pertussis as a SEIRS model with slowly waning immunity | Wearing and Rohani (2009); the structure of Keeling and Rohani (2008) |
| `Polio` | `ss.Disease` | Poliovirus with a paralytic fraction of about 1 in 200 infections | US CDC Pink Book; Fine and Carneiro (1999) |
| `RSV` | `ss.Disease` | RSV as a seasonal MSEIRS model, with babies born protected by maternal antibodies | Weber et al. (2001) |
| `Syphilis` | `ss.Disease` | Staged syphilis, with the infectiousness of latent syphilis written as code | STIsim's syphilis module |
| `TB` | `ss.Disease` | TB across the spectrum of disease, with reinfection | The LSHTM model (Schwalb et al. 2025), as TBsim's reference ODE |
| `Typhoid` | `ss.Disease` | Typhoid with chronic carriers who shed for life | Pitzer et al. (2014) |

### Networks (`networks/`)

| Class | Base | Description |
| --- | --- | --- |
| `HouseholdNet` | `ss.Network` | Households built from DHS-style survey data, optionally evolving over time as women move out and give birth |
| `ClusterNet` | `ss.Network` | Fully connected clusters, such as households |
| `HybridNet` | `ss.Network` | Covasim's households, schools, workplaces, and community |
| `DiskNet` | `ss.Network` | Agents move within a unit square and connect to others within a given radius |
| `GeoNet` | `ss.RandomNet` | Agents in real places, with most contacts near home and the rest in places that are large and near |
| `GridNet` | `StaticNet` | Agents on a square grid, in contact with their neighbors |
| `SmallWorldNet` | `StaticNet` | A Watts-Strogatz graph: mostly local contacts, with a few long-range ones |
| `ScaleFreeNet` | `StaticNet` | A Barabási-Albert graph: a few agents have many contacts |
| `ErdosRenyiNet` | `ss.DynamicNetwork` | Every possible edge is created with probability `p` each timestep |
| `NullNet` | `ss.Network` | Self-connections only, with zero transmission; useful as a placeholder or for debugging |
| `StaticNet` | `ss.Network` | A fixed network from a NetworkX graph, or a function that makes one |

### MNCH (`mnch/`)

Maternal, newborn, and child health examples, demonstrating patterns that are more involved than a standard disease module. See [`mnch/README.md`](mnch/README.md) for details.

| Class | Base | Description |
| --- | --- | --- |
| `CongenitalDisease` | `AgentSIR` | Congenital outcomes (stillbirth, congenital infection, normal), scheduled around delivery by its `set_congenital()` and `step_congenital()` |
| `NeonatalSepsis` | `AgentSIR` | Infects newborns at birth and kills some within days, exercising the Pregnancy module's passive neonatal death detection |
| `FetalHealth` | `ss.Module` | Tracks fetal growth and birth weight outcomes (LBW, VLBW, SGA); disease-agnostic, modified by other modules via callbacks |
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

- User guide: [Library modules](https://docs.starsim.org/user_guide/io_library.html)
- API reference: [starsim.library](https://docs.starsim.org/api/)

## Contributing

New library modules are welcome, especially ones that demonstrate a modeling pattern that is otherwise hard to discover. Guidelines:

1. Put the module in the most specific existing subfolder, or create a new one with an `__init__.py` that has a module docstring.
2. Define `__all__` in each file, and re-export its contents explicitly from the subpackage `__init__.py` and from `library/__init__.py` — no star imports, so that the public API is unambiguous.
3. Follow the [Starsim style guide](https://github.com/starsimhub/styleguide).
4. Document each class with a short description, a `Pars:` section, a `States:` section, and a runnable `Examples:` block.
5. Add a test to `tests/test_library.py`, and a row to the tables above.
6. Add the class to the relevant `library.*` page in the `quartodoc` section of `docs/_quarto.yml` so it appears in the API reference. (Members are listed explicitly there, since quartodoc's static analysis doesn't follow re-exports.)
