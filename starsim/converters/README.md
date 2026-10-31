# Importing and exporting

Reading models from other frameworks into Starsim, writing Starsim models out to them, and rewriting Starsim 3 code. See the user guide's [Importing and exporting models](https://docs.starsim.org/user_guide/io_import.html).

```python
sim = ss.import_from('model.yaml')                       # The format is told from the file where it can be
ss.export_to(sim, 'sbml', filename='model.xml')
sim = ss.converters.convert('model.yaml', filename='model.py')   # Import, and save the Starsim code it wrote
```

## Importing

Each importer reads its framework's model into an `ss.converters.Source`, and `core.py` writes that as Starsim code by the same rules for every format, then runs the code to make the sim (the code is kept in `sim.imported.code`). `ss.converters.importers` lists the formats:

| Framework | File |
|---|---|
| Atomica | `atomica.py` |
| camdl | `camdl.py` |
| EMULSION | `emulsion.py` |
| EpiModel | `epimodel.py`, with `dump_epimodel.R` |
| epipack | `epipack.py` |
| epydemix | `epydemix.py` |
| epymorph | `epymorph.py` |
| flepiMoP | `flepimop.py` |
| MEmilio | `memilio.py` |
| odin | `odin.py` |
| SBML | `sbml.py` |
| SimInf | `siminf.py` |
| summer2 | `summer2.py` |

Some frameworks can't be installed alongside Starsim, so their models are first written as JSON in the framework's own environment. For epipack, epydemix, epymorph, MEmilio, and summer2, the importer does this when it is run as a script there (it needs neither Starsim nor Sciris, so it can be copied anywhere), e.g. `python memilio.py my_model.py model.json`; see the docstring of each. For EpiModel, `dump_epimodel.R` does the same in R, and `ss.import_from()` runs it if R is installed.

## Exporting

`export.py` writes a sim's declared diseases as SBML or odin (`ss.export_to()`, with the formats in `ss.converters.exporters`), listing what the format can't carry, and a MultiSim's runs as a submission to a forecast hub (`ss.converters.to_hub()`).

## Migrating

`migrate.py` rewrites Starsim 3 code for Starsim 4: `python -m starsim.converters.migrate mypackage/`.
