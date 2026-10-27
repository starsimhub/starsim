# The model as code

There is no model spec (decided 2026-10-27, replacing the JSON spec with `sim.to_spec()`, `sim.hash()`, `ss.diff()`, and `sim.to_code()`). A model is its code: the script that builds the sim is the thing to read, share, version, and compare, and two models are compared by their results, by `describe()`, or by a text diff of their code. There is no code-to-code translator: Starsim never turns a sim back into code.

| What | How | Holds | Read back |
|---|---|---|---|
| The model | the user's script; for an import, the code the importer writes (`sim.imported.code`, [16-interop.md](16-interop.md)) | the model as Python | it is run |
| The results as JSON | `sim.to_json()` | the sim's parameters (a module among them as a reference, its type and name), its summary, and its results, for reading elsewhere | no |
| A saved sim (a checkpoint) | `sim.save()`, `ss.save('run.sim', sim)` | the sim itself, as a pickle, with its results; saved partway through a run, with its full state, so that it can continue | yes, with `ss.load()`, by the same code |

An import is the one place code is written: each importer reads its format into an `ss.interop.Source`, `ss.interop.write()` writes that as a declared `ss.Disease` and its `ss.Sim`, and the imported sim is that code run, so there is one path (foreign model, then code, then sim). The exporters (SBML, odin) read the built model directly: each disease's graph (its states, transitions, parameters, and expressions) and engine ([16-interop.md](16-interop.md#export)). The record of what was run is the code, its commit or tag, and the Starsim version that a saved sim stores ([11-random.md](11-random.md#run-identity)).
