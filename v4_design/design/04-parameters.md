# Parameters and data

Declaration, typing, provenance, shapes, and the requirement contract.

Justified by [best-practices/parameters-and-distributions.md](../best-practices/parameters-and-distributions.md).

## The default: no parameter block at all

In a two-transition model, a parameters block is ceremony:

```python
sir = ss.Disease(
    infection = ss.transmission('S -> I', beta=0.05),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)
```

The values live in the transitions that use them, and both are addressable — `sir.infection.beta` and `sir.recovery.dur` are the canonical paths, usable as effect targets and as calibration targets. Naming a parameter is for when it is used twice, fitted, or documented.

## `ss.Par`

```python
ss.Par(
    value  = None,        # scalar, ss.Quantity, ss.Dist, ss.timeseries, or callable
    *,
    kind   = None,        # 'rate' | 'dur' | 'prob' | 'freq' | 'count' | None (inferred from use)
    bounds = None,        # [lo, hi]: the admissible range, and the calibration search space
    prior  = None,        # an ss.Dist: the prior for Bayesian inference
    source = None,        # free text: where this number came from
    doc    = None,        # free text: what it means
    unit   = None,        # overrides the sim's unit for this parameter only
)
```

```python
sir = ss.Disease(
    beta      = ss.Par(0.05, bounds=[0.001, 0.5], source='Ferguson et al. 2020, Table 2'),
    dur_inf   = ss.Par(ss.days(10), bounds=[3, 30]),
    infection = ss.transmission('S -> I', beta=beta),
    recovery  = ss.progression('I -> R', dur=dur_inf),
)
```

`bounds` is what makes calibration declarative ([13-inference.md](13-inference.md)): the inference layer reads the range and the kind, and transforms automatically — probabilities on a logit scale, rates on a log scale — without the user specifying it.

`source` is one optional line that answers the question every reviewer of a modeling analysis asks. It costs nothing when unused, it is carried into `summary()` and into the parameter hash, and the models that matter most are the ones where someone bothered.

`kind` is normally inferred from the slot the parameter is used in ([02-types.md](02-types.md)). It is stated explicitly when a parameter is declared before it is used, or is used in more than one slot. A parameter used in two slots of different kinds raises `E104 InconsistentParameterKind`.

### Declaration is required; unknown names are refused

Assigning an undeclared parameter name on a module MUST raise `E602 UnknownParameter`, naming the module and listing the closest declared names. This is Starsim's `define_pars` locking, kept: a misspelled parameter name is a silent behavioral difference in every framework that permits it.

```text
E602 UnknownParameter: 'sir' has no parameter 'dur_inff'.
     Did you mean 'dur_inf'?
     Declared parameters: beta, dur_inf
```

## The parameter ladder

Every parameter slot accepts every rung. The slot's *kind* is fixed; its *form* is not.

```python
beta = 0.05                                      # bare number, unit from context
beta = ss.perday(0.05)                           # typed
beta = ss.lognorm(mean=0.05, std=0.01)           # a distribution: heterogeneity
beta = ss.timeseries({2020: 0.05, 2021: 0.02})   # time-varying
beta = ss.timeseries('beta.csv', mode='linear')  # from a file
beta = ss.seasonal(mean=0.05, amplitude=0.3, peak='January')
beta = ss.by(age=[0.08, 0.05, 0.03])             # one per stratum
beta = lambda mod, sim, uids: ...                # escape hatch, priced
```

`ss.timeseries` interpolation modes are `constant`, `linear`, and `spline`, which is odin's set and is sufficient.

Every backend MUST handle every rung or refuse by name. In particular, a time-varying rate under an exact continuous-time method requires the inhomogeneous-Poisson algorithm, and a backend that cannot do it MUST raise `E803 TimeVaryingRateUnsupported` rather than freezing the rate at its initial value. A rate is an expression in time and state, and each backend is responsible for handling that correctly or refusing.

### Heterogeneity versus uncertainty

A distribution in a parameter slot means one of two things, and the **slot** disambiguates. They MUST NOT be conflated:

```python
ss.progression('I -> R', dur=ss.lognorm(mean=ss.days(10), std=ss.days(3)))   # heterogeneity
ss.Par(prior=ss.lognorm(mean=0.05, std=0.01))                                 # uncertainty
```

The first gives each agent its own dwell time and becomes Erlang stages in a compartmental backend. The second is a prior over a single value and is used only by calibration. A vector-valued parameter is never implicitly a time series in one place and a sensitivity sweep in another.

## Shapes

Dimensional types say what a number means; shapes say how many of them there are and along which axis. A language spanning agents, patches, strata, and time needs both.

```python
beta = 0.05                            # scalar: the same everywhere
beta = ss.by(age=[0.08, 0.05, 0.03])   # shape (age,)
beta = ss.by(patch=matrix)             # shape (patch, patch)
beta = ss.by(age=..., patch=...)       # shape (age, patch)
```

Shapes are inferred from the value and checked against the model's dimensions at build time. Broadcasting over omitted dimensions is permitted and is named in the error when a shape does not fit:

```text
E603 ShapeError: 'beta' for transition 'infection' has shape (age=3,)
      but the model is stratified by (age=3, patch=47).
      Broadcasting over 'patch' is assumed. To be explicit, write
        ss.by(age=[...], patch=[...])
      or leave 'patch' out to broadcast, which is what is happening now.
```

A shape that is wrong rather than merely incomplete — `(age=4,)` against `age=3` — is an error with no fix offered except the correct shape.

`ss.by` supersedes the parallel global/local parameter concepts (SimInf's `gdata`/`ldata`, MetaCast's `universal_params`/`subpop_params`) with one mechanism, and it supersedes string-concatenated parameter names entirely. A name like `'beta_[high,vaccinated]'` MUST NOT appear anywhere in the language.

## Where values live

Values live inline by default. The split is available and is not required:

```python
sir = ss.Disease(
    beta      = ss.Par(kind='rate', bounds=[0.001, 2.0], doc='transmission rate'),
    infection = ss.transmission('S -> I', beta=beta),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
)

sim = ss.Sim(sir, pars='params/kenya_2024.csv')
```

A `ss.Par` with no value and no external source raises `E604 UnvaluedParameter` at build time, naming the parameter and the files searched. External parameter files may be CSV, TOML, JSON, or a dict; the format is a convenience and carries no semantics.

Either way, `sim.hash()` reports the factored run identity ([11-random.md](11-random.md)):

```text
model=8f3a21  pars=c47e90  config=1b6d55  scenario=baseline  seed=1
```

Separating structural identity from parameter values is right; requiring two files for an SIR is not. The hashes are computed the same way whichever form was used, so the reviewer's question — *did the model change, or did the numbers?* — is answered without mandating a file layout.

## Requirements

A module declares its data interface as data, so it can be checked before anything runs and so a data-entry form can be generated from it.

```python
ss.Needs(
    name,
    *,
    shape  = None,     # 'scalar' | 'age' | 'patch' | 'age x age' | 'T x patch' | ...
    kind   = None,     # a type from 02-types.md
    source = None,     # a suggested provider, e.g. 'socialmixr:POLYMOD'
    doc    = None,
    default = None,    # if given, the requirement is optional
)
```

```python
class MyNetwork(ss.Network):
    requires = [
        ss.Needs('population',     shape='patch', source='UN WPP 2024'),
        ss.Needs('contact_matrix', shape='age x age', source='socialmixr:POLYMOD'),
    ]
```

```python
sim.needs()                 # the requirements table for this model
sim.needs(to='form.csv')    # a data-entry form with exactly these fields, units, and labels
```

`needs(to=...)` is Atomica's databook generated from the framework, which is the single most effective accessibility mechanism in the landscape. The generated form is a contract: filling it in is sufficient to run the model.

Requirements are **enforced**, not advisory. A module reads through a view containing exactly what it declared; reading anything else raises `E601 MissingRequirement`. This is what keeps the generated form honest — a module that quietly reads something it did not declare would produce a form that is missing a field.

## Named data

A reference to data is part of the model. A language that can express a contact matrix but cannot say *which* contact matrix, from *which* source, for *which* place, has moved the problem rather than solved it.

```python
ss.People('Kenya', year=2024)                     # size, age structure, sex ratio
ss.contacts('Kenya', source='prem_2021')          # a contact matrix
ss.geo('Kenya', level='county')                   # names, populations, centroids, adjacency
ss.data('data/weekly_cases.csv')                  # an observation series
```

Requirements:

- Resolution is **pull-with-cache**, not bundle. Data is fetched on first use, cached locally, and the cache key includes the source and version.
- The resolved source, version, and retrieval date MUST appear in `summary()` and MUST contribute to the parameter hash, so that a re-run against changed upstream data is *detectable* rather than silent.
- The provider interface MUST be pluggable. Bundled providers MUST include at least one country that is not the one the developers live in.
- A named location that cannot be resolved raises `E605 UnknownLocation` listing near matches and the providers searched.

## Rejected

- **Open parameter lists.** A parameter that springs into existence on first use is a typo that becomes a silent behavioral difference.
- **Mandatory value/model separation.** The capability is right; requiring two files for an SIR is not.
- **String-concatenated parameter names.** Shapes do this job properly.
- **A `use_defaults` flag** that leaves much of a model's behavior invisible in its files. Defaults here are guessed aggressively and printed exhaustively, which sounds similar and is the opposite policy.
- **More than eleven core distributions.** See [17-vocabulary.md](17-vocabulary.md).
