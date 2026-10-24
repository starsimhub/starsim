# One sexual network class (decision 18)

A design study for Cliff's answer to decision 18 in [plan.md](plan.md): "can we have just `ss.SexualNetwork`, which is easily configurable to turn into `ss.MFNet`, and get rid of `PartnerNet`?" Report only: nothing in `starsim/`, the tests, or the spec was changed. Line numbers are of `starsim/networks.py` on `v4-simplify` (uncommitted) unless given.

## Short answer

Yes, and the smallest version is mostly a move, not a rewrite: `ss.PartnerNet` becomes `ss.SexualNetwork` (taking in the 44-line base it would replace), `ss.MFNet`'s two code paths move into it verbatim behind arguments, and `ss.MFNet` stays as a three-line preset so that its name (`'mfnet'`, `'mf'`), Starsim 3 code and every MFNet result are unchanged. The real cost is not in core but in the contract with subclasses: STIsim and HPVsim subclass `ss.SexualNetwork` as a thin base, and a base that forms partnerships will form them for its subclasses too unless it is guarded (five specific rules below). With the guards, STIsim needs no change and HPVsim none (or one line), and their results should be bit-identical.

## 1. What the three classes do now

| Class | Lines | What it is |
|---|---|---|
| `ss.SexualNetwork` | 44 (744–787) | A base class with no formation of its own: edges carry `acts` and `condoms`; a `debut` state; `sides = dict(m='male', f='female')` as a class attribute (so a disease's beta on it is `ss.peract(m2f=, f2m=)`); `active()` (participant, past debut, alive) and `available(sex)` (active and unpartnered). Subclassed by `ss.MFNet`, `ssl.MSMNet`, STIsim's `BaseNetwork` and HPVsim's `hpv.SexualNetwork`. |
| `ss.PartnerNet` | 260 (1141–1400) | Generic lasting partnerships, new in v4 (not in Starsim 3): `n_contacts` (mean partners at a time), `dur`, optional `acts` (no acts means beta per contact), `sides=` (`'sex'` for anyone with four directions, or a dict), `between=(a, b)` (two groups, one of each per partnership, with per-group `n_contacts`), `where=` (who takes part), `beta`. People seek a partner with a constant probability `n/D̄` per step and seekers are paired at random; it starts at a Poisson degree distribution. With `between=` and no `dur=`, contacts are one-off (a contact matrix, exact under the ODE solver). Not a `SexualNetwork`. |
| `ss.MFNet` | 229 (1402–1630) | `DegreeTargets` + `SexualNetwork`. Two modes. Default (Starsim 3): debut (`ss.normal(16)`), participation (`ss.bernoulli(0.9)`, drawn at debut), `rel_part_rates`, durations lognormal (mean 15 y), acts lognormal (80/y); every unpartnered active man and woman is paired each step (`ss.choice`, not CRN-safe), so at most one partner each. With `n_contacts` (and optionally `concurrency`, `max_contacts`): a target degree distribution held by degree-dependent seeking, an open-population "boost", and a start at the target with residual durations. |

Related: `DegreeTargets` (142 lines, 998–1139) is the target-degree machinery shared by `ss.MFNet` and `ssl.DegreeNet`; `mode_sides()` and `check_acts()` (~35 lines) are shared helpers; `ssl.DegreeNet` (`starsim/library/networks/degree.py`, 242 lines) is `DegreeTargets` + `PartnerNet` adding `match=` (nodematch), `ss.by` degrees (nodefactor) and the d.rate correction, and is what the EpiModel importer emits; `ssl.MSMNet` (78 lines) is the Starsim 3 pairing rule for men only.

What they share already: `DynamicNetwork.end_pairs()`; edge acts and `check_acts()`; sides and `side_raw()` (MFNet overrides it with `female.raw`); a waiting list for the surplus side; de-duplication of existing pairs (`PartnerNet.pair()`, `MFNet.pair_sexes()`). One finding worth having: PartnerNet's seeking rule is a special case of `DegreeTargets`. For a Poisson target, `π(k+1)/π(k) = λ/(k+1)`, so `DegreeTargets.init_seek()` gives `p_seek[k] = (k+1)·p_end·λ/(k+1) = λ/D̄`, the constant PartnerNet uses, and both start from the same inverse CDF of `rng_degree`. So `PartnerNet(n_contacts=x)` and `MFNet(n_contacts=x)` are the same network up to MFNet's boost, its residual starting durations, and the pairing by sex.

## 2. Proposed design

One class, `ss.SexualNetwork(DegreeTargets, DynamicNetwork)`, with PartnerNet's code as the body, MFNet's two paths added verbatim, and MFNet's defaults:

| Argument | Default | Meaning |
|---|---|---|
| `dur` | `ss.lognorm_ex(ss.years(15), ss.years(1))` | As both today |
| `acts` | `ss.lognorm_ex(ss.freqperyear(80), ss.freqperyear(30))` | `None` for no acts (beta per contact), as PartnerNet |
| `n_contacts` | `None` | `None`: Starsim 3 pairing, at most one partner (MFNet default). A number: constant seeking (PartnerNet). A distribution, or with `concurrency=`/`max_contacts=`: targets (MFNet's targeted mode). A dict with `between=`: per group |
| `concurrency`, `max_contacts` | `None` | As MFNet |
| `sides` | `'mf'` | New value: each partnership is a man and a woman (directions `m2f`, `f2m`), as MFNet. `'sex'`: anyone, by sex (four directions). `None`: no sides, one beta. A dict: conditions |
| `between`, `where` | `None` | As PartnerNet; `between=` without `dur=` keeps meaning one-off contacts |
| `debut` | `ss.normal(loc=16)` | `None` for none |
| `participation`, `rel_part_rates` | `ss.bernoulli(0.9)`, `1.0` | `None` (or 1) for everyone; `rel_part_rates` kept for Starsim 3 code |

Dropped: PartnerNet's `beta` parameter (edges get `beta=1`, as MFNet; see rule 1 below for why a `beta` par is harmful). Four formation paths remain (Starsim 3 pairing, constant seeking, targets, one-off), selected by `n_contacts`, `concurrency`/`max_contacts` and `dur`; the solver code already distinguishes them.

`ss.MFNet` stays as a preset: `class MFNet(SexualNetwork): """ ss.SexualNetwork with its defaults, as in Starsim 3 """`. It costs three lines and keeps the default name `'mfnet'` and the `'mf'` lookup (module names come from the class name, `modules.py:442`; the `'net'`-suffix alias at `modules.py:130`), `migrate.py`'s `mf_keys`, beta dicts keyed `mfnet`, the 83 test lines and 46 docs lines that use it, and its random streams (seeds hash the trace, e.g. `networks.mfnet.rng_seek`, `distributions.py:777`). Removing it instead would rename every default MF network to `sexualnetwork` and break `beta=dict(mfnet=...)`, `networks='mf'` and Starsim 3 scripts, for three lines saved. Docs and the spec would present `ss.SexualNetwork` as the class and mention `ss.MFNet` once.

`DegreeTargets` stays in core as a mixin of `ss.SexualNetwork` (Cliff already decided MFNet keeps its targets). `ssl.DegreeNet` becomes `class DegreeNet(ss.SexualNetwork)` with generic defaults (`sides=None, debut=None, participation=None, n_contacts=1`), so the EpiModel importer's output (`ssl.DegreeNet(...)`, `interop/epimodel.py:119`) and its code string are unchanged. `ssl.MSMNet` could later become `ss.SexualNetwork(sides=None, where='male', ...)`, but that is out of scope.

### Spellings, before and after

```python
# Starsim 3 / default MF network: unchanged
ss.MFNet()                                    →  ss.SexualNetwork()  (or ss.MFNet(), still the same network, named 'mfnet')
ss.MFNet(n_contacts=ss.poisson(0.7), concurrency=0.1, dur=d)  →  ss.SexualNetwork(n_contacts=ss.poisson(0.7), concurrency=0.1, dur=d)

# Generic partnerships (PartnerNet's default had no sexes, debut or participation)
ss.PartnerNet(n_contacts=0.7, dur=50)         →  ss.SexualNetwork(n_contacts=0.7, dur=50, sides=None, debut=None, participation=None, acts=None)
ss.PartnerNet(sides='sex', n_contacts=1, dur=ss.years(2), acts=ss.freqperyear(10))
                                              →  ss.SexualNetwork(sides='sex', n_contacts=1, dur=ss.years(2), acts=ss.freqperyear(10), debut=None, participation=None)

# Non-sexual: needle sharing among people who inject
ss.PartnerNet(name='needles', where=ss.ppl.pwid, n_contacts=2, dur=ss.months(6), acts=ss.freqpermonth(20))
                                              →  ss.SexualNetwork(name='needles', where=ss.ppl.pwid, sides=None, debut=None, participation=None, n_contacts=2, dur=ss.months(6), acts=ss.freqpermonth(20))

# Sex work between two groups: lasting, and one-off
ss.PartnerNet(name='regular', between=('fsw', 'client'), n_contacts=dict(fsw=3, client=0.5), dur=ss.months(3))
                                              →  ss.SexualNetwork(name='regular', between=('fsw', 'client'), n_contacts=dict(fsw=3, client=0.5), dur=ss.months(3), acts=None)
ss.PartnerNet(name='sw', between=('fsw', 'client'), n_contacts=dict(client=ss.permonth(1)), acts=3)
                                              →  ss.SexualNetwork(name='sw', between=('fsw', 'client'), n_contacts=dict(client=ss.permonth(1)), acts=3)

# EpiModel-style targets: unchanged
ssl.DegreeNet(n_contacts=0.7, concurrency=0.22, max_contacts=3, dur=50)
```

The generic and non-sexual spellings get longer, because MFNet's defaults (debut, participation, acts) are now the class's. That is the price of one class with sexual defaults; the alternative (generic defaults, MFNet as a preset carrying its own) keeps them short but makes `ss.SexualNetwork()` alone a network without sexes or debut, which reads wrong for the name. A `between=` with sex workers keeps debut and participation by default, which is arguably right for them.

## 3. Cost

Package (`starsim/`):

- `networks.py`: three classes of 533 lines become one of roughly 410 plus a 3-line preset (one docstring and `__init__` instead of three; MFNet's debut/participation (~30 lines), Starsim 3 pairing (~40) and targeted mode (~50) move in; PartnerNet's body stays). About −120 lines. Merging MFNet's targeted mode onto `DegreeTargets` + PartnerNet's seek/pair path (one targets implementation, choosing the boost or DegreeNet's departure correction) would save ~50 more but change the results of `MFNet(n_contacts=...)`; leave it for later.
- `solvers/mixing.py` (lines 138–140, 186, 231–237, 276): `isinstance(net, ss.MFNet)` becomes `p.sides == 'mf'` and debut/participation present; `(ss.PartnerNet, ss.MFNet)` becomes the guarded test of rule 4. About 8 lines. `solvers/choose.py:530–532` the same (2 lines). `sim.py:151` (rule 3, 1 line). `__init__.py` drops `PartnerNet`. Docstrings in `time.py`, `utils.py`, `solvers/compartmental.py`, `library/networks/__init__.py`, `library/README.md`, `interop/epimodel.py`. `interop/migrate.py`: none, since `ss.MFNet` stays.
- `library/networks/degree.py`: new base and generic defaults (~10 lines); `sexual.py` (`MSMNet`): check its `init_post()`, which calls `super().init_post()` (rule 2).

Tests: PartnerNet appears in 14 lines of 4 files (`test_networks.py` 544–602, 739, 767, 825–887; `test_interop.py:299`; `test_spec.py:129` twice); each is a rename plus the `debut=None, participation=None` arguments. Exact-value assertions on those networks may move, since the default name, and so the stream traces, change from `partnernet` to `sexualnetwork`. The two `Couples(ss.SexualNetwork)` tests (`test_networks.py:740`, `1076`) override only `step()`, so they are the in-repo check of rule 2. MFNet tests and `test_baselines.py` are untouched. Add: `ss.SexualNetwork()` and `ss.MFNet()` give identical results (with `name='mfnet'`), and a subclass overriding `step()` has no partnerships after `init()`.

Spec: 22 PartnerNet lines in 7 files, mostly `06-routes.md` (rows 185–186, 201, the example at 226 and the bullets 234–237), `17-vocabulary.md` (92, 187, 207, 208), `12-backends.md` (63, 135, 154), and one each in `01-objects.md`, `02-types.md`, `19-open-questions.md`; plus `docs/user_guide/modules_networks.qmd` (4 lines), `CHANGELOG.md`, and two skills (`skills/import-model/references/idioms.md`, `skills/migrate-starsim3/SKILL.md`). About half a day in all with the package change.

STIsim (`stisim/networks/base.py:51`, `BaseNetwork(ss.SexualNetwork)`) relies on: `meta.acts`/`meta.condoms`, the m/f `sides` class attribute, `active()` (`diseases/sti.py:189`, `syphilis.py:588`), `isinstance(net, ss.SexualNetwork)` (`diseases/bv.py:523`, `560`), and `init_post(add_pairs=False)`; it overrides `__init__` (no arguments to super), `init_pre` (calls super), `step`, `add_pairs`, `end_pairs`, `net_beta`, and defines its own `participant`/`debut` states and pars (`acts`, `debut_f`, `debut_m`, ...). HPVsim (`hpvsim/network.py:56`) relies on `active()` (super), `debut`, `participant`, `edges.acts`, `end_pairs()`, and calls `super().init_post()` (with pairing) and `super().init_pre()`; it overrides `step`, `set_network_states` and `net_beta`, not `add_pairs`. Neither subclasses `ss.MFNet` or `ss.PartnerNet`. TBsim does not use these classes.

Their partnership formation is their own (`step()` overridden), so it does not change, provided the base keeps five rules:

1. No `beta` parameter on the base. HPVsim's `route_pars` broadcasts a bare key to every module whose pars have it, using `set(SexualNetwork().pars.keys())` (`hpvsim/parameters.py:370`); HPV's own `beta` (`parameters.py:175`) would then also set the network's. The other new keys (`dur`, `acts`, `debut`, `participation`, `n_contacts`, ...) do not collide with HPVsim or STIsim names, but they will show, unused, in those networks' `pars` and `describe()`.
2. A subclass that overrides `step()` forms its own partnerships: the base's `init_post()` then starts none and draws no debut or participation (only `Network.init_post`). Without this, HPVsim's `super().init_post()` would run HPVsim's own `set_network_states()` and then the Starsim 3 pairing, adding MF partnerships at the start with the base's 15-year durations, and results would change; likewise for `ssl.MSMNet` and the `Couples` tests. (Alternative: an explicit class flag, which needs a line in HPVsim; or HPVsim passes `add_pairs=False`, as STIsim already does.) `init_post(add_pairs=False)` must keep its signature, which PartnerNet's lacks today.
3. Naming from the variable (`sim.py:151`, `isinstance(mod, ss.PartnerNet)`) must become `type(mod) is ss.SexualNetwork`; with `isinstance`, `net = sti.StructuredSexual()` would be renamed `'net'` and STIsim's `sim.networks.structuredsexual` would break.
4. The solvers must not treat a subclass as approximable: with `isinstance(net, ss.SexualNetwork)`, an STIsim network under `solver='ode'` would be approximated from the base's unused defaults instead of raising `CapabilityError`. Use the same test as rule 2 (an exact `SexualNetwork`/`MFNet`, or one not overriding `step()`).
5. Keep `active()`, `available()`, `debut`, `participant`, `meta.acts`, `meta.condoms` and the m/f class-level `sides` for subclasses that set none (`sides='mf'` must resolve to the same dict).

Results and CRN: with the rules, STIsim and HPVsim should be bit-identical; the base's extra random streams (`rng_seek`, `rng_end`, ...) do not move theirs, since each stream's seed hashes its own trace, and none of the new attribute names clash with theirs (checked: STIsim's `concurrency` is a state, not a par). This should be confirmed by running both suites after the change; HPVsim does not import against the current branch (`ss.Calibration` is missing), so that comes first. In core, `ss.MFNet()` and `ss.MFNet(n_contacts=...)` keep their attribute names and default name, so their draws and the baselines are unchanged; ex-PartnerNet networks get new draws (same statistics) only where their default name changes. Nothing here fixes or worsens the S5.5 issue (adding a declared disease changes MFNet's initial partnerships) or the Starsim 3 path's non-CRN `ss.choice`. A side effect: a needle network made with `ss.SexualNetwork` is now `isinstance(..., ss.SexualNetwork)`, so STIsim's BV (`bv.py:523`) would count it as sexual.

## 4. Risks and open questions

1. Defaults: MFNet's (proposed; `ss.SexualNetwork()` is today's `ss.MFNet()`) or generic ones (shorter generic spellings; `ss.MFNet` then carries the Starsim 3 defaults)?
2. Keep `ss.MFNet` as a 3-line preset (proposed), or remove it (renames `'mfnet'`, breaks `networks='mf'`, beta dicts and Starsim 3 code; `migrate` would rewrite to `ss.SexualNetwork(name='mfnet')`)?
3. The new `sides='mf'` next to the existing `sides='sex'` may confuse; alternatives are `sides='sex'` meaning MF and `'any'` for the four-direction case, or spelling MF as `between=dict(m='male', f='female')` (which works with today's code, but is long for the default).
4. Non-sexual partnerships (needle sharing, PartnerNet's own docstring example) under the name `SexualNetwork`: acceptable, or point those at `ss.RandomNet(dur=...)` in the docs?
5. Rule 2's opt-out: implicit (overriding `step()`) or an explicit class flag? Implicit needs no downstream change; explicit is clearer.
6. One-off contacts are signalled by `between=` without `dur=`; with `dur` now having a default, that rests on "was `dur` given", which is subtle. An explicit `dur=0` or `one_off=True` would be clearer, but is scope creep.
7. Later, not now: one targets implementation for MFNet and DegreeNet (MFNet's boost versus DegreeNet's departure correction; changes `MFNet(n_contacts=)` results), and folding `ssl.MSMNet` into a preset.

Recommendation: do the smallest version. Rename `PartnerNet` into `ss.SexualNetwork`, move MFNet's two code paths in verbatim under the arguments above with MFNet's defaults, keep `ss.MFNet` as a 3-line preset, put `ssl.DegreeNet` on the new class with generic defaults, and implement the five subclass rules with a test for each. That gets to one class with about −120 lines in core, no change to any MFNet result or to Starsim 3 compatibility, and no required change to STIsim or HPVsim. Run the STIsim and HPVsim suites before and after to confirm bit-identical results.
