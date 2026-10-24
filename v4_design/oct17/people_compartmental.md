# `sim.people` in a compartmental run: the alternatives

For decision 13 in [decisions_oct18.md](decisions_oct18.md). Cliff: "not sure about one row ... seems like it could be used for compartments, metapopulations, etc. but i would need to see the alternatives in more detail." [CK: yes, i like alternative B, as already implemented -- can close this]

## What the code does now

It is not one row: it is **one row per group**. With no stratification there is one group, so one row; with `stratify(age=[0, 15, 65], patch=['a', 'b'])` there are 6 rows. Each row has the group's values of its stratifications (`people.age` is the middle of the age group: `[7.5, 7.5, 40, 40, 82.5, 82.5]`), and `alive` (all True), `scale` (all 1), `ti_dead`, `ti_removed`. A per-agent attribute that is not a stratification raises `CapabilityError` naming `solver='abm'` (`people.female` when sex is not a dimension). What uses it: conditions written on people (`where=ppl.age < 15`, tried on the groups at the first step, so `auto` can tell whether a function of people has a compartmental form), the age midpoints the engines read, and `ss.People(1)` in the docs as a stand-in to write conditions on.

## The alternatives

**A. `None`.** `sim.people` is `None`; reading it raises with a pointer to `sim.groups` and `solver='abm'`; group values live only on `sim.groups`.
```python
sim.people            # CapabilityError: a compartmental run has no people; see sim.groups, or run with solver='abm'
sim.groups.strats[0]            # the age stratification, with its levels
```
Breaks: conditions on people (`where=ppl.age < 15`) need another object to be written on (`ss.People.age < 15` at class level, or `ss.age`); the try-it rule needs a stand-in anyway; every module that reads `sim.people` in `init_post()` must branch on the solver. Clearest about what is there, but moves the problem rather than removing it.

**B. One row per group (now).** As above.

```python
ppl = sim.people; len(ppl)       # 6 groups
mask = ppl.age < 15              # [True, True, False, False, False, False], a mask over groups
```
Breaks: `len(sim.people)` is the number of groups, not people, and `people.scale` is 1, so `(ppl.age < 15).sum()` counts groups, not children; `alive` and `ti_dead` are placeholders. Code that loops over uids works but means something else.

**C. One row per compartment and group.** Rows are (state, group) pairs, with `scale` the number of people in each, so a sum over rows is a population count.
```python
ppl.scale[ppl.age < 15].sum()    # children, from the compartments
sir.infected                     # a mask over rows: True for the I rows
```
Breaks: rows are created by the model's states, so they change when a state is added; counts are fractional (an ODE) and change every step, so `scale` must be rewritten each step (a per-step cost); several diseases tracked jointly multiply the rows (a crossing); `uids` would index compartments, which is the confusion the try-it rule exists to avoid. Most like agents, and most misleading.

**D. A metapopulation view.** Rows are places (one stratification, usually `ss.geo()`), with each place's attributes (`pop`, `lat`, `lon`, the live size), and the other dimensions nested inside.
```python
places = sim.people              # one row per district
places.pop, places.lat           # per place
(places.pop > 1e5)               # a mask over places, for where=
```
Breaks: only one dimension can be the rows, so age conditions need another route; a model with no places has nothing to show. Good for spatial work (the LASER and Ghana-style users), but a special case of B.

**E. A People-like view of `sim.groups`.** Not an `ss.People`: a light object with the stratifications as attributes, `n` as the live size of each group (from the compartments), and the same conditions as people.
```python
g = sim.people                   # GroupView over the 6 groups
g.age, g.patch, g.n              # midpoints, labels, live sizes
(g.n * (g.age < 15)).sum()       # children
```
Breaks: code that expects an `ss.People` (its arrays, `uids`, `people.states`) gets an object with fewer methods; the engines and the try-it rule would read the view instead of `People` (a moderate refactor in `people.py`, `dimensions.py`, and `solvers/choose.py`).

## Recommendation

Keep **B** (one row per group, which already serves compartments and metapopulations: each row is a patch × age group), with two small changes that remove what is misleading in it: `people.n` gives the live size of each group, computed from the compartments when it is read (so that `(ppl.n * (ppl.age < 15)).sum()` counts people, with no write each step), and `repr(sim.people)` says `People(6 groups; ...)` rather than `People(n=6; ...)`. That is E's main benefit without a new class. D is then a matter of `ppl.patch == 'a'` on the same object. A and C are not recommended: A only moves conditions elsewhere, and C ties the rows to the model's states and costs a write each step.
