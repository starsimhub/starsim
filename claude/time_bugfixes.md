# `starsim/time.py` bug audit

Audit of `starsim/time.py` (all 2535 lines: `DateArray`, `date`, the unit factors, `TimePar`/`dur`/`datedur`, `Rate`/`prob`/`per`/`freq`, the convenience classes and class maps, the backwards-compatibility shims, and the matplotlib locator/formatter/converter) at commit `3d8dc9d5` on branch `rc3.6.2`, working-tree version as-is. Method: line-by-line reading, then every candidate was reproduced with a script run against the editable install (Starsim 3.6.1, numpy 2.4.6, `MPLBACKEND=agg`); scripts are in the session scratchpad under `time/`. Candidates were checked against the docstrings, `tests/test_time.py`, and `docs/user_guide/advanced_time.qmd` before being recorded. Only unambiguous defects on in-contract input are listed.

**Nothing in this document has been applied.**

## Summary

| # | Severity | Function/method | Defect | Line |
|---|----------|-----------------|--------|------|
| 1 | High | `prob.to_prob()` | With no argument, ignores the module-linked `default_dur` and returns the raw per-unit probability | 1990-1992 |
| 2 | High | `Rate.__add__()`, `Rate.__sub__()` | Adding or subtracting `ss.per` rates converts the other operand to a probability, giving wrong rates | 1708, 1725 |
| 3 | Medium | `FloatYearLocator.__init__()` | Compares the unit class to the string `'ss.date'`, so the date-aware tick locator is never used | 2382 |
| 4 | Medium | `Rate.__eq__()` | Unitless `ss.prob` objects always compare equal; unit-bearing probs compare by linear rescaling, not the underlying rate | 1738, 1744 |
| 5 | Low | `datedur.__truediv__()` | The precision workaround doesn't work: `datedur(weeks=1)/datedur(days=1)` still gives `6.999999999999999` | 1511-1516 |
| 6 | Low | `years.__init__()` | Date-string arrays are converted in place: numpy string arrays end up holding strings, and the caller's list is mutated | 2185-2188 |
| 7 | Low | `datedur.__init__()` (via `date.__sub__()`) | Constructing from a `relativedelta` drops hours/minutes/seconds, so subtracting sub-daily dates loses the time part | 1272 |
| 8 | Low | `DateArray.to_date()` | `inplace=True` silently does nothing on a float-backed `DateArray` | 196-197 |

## High severity

### 1. `prob.to_prob()` ignores the linked timestep — `time.py:1990`

`Rate.to_prob()` falls back to `self.default_dur` when `dur` is `None`, and `Module.link_timepars()` sets `default_dur` to the module's `dt` so that "`self.pars.death_rate.to_prob()`" works with no argument (the docstrings of `link_timepars()` and `set_default_dur()` say so, as does `advanced_time.qmd`: "`death_rate.to_prob()` is simply a shortcut for `death_rate * self.dt`"). `prob.to_prob()` overrides this method but leaves out the fallback: with `dur=None` it returns `self.value` unchanged. So any `ss.prob` subclass (`probperyear`, `probpermonth`, etc.) resolved with `.to_prob()` inside a module gives the probability per its own unit, not per timestep. `.to_dt()` and `* self.dt` give the right answer, which makes the inconsistency easy to miss.

```python
class M(ss.Module):
    def __init__(self, **kw):
        super().__init__()
        self.define_pars(p_die=ss.probperyear(0.5), r_die=ss.peryear(0.5))
        self.update_pars(**kw)
    def step(self):
        if self.ti == 0:
            print(f'prob.to_prob()={self.pars.p_die.to_prob():.4f}  prob*dt={self.pars.p_die*self.t.dt:.4f}  prob.to_dt()={self.pars.p_die.to_dt():.4f}  per.to_prob()={self.pars.r_die.to_prob():.4f}')
ss.Sim(n_agents=100, dt=ss.months(1), start=2000, stop=2001, modules=M(), verbose=0).run()
```

Actual: `prob.to_prob()=0.5000  prob*dt=0.0561  prob.to_dt()=0.0561  per.to_prob()=0.0408`. Expected: `prob.to_prob()=0.0561`, matching `prob*dt` and `to_dt()`.

Blast radius: anyone who writes a custom module that follows the documented `self.pars.x.to_prob()` pattern with a `probper*` parameter. With a monthly `dt` the per-step probability is about 9x too high here (0.5 vs 0.056), with no warning. `ss.per` parameters are unaffected, which is why the built-in `diseases.py:878` waning call happens to be fine.

**Fix**: in `prob.to_prob()`, add the same fallback as `Rate.to_prob()` at the top, `if dur is None: dur = self.default_dur`, before the `dur is None` branch.

### 2. Adding or subtracting `ss.per` rates gives wrong values — `time.py:1708`, `time.py:1725`

`Rate.__add__()` computes `self.value + other*self.unit`. The intent is "`other` expressed per `self.unit`", which works for `ss.freq` (`freq*dur` returns a number of events). But `per.__mul__(dur)` is `to_prob()`, so for `ss.per` the second operand is turned into the probability `1 - exp(-rate*factor)` before being added to a rate. `Rate.__sub__()` has the same expression.

```python
ss.peryear(1) + ss.peryear(1)     # actual peryear(1.63212)   expected peryear(2)
ss.peryear(0.1) + ss.perday(0.01) # actual peryear(1.07401)   expected peryear(3.75)
ss.peryear(3) - ss.peryear(1)     # actual peryear(2.36788)   expected peryear(2)
ss.freqperyear(1) + ss.freqperyear(1) # freqperyear(2), correct
```

Blast radius: any user who combines hazards by addition, e.g. background plus excess mortality, or several exit rates from a compartment. That is the natural thing to do with instantaneous rates, and the result is silently wrong by a large factor. Nothing in `tests/` covers rate addition.

**Fix**: convert `other` onto `self.unit` via the instantaneous rate rather than via `__mul__`, e.g. `self.__class__(self.value + self._convert_rate(other), self.unit)` for `per`/`freq` (`_convert_rate()` already does `other.rate*(self.unit/other.unit)`). The same goes for `__sub__()`. `prob` needs its own decision about whether addition means adding rates or probabilities, but it should not route through `per`'s probability conversion either.

## Medium severity

### 3. The date tick locator is never used on date axes — `time.py:2382`

`FloatYearLocator.__init__()` sets `self._convert_dates = unit == 'ss.date'`. That was right when `DateConverter.default_units()` returned the string `'ss.date'`. It now returns the class `ss.date` (and `axisinfo()` tests `issubclass(unit, ss.date)`), so the comparison is always `False`. Every date axis therefore falls back to a plain `AutoLocator` on float years. You get ticks at 2020.1, 2020.2, etc., which the formatter labels as arbitrary days ("Feb-07", "Mar-14"), instead of month or year boundaries.

```python
sim = ss.Sim(n_agents=1000, start='2020-01-01', stop='2020-07-01', dt=ss.days(7), diseases='sir', networks='random', verbose=0)
sim.run()
fig = sim.results.sir.n_infected.plot(); fig.canvas.draw()
ax = fig.axes[0]
ax.xaxis.get_major_locator()._convert_dates, [t.get_text() for t in ax.get_xticklabels()]
```

Actual: `False`, `['Nov-26\n2019', 'Jan-01\n2020', 'Feb-07', 'Mar-14', 'Apr-20', 'May-26', 'Jul-02', 'Aug-08']`. With `FloatYearLocator('ss.date')` substituted, the output is `['Jan\n2020', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']`.

Blast radius: every Starsim date-axis plot (`sim.plot()`, `Result.plot()`, and user `plt.plot(timevec, ...)`) on spans where the float ticks don't happen to land on year boundaries, i.e. any sim shorter than a few years. It is cosmetic but visible everywhere.

**Fix**: `self._convert_dates = isinstance(unit, type) and issubclass(unit, ss.date)`, optionally also accepting the legacy string.

### 4. `Rate.__eq__()` is wrong for probabilities — `time.py:1738`, `time.py:1744`

In the unitless branch, `self.value == other.value` is evaluated but its result is discarded (there is no `assert`), so any two unitless probs compare equal. In the unit branch, the comparison rescales `other.value` linearly (`other.value/other.unit*self.unit`). For `ss.prob` that contradicts the class's own conversion (`to()`, `to_prob()`, `_convert_rate()`), which goes through the underlying rate.

```python
ss.prob(0.1) == ss.prob(0.9)                       # actual True   expected False
p = ss.probperyear(0.5)
p == p.to(ss.probpermonth)                         # actual False  expected True  (probpermonth(0.0561257))
p == ss.probpermonth(0.5/12)                       # actual True   expected False
```

Blast radius: any code that compares probability parameters, e.g. checking whether a parameter was changed from its default, deduplicating, or asserting in tests. Unitless `ss.prob` is common for per-event probabilities such as `p_symp`.

**Fix**: in the unitless branch, `return other.unit is None and np.all(self.value == other.value)`. In the unit branch, compare `self.value` against `self._convert_rate(other)`, which dispatches correctly for `prob` vs `per`/`freq`.

## Low severity

### 5. `datedur` division still loses precision — `time.py:1511`

The comment in `datedur.__truediv__()` says "`datedur(weeks=1)/datedur(days=1)` should return 7, but we get 6.9999999 if we convert both to years", and describes a fix. But the loop still divides every component by `factor_vals[i]`, i.e. it still converts to years, so the result is unchanged.

```python
ss.datedur(weeks=1)/ss.datedur(days=1)   # actual 6.999999999999999   expected 7.0 (as the comment states)
ss.datedur(days=14)/ss.datedur(weeks=1)  # actual 2.0000000000000004
ss.datedur(weeks=1).to_dt(ss.datedur(days=1))  # 6.999999999999999; int() -> 6
```

Blast radius: users with a `datedur` timestep who floor or `int()` the result of `to_dt()` or of `datedur/datedur` get an off-by-one number of steps.

**Fix**: accumulate in the finest nonzero unit instead of in years. Take `finest = max(nonzero indices)`, then `a += self_array[i]*factor_vals[finest]/factor_vals[i]` (and the same for `b`), so whole weeks become exactly 7 days.

### 6. `ss.years()` on date-string arrays corrupts or mutates the input — `time.py:2185`

`years.__init__()` converts date strings with `value[i] = sc.datetoyear(val)`, writing into the caller's object. For a numpy string array the floats are coerced back to strings (`<U10`), so the resulting duration holds strings and crashes on first use. For a list, the conversion works but the user's list is modified in place.

```python
y = ss.years(np.array(['2020-01-01', '2021-07-02']))
repr(y), y.value.dtype  # years(['2020.0' '2021.49863']), <U10
y.years                 # UFuncTypeError: ufunc 'multiply' did not contain a loop ...
lst = ['2020-01-01', '2021-07-02']; ss.years(lst); lst   # [2020.0, 2021.4986301369863]
```

Blast radius: users building durations from date columns, e.g. from a pandas/numpy array of date strings. Uncommon, but the docstring comment explicitly offers "date-string input(s)".

**Fix**: build a new float array, e.g. `value = np.array([sc.datetoyear(v) if isinstance(v, to_convert) else v for v in value], dtype=float)`, rather than assigning into `value`.

### 7. Subtracting sub-daily dates drops the time of day — `time.py:1272`

`date.__sub__(date)` returns `ss.datedur(relativedelta(...))`, and the `relativedelta` branch of `datedur.__init__()` keeps only `years`, `months` and `days`. The hours, minutes and seconds are discarded, even though `datedur` otherwise supports them (e.g. `datedur(weeks=2.5)` has `+12:00:00`).

```python
ss.date('2020-01-02 06:00') - ss.date('2020-01-01')   # actual datedur(days=1), .days = 1.0   expected 1.25 days
ss.date('2020-01-01 12:00') - ss.date('2020-01-01')   # actual datedur(0)
```

Blast radius: only sims with explicitly sub-daily timesteps (for example `dt=ss.days(0.25)`) that difference dates. That is rare, but the answer is silently wrong.

**Fix**: pass all fields through, e.g. `pd.DateOffset(years=arg.years, months=arg.months, days=arg.days, hours=arg.hours, minutes=arg.minutes, seconds=arg.seconds)`.

### 8. `DateArray.to_date(inplace=True)` is a no-op on float arrays — `time.py:196`

A `DateArray` built from float years ≥ 1 gets `unit=ss.date` but a float64 buffer. `self[:] = vals` writes the `ss.date` objects back through `date.__float__()`, so the array is unchanged, and no error is raised.

```python
x = ss.DateArray(np.array([2020., 2021.]))
x.to_date(inplace=True); x   # actual [2020. 2021.]   expected DateArray([<2020.01.01>, <2021.01.01>])
```

Blast radius: small, since nothing inside Starsim calls it with `inplace=True`. It is a documented argument that silently does nothing.

**Fix**: when `self.dtype != object`, raise (an ndarray view can't change dtype in place), or document that `inplace` requires an object-dtype array. The same applies to `to_float(inplace=True)` on an object array that is later expected to be numeric.

## Verified clean

Tested and found correct or intended: `date` construction from years, strings, `pd.Timestamp`, `np.int64` and `day_round=False`; `from_year`/`to_year` round-trips; date comparisons and hashing against float years; `date ± datedur` with month-end clamping (`<2020.02.29>` both ways); `date - date` for whole days; `date.arange` with float, `ss.days(2)`, `ss.weeks(1)`, `ss.months(1)`, `'month'`, `inclusive=False` and `start=0` (datedur path); `dur.arange`; `datedur` rounding/cascading (`weeks=2.5`, `months=3/2`), `to_dur()` (docstring example `days(375)` holds), `str()`, `+`/`-` with `datedur` and `dur`, and conversion from `ss.days`/`ss.weeks`/`ss.months`; `dur` arithmetic across units (`days(10)+weeks(1)`, `5-years(2)`, `years(2)/days(365)`); `to_dt()` for durations and rates (docstring examples reproduce); `Rate.to()` for per/prob/freq including the non-linear prob conversion; `per*dur`, `per*scalar`, `prob*scalar` (matches the user guide's `prob(0.5)*2 == prob(0.75)`), `freq*dur`, `1/freq`, `Rate/Rate`; `prob.array_to_prob()`; `TimePar.replace()` preserving base and updating prob's rate; `DateArray` pickling/deepcopy preserving `unit`, `years`, `to_float()` and `to_human()`. `date + ss.days(n)` in a leap year lands one day late for n ≥ ~183 (e.g. 2020-01-01 + `days(365)` gives 2021-01-01), because arithmetic goes through float years with a 365-day year. This was not recorded because `advanced_time.qmd` documents that a year is 365 days with "stretching" to align calendar dates, and `ss.datedur` is the documented exact-calendar alternative.
