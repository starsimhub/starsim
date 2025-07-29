# Starsim v2 to v3 migration guide

This guide describes the steps required to migrate code from Starsim v2 (e.g. v2.3.1, the last v2 release) to Starsim v3. It is written for both humans and LLMs.

If possible, a Python migration script is provided to automatically make the change, under the heading "Migration script". The next two headings are "Example input" and "Example output", which show the intended result of the migration script. All migration scripts, including intended input and output, are included in this folder as well. See `validate_migration_scripts.py` for testing.

If immediately below the migration script you find the text "Additional changes", please make those (non-automated) changes too.

## Overview of changes

The main change is regarding time parameters (timepars). These are described in the section "Time changes". Other changes are described in the section "Other changes".

## Time changes

TBC

- `ss.beta()` has been removed; use `ss.prob()` instead for a literal equivalent, although in most cases `ss.per()` is preferable, e.g. `ss.peryear()`.  **#TODOMIGRATION**
- `ss.rate()` has been removed; use `ss.freq()` instead for a literal equivalent, although in most cases `ss.per()` is preferable, e.g. `ss.peryear()`.  **#TODOMIGRATION**
- `unit` has been removed as an argument; use `dt` instead, e.g. `ss.Sim(dt=1, unit='years')` is now `ss.Sim(dt=ss.year)` (or `ss.Sim(dt='years')` or `ss.Sim(dt=ss.years(1))`).  **#TODOMIGRATION**
- Although `ss.dur()` still exists in Starsim v3.0, it is preferable to use named classes instead, e.g. `ss.years(3)` instead of `ss.dur(3, 'years')`.  **#TODOMIGRATION**
- `ss.Time()` is now called `ss.Timeline()` and its internal calculations are handled differently.  **#TODOMIGRATION**
- `ss.time_ratio()` has been removed; time unit ratio calculations (e.g. months to years) are now handled internally by timepars.
- `t.abstvec` has been removed; in most cases, `t.yearvec` should be used instead (although `t.datevec` or `t.timevec` may be preferable in some cases).
- Multiplication by `dt` no longer happens automatically; call `to_prob()` or `p()` to convert from a timepar to a unitless quantity (or `to_events()` or `n()` to convert to a number of events instead). **#TODOMIGRATION**




## Other changes

### Example diseases and networks have moved

These classes have been moved from the `starsim` namespace to the `starsim_examples` namespace:
```
['ART', 'CD4_analyzer', 'Cholera', 'DiskNet', 'Ebola', 'EmbeddingNet', 'ErdosRenyiNet', 'Gonorrhea', 'HIV', 'Measles', 'NullNet', 'Syphilis', 'syph_screening', 'syph_treatment']
```

#### Migration script (`starsim_examples__script.py`)
```py
#!/usr/bin/env python3
import re
import sys
from pathlib import Path

names = [
    'ART', 'CD4_analyzer', 'Cholera', 'DiskNet', 'Ebola', 'EmbeddingNet',
    'ErdosRenyiNet', 'Gonorrhea', 'HIV', 'Measles', 'NullNet',
    'Syphilis', 'syph_screening', 'syph_treatment'
]

pattern = re.compile(r'\bss\.(?:' + '|'.join(map(re.escape, names)) + r')\b')
OLD = 'ss.'
NEW = 'sse.'

files = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else Path('.').rglob('*.py')

for path in files:
    text = path.read_text(encoding='utf-8')
    new_text = pattern.sub(lambda m: m.group(0).replace(OLD, NEW), text)

    if new_text != text:
        lines = new_text.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == 'import starsim as ss':
                lines.insert(i + 1, 'import starsim_examples as sse')
                break
        new_text = '\n'.join(lines)
        path.write_text(new_text, encoding='utf-8')
        print(f"Updated {path}")
```

#### Example input (`starsim_examples__input.py`)
```py
import starsim as ss

hiv = ss.HIV()
net = ss.DiskNet()
sim = ss.Sim(diseases=hiv, networks=net).run()
```

#### Example output (`starsim_examples__output.py`)
```py
import starsim as ss
import starsim_examples as sse

hiv = sse.HIV()
net = sse.DiskNet()
sim = ss.Sim(diseases=hiv, networks=net).run()
```

### `ss.State` has been renamed to `ss.BoolState`

The class `ss.State` has been renamed `ss.BoolState`, with no significant change in functionality.

#### Migration script (`boolstate__script.py`)
```py
#!/usr/bin/env python3
import sys
from pathlib import Path

files = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else Path('.').rglob('*.py')

for path in files:
    text = path.read_text()
    new_text = text.replace('ss.State', 'ss.BoolState')
    if new_text != text:
        path.write_text(new_text)
        print(f"Updated {path}")
```

#### Example input (`boolstate__input.py`)
```py
import starsim as ss

class MySIR:
    def __init__(self):
        super().__init__()
        self.define_states(
            ss.FloatArr('my_attr'),
            ss.State('mystate', label='My new state'),
        )
        return
```

#### Example output (`boolstate__output.py`)
```py
import starsim as ss

class MySIR:
    def __init__(self):
        super().__init__()
        self.define_states(
            ss.FloatArr('my_attr'),
            ss.BoolState('mystate', label='My new state'),
        )
        return
```

## Additional code examples

This section contains additional examples of code before and after porting from v2 to v3.

### Example 1, beta definition

#### v2 (old)
```py
sis = ss.SIS(beta={'random':[0.005, 0.001], 'prenatal':[0.1, 0], 'postnatal':[0.1, 0]})
```

#### v3 (new)
```py
sis = ss.SIS(
    beta = dict(
        random = [ss.permonth(0.005)]*2,
        prenatal = [ss.permonth(0.1), 0],
        postnatal = [ss.permonth(0.1), 0]
    )
)
```

### Example 2, sim definition

#### v2 (old)
```py
sim = ss.Sim(
    n_agents = 1000,
    pars = dict(
      networks = dict(
        type = 'random',
        n_contacts = 4
      ),
      diseases = dict(
        type      = 'sir',
        init_prev = 0.01,
        dur_inf   = ss.dur(10),
        p_death   = 0,
        beta      = ss.beta(0.06),
      )
    ),
    dur = 10,
    dt  = 0.01
)
```

#### v3 (new)
```py
sim = ss.Sim(
    n_agents = 1000,
    pars = dict(
      networks = dict(
        type = 'random',
        n_contacts = 4
      ),
      diseases = dict(
        type      = 'sir',
        init_prev = 0.01,
        dur_inf   = ss.years(10),
        p_death   = 0,
        beta      = ss.peryear(0.06),
      )
    ),
    dur = ss.years(10),
    dt  = ss.years(0.05)
)
```

### Example 3, parameters definition

#### v2 (old)
```py
pars = dict(
    diseases = ss.SIS(unit='day', dt=1.0, init_prev=0.1, beta=ss.beta(0.01)),
    demographics = ss.Births(unit='year', dt=0.25),
    networks = ss.RandomNet(unit='week'),
    n_agents = small,
    verbose = 0,
)
```

#### v3 (new)
```py
pars = dict(
    diseases = ss.SIS(dt=ss.days(1), init_prev=0.1, beta=ss.peryear(0.01)),
    demographics = ss.Births(dt=0.25),
    networks = ss.RandomNet(dt=ss.weeks(1)),
    n_agents = small,
    verbose = 0,
)
```

### Example 4, time units

#### v2 (old)
```py
siskw = dict(dur_inf=ss.dur(50, 'day'), beta=ss.beta(0.01, 'day'), waning=ss.rate(0.005, 'day'))
kw = dict(n_agents=1000, start='2001-01-01', stop='2001-07-01', networks='random', copy_inputs=False, verbose=0)

print('Year-year')
sis1 = ss.SIS(unit='year', dt=1/365, **sc.dcp(siskw))
sim1 = ss.Sim(unit='year', dt=1/365, diseases=sis1, label='year-year', **kw)

print('Day-day')
sis2 = ss.SIS(unit='day', dt=1.0, **sc.dcp(siskw))
sim2 = ss.Sim(unit='day', dt=1.0, diseases=sis2, label='day-day', **kw)

print('Day-year')
sis3 = ss.SIS(unit='day', dt=1.0, **sc.dcp(siskw))
sim3 = ss.Sim(unit='year', dt=1/365, diseases=sis3, label='day-year', **kw)

print('Year-day')
sis4 = ss.SIS(unit='year', dt=1/365, **sc.dcp(siskw))
sim4 = ss.Sim(unit='day', dt=1.0, diseases=sis4, label='year-day', **kw)
```

#### v3 (new)
```py
siskw = dict(dur_inf=ss.datedur(days=50), beta=ss.perday(0.01), waning=ss.perday(0.005))
kw = dict(n_agents=1000, start='2001-01-01', stop='2001-07-01', networks='random', copy_inputs=False, verbose=0)

print('Year-year')
sis1 = ss.SIS(dt=1/365, **sc.dcp(siskw))
sim1 = ss.Sim(dt=1/365, diseases=sis1, label='year-year', **kw)

print('Day-day')
sis2 = ss.SIS(dt=ss.days(1), **sc.dcp(siskw))
sim2 = ss.Sim(dt=ss.days(1), diseases=sis2, label='day-day', **kw)

print('Day-year')
sis3 = ss.SIS(dt=ss.days(1), **sc.dcp(siskw))
sim3 = ss.Sim(dt=1/365, diseases=sis3, label='day-year', **kw)

print('Year-day')
sis4 = ss.SIS(dt=1/365, **sc.dcp(siskw))
sim4 = ss.Sim(dt=ss.days(1), diseases=sis4, label='year-day', **kw)
```

### Example 5, custom disease module

#### v2 (old)
```py
class MySIR(ss.Infection):
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            beta = ss.beta(0.1),
            init_prev = ss.bernoulli(p=0.01),
            dur_inf = ss.lognorm_ex(mean=ss.dur(6)),
            p_death = ss.bernoulli(p=0.01),
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.State('susceptible', default=True, label='Susceptible'),
            ss.State('infected', label='Infectious'),
            ss.State('recovered', label='Recovered'),
            ss.FloatArr('ti_infected', label='Time of infection'),
            ss.FloatArr('ti_recovered', label='Time of recovery'),
            ss.FloatArr('ti_dead', label='Time of death'),
            ss.FloatArr('rel_sus', default=1.0, label='Relative susceptibility'),
            ss.FloatArr('rel_trans', default=1.0, label='Relative transmission'),
        )
        return
```

#### v3 (new)
```py
class MySIR(ss.Infection):
    def __init__(self, pars=None, beta=_, init_prev=_, dur_inf=_, p_death=_, **kwargs):
        super().__init__()
        self.define_pars(
            beta = ss.peryear(0.1),
            init_prev = ss.bernoulli(p=0.01),
            dur_inf = ss.lognorm_ex(mean=ss.years(6)),
            p_death = ss.bernoulli(p=0.01),
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.BoolState('susceptible', default=True, label='Susceptible'),
            ss.BoolState('infected', label='Infectious'),
            ss.BoolState('recovered', label='Recovered'),
            ss.FloatArr('ti_infected', label='Time of infection'),
            ss.FloatArr('ti_recovered', label='Time of recovery'),
            ss.FloatArr('ti_dead', label='Time of death'),
            ss.FloatArr('rel_sus', default=1.0, label='Relative susceptibility'),
            ss.FloatArr('rel_trans', default=1.0, label='Relative transmission'),
        )
        return
```