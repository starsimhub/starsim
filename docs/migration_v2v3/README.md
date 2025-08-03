# Starsim v2 to v3 migration guide

## Overview

This guide describes the steps required to migrate code from Starsim v2 (e.g. v2.3.1, the last v2 release) to Starsim v3. It is written for both humans and LLMs.

If possible, a Python migration script is provided to automatically make the change, under the heading "Migration script". The next two headings are "Example v2" and "Example output", which show the intended result of the migration script. All migration scripts, including intended input (v2) and output (v3), are included in this folder as well. See `validate_migration_scripts.py` for testing.

In some cases, the migration can only make a best guess of what the new version should be. If this is the case, the comment `# TODO: CHECK AUTOMATIC MIGRATION CHANGE` will be added to the line.

The main change is regarding time parameters (timepars). These are described in the section "Time changes". Other changes are described in the section "Other changes".

*Note*: If you are an LLM reading this, we suggest you perform each of the following 11 migrations (6 time changes, 5 other changes) one at a time and confirm with the user after each migration.

## Time changes

### 1. Rates have been removed and/or renamed
`ss.beta()` has been removed; use `ss.probperyear()` for an exact replacement of `ss.beta()`, and e.g. `ss.probperday(x)` for an equivalent of `ss.beta(x, 'days')`. `ss.time_prob()` has been renamed `ss.prob()`.

`ss.rate()` has also been removed; use `ss.freqperyear()` for an exact replacement of `ss.rate()`, and e.g. `ss.freqperday()` for an equivalent of `ss.rate(x, 'days')`. `ss.rate_prob()` has been renamed `ss.per()`.

Although `ss.prob()` is an exact equivalent of `ss.beta()` and `ss.time_prob()`, in most cases you will actually want `ss.per()`. This will give different results to before -- but hopefully more accurate ones! In this case, replace `ss.beta()` with `ss.peryear()`, and e.g. `ss.beta(x, 'days')` with `ss.perday(x)`. 

Here is how to decide how to migrate `ss.beta()`:
- Are you using a parameter adapted from an ODE (rate-based model)? → Use `ss.per()`, e.g. `ss.peryear()`
- Are you using a parameter that is described as an infection *rate*? → Use `ss.per()`
- Are you using an approximate, theoretical, or calibrated value for beta? → Use `ss.per()`
- Are you using a parameter described as a *probability of infection per unit time*, e.b. "probability of infection after one year"? → Use `ss.prob()`

Likewise, although `ss.freq()` is an exact equivalent of `ss.rate()`, in most cases you will actually want `ss.per()` (as with `ss.beta()`/`ss.time_prob()`). This will also give different results to before. In that case, replace `ss.rate()` with `ss.peryear()`, and e.g. `ss.rate(x, 'days')` with `ss.perday(x)`. Note that in Starsim v2, birth rates and death rates were defined as numbers of events per unit time (`ss.rate()`), whereas in v3 they are defined as probability rates per unit time (`ss.per()`). This gives slightly different -- but more accurate! -- results. (Note that in most cases the difference is very small: e.g. for a crude birth rate of 20 per 1000 per year, the difference between these two approaches is only about 1%.) The automatic migration script makes an assumption that if the provided rate is <1, then `ss.per()` is intended; if the rate is >1, it assumes `ss.freq()` is intended.

Here is how to decide how to migrate `ss.rate()`:
- Are you using a parameter for a *probability* of an event occuring, such as probability of death (death rate) or probability of birth (birth rate)? → Use `ss.per()`, e.g. `ss.peryear()`
- Are you using a parameter for a *number* of events occurring, such as number of sexual acts in a year? → Use `ss.freq()`, e.g. `ss.freqperyear()`

#### Migration script (`rates__script.py`)
```py
#!/usr/bin/env python3
import re
import sys
from pathlib import Path

def is_prob(value: str) -> bool:
    """Determine if a value is probably a probability (<1)."""
    try:
        return float(value) < 1
    except:
        return True  # Default to per() if unsure

def migrate_rates(text):
    """
    Migrate v2 ss.rate() and derived classes to appropriate v3 classes based on time units.
    
    - ss.beta:
        - ss.beta(x) -> ss.peryear(x)
        - ss.beta(x, 'days') -> ss.perday(x)
        - ss.beta(x, 'weeks') -> ss.perweek(x)
        - ss.beta(x, 'months') -> ss.permonth(x)
        - ss.beta(x, 'years') -> ss.peryear(x)
    - ss.rate_prob:
        - ss.rate_prob(x) -> ss.peryear(x)
        - ss.rate_prob(x, 'days') -> ss.perday(x)
    - ss.time_prob:
        - ss.time_prob(x) -> ss.probperyear(x)
        - ss.time_prob(x, 'days') -> ss.probperday(x) 
    - ss.rate:
        - For x<1:
            - ss.rate(x) -> ss.peryear(x)
            - ss.rate(x, 'days') -> ss.perday(x)
        - For x>=1:
            - ss.rate(x) -> ss.freqperyear(x)
            - ss.rate(x, 'days') -> ss.freqperday(x)
    """
    lines = text.splitlines()
    new_lines = []

    unit_map = {
        'days': 'perday',
        'weeks': 'perweek',
        'months': 'permonth',
        'years': 'peryear',
    }

    freq_map = {
        'days': 'freqperday',
        'weeks': 'freqperweek',
        'months': 'freqpermonth',
        'years': 'freqperyear',
    }

    # Patterns to match calls with optional time unit
    # Each entry is (pattern, replacement_func, token_name)
    patterns = [
        # Two-arg patterns first (must come first to prevent premature matching)
        (r'ss\.beta\s*\(\s*([^,]+)\s*,\s*[\'"](\w+)[\'"]\s*\)', lambda v, u: f"ss.{unit_map.get(u, 'peryear')}({v})", 'ss.beta'), # ss.beta(x, 'unit')
        (r'ss\.time_prob\s*\(\s*([^,]+)\s*,\s*[\'"](\w+)[\'"]\s*\)', lambda v, u: f"ss.prob{unit_map.get(u, 'peryear')}({v})", 'ss.time_prob'), # ss.time_prob(x, 'unit')
        (r'ss\.rate_prob\s*\(\s*([^,]+)\s*,\s*[\'"](\w+)[\'"]\s*\)', lambda v, u: f"ss.per{unit_map.get(u, 'year')}({v})", 'ss.rate_prob'), # ss.rate_prob(x, 'unit')
        (r'ss\.rate\s*\(\s*([^,]+)\s*,\s*[\'"](\w+)[\'"]\s*\)', lambda v, u: f"ss.{unit_map[u] if is_prob(v) else freq_map[u]}({v})", 'ss.rate'), # ss.rate(x, 'unit')

        # One-arg patterns second
        (r'ss\.beta\s*\(\s*([^)]+?)\s*\)', lambda v: f"ss.peryear({v})", 'ss.beta'), # ss.beta(x)
        (r'ss\.time_prob\s*\(\s*([^)]+?)\s*\)', lambda v: f"ss.probperyear({v})", 'ss.time_prob'), # ss.time_prob(x)
        (r'ss\.rate_prob\s*\(\s*([^)]+?)\s*\)', lambda v: f"ss.peryear({v})", 'ss.rate_prob'), # ss.rate_prob(x)
        (r'ss\.rate\s*\(\s*([^)]+?)\s*\)', lambda v: f"ss.peryear({v})" if is_prob(v) else f"ss.freqperyear({v})", 'ss.rate'), # ss.rate(x)
    ]

    for orig_line in lines:
        line = orig_line
        for pattern, repl_func, token in patterns:
            match = re.search(pattern, line)
            if not match:
                continue
            try:
                groups = match.groups()
                new_expr = repl_func(*[g.strip() for g in groups])
                line = re.sub(pattern, new_expr, line, count=1)
                # Only apply first successful match per line
                if line != orig_line:
                    line += f"  # TODO: Check automatic migration change for {token}"
                break
            except Exception:
                continue
        new_lines.append(line)

    return '\n'.join(new_lines)

# Entry point
files = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else Path('.').rglob('*.py')
for path in files:
    if not path.is_file():
        continue
    text = path.read_text()
    new_text = migrate_rates(text)
    if new_text != text:
        path.write_text(new_text)
        print(f"Updated {path}")
```

#### v2 (old) (`rates__v2.py`)
```py
import starsim as ss

sir = ss.SIR(beta=ss.beta(0.1))
sis = ss.SIS(beta=ss.beta(0.001, 'days'), waning=ss.rate(0.05))
bir = ss.Births(birth_rate=ss.rate(0.02))
net = ss.MFNet(acts=ss.rate(80))
sim = ss.Sim(demographics=bir, diseases=[sir, sis], networks=net)
```

#### v3 (new) (`rates__v3.py`)
```py
import starsim as ss

sir = ss.SIR(beta=ss.peryear(0.1))  # TODO: Check automatic migration change for ss.beta
sis = ss.SIS(beta=ss.perday(0.001), waning=ss.rate(0.05))  # TODO: Check automatic migration change for ss.beta
bir = ss.Births(birth_rate=ss.peryear(0.02))  # TODO: Check automatic migration change for ss.rate
net = ss.MFNet(acts=ss.freqperyear(80))  # TODO: Check automatic migration change for ss.rate
sim = ss.Sim(demographics=bir, diseases=[sir, sis], networks=net)
```

### 2. The `'unit'` argument has been removed
`unit` has been removed as an argument for sims and modules (and `ss.Timeline()`); use `dt` instead, e.g. `ss.Sim(dt=1, unit='years')` is now `ss.Sim(dt=ss.year)` (or `ss.Sim(dt='years')` or `ss.Sim(dt=ss.years(1))`).

If you have `unit=<x>` in v2 code, migrate it to v3 code as follows:
- If `dt` is not defined or `dt=1`: change `unit=<x>` to `dt=<x>`, e.g. `unit='years'` to `dt='years'`
- If `dt=<y>`, change `unit=<x>, dt=<y>` to `dt=ss.<x>(<y>)`, e.g. `dt=2, unit='days'` to `dt=ss.days(2)`

#### Migration script (`unit__script.py`)
```py
#!/usr/bin/env python3
"""
Migrate v2 unit arguments to v3 dt arguments.

Rules:
- If dt is not defined or dt=1: change unit=<x> to dt=<x>
- If dt=<y>, change unit=<x>, dt=<y> to dt=ss.<x>(<y>)

Examples:
- ss.SIS(unit='day', dt=1.0) -> ss.SIS(dt=ss.days(1))
- ss.Births(unit='year', dt=0.25) -> ss.Births(dt=ss.years(0.25))
- ss.RandomNet(unit='week') -> ss.RandomNet(dt='week')
- ss.Sim(unit='day', dt=2) -> ss.Sim(dt=ss.days(2))
"""
import re
import sys
from pathlib import Path

unit_map = {
    'day': 'days',
    'week': 'weeks',
    'month': 'months',
    'year': 'years',
}

# Choose input files
files = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else Path('.').rglob('*.py')

# Handles both orderings: unit then dt, or dt then unit
pattern_unit_dt = re.compile(r'unit\s*=\s*[\'"](\w+)[\'"]\s*,\s*dt\s*=\s*([\d\.]+)')
pattern_dt_unit = re.compile(r'dt\s*=\s*([\d\.]+)\s*,\s*unit\s*=\s*[\'"](\w+)[\'"]')
pattern_unit_only = re.compile(r'\bunit\s*=\s*[\'"](\w+)[\'"]') # Standalone unit

for path in files:
    text = path.read_text()
    original = text

    # Replace unit='x', dt=y → dt=ss.xs(y)
    def repl_unit_dt(m):
        unit, dt = m.group(1), m.group(2)
        ss_unit = unit_map.get(unit, unit + 's')
        return f'dt=ss.{ss_unit}({dt})'

    # Replace dt=y, unit='x' → dt=ss.xs(y)
    def repl_dt_unit(m):
        dt, unit = m.group(1), m.group(2)
        ss_unit = unit_map.get(unit, unit + 's')
        return f'dt=ss.{ss_unit}({dt})'

    # Replace unit='x' → dt='x' (assumes dt=1)
    def repl_unit_only(m):
        unit = m.group(1)
        return f"dt='{unit}'"

    text = pattern_unit_dt.sub(repl_unit_dt, text)
    text = pattern_dt_unit.sub(repl_dt_unit, text)
    text = pattern_unit_only.sub(repl_unit_only, text)

    if text != original:
        path.write_text(text)
        print(f'Updated {path}')
```

#### v2 (old) (`unit__v2.py`)
```py
import starsim as ss

pars = dict(
    diseases = ss.SIS(unit='day', dt=1.0, init_prev=0.1),
    demographics = ss.Births(unit='year', dt=0.25),
    networks = ss.RandomNet(unit='week'),
)

sim = ss.Sim(pars, unit='day', dt=2, start='2000-01-01', stop='2002-01-01')
sim.run()
```

#### v3 (new) (`unit__v3.py`)
```py
import starsim as ss

pars = dict(
    diseases = ss.SIS(dt=ss.days(1), init_prev=0.1),
    demographics = ss.Births(dt=ss.years(0.25)),
    networks = ss.RandomNet(dt='week'),
)

sim = ss.Sim(pars, dt=ss.days(2), start='2000-01-01', stop='2002-01-01')
sim.run()
```

### 3. Multiplication by `dt` is no longer automatic
*Note: no automatic migration script is provided for this change as the code is likely to need refactoring in unpredicable ways.*

Multiplication by `dt` no longer happens automatically; call `to_prob()` to explicitly convert from a timepar to a unitless quantity (or `to_events()` to convert to a number of events instead). This is done so that the correct intermediate calculations are carried through until the final step, preventing probabilities from going <0 or >1.

Note: although you _can_ supply `dt` as an explicit argument to `to_prob()`, if the rate is part of a module (which is almost always the case), then it will already be initialized with a `default_dur` equal to the module's `dt`. In this case, calling `par.to_prob()` and `par.to_prob(self.dt)` are identical.

For example, code such as this:
```py
beta_per_dt = route.net_beta(disease_beta=beta) # From disease.py
new_bacteria = p.shedding_rate * (n_symptomatic + p.asymp_trans * n_asymptomatic) # From starsim/diseases/cholera.py
old_bacteria = old_prev * (1 - p.decay_rate) # From starsim/diseases/cholera.py
p_transmit = res.env_conc[self.ti] * pars.beta_env # From starsim/diseases/cholera.py
```
should be migrated to this:
```py
beta_per_dt = route.net_beta(disease_beta=beta.to_prob(self.t.dt)) # From diseases.py
new_bacteria = (p.shedding_rate * n_symptomatic + p.asymp_trans * n_asymptomatic).to_prob() # From starsim_examples/diseases/cholera.py
old_bacteria = old_prev * np.exp(-p.decay_rate.to_prob()) # From starsim_examples/diseases/cholera.py
p_transmit = (res.env_conc[self.ti] * pars.beta_env).to_prob() # From starsim_examples/diseases/cholera.py
```

### 4. `ss.time_ratio()` has been removed
*Note: no automatic migration script is provided for this change as the code is likely to need refactoring in unpredicable ways.*

`ss.time_ratio()` has been removed; time unit ratio calculations (e.g. months to years) are now handled internally by timepars.

For example, code such as this (from `demographics.py`):
```py
if isinstance(this_birth_rate, ss.TimePar):
    factor = 1.0
else:
    factor = ss.time_ratio(unit1=self.t.unit, dt1=self.t.dt, unit2='year', dt2=1.0)

scaled_birth_prob = this_birth_rate * p.rate_units * p.rel_birth * factor
```
should be migrated to this:
```py
scaled_birth_prob = (this_birth_rate * p.rate_units * p.rel_birth).to_prob()
```

### 5. Timepars no longer require `.init()`
*Note: no automatic migration script is provided for this change as it is relatively straightforward.*

Previously, timepars required knowledge of a parent (typically the module) in order to perform calculations. Since they now use absolute time, they no longer need to be initialized.

For example, code such as this:
```py
param = ss.peryear(1.5).init(parent=self.sim.t)
```
should be migrated to this:
```py
param = ss.peryear(1.5)
```

### 6. `ss.Time()` has been renamed, and `abstvec` has been removed
*Note: no automatic migration script is provided for these changes as they are unlikely to affect many users.*

`ss.Time()` is now called `ss.Timeline()`. Its internal calculations are also handled differently, although this should not affect the user.

`ss.Timeline.abstvec` (commonly accessed as `t.abstvec`) has been removed; in most cases, `t.tvec` should be used instead (although `t.yearvec`, `t.datevec` or `t.timevec` may be preferable in some cases).


## Other changes

### 1. Example diseases and networks have moved
These classes have been moved from the `starsim` namespace to the `starsim_examples` namespace:
```py
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

#### v2 (old) (`starsim_examples__v2.py`)
```py
import starsim as ss

hiv = ss.HIV()
net = ss.DiskNet()
sim = ss.Sim(diseases=hiv, networks=net).run()
```

#### v3 (new) (`starsim_examples__v3.py`)
```py
import starsim as ss
import starsim_examples as sse

hiv = sse.HIV()
net = sse.DiskNet()
sim = ss.Sim(diseases=hiv, networks=net).run()
```

### 2. `ss.State` has been renamed to `ss.BoolState`
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

#### v2 (old) (`boolstate__v2.py`)
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

#### v3 (new) (`boolstate__v3.py`)
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

### 3. `ss.MixingPool(contacts=)` has been renamed `n_contacts`
*Note: no automatic migration script is available for this change since "contacts" is too ambiguous a name.*

For `ss.MixingPool()` and `ss.MixingPools()`, the argument `contacts` has been renamed `n_contacts`.

For example, code such as this:
```py
import starsim as ss

mp_pars = dict(
    src = ss.AgeGroup(0, 15),
    dst = ss.AgeGroup(15, None),
    contacts = ss.poisson(lam=5),
)
mp = ss.MixingPool(**mp_pars)
```
should be migrated to this:
```py
import starsim as ss

mp_pars = dict(
    src = ss.AgeGroup(0, 15),
    dst = ss.AgeGroup(15, None),
    n_contacts = ss.poisson(lam=5),
)
mp = ss.MixingPool(**mp_pars)
```

### 4. `sim.modules` and `module.states` have been renamed

*Note: no automatic migration script is available for this change.*

- `ss.Sim.modules` has been renamed `ss.Sim.module_list`. (Note: `ss.Sim.modules` is now something _different_, so be especially careful with this change.)
- `ss.Module.states` has been renamed `ss.Module.state_list`. `module.statesdict` has been renamed `module.state_dict`.
- These attributes are used in Starsim Core but are rarely used in user code, so are unlikely to need to be migrated.

### 5. `key_dict` has been removed from `ss.Network`

*Note: no automatic migration script is available for this change.*

Previously, extra edge attributes were set on networks via a `key_dict` argument. Now, the `self.meta` dictionary is simply updated directly.

For example, code such as this:
```py
def __init__(self, pars=None, key_dict=None, name=None, **kwargs):
    key_dict = sc.mergedicts({
        'age_p1': ss_float_,
        'age_p2': ss_float_,
        'edge_type': ss_int_,
    }, key_dict)

    super().__init__(key_dict=key_dict, name=name)
```
should be migrated to this:
```py
def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.meta.age_p1 = ss_float
    self.meta.age_p2 = ss_float
    self.meta.edge_type = ss_int
```


## Additional code examples

This section contains additional examples of code before and after porting from v2 to v3.

### Example 1, beta definition

#### v2 (old) (`example_beta__v2.py`)
```py
sis = ss.SIS(beta={'random':[0.005, 0.001], 'prenatal':[0.1, 0], 'postnatal':[0.1, 0]})
```

#### v3 (new) (`example_beta__v3.py`)
```py
sis = ss.SIS(
    beta = dict(
        random = ss.permonth([0.005, 0.001]),
        prenatal = [ss.permonth(0.1), 0],
        postnatal = [ss.permonth(0.1), 0]
    )
)
```

### Example 2, sim definition

#### v2 (old) (`example_sim__v2.py`)
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

#### v3 (new) (`example_sim__v3.py`)
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

#### v2 (old) (`example_parameters__v2.py`)
```py
pars = dict(
    diseases = ss.SIS(unit='day', dt=1.0, init_prev=0.1, beta=ss.beta(0.01)),
    demographics = ss.Births(unit='year', dt=0.25),
    networks = ss.RandomNet(unit='week'),
    n_agents = small,
    verbose = 0,
)
```

#### v3 (new) (`example_parameters__v3.py`)
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

#### v2 (old) (`example_time_units__v2.py`)
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

#### v3 (new) (`example_time_units__v3.py`)
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

#### v2 (old) (`example_custom_module__v2.py`)
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

#### v3 (new) (`example_custom_module__v3.py`)
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