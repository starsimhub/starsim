# Starsim v2 to v3 migration guide

This guide describes the steps required to migrate code from Starsim v2 (e.g. v2.3.1, the last v2 release) to Starsim v3. It is written for both humans and LLMs.

If possible, a Python migration script is provided to automatically make the change, under the heading "Migration script". The next two headings are "Example input" and "Example output", which show the intended result of the migration script. All migration scripts, including intended input and output, are included in this folder as well. See `validate_migration_scripts.py` for testing.

If immediately below the migration script you find the text "Additional changes", please make those (non-automated) changes too.

## Overview of changes

The main change is regarding time parameters (timepars). These are described in the section "Time changes". Other changes are described in the section "Other changes".

## Time changes

TBC

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
from pathlib import Path

names = [
    'ART', 'CD4_analyzer', 'Cholera', 'DiskNet', 'Ebola', 'EmbeddingNet',
    'ErdosRenyiNet', 'Gonorrhea', 'HIV', 'Measles', 'NullNet',
    'Syphilis', 'syph_screening', 'syph_treatment'
]

pattern = re.compile(r'\bss\.(?:' + '|'.join(map(re.escape, names)) + r')\b')
OLD = 'ss.'
NEW = 'sse.'

for path in Path('.').rglob('*.py'):
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