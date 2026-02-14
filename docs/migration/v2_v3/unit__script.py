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