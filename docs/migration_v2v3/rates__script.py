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
