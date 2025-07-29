#!/usr/bin/env python3
import re
import sys
from pathlib import Path

def migrate_beta(text):
    """
    Migrate ss.beta() calls to appropriate ss.per*() functions based on time units.
    
    Handles:
    - ss.beta(x) -> ss.peryear(x)
    - ss.beta(x, 'days') -> ss.perday(x)
    - ss.beta(x, 'weeks') -> ss.perweek(x)
    - ss.beta(x, 'months') -> ss.permonth(x)
    - ss.beta(x, 'years') -> ss.peryear(x)
    """
    
    lines = text.split('\n')
    new_lines = []
    
    for line in lines:
        # Pattern for ss.beta(x, 'unit')
        pattern_with_unit = r'ss\.beta\s*\(\s*([^,]+)\s*,\s*[\'"]([^\'"]+)[\'"]\s*\)'
        
        def replace_beta_with_unit(match):
            value = match.group(1).strip()
            unit = match.group(2)
            
            # Map time units to appropriate ss.per* functions
            unit_mapping = {
                'days': 'perday',
                'weeks': 'perweek', 
                'months': 'permonth',
                'years': 'peryear'
            }
            
            if unit not in unit_mapping:
                # Unknown unit, keep as peryear but add TODO comment
                return f"ss.peryear({value})  # TODO: CHECK AUTOMATIC MIGRATION CHANGE - unknown unit '{unit}'"
            
            function_name = unit_mapping[unit]
            return f"ss.{function_name}({value})"
        
        # Apply the pattern with unit first
        line = re.sub(pattern_with_unit, replace_beta_with_unit, line)
        
        # Then handle cases without time units (ss.beta(x))
        pattern_without_unit = r'ss\.beta\s*\(\s*([^)]+)\s*\)'
        
        def replace_beta_without_unit(match):
            value = match.group(1).strip()
            return f"ss.peryear({value})"
        
        # Apply the pattern without unit
        line = re.sub(pattern_without_unit, replace_beta_without_unit, line)
        
        new_lines.append(line)
    
    return '\n'.join(new_lines)

files = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else Path('.').rglob('*.py')

for path in files:
    text = path.read_text()
    new_text = migrate_beta(text)
    if new_text != text:
        path.write_text(new_text)
        print(f"Updated {path}") 