#!/usr/bin/env python3
import re
import sys
from pathlib import Path

def migrate_rates(text):
    """
    Migrate v2 ss.rate() and derived calls to appropriate v3 functions based on time units.
    
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
    lines = text.split('\n')
    new_lines = []
    
    for orig_line in lines:
        line = orig_line
        changes_made = []
        
        # Handle ss.beta() calls
        # Pattern for ss.beta(x, 'unit')
        pattern_beta_with_unit = r'ss\.beta\s*\(\s*([^,]+)\s*,\s*[\'"]([^\'"]+)[\'"]\s*\)'
        
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
                out = f"ss.peryear({value})"
            else:
                function_name = unit_mapping[unit]
                out = f"ss.{function_name}({value})"
            
            return out
        
        # Apply the pattern with unit first
        if re.search(pattern_beta_with_unit, line):
            line = re.sub(pattern_beta_with_unit, replace_beta_with_unit, line)
            changes_made.append('BETA')
        
        # Then handle cases without time units (ss.beta(x))
        pattern_beta_without_unit = r'ss\.beta\s*\(\s*([^)]+)\s*\)'
        
        def replace_beta_without_unit(match):
            value = match.group(1).strip()
            return f"ss.peryear({value})"
        
        # Apply the pattern without unit
        if re.search(pattern_beta_without_unit, line):
            line = re.sub(pattern_beta_without_unit, replace_beta_without_unit, line)
            if 'BETA' not in changes_made:
                changes_made.append('BETA')
        
        # Handle ss.rate() calls
        # Pattern for ss.rate(x, 'unit')
        pattern_rate_with_unit = r'ss\.rate\s*\(\s*([^,]+)\s*,\s*[\'"]([^\'"]+)[\'"]\s*\)'
        
        def replace_rate_with_unit(match):
            value = match.group(1).strip()
            unit = match.group(2)
            
            # Try to evaluate the value to determine if it's < 1 or >= 1
            try:
                num_value = float(value)
                if num_value < 1:
                    # Use ss.per* for probabilities
                    unit_mapping = {
                        'days': 'perday',
                        'weeks': 'perweek', 
                        'months': 'permonth',
                        'years': 'peryear'
                    }
                    if unit not in unit_mapping:
                        out = f"ss.peryear({value})"
                    else:
                        function_name = unit_mapping[unit]
                        out = f"ss.{function_name}({value})"
                else:
                    # Use ss.freq* for frequencies
                    unit_mapping = {
                        'days': 'freqperday',
                        'weeks': 'freqperweek', 
                        'months': 'freqpermonth',
                        'years': 'freqperyear'
                    }
                    if unit not in unit_mapping:
                        out = f"ss.freqperyear({value})"
                    else:
                        function_name = unit_mapping[unit]
                        out = f"ss.{function_name}({value})"
            except (ValueError, TypeError):
                # If we can't evaluate, default to peryear
                unit_mapping = {
                    'days': 'perday',
                    'weeks': 'perweek', 
                    'months': 'permonth',
                    'years': 'peryear'
                }
                if unit not in unit_mapping:
                    out = f"ss.peryear({value})"
                else:
                    function_name = unit_mapping[unit]
                    out = f"ss.{function_name}({value})"
            
            return out
        
        # Apply the pattern with unit first
        if re.search(pattern_rate_with_unit, line):
            line = re.sub(pattern_rate_with_unit, replace_rate_with_unit, line)
            changes_made.append('RATE')
        
        # Then handle cases without time units (ss.rate(x))
        pattern_rate_without_unit = r'ss\.rate\s*\(\s*([^)]+)\s*\)'
        
        def replace_rate_without_unit(match):
            value = match.group(1).strip()
            
            # Try to evaluate the value to determine if it's < 1 or >= 1
            try:
                num_value = float(value)
                if num_value < 1:
                    return f"ss.peryear({value})"
                else:
                    return f"ss.freqperyear({value})"
            except (ValueError, TypeError):
                # If we can't evaluate, default to peryear
                return f"ss.peryear({value})"
        
        # Apply the pattern without unit
        if re.search(pattern_rate_without_unit, line):
            line = re.sub(pattern_rate_without_unit, replace_rate_without_unit, line)
            if 'RATE' not in changes_made:
                changes_made.append('RATE')
        
        # Handle ss.time_prob() calls
        # Pattern for ss.time_prob(x, 'unit')
        pattern_time_prob_with_unit = r'ss\.time_prob\s*\(\s*([^,]+)\s*,\s*[\'"]([^\'"]+)[\'"]\s*\)'
        
        def replace_time_prob_with_unit(match):
            value = match.group(1).strip()
            unit = match.group(2)
            
            # Map time units to appropriate ss.prob* functions
            unit_mapping = {
                'days': 'probperday',
                'weeks': 'probperweek', 
                'months': 'probpermonth',
                'years': 'probperyear'
            }
            
            if unit not in unit_mapping:
                out = f"ss.probperyear({value})"
            else:
                function_name = unit_mapping[unit]
                out = f"ss.{function_name}({value})"
            
            return out
        
        # Apply the pattern with unit first
        if re.search(pattern_time_prob_with_unit, line):
            line = re.sub(pattern_time_prob_with_unit, replace_time_prob_with_unit, line)
            changes_made.append('TIME_PROB')
        
        # Then handle cases without time units (ss.time_prob(x))
        pattern_time_prob_without_unit = r'ss\.time_prob\s*\(\s*([^)]+)\s*\)'
        
        def replace_time_prob_without_unit(match):
            value = match.group(1).strip()
            return f"ss.probperyear({value})"
        
        # Apply the pattern without unit
        if re.search(pattern_time_prob_without_unit, line):
            line = re.sub(pattern_time_prob_without_unit, replace_time_prob_without_unit, line)
            if 'TIME_PROB' not in changes_made:
                changes_made.append('TIME_PROB')
        
        # Handle ss.rate_prob() calls
        # Pattern for ss.rate_prob(x, 'unit')
        pattern_rate_prob_with_unit = r'ss\.rate_prob\s*\(\s*([^,]+)\s*,\s*[\'"]([^\'"]+)[\'"]\s*\)'
        
        def replace_rate_prob_with_unit(match):
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
                out = f"ss.peryear({value})"
            else:
                function_name = unit_mapping[unit]
                out = f"ss.{function_name}({value})"
            
            return out
        
        # Apply the pattern with unit first
        if re.search(pattern_rate_prob_with_unit, line):
            line = re.sub(pattern_rate_prob_with_unit, replace_rate_prob_with_unit, line)
            changes_made.append('RATE_PROB')
        
        # Then handle cases without time units (ss.rate_prob(x))
        pattern_rate_prob_without_unit = r'ss\.rate_prob\s*\(\s*([^)]+)\s*\)'
        
        def replace_rate_prob_without_unit(match):
            value = match.group(1).strip()
            return f"ss.peryear({value})"
        
        # Apply the pattern without unit
        if re.search(pattern_rate_prob_without_unit, line):
            line = re.sub(pattern_rate_prob_without_unit, replace_rate_prob_without_unit, line)
            if 'RATE_PROB' not in changes_made:
                changes_made.append('RATE_PROB')
        
        # Add warning message if line was changed
        if changes_made:
            line += f"  # TODO: CHECK AUTOMATIC MIGRATION CHANGE FOR {' AND '.join(changes_made)}"
        
        new_lines.append(line)
    
    return '\n'.join(new_lines)

files = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else Path('.').rglob('*.py')

for path in files:
    text = path.read_text()
    new_text = migrate_rates(text)
    if new_text != text:
        path.write_text(new_text)
        print(f"Updated {path}")