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