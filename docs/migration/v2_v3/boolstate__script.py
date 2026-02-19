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