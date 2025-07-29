#!/usr/bin/env python3
"""
Check that the migration script examples work (written mostly by ChatGPT)
"""

import shutil
import tempfile
import difflib
import sciris as sc

root = sc.path(".")

# Find all base names that have __script.py, which should have matching __v2.py and __v3.py files
triplets = []
for script_file in sorted(root.rglob("*__script.py")):
    base = script_file.stem.replace("__script", "")
    input_file = script_file.with_name(f"{base}__v2.py")
    output_file = input_file.with_name(f"{base}__v3.py")
    if input_file.exists() and output_file.exists():
        triplets.append((base, input_file, output_file, script_file))
    else:
        raise FileNotFoundError(f'Expecting {input_file} and {output_file}')

for i, (base, input_file, output_file, script_file) in enumerate(triplets):
    expected = output_file.read_text().splitlines(keepends=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        print(sc.ansi.green(f"Checking example {base} ({i+1} of {len(triplets)}) in {tmpdir}/..."))

        tmpdir = sc.path(tmpdir)
        tmp_example = tmpdir / f"{base}__example.py"
        tmp_script = tmpdir / f"{base}__script.py"

        # Copy files
        shutil.copy(input_file, tmp_example)
        shutil.copy(script_file, tmp_script)

        # Run the script
        cmd = f'python {tmp_script} {tmp_example}'
        sc.runcommand(cmd, cwd=tmpdir)

        # Compare the transformed input with the expected output
        actual = tmp_example.read_text().splitlines(keepends=True)
        if actual == expected:
            print(f"  ✅ {base} matches expected output.")
        else:
            print(f"  ❌ {base} differs from expected output.")
            diff = "".join(difflib.unified_diff(actual, expected, fromfile="actual", tofile="expected"))
            print(diff)
            print('\n\nRaw strings:')
            print('<<<')
            print(f'"{expected}"')
            print('≠≠≠≠≠≠≠≠≠')
            print(f'"{actual}"')
            print('>>>')
