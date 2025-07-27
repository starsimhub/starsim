#!/usr/bin/env python3
"""
Check that the migration script examples work (written mostly by ChatGPT)
"""

import shutil
import subprocess
import tempfile
import difflib
import sciris as sc

root = sc.path(".")

# Find all base names that have __input.py, __output.py, and __script.py
triplets = []
for input_file in root.rglob("*__input.py"):
    base = input_file.stem.replace("__input", "")
    output_file = input_file.with_name(f"{base}__output.py")
    script_file = input_file.with_name(f"{base}__script.py")
    if output_file.exists() and script_file.exists():
        triplets.append((base, input_file, output_file, script_file))

for i, (base, input_file, output_file, script_file) in enumerate(triplets):
    sc.heading(f"Checking example {base} ({i+1} of {len(triplets)})...")
    expected = output_file.read_text().splitlines(keepends=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = sc.path(tmpdir)
        tmp_example = tmpdir / f"{base}__example.py"
        tmp_script = tmpdir / f"{base}__script.py"

        # Copy files
        shutil.copy(input_file, tmp_example)
        shutil.copy(script_file, tmp_script)

        # Run the script
        subprocess.run(["python", str(tmp_script)], cwd=tmpdir, check=True)

        # Compare the transformed input with the expected output
        actual = tmp_example.read_text().splitlines(keepends=True)
        if actual == expected:
            print(f"  ✅ {base} matches expected output.")
        else:
            print(f"  ❌ {base} differs from expected output.")
            diff = "".join(difflib.unified_diff(actual, expected, fromfile="generated", tofile="expected"))
            print(diff)
