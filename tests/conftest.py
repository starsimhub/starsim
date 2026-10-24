"""
Let the tests in subfolders (devtests/, switching/) import the helpers in tests/ (such as paths.py and corpus.py),
and run JAX on the CPU: many test workers can't share one GPU (set JAX_PLATFORMS to test on a GPU)
"""
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
