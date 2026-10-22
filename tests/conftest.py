"""
Let the tests in subfolders (interop/, switching/) import the helpers in tests/ (such as paths.py and corpus.py)
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
