"""
Let the switching demos import the helpers in tests/ (such as paths.py) when run by pytest
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
