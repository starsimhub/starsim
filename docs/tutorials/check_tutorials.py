#!/usr/bin/env python
"""
Run all the Jupyter notebook tutorials. Usage:
  ./check_tutorials.py

See docutils.py in the parent folder for more information.
"""
import os
import sys
import sciris as sc
parent = sc.path(__file__).resolve().parent.parent # Ensure it's the real path if it's symlinked
ut = sc.importbypath(parent / 'docutils.py')

if __name__ == '__main__':
    results = ut.execute_notebooks(*sys.argv[1:], folders=['tutorials'])
