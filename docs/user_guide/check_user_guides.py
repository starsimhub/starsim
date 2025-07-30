#!/usr/bin/env python
"""
Run all the Jupyter notebook user guides. Usage:
  ./check_user_guides.py

See check_notebooks in the parent folder for more information.
"""
import os
import sys
import sciris as sc
parent = sc.path(__file__).resolve().parent.parent # Ensure it's the real path if it's symlinked
cnb = sc.importbypath(parent / 'check_notebooks.py')

if __name__ == '__main__':
    results = cnb.main(*sys.argv[1:], folders=['user_guide'])
