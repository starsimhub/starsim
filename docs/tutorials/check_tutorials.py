#!/usr/bin/env python
"""
Run all the Jupyter notebook tutorials. Usage:
  ./check_tutorials

See check_notebooks in the parent folder for more information.
"""

import os
import sciris as sc
parent = sc.path(__file__).resolve().parent.parent # Ensure it's the real path if it's symlinked
cnb = sc.importbypath(parent / 'check_notebooks.py')

if __name__ == '__main__':
    results = cnb.main(folders=['tutorials'])

