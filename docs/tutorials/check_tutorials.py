#!/usr/bin/env python
"""
Run all the Jupyter notebook tutorials. Usage:
  ./check_tutorials

See check_notebooks in the parent folder for more information.
"""

import os
import sciris as sc
parent = sc.path(__file__).resolve().parent # Ensure it's the real path if it's symlinked
os.chdir(parent)
import check_notebooks as cnb

if __name__ == '__main__':
    results = main(folders=['tutorials'])

