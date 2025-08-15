#!/usr/bin/env python
"""
Run all the Jupyter notebook tutorials and user guide entries. Usage:
  ./check_notebooks.py

It will execute all notebooks in parallel, print out any errors
encountered, and print a summary of results.
"""
import docutils as ut
results = ut.execute_notebooks(*sys.argv[1:])
