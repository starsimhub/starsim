#!/usr/bin/env python
"""
Run all the Jupyter notebook tutorials and user guide entries. Usage:
  ./check_notebooks.py [paths]

It will execute all notebooks in parallel, print out any errors
encountered, and print a summary of results. Same as `qpynb check tutorials user_guide`.
"""
import sys
from quartodoc.qpyd.execute import execute_notebooks
results = execute_notebooks(*(sys.argv[1:] or ['tutorials', 'user_guide']))
