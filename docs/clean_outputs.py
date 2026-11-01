#!/usr/bin/env python
"""
Remove auto-generated files (the qpyd.clean patterns in _quarto.yml). Same as `qpyd clean`.
"""
from quartodoc.qpyd.clean import clean_outputs
clean_outputs()
