"""
Values for the docs, e.g. {{< var version >}}: `qpyd prerender` writes them to _variables.yml.
"""
import quartodoc as qd

version = qd.get_version() # Read from the Starsim source, without importing Starsim
versiondate = qd.get_attr('__versiondate__')
