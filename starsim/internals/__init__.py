"""
How Starsim works under the hood: the machinery that the modules and solvers are built on

The files here that users reach directly (`ss.time`, `ss.arrays`, `ss.parameters`, and others) are also at the top level, as are ss.errors, ss.v3_deprecations, ss.expressions, ss.graph, ss.display, ss.progress, ss.fitting, ss.gradients, ss.debugtools.

This file imports nothing: the modules are attached to the top level by `starsim/__init__.py`, which is
where users reach them (`ss.time`, not `ss.internals.time`), so there is no second path to keep in step.

Used with: agents and compartments; declared and coded modules
"""
