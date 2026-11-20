---
name: starsim
description: Build, run, fit, and debug disease models with Starsim (agent-based and compartmental, in Python). Use when the user works with `import starsim`, or asks for /starsim.
allowed-tools: Bash(starsim ai show:*) Bash(python -m starsim ai show:*) Bash(python3 -m starsim ai show:*) Bash(uv run starsim ai show:*)
---

# Starsim

Load the guide for the version of Starsim that the user's project uses, and follow it. Use the project's Python, since a `starsim` on the path may belong to another environment. Try these in order:

1. With a `.venv` folder in the project: `.venv/bin/python -m starsim ai show` (`.venv\Scripts\python -m starsim ai show` on Windows)
2. With a `uv.lock`: `uv run starsim ai show`
3. Otherwise: `python -m starsim ai show`, or `starsim ai show`

If Starsim isn't in that environment, say so and suggest `pip install starsim`. Its version is in the first line of what's printed; if a command says that `ai` is unknown, the project uses Starsim 3, which has no guide: work from its installed source and docs.

The guide lists further skills; load one (`starsim ai show <skill>`, with the same Python) only when the task needs it.

This file was installed by `starsim ai`; `starsim ai uninstall --global` removes it.
