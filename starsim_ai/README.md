# Starsim's AI skills

What Starsim gives AI coding agents (Claude Code, Codex, and others), installed with Starsim. `starsim ai` installs the `starsim` skill for the user's agents, which points them to `guide.md`; the other skills are read when they are needed, with `starsim ai show <skill>` (see `starsim/internals/cli.py`).

- `guide.md`: what an agent reads first: how to write, check, and run a Starsim model
- `starsim/`: the skill that is installed
- `import-model/`: importing a model from another framework, with notes on each in `references/`
- `validate-import/`: checking an imported model against the original's results
- `migrate-starsim3/`: rewriting Starsim 3 code for Starsim 4
