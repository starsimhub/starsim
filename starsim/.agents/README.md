# Starsim's AI skills

What Starsim gives AI coding agents (Claude Code, Codex, and others), installed with Starsim, so that an agent always has the skills for the version in use. Facts about Starsim belong in the docs, which are tested and read by people too; a skill says how to work with Starsim and which pages to read.

- `skills/`: the skills, in the layout that tools for reading skills from installed packages look for (e.g. `uvx library-skills --claude`)
  - `starsim/`: what an agent reads first: how to write, check, and run a Starsim model
  - `convert-model/`: importing a model from another framework or exporting one to it, with notes on each in `references/`
  - `validate-conversion/`: checking a converted model against the original's results
  - `migrate-starsim3/`: rewriting Starsim 3 code for Starsim 4
- `meta-skill/starsim/`: the short skill that `starsim ai --global` installs for all projects, which loads the skills above from the project's Starsim (`starsim ai show`). It is kept out of `skills/` so that those tools don't list it.

`starsim ai` links the skills into a project, and importing Starsim suggests that where an AI coding agent is installed and has no skills (`starsim ai off` stops this, or `STARSIM_SKILL_NOTE=0`). The commands are in `starsim/internals/cli.py`.
