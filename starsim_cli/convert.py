"""
Converting models between Starsim and other frameworks: `starsim convert`, `starsim import`, and `starsim export`

The conversion itself is `ss.converters`: the model is read into a sim (`ss.import_from()`, or for Starsim, by running
the file), then written out as Starsim code (`sim.imported.code`) or in another format (`ss.export_to()`). With `--ai`,
an AI coding agent does it instead, with Starsim's convert-model skill, for what has no
mechanical conversion (e.g. LASER) or needs judgment.
"""
import os
import sys
import ast
import shutil
import warnings
import subprocess
from pathlib import Path

__all__ = ['convert', 'formats', 'agents', 'choose_agent']

# The coding agents that --ai can run: the command for an interactive session, and for one that runs on its own (--headless), with the prompt where "{}" is
agents = dict(
    claude = dict(interactive=['claude', '{}'], headless=['claude', '-p', '{}', '--permission-mode', 'acceptEdits', '--allowedTools', 'Bash(python:*)', 'Bash(starsim:*)']), # Edits files, and runs Python to check the result
    codex  = dict(interactive=['codex', '{}'], headless=['codex', 'exec', '--sandbox', 'workspace-write', '{}']),
)


def formats():
    """ The formats that can be read and written, as text for the help """
    import starsim as ss
    conv = ss.converters
    lines = [
        f'Import from (--from):  {", ".join(conv.importers)}, or starsim (a Python file that makes a sim)',
        f'Export to (--to):      starsim (the default), {", ".join(conv.exporters)}',
        f'Tested with:           {", ".join(conv.core.tested_with(key) for key in conv.tested)} (other versions may work too)',
        '',
        'With --from=auto (the default), the format is told from the file. Options this help doesn\'t list are passed to the importer (e.g. --databook=IND.xlsx for atomica).',
        'Frameworks without an importer (e.g. LASER), and parts the importer can\'t translate, can be converted by an AI coding agent with --ai.',
    ]
    return '\n'.join(lines)


def parse_extra(extra):
    """ Options not known to the command (e.g. --databook=IND.xlsx), as keyword arguments for the importer """
    kwargs = {}
    items = list(extra)
    while items:
        item = items.pop(0)
        if not item.startswith('--'):
            sys.exit(f'Unexpected argument "{item}"')
        key, eq, value = item[2:].partition('=')
        if not eq:
            value = items.pop(0) if items and not items[0].startswith('--') else 'True'
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError): # A plain string, e.g. a file name
            pass
        kwargs[key.replace('-', '_')] = value
    return kwargs


def load_sim(filename, name='sim'):
    """ The sim that a Python file makes, as the variable `name` (without running the file's `if __name__ == '__main__'` block) """
    import runpy
    import starsim as ss
    scope = runpy.run_path(str(filename), run_name='__starsim_convert__')
    if name in scope and isinstance(scope[name], ss.Sim):
        return scope[name]
    sims = [key for key,value in scope.items() if isinstance(value, ss.Sim)]
    if name == 'sim' and len(sims) == 1: # Not given, and there's only one
        return scope[sims[0]]
    found = f'the sims it makes are {", ".join(sims)}' if sims else 'it makes no ss.Sim'
    sys.exit(f'{filename} makes no sim called "{name}" ({found}); give --sim=<name>')


def convert(args, extra=()):
    """ Convert a model from one format to another (see `formats()`), or with --ai, have an AI coding agent do it """
    if not args.input:
        args.parser.print_help()
        print(f'\n{formats()}')
        return
    source = Path(args.input)
    if not source.exists():
        sys.exit(f'No file {source}')
    if args.ai:
        return run_agent(args)

    import sciris as sc
    import starsim as ss
    conv = ss.converters
    kwargs = parse_extra(extra)
    fmt = args.source
    if fmt == 'auto':
        fmt = conv.sniff(source)
        if fmt is None:
            sys.exit(f'Cannot tell which framework {source} is from: give --from=, one of {", ".join(["starsim", *conv.importers])}\n(Or have an AI agent convert it: starsim convert {source} --ai)')
    if fmt != 'starsim' and fmt not in conv.importers:
        sys.exit(f'{source} is a model for {fmt}, which has no importer (the formats are: starsim, {", ".join(conv.importers)})\nHave an AI agent convert it: starsim convert {source} --ai')
    to = args.to
    if to != 'starsim' and to not in conv.exporters:
        sys.exit(f'Cannot export to "{to}"; the formats are: starsim, {", ".join(conv.exporters)}')
    if fmt == to:
        sys.exit(f'{source} is already {fmt}: give --to=, one of {", ".join(conv.exporters)}' if fmt == 'starsim' else f'{source} is already {fmt}')
    out = Path(args.out) if args.out else source.with_suffix(conv.extensions[to])
    if out == source: # E.g. an epydemix script to Starsim code
        out = source.with_name(f'{source.stem}_{to}{out.suffix}')
    if out.exists() and not args.force:
        sys.exit(f'{out} exists already: give --out=<file>, or --force to replace it')

    warnings.showwarning = lambda message, category, *args, **kwargs: print(f'{category.__name__}: {str(message).strip()}', file=sys.stderr) # Without the line of Starsim that raised it
    if fmt == 'starsim':
        sim = load_sim(source, args.sim)
    else:
        try:
            sim = conv.import_from(source, format=fmt, **kwargs)
        except Exception as e: # E.g. a format without an importer, or a framework that isn't installed; shown without the traceback, which is Starsim's
            sys.exit(f'Could not import {source} ({type(e).__name__}): {e}\n(Or have an AI agent convert it: starsim convert {source} --ai)')
    if to == 'starsim':
        sc.savetext(out, sim.imported.code)
        incomplete = getattr(sim.imported, 'incomplete', None)
        note = f', with {len(incomplete)} part(s) not translated (see the warning above; to finish them: starsim convert {source} --ai)' if incomplete else ''
        print(f'Wrote {out}{note}')
    else:
        conv.export_to(sim, to, filename=out)
        print(f'Wrote {out}')
    return


#%% With an AI coding agent

def choose_agent(name=None):
    """ The agent to run: the one named (by --ai=<name>, or else `STARSIM_AGENT`), or else the only one installed """
    name = name or os.environ.get('STARSIM_AGENT')
    if name:
        if name not in agents:
            sys.exit(f'Unknown agent "{name}"; the agents are: {", ".join(agents)}')
        if not shutil.which(agents[name]['interactive'][0]):
            sys.exit(f'"{agents[name]["interactive"][0]}" is not on the path')
        return name
    found = [key for key,agent in agents.items() if shutil.which(agent['interactive'][0])]
    if not found:
        sys.exit(f'No AI coding agent found on the path: install one of {", ".join(agents)}')
    elif len(found) > 1:
        sys.exit(f'Several AI coding agents are installed ({", ".join(found)}): choose one with --ai=<name>, or set STARSIM_AGENT')
    return found[0]


def prompt(args):
    """ What to ask the agent """
    source = Path(args.input)
    fmt = '' if args.source == 'auto' else f' (a model for {args.source})'
    out = args.out
    if not out and args.to == 'starsim': # As the mechanical conversion names it
        out = source.with_suffix('.py') if source.suffix != '.py' else source.with_name(f'{source.stem}_starsim.py')
    to = 'Starsim code' if args.to == 'starsim' else args.to
    out = f', writing the result to {out}' if out else ''
    sim = f' The sim is the variable "{args.sim}".' if args.sim != 'sim' else ''
    return (f'Convert the model in {source}{fmt} to {to}{out}.{sim} Use Starsim\'s convert-model skill (if it isn\'t loaded, '
            f'run "starsim ai show convert-model"): use the mechanical conversion where there is one (starsim convert), and translate '
            f'by hand what it can\'t. Then check the result against the original, as the skill says, and finish '
            f'by summarizing what was converted, what changed, and what could not be.')


def run_agent(args):
    """ Run an AI coding agent on the conversion, interactively or (with --headless) on its own """
    name = choose_agent(None if args.ai is True else args.ai)
    text = prompt(args)
    command = [text if word == '{}' else word for word in agents[name]['headless' if args.headless else 'interactive']]
    print(f'Starting {name}{" (headless)" if args.headless else ""} to convert {args.input} ...')
    sys.exit(subprocess.run(command).returncode)
