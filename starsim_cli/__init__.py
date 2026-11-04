"""
The starsim command, available after installing Starsim

    starsim init                  # "init ai" (if an AI coding agent is installed), then "init compile"
    starsim init compile          # Compile the Numba kernels ahead of time (optional; see below)
    starsim init compile --rust   # In a clone of Starsim, build the Rust kernels instead (installing Rust if needed)
    starsim init compile --check  # Which kernels are in use, and whether they are compiled
    starsim init ai               # Link Starsim's skills for AI agents into this project
    starsim init dev              # Set up a clone of Starsim for development

    starsim ai               # Where the skills are installed
    starsim ai install       # Link them into this project, as "init ai" does (--global: all projects)
    starsim ai list          # List the skills, with what each is for
    starsim ai show [skill]  # Show the guide for AI agents, or one of its skills
    starsim ai uninstall     # Remove them
    starsim ai off           # Stop the other commands from suggesting them ("on" undoes it)

    starsim convert model.R       # Convert to Starsim code (model.py); the format is told from the file
    starsim convert model.camdl --to=epimodel # Or to another framework
    starsim convert laser_model.py --ai # Or have an AI coding agent do it
    starsim import model.camdl    # Short for: convert model.camdl --to=starsim
    starsim export model.py sbml  # Short for: convert model.py --from=starsim --to=sbml
    starsim convert               # The formats, and the options

Compiling: Starsim runs its Rust kernels where they are built (as in the wheels on PyPI), and
otherwise its Numba kernels, which are compiled on first use (a few minutes, once per install).
Compiling them ahead of time is optional: it saves that wait in the first runs, e.g. in a Docker
image or in CI. "starsim init" skips it when the Rust kernels are in use, and "starsim init
compile" compiles Numba's anyway. In a clone of Starsim whose Rust kernels are not built, both
offer to build them instead, saying what that takes (--numba skips asking).

`python -m starsim` does the same, e.g. where the command is not on the path.
"""
# Only the commands that need Starsim import it (compiling, converting), so that the rest start at once
import sys
import argparse

__all__ = ['main']


def version():
    """ The version of Starsim, for --version """
    from .ai import version
    return f'Starsim {version()}'


class LazyVersion(argparse.Action):
    """ --version, which reads the version only when asked """
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, help='Show the version and exit')

    def __call__(self, parser, namespace, values, option_string=None):
        print(version())
        parser.exit()


def run(module, func):
    """ A command, whose module is only imported when it runs """
    def command(args, extra=()):
        import importlib
        mod = importlib.import_module(f'.{module}', __name__)
        return getattr(mod, func)(args, extra) if func == 'convert' else getattr(mod, func)(args)
    return command


def init(args):
    """ Set Starsim up after installing it: the skills for AI agents, if one is installed, then compile """
    from . import ai, build
    if ai.detected():
        ai.install(args)
    else:
        print('No AI coding agent found, so no skills installed ("starsim init ai" installs them anyway)')
    print()
    build.compile(args, needed=True)
    return


def main(argv=None):
    """ Run the starsim command; see this module's docstring """
    usage = __doc__.strip().split('\n', 2)[2] # The examples and the note on compiling
    parser = argparse.ArgumentParser(prog='starsim', usage='starsim <command> ...', description=usage, formatter_class=argparse.RawDescriptionHelpFormatter, epilog='Run "starsim <command> -h" for the options of each.')
    parser.add_argument('--version', action=LazyVersion)
    commands = parser.add_subparsers(dest='command', prog='starsim', help=argparse.SUPPRESS) # Listed in the usage above

    def add_target_args(p, ask=True):
        p.add_argument('--to', help='Where to install the skills, as a comma-separated list of: claude, agents (default: each whose agent is installed)')
        p.add_argument('--force', action='store_true', help='Replace a skill of the same name that was not installed by starsim')
        p.add_argument('--global', dest='all', action='store_true', help='Install a short skill for all projects, rather than linking the skills into this project')
        if ask:
            p.add_argument('-y', '--yes', action='store_true', help='Install without asking "Proceed?"')
        return

    def add_compile_args(p):
        p.add_argument('--check', action='store_true', help='Say whether the kernels are compiled, rather than compiling them, and exit with 1 if not')
        which = p.add_mutually_exclusive_group()
        which.add_argument('--rust', action='store_true', help='In a clone of Starsim, build the Rust kernels rather than compile the Numba ones, installing Rust if it is missing')
        which.add_argument('--numba', action='store_true', help='Compile the Numba kernels without offering to build the Rust ones')
        return

    # starsim init
    p = commands.add_parser('init', help='Set Starsim up, once after installing it: "init ai" if an AI coding agent is installed, then "init compile"')
    add_target_args(p)
    p.set_defaults(func=init)
    sub = p.add_subparsers(dest='what', metavar='<what>')
    a = sub.add_parser('compile', help='Compile the Numba kernels ahead of time (optional: they are otherwise compiled on first use), or in a clone without the Rust kernels, offer to build those instead')
    add_compile_args(a)
    a.set_defaults(func=run('build', 'compile'))
    a = sub.add_parser('ai', help='Link the skills for AI coding agents into this project (the same as "starsim ai install")')
    add_target_args(a)
    a.set_defaults(func=run('ai', 'install'))
    a = sub.add_parser('dev', help='Set up a clone of Starsim for development: the dev dependencies, Rust if it is missing, both sets of kernels, and the skills')
    add_target_args(a)
    a.set_defaults(func=run('build', 'init_dev'))

    # starsim ai
    p = commands.add_parser('ai', help='The skills for AI coding agents: where they are installed, install them, or show them')
    p.set_defaults(func=run('ai', 'status'))
    sub = p.add_subparsers(dest='action', metavar='<action>')
    for name,help in dict(install='Link the skills into this project (--global: a short skill for all projects)', uninstall='Remove the links in this project (--global: the short skill for all projects)').items():
        a = sub.add_parser(name, help=help)
        add_target_args(a, ask=name == 'install')
        a.set_defaults(func=run('ai', name))
    sub.add_parser('list', help='List the skills, with what each is for').set_defaults(func=run('ai', 'list_skills'))
    for name,help in dict(status='Where the skills are installed', off='Stop the other starsim commands from suggesting the skills', on='Let the other starsim commands suggest the skills again').items():
        sub.add_parser(name, help=help).set_defaults(func=run('ai', name))
    a = sub.add_parser('show', help='Show the guide for AI agents (which lists the skills), or one of its skills')
    a.add_argument('skill', nargs='?', help='The skill to show (default: the guide)')
    a.set_defaults(func=run('ai', 'show'))

    # starsim convert, import, export
    def add_convert_args(p, source=True, to=True):
        p.add_argument('--out', help='The file to write (default: the input\'s name, with the extension of the format)')
        p.add_argument('--sim', default='sim', help='For a Starsim file, the variable that holds the sim (default: sim)')
        p.add_argument('--force', action='store_true', help='Replace the output file if it exists')
        p.add_argument('--ai', nargs='?', const=True, default=False, metavar='AGENT', help='Have an AI coding agent do the conversion, with Starsim\'s skills (optionally naming the agent, e.g. --ai=claude; default: STARSIM_AGENT, or the only one installed)')
        p.add_argument('--headless', action='store_true', help='With --ai, run the agent on its own, rather than in an interactive session')
        p.set_defaults(func=run('convert', 'convert'), parser=p)
        return

    p = commands.add_parser('convert', help='Convert a model between Starsim and another framework, or between two others ("starsim convert" lists the formats)')
    p.add_argument('input', nargs='?', help='The model\'s file')
    p.add_argument('--in', dest='input_opt', metavar='INPUT', help='The model\'s file (as an option)')
    p.add_argument('--from', dest='source', default='auto', help='The format of the input (default: auto, to tell it from the file)')
    p.add_argument('--to', default='starsim', help='The format to write (default: starsim)')
    add_convert_args(p)

    p = commands.add_parser('import', help='Convert a model from another framework to Starsim code (the same as "starsim convert --to=starsim")')
    p.add_argument('input', help='The model\'s file')
    p.add_argument('source', nargs='?', default='auto', metavar='format', help='Its format (default: told from the file)')
    p.set_defaults(to='starsim')
    add_convert_args(p)

    p = commands.add_parser('export', help='Convert a Starsim model (a Python file that makes a sim) to another framework (the same as "starsim convert --from=starsim")')
    p.add_argument('input', help='The Python file that makes the sim')
    p.add_argument('to', metavar='format', help='The format to write')
    p.set_defaults(source='starsim')
    add_convert_args(p)

    args, extra = parser.parse_known_args(argv)
    if args.command is None: # Bare "starsim": show the help rather than an error
        parser.print_help()
        out = None
    elif args.command in ['convert', 'import', 'export']:
        args.input = getattr(args, 'input_opt', None) or args.input
        out = args.func(args, extra)
    elif extra:
        parser.error(f'unrecognized arguments: {" ".join(extra)}')
    else:
        out = args.func(args)
    if args.command not in ['init', 'ai']: # Those deal with the skills themselves
        from . import ai
        ai.note_skills()
    return out
