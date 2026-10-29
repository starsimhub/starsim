"""
The starsim command, available after installing Starsim:

    starsim init                # Both of the below: what most users run once, after pip install starsim
    starsim compile             # Compile Starsim's numerical code (once per install)
    starsim ai                  # Install the Starsim skill for AI coding agents (Claude Code, Codex, Kimi, ...)
    starsim ai show [skill]     # Show the guide for AI agents, or one of its skills
    starsim ai status           # Where the skill is installed
    starsim ai uninstall        # Remove it

`python -m starsim` does the same, e.g. where the command is not on the path.
"""
import sys
import shutil
import argparse
from pathlib import Path
import sciris as sc
import starsim as ss
import starsim_ai

__all__ = ['main']

ai_folder = starsim_ai.folder # The skills, and the guide that the installed skill points to
skill_name = 'starsim' # The one skill that is installed; the others are served from ai_folder by "starsim ai show"
placeholder = '<!-- The list of skills is added here by `starsim ai show` -->' # In guide.md
marker = 'installed by `starsim ai`' # In the installed SKILL.md, so that a skill of the same name from elsewhere is not overwritten

# Where coding agents look for a user's skills: each is a folder of skills, one per subfolder with a SKILL.md
targets = sc.objdict(
    claude = sc.objdict(folder='~/.claude/skills', agents='Claude Code (as /starsim)', needs='~/.claude'),
    agents = sc.objdict(folder='~/.agents/skills', agents='Codex ($starsim), Kimi Code (/skill:starsim), Deep Code, and other agents that read the shared folder', needs=None),
)


def compile(args=None):
    """ Compile Starsim's numerical code now, so that no run waits for it: see `ss.compile()` """
    ss.compile()
    return


def skills():
    """ The skills that "starsim ai show" serves, by name, with their descriptions """
    out = sc.objdict()
    for file in sorted(ai_folder.glob('*/SKILL.md')):
        name = file.parent.name
        if name != skill_name:
            meta = frontmatter(file.read_text(encoding='utf-8'))
            out[name] = meta.get('description', '')
    return out


def frontmatter(text):
    """ The fields of a SKILL.md's frontmatter (one line each), as a dict """
    lines = text.split('\n')
    out = {}
    if lines and lines[0].strip() == '---':
        for line in lines[1:]:
            if line.strip() == '---':
                break
            key, _, value = line.partition(':')
            out[key.strip()] = value.strip()
    return out


def show(args):
    """ Print the guide for AI agents (with the skills it can load), or one of those skills """
    if not args.skill:
        guide = (ai_folder / 'guide.md').read_text(encoding='utf-8')
        listing = '\n'.join(f'- `{name}`: {desc}' for name,desc in skills().items())
        print(f'<!-- Starsim {ss.__version__}; skills folder: {ai_folder} -->\n')
        print(guide.replace(placeholder, listing))
        return
    file = ai_folder / args.skill / 'SKILL.md'
    if args.skill == skill_name or not file.exists():
        sys.exit(f'No skill "{args.skill}"; the skills are: {sc.strjoin(skills().keys())}')
    print(f'<!-- Skill folder (for the files it refers to): {file.parent} -->\n')
    print(file.read_text(encoding='utf-8'))
    return


def chosen(args):
    """ The targets to install to: those given, or else those whose agent is installed (the shared folder always) """
    if args.to:
        keys = args.to.split(',')
        unknown = [k for k in keys if k not in targets]
        if unknown:
            sys.exit(f'Unknown target(s) {sc.strjoin(unknown)}; choose from {sc.strjoin(targets.keys())}')
        return {k:targets[k] for k in keys}
    return {k:t for k,t in targets.items() if t.needs is None or Path(t.needs).expanduser().exists()}


def install(args):
    """ Install the Starsim skill for AI coding agents: a short skill that loads the guide from the installed package when used """
    source = ai_folder / skill_name
    for key,target in chosen(args).items():
        dest = Path(target.folder).expanduser() / skill_name
        file = dest / 'SKILL.md'
        if file.exists() and marker not in file.read_text(encoding='utf-8') and not args.force:
            print(f'Skipped {dest}: it has a skill named "{skill_name}" that starsim did not install (use --force to replace it)')
            continue
        shutil.copytree(source, dest, dirs_exist_ok=True)
        print(f'Installed {dest}, for {target.agents}')
    print('Start a new session of your agent to load it; it adds one line to the agent\'s context until used.')
    return


def uninstall(args):
    """ Remove the Starsim skill installed by "starsim ai" """
    for key,target in chosen(args).items():
        dest = Path(target.folder).expanduser() / skill_name
        file = dest / 'SKILL.md'
        if file.exists() and marker in file.read_text(encoding='utf-8'):
            shutil.rmtree(dest)
            print(f'Removed {dest}')
    return


def status(args):
    """ Say where the Starsim skill is installed, and whether it is current """
    current = (ai_folder / skill_name / 'SKILL.md').read_text(encoding='utf-8')
    for key,target in targets.items():
        file = Path(target.folder).expanduser() / skill_name / 'SKILL.md'
        if not file.exists():
            state = 'not installed'
        elif marker not in (text := file.read_text(encoding='utf-8')):
            state = 'a skill of the same name, not installed by starsim'
        else:
            state = 'installed' if text == current else 'installed, out of date (run "starsim ai" to update)'
        print(f'{key:8s} {file.parent}: {state}')
    print(f'Guide and skills: {ai_folder}')
    return


def init(args):
    """ Set Starsim up after installing it: install the skill for AI agents, then compile """
    install(args)
    print()
    compile(args)
    return


def main(argv=None):
    """ Run the starsim command; see this module's docstring """
    parser = argparse.ArgumentParser(prog='starsim', description='Set up and use Starsim from the command line')
    parser.add_argument('--version', action='version', version=f'Starsim {ss.__version__} ({ss.__versiondate__})')
    commands = parser.add_subparsers(dest='command')

    def add_target_args(p):
        p.add_argument('--to', help=f'Where to install the skill, as a comma-separated list of: {sc.strjoin(targets.keys())} (default: each whose agent is installed)')
        p.add_argument('--force', action='store_true', help='Replace a skill named "starsim" that was not installed by starsim')
        return

    p = commands.add_parser('init', help='Install the skill for AI agents and compile Starsim (once, after installing it)')
    add_target_args(p)
    p.set_defaults(func=init)

    p = commands.add_parser('compile', help='Compile Starsim\'s numerical code (once per install; a few minutes)')
    p.set_defaults(func=compile)

    p = commands.add_parser('ai', help='Install the Starsim skill for AI coding agents, or show its guide')
    actions = p.add_subparsers(dest='action')
    p.set_defaults(func=install, to=None, force=False)
    for name,func in dict(install=install, uninstall=uninstall).items():
        a = actions.add_parser(name, help=func.__doc__.strip())
        add_target_args(a)
        a.set_defaults(func=func)
    actions.add_parser('status', help=status.__doc__.strip()).set_defaults(func=status)
    a = actions.add_parser('show', help=show.__doc__.strip())
    a.add_argument('skill', nargs='?', help='The skill to show (default: the guide, which lists them)')
    a.set_defaults(func=show)

    args = parser.parse_args(argv)
    if args.command is None: # Bare "starsim": show the help rather than an error
        parser.print_help()
        return
    args.func(args)
    return
