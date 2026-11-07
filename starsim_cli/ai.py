"""
The skills for AI coding agents, and asking an agent about Starsim: `starsim ai` (and `starsim init ai`)

Starsim's skills are in its package, in `.agents/skills/` (one folder per skill). In a project, they are linked into
the folders where coding agents look for skills, so that they always match the Starsim installed; for all projects,
a short skill is installed instead, which loads them from the Starsim in use (with `starsim ai show`).
`starsim ai "<question>"` asks an agent a question about Starsim (with `-i`, in an interactive session).

This module doesn't import Starsim (or Sciris), so that the command starts at once. The starsim command calls
`note_skills()` from here after its other commands, to suggest the skills where an agent has none.
"""
import os
import sys
import shutil
import subprocess
import importlib.util
from pathlib import Path

__all__ = ['skills', 'list_skills', 'show', 'install', 'uninstall', 'status', 'off', 'on', 'note_skills', 'agents', 'choose_agent', 'ask']

package_folder = Path(importlib.util.find_spec('starsim').submodule_search_locations[0]) # Found without importing it
agents_folder = package_folder / '.agents' # Where tools that read installed packages look for skills (e.g. library-skills)
ai_folder = agents_folder / 'skills' # The skills, by folder
meta_folder = agents_folder / 'meta-skill' # The short skill for all projects, which loads the others with "starsim ai show"; apart, so that those tools don't list it
skill_name = 'starsim' # The name of both the short skill and the main skill (the guide)
marker = 'installed by `starsim ai`' # In the installed SKILL.md, so that a skill of the same name from elsewhere is not overwritten
off_file = '.starsim/skill_note_off' # In the home folder: if there, the starsim command doesn't suggest the skills (see note_skills())


# The coding agents that the starsim command can run, each as a command with the prompt where "{}" is: for an interactive
# session; for one that runs on its own and may edit files (starsim convert --headless); and for one that only answers a question (starsim ai "...")
agents = dict(
    claude = dict(
        interactive = ['claude', '{}'],
        headless = ['claude', '-p', '{}', '--permission-mode', 'acceptEdits', '--allowedTools', 'Bash(python:*)', 'Bash(starsim:*)'], # Edits files, and runs Python to check the result
        ask = ['claude', '-p', '{}', '--allowedTools', 'Bash(starsim:*)', 'WebFetch(domain:starsim.org)'], # Reads, and looks things up in the docs
    ),
    codex = dict(
        interactive = ['codex', '{}'],
        headless = ['codex', 'exec', '--sandbox', 'workspace-write', '{}'],
        ask = ['codex', 'exec', '--sandbox', 'read-only', '--skip-git-repo-check', '{}'], # Read-only, so safe outside a repo
    ),
)


class Target:
    """ A folder where coding agents look for skills (in a project, and in the home folder), and how to tell whether one of those agents is installed """
    def __init__(self, folder, label, commands, paths):
        self.folder = folder
        self.label = label # Who reads it, as printed
        self.commands = commands
        self.paths = paths

    @property
    def present(self):
        """ Whether one of the agents that read this folder is installed """
        return any(shutil.which(c) for c in self.commands) or any(Path(p).expanduser().exists() for p in self.paths)

targets = dict(
    claude = Target('.claude/skills', 'Claude Code', commands=['claude'], paths=['~/.claude', '~/.claude.json']),
    agents = Target('.agents/skills', 'everything else', commands=['codex', 'kimi'], paths=['~/.codex', '~/.kimi']), # Codex, Kimi Code, Deep Code, and other agents that read the shared folder
)


#%% Showing the skills

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


def skill_folders():
    """ The folders of Starsim's skills, the guide included """
    return sorted(path.parent for path in ai_folder.glob('*/SKILL.md'))


def skills():
    """ The skills that "starsim ai show" serves, by name, with their descriptions (the guide lists them) """
    return {folder.name: frontmatter((folder/'SKILL.md').read_text(encoding='utf-8')).get('description', '') for folder in skill_folders() if folder.name != skill_name}


def list_skills(args=None):
    """ List the skills, each with its description """
    import textwrap
    width = min(shutil.get_terminal_size().columns, 100)
    print(f'Skills (from {ai_folder.relative_to(package_folder.parent).as_posix()}; "starsim ai show <skill>" shows one):')
    for folder in skill_folders():
        description = frontmatter((folder/'SKILL.md').read_text(encoding='utf-8')).get('description', '')
        print(f'\n    {folder.name}')
        print(textwrap.fill(description, width=width, initial_indent=' '*8, subsequent_indent=' '*8))
    return


def version():
    """ The version of Starsim and its date, read without importing it (from its version.py, which is current in a clone too) """
    import runpy
    v = runpy.run_path(str(package_folder/'version.py'))
    return f"{v['__version__']} ({v['__versiondate__']})"


def show(args):
    """ Print the guide for AI agents (with the skills it can load), or one of those skills """
    if not args.skill:
        print(f'<!-- Starsim {version()}; skills folder: {ai_folder} -->\n')
        print((ai_folder/skill_name/'SKILL.md').read_text(encoding='utf-8'))
        return
    file = ai_folder / args.skill / 'SKILL.md'
    if args.skill == skill_name or not file.exists():
        sys.exit(f'No skill "{args.skill}"; the skills are: {", ".join(skills())}')
    print(f'<!-- Skill folder (for the files it refers to): {file.parent} -->\n')
    print(file.read_text(encoding='utf-8'))
    return


#%% Installing them

def chosen(args, every=False):
    """ The targets to install to: those given, or else those whose agent is installed, and the shared folder always (or with `every`, all of them, e.g. to uninstall) """
    if getattr(args, 'to', None):
        keys = args.to.split(',')
        unknown = [k for k in keys if k not in targets]
        if unknown:
            sys.exit(f'Unknown target(s) {", ".join(unknown)}; choose from {", ".join(targets)}')
        return {k:targets[k] for k in keys}
    return {k:t for k,t in targets.items() if every or k == 'agents' or t.present}


def detected():
    """ Whether any AI coding agent is installed, for `starsim init` """
    return any(t.present for t in targets.values())


def project_root():
    """ The folder of the current project: the nearest one that looks like a project's, or else the current folder; None if that's the home folder, which holds every project """
    cwd, home = Path.cwd(), Path.home()
    if cwd == home:
        return None
    for folder in [cwd, *cwd.parents]:
        if folder == home:
            break
        if folder != package_folder and any((folder/name).exists() for name in ['.git', '.agents', '.claude', 'pyproject.toml']): # Starsim's own folder isn't a project, though it has .agents
            return folder
    return cwd


def is_link(dest):
    """ Whether a project's skill is a link to Starsim's, even if the Starsim it pointed to has since moved """
    return dest.is_symlink() and '.agents' in dest.readlink().parts


def where(chosen_targets, folder=lambda t: t.folder.split('/')[0]):
    """ The folders installed to, as printed: e.g. ".claude (for Claude Code) and .agents (for everything else)" """
    return ' and '.join(f'{folder(t)} (for {t.label})' for t in chosen_targets.values())


def unignored(root, paths):
    """ Those of the paths (relative to the root) that git doesn't ignore, if the root is a git repository """
    if not (root/'.git').exists() or not shutil.which('git'):
        return []
    out = subprocess.run(['git', '-C', str(root), 'check-ignore', *paths], capture_output=True, text=True)
    ignored = set(out.stdout.split())
    return [path for path in paths if path not in ignored]


def proceed(args):
    """ Whether to go ahead: yes with --yes, else as the user answers (no if there is no one to ask) """
    if getattr(args, 'yes', False):
        return True
    if not sys.stdin.isatty():
        print('Not installed: there is no terminal to ask "Proceed?" in (--yes installs without asking)')
        return False
    answer = input('Proceed? [Y/n] ').strip().lower()
    args.yes = answer in ['', 'y', 'yes'] # So that falling back to all projects doesn't ask again
    return args.yes


def install_project(args):
    """ Link each skill into the current project, so that it follows the installed Starsim; with no project, or if links can't be made, install for all projects instead """
    root = project_root()
    if root is None:
        print('No project here (this is the home folder), so installing for all projects')
        return install_global(args)
    use = chosen(args)
    current = [folder.name for folder in skill_folders()]
    links, skipped = [], []
    stale = [dest for target in use.values() for dest in sorted((root/target.folder).glob('*')) if is_link(dest) and dest.name not in current] # Skills this Starsim no longer has
    for target in use.values():
        for source in skill_folders():
            dest = root / target.folder / source.name
            if (dest.is_symlink() or dest.exists()) and not (args.force or is_link(dest)):
                skipped.append(dest.relative_to(root).as_posix())
            else:
                links.append((source, dest))
    print(f'Root: {root}')
    print(f'Installing in: {where(use)}')
    print(f'Skills to link (from {ai_folder.relative_to(package_folder.parent).as_posix()}):')
    for name in sorted({source.name for source,dest in links}):
        print(f'    {name}')
    if skipped:
        print('Skipping, since they were not installed by starsim (--force replaces them):')
        for path in skipped:
            print(f'    {path}')
    if stale:
        print('Removing links to skills that this Starsim no longer has:')
        for dest in stale:
            print(f'    {dest.relative_to(root).as_posix()}')
    if not links or not proceed(args):
        return

    for dest in stale:
        dest.unlink()
    for source,dest in links:
        if dest.is_symlink() or dest.exists():
            dest.unlink() if dest.is_symlink() else shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.symlink_to(source, target_is_directory=True)
        except OSError as e: # E.g. on Windows without Developer Mode
            print(f'Could not make links in this project ({e}), so installing for all projects')
            return install_global(args)
    print('Linked')
    new = unignored(root, [dest.relative_to(root).as_posix() for source,dest in links])
    if new:
        print('The links are specific to this Python environment, so they generally should not be committed. For .gitignore:')
        for path in new:
            print(f'    {path}')
    return


def install_global(args):
    """ Install a short skill for all projects, which loads the guide from the installed package when used """
    use = {}
    for key,target in chosen(args).items():
        file = Path.home() / target.folder / skill_name / 'SKILL.md'
        if file.exists() and marker not in file.read_text(encoding='utf-8') and not args.force:
            print(f'Skipping ~/{target.folder}/{skill_name}: a skill named "{skill_name}" that starsim did not install is there (--force replaces it)')
        else:
            use[key] = target
    if not use:
        return
    print(f'Installing in: {where(use, lambda t: f"~/{t.folder}/{skill_name}")}')
    if not proceed(args):
        return
    for target in use.values():
        shutil.copytree(meta_folder/skill_name, Path.home()/target.folder/skill_name, dirs_exist_ok=True)
    print('Installed. Start a new session of your agent to load it; it adds one line to the agent\'s context until used.')
    return


def install(args):
    """ Link the skills for AI coding agents into this project; with --global, install a short skill for all projects instead """
    return install_global(args) if args.all else install_project(args)


def uninstall(args):
    """ Remove the links in this project, or with --global, the short skill for all projects """
    root = project_root()
    for target in chosen(args, every=True).values():
        if not args.all:
            for dest in sorted((root/target.folder).glob('*')) if root else []:
                if is_link(dest):
                    dest.unlink()
                    print(f'Removed {dest.relative_to(root).as_posix()}')
            continue
        dest = Path.home() / target.folder / skill_name
        file = dest / 'SKILL.md'
        if file.exists() and marker in file.read_text(encoding='utf-8'):
            shutil.rmtree(dest)
            print(f'Removed ~/{target.folder}/{skill_name}')
    return


#%% Status, and the note about the skills

def status(args=None):
    """ Say where the Starsim skills are installed, and whether they are current """
    current = (meta_folder/skill_name/'SKILL.md').read_text(encoding='utf-8')
    root = project_root()
    print(f'Skills (from {ai_folder}):')
    for folder in skill_folders():
        print(f'    {folder.name}')
    if root:
        print(f'Root: {root}')
    for target in targets.values():
        states = []
        links = [dest for dest in sorted((root/target.folder).glob('*')) if is_link(dest)] if root else []
        if links:
            broken = sum(not dest.exists() for dest in links)
            states.append(f'{len(links)} linked in this project' + (f', {broken} broken (run "starsim ai install" to repair)' if broken else ''))
        file = Path.home() / target.folder / skill_name / 'SKILL.md'
        if file.exists():
            if marker not in (text := file.read_text(encoding='utf-8')):
                states.append(f'~/{target.folder}/{skill_name} is a skill of the same name, not installed by starsim')
            else:
                states.append(f'installed for all projects' + ('' if text == current else ', out of date (run "starsim ai install --global" to update)'))
        print(f'{target.folder.split("/")[0]} (for {target.label}): {"; ".join(states) or "not installed"}')
    print(f'Suggesting them after other starsim commands: {"off" if note_off() else "on"}')
    return


def off(args=None):
    """ Stop the starsim command from suggesting the skills, on this computer """
    file = Path.home() / off_file
    file.parent.mkdir(exist_ok=True)
    file.touch()
    print('Starsim won\'t suggest the skills; "starsim ai on" undoes this')
    return


def on(args=None):
    """ Let the starsim command suggest the skills again """
    (Path.home() / off_file).unlink(missing_ok=True)
    return


def note_off():
    """ Whether suggesting the skills has been switched off, by "starsim ai off" or `STARSIM_SKILL_NOTE=0` """
    enabled = os.environ.get('STARSIM_SKILL_NOTE', '1').strip().lower() not in ['0', 'false', 'no', 'off']
    return not enabled or (Path.home() / off_file).exists()


def skills_found():
    """ Whether an agent would find the Starsim skill here, however it got there """
    root = project_root()
    folders = [Path.home()/t.folder for t in targets.values()] + ([root/t.folder for t in targets.values()] if root else [])
    return any((folder/skill_name).exists() for folder in folders)


def note_skills():
    """
    After a starsim command, say that an AI coding agent here has no Starsim skills: a few checks of files, so quick. Only
    where an agent is installed, and not if switched off (see `note_off()`)
    """
    if not note_off() and targets['claude'].present and not skills_found():
        print('\nUsing an AI coding agent? Run "starsim init ai" to give it Starsim\'s skills ("starsim ai off" to stop this note)')
    return


#%% Asking an agent

def choose_agent(name=None, option='--agent'):
    """ The agent to run: the one named (by `option`, or else `STARSIM_AGENT`), or else the first installed, in the order of `agents` """
    name = name or os.environ.get('STARSIM_AGENT')
    if name:
        if name not in agents:
            sys.exit(f'Unknown agent "{name}"; the agents are: {", ".join(agents)}')
        if not shutil.which(agents[name]['interactive'][0]):
            sys.exit(f'"{agents[name]["interactive"][0]}" is not on the path')
        return name
    found = [key for key,agent in agents.items() if shutil.which(agent['interactive'][0])]
    if not found:
        sys.exit(f'No AI coding agent found on the path: install one of {", ".join(agents)} (or look it up with "starsim help <term>")')
    elif len(found) > 1:
        print(f'Using {found[0]} (also installed: {", ".join(found[1:])}; choose with {option}=<name>, or set STARSIM_AGENT)', file=sys.stderr)
    return found[0]


def ask(args):
    """ Ask an AI coding agent a question about Starsim: answered on its own, or with -i, in an interactive session """
    question = ' '.join(args.question).strip()
    if not question:
        sys.exit('Ask a question, e.g. starsim ai "how would I model a leaky vaccine?"')
    name = choose_agent(args.agent)
    text = (f'{question}\n\n(This is a question about Starsim {version()}, the disease modeling framework installed here. '
            f'Use Starsim\'s skills (if they aren\'t loaded, run "starsim ai show"), and look up names and topics with '
            f'"starsim help <term>". Answer with working code where it helps; don\'t change any files unless asked to.)')
    command = [text if word == '{}' else word for word in agents[name]['interactive' if args.interactive else 'ask']]
    if args.interactive:
        print(f'Starting {name} ...')
    sys.exit(subprocess.run(command, stdin=None if args.interactive else subprocess.DEVNULL).returncode) # Nothing more to read, so an agent waits for no input
