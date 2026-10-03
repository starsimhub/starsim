"""
Generate a machine-readable index of the Starsim API.

The whole public Starsim API fits in a single file, which makes it possible for a
tool (an IDE, a documentation search, or an LLM agent) to load the entire map of
the library in one shot. This script introspects `starsim` and `starsim.library`
and projects them into three artifacts, all generated from the same data and all
published with the docs (e.g. <https://docs.starsim.org/llms.txt>):

- `api.json`: the canonical index, as structured data
- `llms.txt`: a compact Markdown index (name, signature, summary, default parameters, aliases)
- `llms-full.txt`: as above, plus the canonical example for each entry

This is a docs build tool: nothing here is part of the Starsim package itself.

To regenerate the artifacts::

    cd docs && python make_api.py

To check that they are up to date (used by the test suite, hence CI)::

    cd docs && python make_api.py --check
"""

import re
import sys
import inspect
import textwrap
import warnings
import sciris as sc
import starsim as ss
import starsim.library as ssl

__all__ = ['make_index', 'load', 'write', 'check']

thisdir = sc.thispath(__file__)
jsonfile = thisdir / 'api.json'
llmsfile = thisdir / 'llms.txt'
llmsfullfile = thisdir / 'llms-full.txt'

max_summary = 300 # Maximum number of characters in a summary
max_par = 80 # Maximum number of characters in a default parameter value
max_example = 12 # Maximum number of lines in an example

description = 'A fast, flexible agent-based disease modeling framework' # Matches pyproject.toml
notes = [
    'Starsim is an agent-based modeling framework for simulating disease spread among agents via dynamic transmission networks, including co-transmission of multiple diseases and the effect of interventions.',
    'Everything in the core package is available from the top level: `import starsim as ss`, then e.g. `ss.Sim()`. Do not import submodules directly.',
    'Example and reference modules (e.g. `ssl.Cholera`) are in the library: `import starsim.library as ssl`. They are illustrative rather than validated.',
    'Signatures are as introspected from the current version; summaries are the first paragraph of each docstring.',
    'Module arguments default to `None` in the signature; the actual defaults are listed as "Default pars", and can be overridden by keyword or via `pars=dict(...)`.',
    'Aliases are alternative names for the same object; the canonical name is the one listed, and is the one to prefer when writing new code.',
]
links = [ # Extra pointers, following the llms.txt convention
    ('Documentation', 'https://docs.starsim.org', 'tutorials, user guide, and API reference'),
    ('Source', 'https://github.com/starsimhub/starsim', 'the Starsim repository'),
    ('Style guide', 'https://github.com/starsimhub/styleguide', 'the Starsim style guide'),
    ('Sciris', 'https://docs.sciris.org/llms.txt', 'the equivalent index for Sciris (`sc`), which Starsim uses extensively'),
]

# The order the modules are listed in, from most to least commonly used; anything else is appended alphabetically
module_titles = sc.objdict(
    sim           = 'Simulations',
    people        = 'People',
    modules       = 'Modules',
    diseases      = 'Diseases',
    networks      = 'Networks',
    demographics  = 'Demographics',
    interventions = 'Interventions',
    products      = 'Products',
    connectors    = 'Connectors',
    analyzers     = 'Analyzers',
    time          = 'Time, durations, and rates',
    timeline      = 'Timelines',
    distributions = 'Distributions',
    parameters    = 'Parameters',
    results       = 'Results',
    run           = 'Running multiple simulations',
    calibration   = 'Calibration',
    samples       = 'Samples',
    arrays        = 'Arrays and agent indexing',
    loop          = 'Integration loop',
    utils         = 'Utilities',
    debugtools    = 'Debugging tools',
    settings      = 'Settings',
    other         = 'Other',
)
module_order = list(module_titles.keys())

# The library subpackages, each listed after the core modules
libraries = sc.objdict(
    diseases = 'Library: diseases',
    mnch     = 'Library: maternal, newborn, and child health',
    networks = 'Library: networks',
)


def _getdoc(obj):
    """ Get an object's own docstring (not an inherited one, which may be from e.g. numpy) """
    doc = getattr(obj, '__doc__', None)
    if not isinstance(doc, str) and inspect.isclass(obj): # Some classes document themselves in __init__
        doc = getattr(getattr(obj, '__init__', None), '__doc__', None)
    if not isinstance(doc, str):
        return ''
    return inspect.cleandoc(doc) # Dedent, since Python ≥3.13 does this automatically but earlier versions do not


def _getsummary(obj, doc):
    """ Extract the first paragraph of a docstring, collapsed onto a single line """
    if not doc:
        if inspect.isclass(obj) and obj.__base__.__module__.startswith('starsim'): # e.g. ss.peryear, a subclass of ss.per
            return f'Subclass of `ss.{obj.__base__.__name__}`; see that entry for details.'
        return ''
    paragraphs = re.split(r'\n\s*\n', doc.strip())
    summary = ' '.join(paragraphs[0].split())
    if len(summary) > max_summary:
        summary = summary[:max_summary].rsplit(' ', 1)[0] + ' […]'
    return summary


def _getsignature(obj):
    """ Get the call signature of a function or class, falling back gracefully """
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError): # pragma: no cover # e.g. some C-implemented callables
        return '(...)'


def _getpars(obj):
    """ Get the default parameters of a module (or sim), by creating one with no arguments """
    if not (inspect.isclass(obj) and issubclass(obj, (ss.Module, ss.Sim))):
        return {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pars = obj().pars
    except Exception: # Some modules need arguments, e.g. a product for a vaccination intervention
        return {}

    out = {}
    for key,val in pars.items():
        valstr = ' '.join(repr(val).split()) # Collapse e.g. multiline array reprs
        valstr = re.sub(r' at 0x[0-9a-f]+', '', valstr) # Remove memory addresses, e.g. from functions, so the output is reproducible
        if type(val).__module__.startswith('starsim') and not valstr.startswith(('ss.', 'ssl.')):
            valstr = 'ss.' + valstr # e.g. peryear(0.1) -> ss.peryear(0.1)
        if len(valstr) > max_par:
            valstr = valstr[:max_par] + ' […]'
        out[key] = valstr
    return out


def _getexample(doc):
    """ Extract the canonical example: the first code block, preferring one under an "Example" heading """
    if not doc:
        return ''
    blocks = [(m.start(), m.group(1)) for m in re.finditer(r'```(?:python)?\n(.*?)```', doc, flags=re.DOTALL)]
    if not blocks:
        return ''
    heading = re.search(r'(?m)^\s*(?:\*\*)?Examples?(?:\*\*)?:', doc)
    example = None
    if heading:
        for start, block in blocks:
            if start > heading.start():
                example = block
                break
    if example is None:
        example = blocks[0][1]
    example = textwrap.dedent(example)
    lines = [line for line in example.strip('\n').rstrip().splitlines()]
    if len(lines) > max_example:
        lines = lines[:max_example] + ['# [...]']
    return '\n'.join(lines)


_deprecated = re.compile(r'(?i)^(?:\*{0,2}note:?\*{0,2}\s*)?(?:this (?:function|class|method) is deprecated|deprecated\b)')

def _isdeprecated(doc):
    """ Whether the docstring marks the object as deprecated (in the summary, or in a paragraph of its own) """
    if not doc:
        return False
    if 'deprecat' in _getsummary(None, doc).lower():
        return True
    paragraphs = re.split(r'\n\s*\n', doc.strip())
    return any(_deprecated.match(' '.join(paragraph.split())) for paragraph in paragraphs)


def _modulekey(module, prefix):
    """ Key for grouping entries by module, e.g. 'diseases' or 'library.diseases' """
    if prefix == 'ssl':
        parts = module.split('.') # e.g. starsim.library.diseases.cholera
        sub = parts[2] if len(parts) > 2 else 'other'
        return f'library.{sub}'
    short = module.rsplit('.', 1)[-1] if module else 'other'
    if short not in module_titles:
        short = 'other'
    return short


def _getnames(prefix):
    """ Get the public names to index for a namespace """
    if prefix == 'ss':
        return [name for name in dir(ss) if not name.startswith('_')]
    else:
        return [name for name in ssl.__all__ if name not in libraries]


def make_index():
    """
    Introspect Starsim and return the API index as a dictionary.

    The index has the keys `version`, `description`, `notes`, `links`, `n_entries`,
    `aliases` (a mapping of alias name to canonical name), and `entries` (a list of
    records, each with `name`, `kind`, `module`, `signature`, `summary`, `pars`,
    `example`, `aliases`, and `deprecated`). Names include the namespace, e.g.
    `ss.SIR` or `ssl.Cholera`.

    Examples:
        ```python
        import make_api
        index = make_api.make_index()
        print(index['entries'][0])
        ```
    """
    entries = []
    aliasmap = {}
    for prefix, namespace in [('ss', ss), ('ssl', ssl)]:

        # Group names by object, so aliases collapse into one entry
        groups = {} # Map id(obj) to the list of names pointing at it
        objs = {}
        for name in _getnames(prefix):
            obj = getattr(namespace, name)
            if inspect.ismodule(obj) or not callable(obj): # Skip submodules and data (e.g. ss.options)
                continue
            key = id(obj)
            groups.setdefault(key, []).append(name)
            objs[key] = obj

        for key, group in groups.items():
            obj = objs[key]
            realname = getattr(obj, '__name__', None)
            canonical = realname if realname in group else sorted(group, key=len)[0] # Prefer the object's own name
            aliases = sorted(f'{prefix}.{name}' for name in group if name != canonical)
            canonical = f'{prefix}.{canonical}'
            for alias in aliases:
                aliasmap[alias] = canonical
            doc = _getdoc(obj)
            entries.append(dict(
                name       = canonical,
                kind       = 'class' if inspect.isclass(obj) else 'function',
                module     = _modulekey(getattr(obj, '__module__', ''), prefix),
                signature  = _getsignature(obj),
                summary    = _getsummary(obj, doc),
                pars       = _getpars(obj),
                example    = _getexample(doc),
                aliases    = aliases,
                deprecated = _isdeprecated(doc),
            ))

    entries = sorted(entries, key=lambda entry: entry['name'].lower())
    index = dict(
        version     = ss.__version__,
        description = description,
        notes       = notes,
        links       = [dict(title=title, url=url, description=desc) for title,url,desc in links],
        n_entries   = len(entries),
        aliases     = dict(sorted(aliasmap.items())),
        entries     = entries,
    )
    return index


def _sortmodules(index):
    """ Return the module keys present in the index: core modules in order, then the library """
    present = {entry['module'] for entry in index['entries']}
    modules = [mod for mod in module_order if mod in present]
    modules += [f'library.{lib}' for lib in libraries.keys() if f'library.{lib}' in present]
    modules += sorted(present - set(modules))
    return modules


def _moduletitle(module):
    """ The section heading for a module, e.g. 'Diseases (ss.diseases)' """
    if module.startswith('library.'):
        lib = module.split('.', 1)[1]
        return f'{libraries.get(lib, module)} (ssl.{lib})'
    elif module in module_titles and module != 'other':
        return f'{module_titles[module]} (ss.{module})'
    else:
        return module_titles.get(module, module)


def make_llms_txt(index=None, examples=False):
    """
    Render the API index as an llms.txt-style Markdown document.

    Args:
        index (dict): the index from `make_index()` (default: generate it)
        examples (bool): whether to include the canonical example for each entry (i.e. llms-full.txt)

    Returns:
        The document, as a string.
    """
    index = sc.ifelse(index, make_index())
    filename = 'llms-full.txt' if examples else 'llms.txt'

    lines = [f'# Starsim v{index["version"]}', '', f'> {index["description"]}', '']
    for note in index['notes']:
        lines += [f'- {note}']
    if not examples:
        lines += ['- A version of this file including a usage example for each function is available at llms-full.txt.']
    lines += ['', f'This file lists all {index["n_entries"]} public Starsim functions and classes. It is generated from the source '
              f'by `python make_api.py`; do not edit it by hand.', '']

    lines += ['## Links', '']
    for link in index['links']:
        lines += [f'- [{link["title"]}]({link["url"]}): {link["description"]}']
    lines += ['']

    bymodule = {}
    for entry in index['entries']:
        bymodule.setdefault(entry['module'], []).append(entry)

    for module in _sortmodules(index):
        lines += [f'## {_moduletitle(module)}', '']
        for entry in bymodule[module]:
            extras = []
            if entry['aliases']:
                extras.append('aliases: ' + ', '.join(f'{alias}()' for alias in entry['aliases']))
            if entry['deprecated']:
                extras.append('DEPRECATED')
            suffix = f' [{"; ".join(extras)}]' if extras else ''
            summary = entry['summary'] or '(no description available)'
            lines += [f'- `{entry["name"]}{entry["signature"]}`: {summary}{suffix}']
            if entry['pars']:
                pars = ', '.join(f'`{key}={val}`' for key,val in entry['pars'].items())
                lines += [f'  - Default pars: {pars}']
            if examples and entry['example']:
                lines += ['', '  ```python']
                lines += [f'  {line}'.rstrip() for line in entry['example'].splitlines()]
                lines += ['  ```', '']
        lines += ['']

    lines += [f'<!-- Generated by `python make_api.py` for Starsim v{index["version"]}: {filename} -->', '']
    return '\n'.join(lines)


def load():
    """
    Load the generated API index (`docs/api.json`).

    Examples:
        ```python
        import make_api
        index = make_api.load()
        print(index['n_entries'])
        ```
    """
    return sc.loadjson(jsonfile)


def write(verbose=True):
    """
    Regenerate all the API artifacts: `api.json`, `llms.txt`, and `llms-full.txt`.

    Args:
        verbose (bool): whether to print progress

    Returns:
        The list of files written.
    """
    index = make_index()
    sc.savejson(jsonfile, index)
    written = [jsonfile]

    for path,examples in [(llmsfile, False), (llmsfullfile, True)]:
        sc.savetext(path, make_llms_txt(index, examples=examples))
        written.append(path)

    if verbose:
        for path in written:
            print(f'  Wrote {path.name} ({path.stat().st_size/1e3:0.1f} kB)')
        print(f'Indexed {index["n_entries"]} Starsim functions and classes for v{index["version"]}')
    return written


def check(verbose=True):
    """
    Check whether the generated artifacts match the current API.

    Args:
        verbose (bool): whether to print which files are out of date

    Returns:
        The list of files that are out of date (empty if everything matches).
    """
    index = make_index()
    stale = []

    def compare(path, expected):
        """ Compare a file to what it should contain, treating a missing file as stale """
        if not path.exists():
            return 'missing'
        actual = sc.loadjson(path) if path.suffix == '.json' else sc.loadtext(path)
        return None if actual == expected else 'out of date'

    for path,expected in [(jsonfile, index),
                          (llmsfile, make_llms_txt(index, examples=False)),
                          (llmsfullfile, make_llms_txt(index, examples=True))]:
        reason = compare(path, expected)
        if reason:
            stale.append(path)
            if verbose:
                print(f'  {path.name} is {reason}')
    if verbose and not stale:
        print(f'All API artifacts are up to date ({index["n_entries"]} entries, v{index["version"]})')
    return stale


if __name__ == '__main__':
    if '--check' in sys.argv:
        stale = check()
        if stale:
            print('Run "python make_api.py" to regenerate.')
            sys.exit(1)
    else:
        write()
