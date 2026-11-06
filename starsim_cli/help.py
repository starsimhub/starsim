"""
Looking things up in the docs: `starsim help <term>`

A term that is a name in Starsim's API (`RandomNet`, `ss.Sim.run`, `ssl.cholera`), in any case, prints its docstring with
a link to its page in the docs. Any other term (or phrase) lists the names whose docstrings mention it, and the pages of
the docs online (the tutorials, the user guide, and the rest of the site) where it appears.

The names and docstrings are those of the Starsim installed, so they always match it; the pages are searched in the
site's own search index (`search.json`, written by Quarto), so they are those of the version published.
"""
import re
import sys
import json
import shutil
import inspect
import urllib.request

__all__ = ['help', 'names', 'lookup', 'search_names', 'search_pages']

docs_url = 'https://starsim.org/v4' # V4-PREVIEW: restore to https://docs.starsim.org for normal publishing (as in docs/make_api.py)
max_names = 30 # The most names to list for a term that isn't one
max_pages = 20 # The most sections of pages to list
whole = ['starsim.converters', 'starsim.library.diseases', 'starsim.library.networks', 'starsim.library.mnch', 'starsim.library.econ'] # Packages documented as a whole, on one page each, rather than by module


def names():
    """
    Every public name, as `{'ss.RandomNet': obj}`: the top level of Starsim (`ss.X`) and of its library (`ssl.X`), the
    names in Starsim's own submodules that are reached from the top (`ss.geo.geography`), and the public methods and
    properties that each class defines itself (`ss.Sim.run`)
    """
    import starsim as ss
    import starsim.library as ssl
    out = {}
    def add(name, obj):
        if not (callable(obj) or type(obj).__module__.startswith('starsim')): # Data (e.g. ss.data.world) has no docs of its own, but Starsim's objects (ss.options) do
            return
        out[name] = obj
        if inspect.isclass(obj):
            for key, value in vars(obj).items(): # As defined in the class, since getattr() on some classes (e.g. dicts with attribute access) gives something else
                if not key.startswith('_') and (inspect.isfunction(value) or isinstance(value, (property, classmethod, staticmethod))):
                    out[f'{name}.{key}'] = value
        return

    for prefix, namespace in dict(ss=ss, ssl=ssl).items():
        for key in dir(namespace):
            obj = getattr(namespace, key)
            if key.startswith('_'):
                continue
            elif inspect.ismodule(obj):
                if obj.__name__.startswith(namespace.__name__) and hasattr(obj, '__all__'): # e.g. ss.geo, but not ss.np
                    for sub in obj.__all__:
                        if hasattr(obj, sub):
                            add(f'{prefix}.{key}.{sub}', getattr(obj, sub))
            else:
                add(f'{prefix}.{key}', obj)
    return out


def canonical(found):
    """ The same objects found under several names (aliases, and names reached from more than one module) once each, under the shortest name """
    out = {}
    for name in sorted(found, key=lambda n: (n.count('.'), len(n), n)):
        if not any(obj is found[name] for obj in out.values()):
            out[name] = found[name]
    return out


def normalize(term):
    """ The term as matched: lowercase, without a trailing "()", and with "starsim." and "starsim.library." as "ss." and "ssl." """
    term = ' '.join(term.split()).lower().removesuffix('()')
    for long, short in [('starsim.library.', 'ssl.'), ('starsim.', 'ss.')]:
        if term.startswith(long):
            term = short + term.removeprefix(long)
    return term


def lookup(term, found=None):
    """ The names that are the term, in any case, with or without their "ss." or "ssl." """
    found = names() if found is None else found
    term = normalize(term)
    return canonical({name: obj for name, obj in found.items() if term in [name.lower(), name.lower().split('.', 1)[1]]})


def documented(obj):
    """ What has the docstring and the module: the function of a property or method, or the class of an object such as ss.options """
    if isinstance(obj, property):
        obj = obj.fget
        if getattr(obj, '__name__', '') == '__get__': # A method kept as a property (e.g. in ss.Pars, so that no key hides it)
            obj = obj.__self__
    obj = getattr(obj, '__func__', obj)
    return obj if inspect.isclass(obj) or inspect.isroutine(obj) else type(obj)


def url(obj):
    """ The link to an object's entry in the API reference (its module's page, e.g. api/networks.html#starsim.networks.RandomNet) """
    target = documented(obj)
    module = getattr(target, '__module__', None) or ''
    qualname = getattr(target, '__qualname__', getattr(target, '__name__', ''))
    qualname = qualname.split('.')[0] # A method or property is linked to its class, since the reference doesn't give every one its own entry
    parts = module.split('.')
    for package in whole: # e.g. starsim.library.diseases.cholera.Cholera is documented as starsim.library.diseases.Cholera
        if module.startswith(package):
            parts = package.split('.')
    page = '.'.join(parts[1:])
    if parts[0] != 'starsim' or not page:
        return f'{docs_url}/api/'
    return f'{docs_url}/api/{page}.html#{".".join(parts)}.{qualname}'


def signature(name, obj):
    """ The name with its signature, e.g. ss.RandomNet(n_contacts=None, ...) """
    if isinstance(obj, property) and documented(obj) is obj.fget:
        return f'{name} (property)'
    elif not (inspect.isclass(obj) or inspect.isroutine(documented(obj))): # An object, e.g. ss.options
        return f'{name} ({type(obj).__name__})'
    try:
        sig = inspect.signature(documented(obj))
        pars = list(sig.parameters.values())
        if pars and pars[0].name == 'self': # A method, as called on an instance
            sig = sig.replace(parameters=pars[1:])
        return f'{name}{sig}'
    except (TypeError, ValueError):
        return name


def show(name, obj):
    """ Print a name's signature, docstring, and link """
    doc = inspect.getdoc(documented(obj)) or '(No docstring)'
    print(signature(name, obj))
    print()
    print(doc)
    print()
    print(f'Docs: {url(obj)}')
    return


def snippet(text, term, width=100):
    """ The text around the first place the term appears, on one line """
    text = ' '.join(text.split())
    i = text.lower().find(term)
    if i < 0:
        return text[:width]
    start = max(0, i - width//3)
    end = start + width
    return ('...' if start else '') + text[start:end].strip() + ('...' if end < len(text) else '')


def search_names(term, found=None):
    """ The names that contain the term, then those whose docstrings mention it, as `{name: text}`: the summary, or the docstring """
    found = canonical(names() if found is None else found)
    term = normalize(term)
    named, mentioned = {}, {}
    for name, obj in found.items():
        doc = inspect.getdoc(documented(obj)) or ''
        summary = doc.strip().split('\n\n')[0]
        if term in name.lower():
            named[name] = summary
        elif term in ' '.join(doc.split()).lower():
            mentioned[name] = doc
    return named | mentioned


def search_pages(term, timeout=10):
    """ The sections of the pages of the docs online where the term appears, as a list of (heading, url, text); None if the site can't be reached """
    try:
        with urllib.request.urlopen(f'{docs_url}/search.json', timeout=timeout) as response:
            index = json.loads(response.read().decode('utf-8'))
    except Exception:
        return None
    term = normalize(term)
    out = []
    for entry in index:
        href = entry.get('href', '')
        if href.startswith('api/'): # The API reference is the docstrings, searched already
            continue
        text = ' '.join(entry.get('text', '').split())
        title = entry.get('title', '')
        section = entry.get('section', '')
        if term in f'{title} {section} {text}'.lower():
            heading = f'{title}: {section}' if section and section != title else title
            out.append((heading, f'{docs_url}/{href}', text))
    order = ['tutorials/', 'user_guide/', 'gallery', 'whatsnew', 'migration/'] # The guides first, and the history last
    rank = lambda link: next((i for i, prefix in enumerate(order) if link.removeprefix(f'{docs_url}/').startswith(prefix)), 2.5)
    return sorted(out, key=lambda entry: rank(entry[1]))


def help(args):
    """ Look up a term: the docstring of the name it is, or else the names and pages where it appears """
    term = ' '.join(args.term)
    found = names()
    exact = lookup(term, found)
    for i, (name, obj) in enumerate(exact.items()):
        if i:
            print('\n' + '-'*40 + '\n')
        show(name, obj)
    if exact:
        return

    width = min(shutil.get_terminal_size().columns, 120) - 8
    matches = search_names(term, found)
    pages = None if args.offline else search_pages(term)
    if not matches and not pages:
        offline = ' (the pages online were not searched: the site could not be reached)' if pages is None and not args.offline else ''
        sys.exit(f'Nothing in the docs mentions "{term}"{offline}')

    if matches:
        print(f'Names that mention "{term}" ("starsim help <name>" shows one):')
        for name, text in list(matches.items())[:max_names]:
            print(f'    {name}')
            if text:
                print(f'        {snippet(text, normalize(term), width)}')
        if len(matches) > max_names:
            print(f'    ... and {len(matches) - max_names} more')
    if pages:
        print(f'\nPages that mention "{term}":' if matches else f'Pages that mention "{term}":')
        for heading, link, text in pages[:max_pages]:
            print(f'    {heading}')
            print(f'        {link}')
            print(f'        {snippet(text, normalize(term), width)}')
        if len(pages) > max_pages:
            print(f'    ... and {len(pages) - max_pages} more')
    elif pages is None and not args.offline:
        print(f'\n(The pages of the docs were not searched: {docs_url} could not be reached)')
    return
