"""
Import models from other frameworks, and export sims to them

`ss.import_from(filename, format=...)` reads another framework's model into a sim, and
`ss.export_to(sim, format)` writes a sim in another framework's format; see
`docs/design/16-interop.md` in the lingua_franca_design repo. Each importer reads its format into an
`ss.converters.Source`, and `ss.converters.build()` writes that as Starsim code by the
same rules for every format (`ss.converters.write()`) and runs the code to make the sim:
the code is kept in `sim.imported.code`, how each rate was read is printed, and what
could not be translated is listed as `ss.errors.ImportIncomplete` and kept in `sim.imported`.

Used with: agents and compartments; declared and coded modules
"""
import os
import re
import sys
import tempfile
import subprocess
import importlib.util
import sciris as sc
from .core import Source, Code, write, build, tested, check_version
from .camdl import load_camdl
from .emulsion import load_emulsion
from .epimodel import load_epimodel
from .epipack import load_epipack
from .epymorph import load_epymorph
from .epydemix import load_epydemix
from .atomica import load_atomica
from .flepimop import load_flepimop
from .memilio import load_memilio
from .odin import load_odin
from .sbml import load_sbml
from .siminf import load_siminf
from .summer2 import load_summer2
from .starsim_json import load_json
from .export import exporters, extensions, export_to, to_hub

__all__ = ['Source', 'Code', 'write', 'build', 'importers', 'tested', 'sniff', 'import_from', 'convert', 'exporters', 'extensions', 'export_to', 'to_hub']

importers = dict(
    atomica  = load_atomica,
    camdl    = load_camdl,
    emulsion = load_emulsion,
    epimodel = load_epimodel,
    epipack  = load_epipack,
    epymorph = load_epymorph,
    epydemix = load_epydemix,
    flepimop = load_flepimop,
    json     = load_json,
    memilio  = load_memilio,
    odin     = load_odin,
    sbml     = load_sbml,
    siminf   = load_siminf,
    summer2  = load_summer2,
)

# The Python packages of the frameworks, as imported in a file of their models, by format; LASER has no importer, but is named so that the error can say so
py_modules = dict(starsim='starsim', summer2='summer2', epydemix='epydemix', epipack='epipack', epymorph='epymorph', memilio='memilio', laser_core='laser', laser='laser')
scripted = ['summer2', 'epydemix', 'epipack', 'epymorph', 'memilio'] # The formats whose importer reads a JSON file, written by running the importer's module as a script in the framework's environment


def sniff(source):
    """ The format of a file of another framework's model, if it can be told from the file (else None) """
    if not isinstance(source, (str, sc.path().__class__)):
        if all(hasattr(source, attr) for attr in ['compartments', 'flows', '_stratifications']): # A live summer2 model
            return 'summer2'
        elif all(hasattr(source, attr) for attr in ['transitions_list', 'population']): # A live epydemix model
            return 'epydemix'
        elif hasattr(source, 'set_processes') or hasattr(source, 'node_transition_events'): # A live epipack model
            return 'epipack'
        return None
    elif not sc.path(source).is_file():
        return None
    path = sc.path(source)
    ext = path.suffix.lower()
    if ext == '.xlsx': # An Atomica framework has these sheets
        try:
            import openpyxl
            sheets = openpyxl.load_workbook(path, read_only=True).sheetnames
        except Exception:
            return None
        return 'atomica' if {'compartments', 'transitions'} <= {name.lower() for name in sheets} else None # In any case, as Atomica reads them
    elif ext == '.camdl':
        return 'camdl'
    elif ext == '.py': # By what it imports: Starsim, or a framework with an importer (which reads the model by running the file; see import_from()), or one without
        names = set(re.findall(r'^\s*(?:import|from)\s+(\w+)', path.read_text(errors='ignore'), re.M))
        found = {fmt for module,fmt in py_modules.items() if module in names}
        if len(found) > 1:
            found -= {'starsim'} # E.g. a script that compares a model with Starsim's copy of it
        return found.pop() if len(found) == 1 else None
    text = path.read_text(errors='ignore') if ext in ['.yaml', '.yml', '.r', '.xml', '.sbml', '.json'] else ''
    top = lambda key: re.search(rf'^{key}\s*:', text, re.M) # A key at the top level of a YAML file
    if ext in ['.yaml', '.yml']:
        if top('state_machines'):
            return 'emulsion'
        elif top('compartments') and top('seir'):
            return 'flepimop'
    elif ext == '.r':
        if re.search(r'EpiModel', text) and re.search(r'\b(netsim|dcm|icm)\s*\(', text):
            return 'epimodel'
        elif re.search(r'^\s*(deriv|update|initial)\s*\(', text, re.M) and 'mparse(' not in text:
            return 'odin'
        elif 'mparse(' in text or (re.search(r'\b(SIR|SIS|SEIR|SISe3?)\s*\(', text) and 'tspan' in text): # SimInf's own models need tspan
            return 'siminf'
    elif ext in ['.xml', '.sbml'] and '<sbml' in text:
        return 'sbml'
    elif ext == '.json' and re.match(r'\s*\{\s*"starsim"\s*:', text): # Starsim's own JSON form, which starts with the version that wrote it
        return 'json'
    elif ext == '.json': # As written by summer2.py, epydemix.py, epipack.py, epymorph.py, or memilio.py (run as scripts), or dump_epimodel.R; or camdlc's JSON form
        match = re.search(r'"format"\s*:\s*"(summer2|epydemix|epimodel|epipack|epymorph|memilio)"', text)
        if match:
            return match.group(1)
        elif '"ir_version"' in text and '"model"' in text:
            return 'camdl'
    return None


def import_from(source, format=None, **kwargs):
    """
    Read another framework's model into a sim, e.g. `ss.import_from('workbook.xlsx')`

    Args:
        source (str/any): the file (or, for some formats, the model object)
        format (str): the framework, one of `ss.converters.importers`; if not given, told from the file or object (see `sniff()`)
        kwargs (dict): passed to the importer

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated
    """
    if format is None:
        format = sniff(source)
        if format is None:
            errormsg = f'Cannot tell which framework {source} is from; give format=, one of {sc.strjoin(importers)}'
            raise ValueError(errormsg)
    key = str(format).lower().replace('-', '').replace('_', '')
    if key not in importers:
        errormsg = f'Cannot import format "{format}"; the formats are {sc.strjoin(importers)}'
        raise ValueError(errormsg)
    check_version(key)
    if key in scripted and isinstance(source, (str, os.PathLike)) and sc.path(source).suffix == '.py': # The model's own script: read by running it
        with tempfile.TemporaryDirectory() as folder:
            return importers[key](dump(source, key, sc.path(folder)/'model.json'), **kwargs)
    return importers[key](source, **kwargs)


def dump(source, format, filename):
    """ Write the JSON that the importer of a framework reads (see `scripted`) by running its module on a script of the model, which needs the framework installed in this Python """
    script = sc.thispath() / f'{format}.py'
    if importlib.util.find_spec(format) is None:
        errormsg = (f'{source} is a model for {format}, which is read by running it, but {format} is not installed in this Python. '
                    f'In an environment that has it, run:\n  python {script} {source} model.json\nand then import model.json')
        raise ValueError(errormsg)
    subprocess.run([sys.executable, str(script), str(source), str(filename)], check=True)
    return str(filename)


def convert(source, filename=None, format=None, **kwargs):
    """
    Read another framework's model, as `import_from()` does, and save the Starsim code for it: read the model once, then run the code (which takes a fraction of the time)

    Args:
        source (str/any): the file (or, for some formats, the model object)
        filename (str): where to save the code (default: beside the source, with its name and .py)
        format (str): the framework (see `import_from()`)
        kwargs (dict): passed to the importer

    Returns:
        The `ss.Sim`, as `import_from()` gives it, with `sim.imported.filename` the file the code was saved to

    Example:
        ```python
        sim = ss.converters.convert('tb_framework.xlsx', format='atomica', databook='IND.xlsx', filename='tb_india.py')
        sim.run() # Or later: python tb_india.py, or runpy.run_path('tb_india.py')['sim']
        ```
    """
    if filename is None:
        if not isinstance(source, (str, os.PathLike)):
            errormsg = f'Give filename= for the code, since the model ({type(source).__name__}) is not a file'
            raise ValueError(errormsg)
        filename = sc.path(source).with_suffix('.py')
    sim = import_from(source, format=format, **kwargs)
    sc.savetext(filename, sim.imported.code)
    sim.imported.filename = sc.path(filename)
    return sim
