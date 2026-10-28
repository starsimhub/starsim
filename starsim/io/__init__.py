"""
Import models from other frameworks, and export sims to them

`ss.import_from(filename, format=...)` reads another framework's model into a sim, and
`ss.export_to(sim, format)` writes a sim in another framework's format; see
`v4_design/design/16-interop.md`. Each importer reads its format into an
`ss.io.Source`, and `ss.io.build()` writes that as Starsim code by the
same rules for every format (`ss.io.write()`) and runs the code to make the sim:
the code is kept in `sim.imported.code`, how each rate was read is printed, and what
could not be translated is listed as `ss.errors.ImportIncomplete` and kept in `sim.imported`.
"""
import os
import re
import sciris as sc
from .core import Source, Code, write, build
from .camdl import load_camdl
from .emulsion import load_emulsion
from .epimodel import load_epimodel
from .epydemix import load_epydemix
from .atomica import load_atomica
from .flepimop import load_flepimop
from .memilio import load_memilio
from .odin import load_odin
from .sbml import load_sbml
from .siminf import load_siminf
from .summer2 import load_summer2
from .export import exporters, export_to, to_hub

__all__ = ['Source', 'Code', 'write', 'build', 'importers', 'sniff', 'import_from', 'convert', 'exporters', 'export_to', 'to_hub']

importers = dict(
    atomica  = load_atomica,
    camdl    = load_camdl,
    emulsion = load_emulsion,
    epimodel = load_epimodel,
    epydemix = load_epydemix,
    flepimop = load_flepimop,
    memilio  = load_memilio,
    odin     = load_odin,
    sbml     = load_sbml,
    siminf   = load_siminf,
    summer2  = load_summer2,
)


def sniff(source):
    """ The format of a file of another framework's model, if it can be told from the file (else None) """
    if not isinstance(source, (str, sc.path().__class__)):
        if all(hasattr(source, attr) for attr in ['compartments', 'flows', '_stratifications']): # A live summer2 model
            return 'summer2'
        elif all(hasattr(source, attr) for attr in ['transitions_list', 'population']): # A live epydemix model
            return 'epydemix'
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
    elif ext == '.json': # As written by summer2.py, epydemix.py, or memilio.py (run as scripts), or dump_epimodel.R; or camdlc's JSON form
        match = re.search(r'"format"\s*:\s*"(summer2|epydemix|epimodel|memilio)"', text)
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
        format (str): the framework, one of `ss.io.importers`; if not given, told from the file or object (see `sniff()`)
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
    return importers[key](source, **kwargs)


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

    **Example**:

        sim = ss.io.convert('tb_framework.xlsx', format='atomica', databook='IND.xlsx', filename='tb_india.py')
        sim.run() # Or later: python tb_india.py, or runpy.run_path('tb_india.py')['sim']
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
