"""
Import models from other frameworks

`ss.load(filename, format=...)` reads another framework's model into a sim; see
`v4_design/design/16-interop.md`. Each importer reads its format into an
`ss.interop.Source`, and `ss.interop.build()` turns that into a sim by the same
rules for every format: how each rate was read is printed, and what could not be
translated is listed as `W1601 ImportIncomplete` and kept in `sim.imported`.
"""
import re
import sciris as sc
from .core import Source, build
from .emulsion import load_emulsion
from .atomica import load_atomica
from .flepimop import load_flepimop
from .odin import load_odin
from .sbml import load_sbml
from .siminf import load_siminf
from .summer2 import load_summer2

__all__ = ['Source', 'build', 'formats', 'sniff', 'load']

formats = dict(
    atomica  = load_atomica,
    emulsion = load_emulsion,
    flepimop = load_flepimop,
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
        return 'atomica' if {'Compartments', 'Transitions'} <= set(sheets) else None
    text = path.read_text(errors='ignore') if ext in ['.yaml', '.yml', '.r', '.xml', '.sbml', '.json'] else ''
    top = lambda key: re.search(rf'^{key}\s*:', text, re.M) # A key at the top level of a YAML file
    if ext in ['.yaml', '.yml']:
        if top('state_machines'):
            return 'emulsion'
        elif top('compartments') and top('seir'):
            return 'flepimop'
    elif ext == '.r':
        if re.search(r'^\s*(deriv|update|initial)\s*\(', text, re.M) and 'mparse(' not in text:
            return 'odin'
        elif 'mparse(' in text or (re.search(r'\b(SIR|SIS|SEIR|SISe3?)\s*\(', text) and 'tspan' in text): # SimInf's own models need tspan
            return 'siminf'
    elif ext in ['.xml', '.sbml'] and '<sbml' in text:
        return 'sbml'
    elif ext == '.json' and re.search(r'"format"\s*:\s*"summer2"', text): # As written by dump_summer2.py
        return 'summer2'
    return None


def load(source, format, **kwargs):
    """
    Read another framework's model into a sim; usually called as `ss.load(source, format=...)`

    Args:
        source (str/any): the file (or, for some formats, the model object)
        format (str): the framework, one of `ss.interop.formats`
        kwargs (dict): passed to the importer

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated
    """
    key = str(format).lower().replace('-', '').replace('_', '')
    if key not in formats:
        errormsg = f'Cannot import format "{format}"; the formats are {sc.strjoin(formats)}'
        raise ValueError(errormsg)
    return formats[key](source, **kwargs)
