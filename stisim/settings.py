"""
Define options for STIsim.
All options should be set using set() or directly, e.g.::

    ss.options(verbose=False)
"""

import os
import numpy as np
import sciris as sc


__all__ = ['options', 'float_', 'int_']


# Not public to avoid confusion with ss.options
class Options(sc.objdict):
    """
    Set options for STIsim.

    Use ``ss.options.set('defaults')`` to reset all values to default, or ``ss.options.set(dpi='default')``
    to reset one parameter to default. See ``ss.options.help(detailed=True)`` for
    more information.

    Options can also be saved and loaded using ``ss.options.save()`` and ``ss.options.load()``.
    See ``ss.options.context()`` and ``ss.options.with_style()`` to set options
    temporarily.

    Common options are (see also ``ss.options.help(detailed=True)``):

        - verbose:        default verbosity for simulations to use
        - warnings:       how to handle warnings (e.g. print, raise as errors, ignore)

    **Examples**::

        ss.options(verbose=True) # Set more verbosity
        ss.options(warn='error') # Be more strict about warnings
    """

    def __init__(self):
        super().__init__()
        optdesc, options = self.get_orig_options()  # Get the options
        self.update(options)  # Update this object with them
        self.setattribute('optdesc', optdesc)  # Set the description as an attribute, not a dict entry
        self.setattribute('orig_options', sc.dcp(options))  # Copy the default options
        return
    
    @staticmethod
    def get_orig_options():
        """
        Set the default options for STIsim -- not to be called by the user, use
        ``ss.options.set('defaults')`` instead.
        """

        # Options acts like a class, but is actually an objdict for simplicity
        optdesc = sc.objdict()  # Help for the options
        options = sc.objdict()  # The options

        optdesc.multirng = 'Set True to use multiple random number generators\
            via the MultiRNG class, which is compatible with common random\
            numbers. The default value is False, which instead uses a single\
            centralized random number generator as implemented in the SingleRNG\
            class.'
        options.multirng = False

        optdesc.verbose = 'Set default level of verbosity (i.e. logging detail): e.g., 0.1 is an update every 10 simulated timesteps.'
        options.verbose = float(os.getenv('STISIM_VERBOSE', 0.1))

        optdesc.warnings = 'How warnings are handled: options are "warn" (default), "print", and "error"'
        options.warnings = str(os.getenv('STISIM_WARNINGS', 'warn'))

        optdesc.sep = 'Set thousands seperator for text output'
        options.sep = str(os.getenv('STISIM_SEP', ','))

        optdesc.precision = 'Set arithmetic precision -- 32-bit by default for efficiency'
        options.precision = int(os.getenv('STISIM_PRECISION', 32))

        return optdesc, options

    def __call__(self, *args, **kwargs):
        """Allow ``ss.options(dpi=150)`` instead of ``ss.options.set(dpi=150)`` """

        return self.set(*args, **kwargs)

    def __repr__(self):
        """ Brief representation """
        output = sc.objectid(self)
        output += 'STIsim options (see also ss.options.disp()):\n'
        output += sc.pp(self.to_dict(), output=True)
        return output

    def __enter__(self):
        """ Allow to be used in a with block """
        return self

    def __exit__(self, *args, **kwargs):
        """ Allow to be used in a with block """
        try:
            reset = {}
            for k, v in self.on_entry.items():
                if self[k] != v:  # Only reset settings that have changed
                    reset[k] = v
            self.set(**reset)
            self.delattribute('on_entry')
        except AttributeError as E:
            errormsg = 'Please use ss.options.context() if using a with block'
            raise AttributeError(errormsg) from E
        return

    def disp(self):
        """ Detailed representation """
        output = 'STIsim options (see also ss.options.help()):\n'
        keylen = 10  # Maximum key length  -- "interactive"
        for k, v in self.items():
            keystr = sc.colorize(f'  {k:>{keylen}s}: ', fg='cyan', output=True)
            reprstr = sc.pp(v, output=True)
            reprstr = sc.indent(n=keylen + 4, text=reprstr, width=None)
            output += f'{keystr}{reprstr}'
        print(output)
        return

    def set(self, key=None, value=None, use=False, **kwargs):
        """
        Actually change the style. See ``ss.options.help()`` for more information.

        Args:
            key    (str):    the parameter to modify, or 'defaults' to reset everything to default
            value  (varies): the value to specify; use None or 'default' to reset to default
            use (bool): whether to use the chosen style
            kwargs (dict):   if supplied, set multiple key-value pairs

        **Example**::
            ss.options.set(dpi=50) # Equivalent to ss.options(dpi=50)
        """

        # Reset to defaults
        if key in ['default', 'defaults']:
            kwargs = self.orig_options  # Reset everything to default

        # Handle other keys
        elif key is not None:
            kwargs = sc.mergedicts(kwargs, {key: value})
        
        # Reset options
        for key, value in kwargs.items():
            if key not in self:
                keylist = self.orig_options.keys()
                keys = '\n'.join(keylist)
                errormsg = f'Option "{key}" not recognized; options are "defaults" or:\n{keys}\n\nSee help(ss.options.set) for more information.'
                raise sc.KeyNotFoundError(errormsg)
            else:
                if value in [None, 'default']:
                    value = self.orig_options[key]
                self[key] = value

        return

    def context(self, **kwargs):
        """
        Alias to set() for non-plotting options, for use in a "with" block.

        Note: for plotting options, use ``ss.options.with_style()``, which is linked
        to Matplotlib's context manager. If you set plotting options with this,
        they won't have any effect.

        **Examples**::

            # Silence all output
            with ss.options.context(verbose=0):
                ss.Sim().run()

            # Convert warnings to errors
            with ss.options.context(warnings='error'):
                ss.Sim(location='not a location').initialize()

            # Use with_style(), not context(), for plotting options
            with ss.options.with_style(dpi=50):
                ss.Sim().run().plot()

        """

        # Store current settings
        on_entry = {k: self[k] for k in kwargs.keys()}
        self.setattribute('on_entry', on_entry)

        # Make changes
        self.set(**kwargs)
        return self

    def get_default(self, key):
        """ Helper function to get the original default options """
        return self.orig_options[key]

    def changed(self, key):
        """ Check if current setting has been changed from default """
        if key in self.orig_options:
            return self[key] != self.orig_options[key]
        else:
            return None


# Create the options on module load, and load the fonts
options = Options()

# Default for precision
# Used in various places throughout the code, generally as:
#   import stisim.settings as sss
#   arr = np.full(100, 0, sss.float_)
result_float = np.float64  # Always use float64 for results, for simplicity
if options.precision == 32:
    float_ = np.float32
    int_ = np.int32
elif options.precision == 64:  # pragma: no cover
    float_ = np.float64
    int_ = np.int64
else:
    raise NotImplementedError(f'Precision must be either 32 bit or 64 bit, not {options.precision}')