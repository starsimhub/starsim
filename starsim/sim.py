"""
Define core Sim classes
"""
import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

__all__ = ['Sim', 'AlreadyRunError', 'demo', 'diff_sims', 'check_sims_match']


class Sim(ss.Base):
    """
    The Sim object

    All Starsim simulations run via the Sim class. It is responsible for initializing
    and running all modules and generating results.

    Args:
        pars (SimPars/dict): either an ss.SimPars object, or a nested dictionary; can include all other arguments
        label (str): the human-readable name of the simulation
        people (People): if provided, use this ss.People object
        modules (Module/list): if provided, use these modules (and divide among demographics, diseases, etc. based on type)
        demographics (str/Demographics/list): a string naming the demographics module to use, the module itself, or a list
        connectors (str/Connector/list): as above, for connectors
        networks (str/Network/list): as above, for networks
        interventions (str/Intervention/list): as above, for interventions
        diseases (str/Disease/list): as above, for diseases
        analyzers (str/Analyzer/list): as above, for analyzers
        copy_inputs (bool): if True, copy modules as they're inserted into the sim (allowing reuse in other sims, but meaning they won't be updated)
        data (df): a dataframe (or dict) of data, with a column "time" plus data of the form "module.result", e.g. "hiv.new_infections" (used for plotting only)
        kwargs (dict): merged with pars; see ss.SimPars for all parameter values

    **Examples**:

        sim = ss.Sim(diseases='sir', networks='random') # Simplest Starsim sim; equivalent to ss.demo()
        sim = ss.Sim(diseases=ss.SIR(), networks=ss.RandomNet()) # Equivalent using objects instead of strings
        sim = ss.Sim(diseases=['sir', ss.SIS()], networks=['random', 'mf']) # Example using list inputs; can mix and match types
    """
    def __init__(self, pars=None, label=None, people=None, modules=None, demographics=None, diseases=None, networks=None,
                 interventions=None, analyzers=None, connectors=None, copy_inputs=True, data=None, **kwargs):
        self.pars = ss.SimPars() # Make default parameters (using values from parameters.py)
        args = dict(label=label, people=people, modules=modules, demographics=demographics, connectors=connectors,
                    networks=networks, interventions=interventions, diseases=diseases, analyzers=analyzers)
        args = {key:val for key,val in args.items() if val is not None} # Remove None inputs
        input_pars = sc.mergedicts(pars, args, kwargs, _copy=copy_inputs)
        self.pars.update(input_pars)  # Update the parameters

        # Set attributes; see also sim.init() for more
        self.created = sc.now()  # The datetime the sim was created
        self.version = ss.__version__ # The Starsim version
        self.metadata = ss.metadata()
        self.dists = ss.Dists(obj=self) # Initialize the random number generator container
        self.loop = ss.Loop(self) # Initialize the integration loop
        self.results = ss.Results(module='Sim')  # For storing results
        self.data = data # For storing input data
        self.initialized = False  # Whether initialization is complete
        self.complete = False  # Whether a simulation has completed running
        self.results_ready = False  # Whether results are ready
        self.elapsed = None # The time required to run
        self.summary = None  # For storing a summary of the results
        self.filename = None # Store the filename, if saved
        return

    def __getitem__(self, key):
        """ Allow dict-like access, e.g. sim['diseases'] """
        return getattr(self, key)

    def __setitem__(self, key, value):
        """ Allow dict-like access, e.g. sim['created'] = sc.now() """
        return setattr(self, key, value)

    def __repr__(self):
        """ Show a quick version of the sim """
        # Try a more custom repr first
        try:
            labelstr = f'{self.label}; ' if self.label else ''
            n = int(self.pars.n_agents)
            if self.initialized:
                timestr = f'{self.t.start}—{self.t.stop}'
            else:
                timestr = f'{self.pars.start}—{self.pars.stop}'

            moddict = {}
            for modkey in ss.module_map().keys():
                if hasattr(self, modkey):
                    thismodtype = self[modkey]
                elif modkey in self.pars:
                    thismodtype = self.pars[modkey]
                else:
                    thismodtype = {}

                if isinstance(thismodtype, dict) and len(thismodtype):
                    moddict[modkey] = sc.strjoin(thismodtype.keys())
                elif isinstance(thismodtype, str):
                    moddict[modkey] = thismodtype

            if len(moddict):
                modulestr = ''
                for k,mstr in moddict.items():
                    modulestr += f'; {k}={mstr}'
            else:
                modulestr = ''
            if not self.initialized:
                modulestr += '; not initialized'

            string = f'Sim({labelstr}n={n:n}; {timestr}{modulestr})'

        # Or just use default
        except Exception as E:
            ss.warn(f'Error displaying custom sim repr, falling back to default: {E}')
            string = sc.prepr(self, vals=False)

        return string

    def products(self):
        """ List all products across interventions; not an ndict like the other module types """
        products = []
        for intv in self.interventions():
            if intv.has_product:
                products.append(intv.product)
        return products

    @property
    def module_list(self):
        """ Return a list of all Module instances (stored in standard places) in the Sim; see `sim.module_dict` for the dict version """
        out = sc.mergelists(
            self.modules(),
            self.demographics(),
            self.connectors(),
            self.networks(),
            self.interventions(),
            self.products(),
            self.diseases(),
            self.analyzers(),
        )
        return out

    @property
    def module_dict(self):
        """ Return a dictionary of all Module instances; see `sim.module_list` for the list version """
        return ss.utils.nlist_to_dict(self.module_list)

    def init(self, force=False, timer=False, **kwargs):
        """
        Perform all initializations for the sim

        Args:
            force (bool): whether to overwrite sim attributes even if they already exist
            timer (bool): if True, count the time required for initialization (otherwise just count run time)
            kwargs (dict): passed to ss.People()
        """
        # Handle the timer
        self.timer = sc.timer() # Store a timer for keeping track of how long the run takes
        if timer:
            self.timer.start()

        # Validation and initialization -- this is "pre"
        np.random.seed(self.pars.rand_seed) # Reset the seed before the population is created -- shouldn't matter if only using Dist objects
        self.pars.validate() # Validate parameters
        self.init_time() # Initialize time
        self.init_people(**kwargs) # Initialize the people
        self.init_module_attrs(force=force) # Load specified modules from parameters, initialize them, and move them to the Sim object
        self.init_modules_pre()

        # Final initializations -- this is "post"
        self.init_dists() # Initialize distributions
        self.init_people_vals() # Initialize the values in all the states and networks
        self.init_modules_post() # Initialize the module values
        self.init_results() # Initialize the results
        self.init_data() # Initialize the data
        self.loop.init() # Initialize the integration loop

        self.verbose = self.pars.verbose # Store a run-specific value of verbose

        # It's initialized
        self.initialized = True
        if timer:
            self.elapsed = self.timer.toc(label='init', output=True, verbose=self.verbose)
        return self

    def init_time(self):
        """ Time indexing; derived values live in the sim rather than in the pars """
        self.t = ss.Timeline(name='sim')
        self.t.init(self)
        self.results.timevec = self.timevec # Store the timevec in the results for plotting
        return

    def init_people(self, verbose=None, **kwargs):
        """
        Initialize people within the sim
        Sometimes the people are provided, in which case this just adds a few sim properties to them.
        Other time people are not provided and this method makes them.

        Args:
            verbose (int):  detail to print
            kwargs  (dict): passed to ss.People()
        """
        # Handle inputs
        people = self.pars.pop('people')
        n_agents = self.pars.n_agents
        verbose = sc.ifelse(verbose, self.pars.verbose)
        if verbose > 0:
            labelstr = f' "{self.label}"' if self.label else ''
            print(f'Initializing sim{labelstr} with {n_agents:0n} agents')

        # If people have not been supplied, make them -- typical use case
        if people is None:
            people = ss.People(n_agents=n_agents, **kwargs)  # This just assigns UIDs and length

        # Finish up (NB: the People object is not yet initialized)
        self.people = people
        self.people.link_sim(self)
        return self.people

    @property
    def label(self):
        """
        Get the sim label from the parameters, if available.

        Note: for `ss.Sim` objects, the label is stored as `sim.pars.label`,
        and `sim.label` is an alias to it. Sims do not have names separate from
        their labels. For `ss.Module` objects, the name and label are both attributes
        (`mod.name` and `mod.label`), with the difference being that the names
        are machine-readable (e.g. `'my_sir'`) while the labels are human-readable
        (e.g. `'My SIR module'`).
        """
        try:    return self.pars.label
        except: return None

    @label.setter
    def label(self, label):
        """
        Set the sim label (actually sets `sim.pars.label`)
        """
        self.pars['label'] = label
        return

    def init_module_attrs(self, force=False):
        """ Move initialized modules to the sim """
        module_types = ss.modules.module_types()
        for attr in module_types:
            orig = getattr(self, attr, None)
            if not force and orig is not None:
                warnmsg = f'Skipping key "{attr}" in parameters since already present in sim and force=False'
                ss.warn(warnmsg)
            else:
                modules = self.pars.pop(attr) # Remove module type (e.g. 'diseases') from the parameters
                setattr(self, attr, modules) # Add the modules ndict to the sim
                self.pars[attr] = sc.objdict() # Recreate with just the module parameters
                for key,module in modules.items():
                    self.pars[attr][key] = module.pars

        return

    def init_modules_pre(self):
        """ Initialize all the modules with the sim """
        for mod in self.module_list:
            mod.init_pre(self)
        return

    def init_dists(self):
        """ Initialize the distributions """
        # Initialize all distributions now that everything else is in place
        self.dists.init(obj=self, base_seed=self.pars.rand_seed, force=True)

        # Copy relevant dists to each module
        for mod in self.module_list:
            self.dists.copy_to_module(mod)
        return

    def init_people_vals(self):
        """ Initialize the People states with actual values """
        self.people.init_vals()
        return

    def init_modules_post(self):
        """ Initialize values in other modules, including networks and time parameters, and do any other post-processing """
        for mod in self.module_list:
            mod.init_post()
        return

    def reset_time_pars(self, force=True):
        """ Reset the time parameters in the modules; used for imposing the sim's timestep on the modules """
        for mod in self.module_list:
            mod.init_time_pars(force=force)
        return

    def init_results(self):
        """ Create initial results that are present in all simulations """
        kw = dict(module='Sim', shape=self.t.npts, timevec=self.t.timevec, dtype=int, scale=True)
        self.results += [
            ss.Result('n_alive',    label='Number alive', **kw),
            ss.Result('new_deaths', label='Deaths', **kw),
            ss.Result('cum_deaths', label='Cumulative deaths', **kw),
        ]
        return

    def init_data(self, data=None):
        """ Initialize or add data to the sim """
        data = data if data is not None else self.data
        self.data = ss.validate_sim_data(data)
        return

    def start_step(self):
        """ Start the step -- only print progress; all actual changes happen in the modules """

        # Set the time and if we have reached the end of the simulation, then do nothing
        if self.complete:
            errormsg = 'Simulation already complete (call sim.init() to re-run)'
            raise AlreadyRunError(errormsg)

        # Print out progress if needed
        self.elapsed = self.timer.toc(output=True)
        if self.verbose: # Print progress
            t = self.t
            simlabel = f'"{self.label}": ' if self.label else ''
            string = f'  Running {simlabel}{t.now("str")} ({t.ti:2.0f}/{t.npts}) ({self.elapsed:0.2f} s) '
            if self.verbose >= 1:
                sc.heading(string)
            elif self.verbose > 0:
                if not (t.ti % int(1.0 / self.verbose)):
                    sc.progressbar(t.ti + 1, t.npts, label=string, length=20, newline=True)
        return

    def finish_step(self):
        """ Finish the simulation timestep """
        self.t.ti += 1
        return

    def run_one_step(self, verbose=None):
        """
        Run a single sim step; only used for debugging purposes.

        Note: sim.run_one_step() runs a single simulation timestep, which involves
        multiple function calls. In contrast, loop.run_one_step() runs a single
        function call.

        Note: the verbose here is only for the Loop object, not the sim.
        """
        self.loop.run(self.t.now(), verbose)
        return self

    def run(self, until=None, verbose=None, check_method_calls=True):
        """
        Run the model -- the main method for running a simulation.

        Args:
            until (date/str/float): the date to run the sim until
            verbose (float): the level of detail to print (default 0.1, i.e. output once every 10 steps)
            check_method_calls (bool): whether to check that all required methods were called
        """
        # Initialization steps
        if not self.initialized: self.init()
        if verbose is not None:
            self._orig_verbose = self.verbose
            self.verbose = verbose
        self.timer.start()

        # Check for AlreadyRun errors
        errormsg = None
        if self.complete:
            errormsg = 'Simulation is already complete (call sim.init() to re-run)'
        if errormsg:
            raise AlreadyRunError(errormsg)

        # Main simulation loop -- just one line!!!
        self.loop.run(until)

        # Check if the simulation is complete
        if self.loop.index == len(self.loop.plan):
            self.complete = True

        # If simulation reached the end, finalize the results
        if self.complete:
            self.finalize()
            if check_method_calls:
                self.check_method_calls()
            sc.printv(f'Run finished after {self.elapsed:0.2f} s.\n', 1, self.verbose)
        return self # Allows e.g. ss.Sim().run().plot()

    def finalize(self):
        """ Compute final results """
        if self.results_ready:
            # Because the results are rescaled in-place, finalizing the sim cannot be run more than once or
            # otherwise the scale factor will be applied multiple times
            raise AlreadyRunError('Simulation has already been finalized')

        # Reset the time index (done in the modules as well in mod.finalize())
        self.t.ti -= 1  # During the run, this keeps track of the next step; restore this be the final day of the sim

        # Finalize the results
        self.finalize_results()

        # Finalize each module, including the results
        for module in self.module_list:
            module.finalize()

        # Resets verbose if needed
        if hasattr(self, '_orig_verbose'):
            self.verbose = self._orig_verbose
            delattr(self, '_orig_verbose')

        # Generate the summary and finish up
        self.summarize() # Create summary
        return

    def finalize_results(self):
        """ Scale the results and remove any "unused" results """
        for reskey, res in self.results.items():
            if isinstance(res, ss.Result): # Note: since Result is a NumPy array, "res" and self.results[key] are not the same object
                if res.scale: # Scale results; NB: disease-specific results are scaled in module.finalize() below
                    self.results[reskey] = self.results[reskey] * self.pars.pop_scale
                if np.all(res == res[0]): # Results were not modified during the sim
                    self.results[reskey].auto_plot = False
        self.results_ready = True # Results are ready to use
        return

    def summarize(self, how='default'):
        """
        Provide a quick summary of the sim; returns the last entry for count and
        cumulative results, and the mean otherwise.

        Args:
            how (str): how to summarize: can be 'mean', 'median', 'last', or a mapping of result keys to those
        """
        def get_func(key, how, default='mean'):
            """
            Find the right function by matching the "how" key with the result key

            For example, hkey="cum_ " will match result key "cum_infections"
            """
            func = None
            for hkey,hfunc in how.items():
                if hkey in key:
                    func = hfunc
                    break
            if func is None:
                func = default
            return func

        def get_result(res, func):
            """ Convert a string to the actual function to use, e.g. "median" maps to np.median() """
            if   func == 'mean':   return res.mean()
            elif func == 'median': return np.median(res)
            elif func == 'last':   return res[-1]
            elif callable(func):   return func(res)
            else: raise Exception(f'"{func}" is not a valid function')

        # Convert "how" from a string to a dict
        if how == 'default':
            how = {'n_':'mean', 'new_':'mean', 'cum_':'last', 'timevec':'last', '':'mean'}
        elif isinstance(how, str):
            how = {'':how} # Match everything

        summary = sc.objdict()
        flat = sc.flattendict(self.results, sep='_')
        for key, res in flat.items():
            if 'timevec' not in key: # Skip module-specific time vectors
                try:
                    func = get_func(key, how)
                    entry = get_result(res, func)
                except Exception as E:
                    entry = f'N/A {E}'
                summary[key] = entry
        self.summary = summary
        return summary

    def shrink(self, inplace=True, size_limit=1.0, intercept=10, die=True):
        """
        "Shrinks" the simulation by removing the people and other memory-intensive
        attributes (e.g., some interventions and analyzers), and returns a copy of
        the "shrunken" simulation. Used to reduce the memory required for RAM or
        for saved files.

        Args:
            inplace (bool): whether to perform the shrinking in place (default), or return a shrunken copy instead
            size_limit (float): print a warning if any module is larger than this size limit, in units of KB per timestep (set to None to disable)
            intercept (float): the size (in units of size_limit) to allow for a zero-timestep sim
            die (bool): whether to raise an exception if the shrink failed

        Returns:
            shrunken (Sim): a Sim object with the listed attributes removed
        """
        # Create the new object, and copy original dict, skipping the skipped attributes
        if inplace:
            sim = self
        else:
            sim = self.copy() # We need to do a deep copy to avoid modifying other objects

        # Shrink the people and loop
        shrunk = ss.utils.shrink()
        sim.people = shrunk
        with sc.tryexcept(die=die):
            sim.loop.shrink()

        # If the sim is not initialized, we're done (ignoring the corner case where initialized modules are passed to an uninitialized sim)
        if sim.initialized:

            # Shrink the distributions
            sim.dists.sim = shrunk
            sim.dists.obj = shrunk
            for dist in sim.dists.dists.values():
                with sc.tryexcept(die=die):
                    dist.shrink()

            # Finally, shrink the modules
            for mod in sim.module_list:
                with sc.tryexcept(die=die):
                    mod.shrink()

            # Check that the module successfully shrunk
            if size_limit:
                max_size = size_limit*(len(sim)+intercept) # Maximum size in KB
                for mod in sim.module_list:
                    size = sc.checkmem(mod, descend=0).bytesize[0]/1e3 # Size in KB
                    if size > max_size:
                        errormsg = f'Module {mod.name} did not successfully shrink: {size:n} KB > {max_size:n} KB; use die=False to turn this message into a warning, or change size_limit to a larger value'
                        if die:
                            raise RuntimeError(errormsg)
                        else:
                            ss.warn(errormsg)

        # Finally, set a flag that the sim has been shrunken
        sim.shrunken = True
        return sim

    def check_results_ready(self, errormsg=None):
        """ Check that results are ready """
        if errormsg is None:
            errormsg = 'Please run the sim first'
        if not self.results_ready:  # pragma: no cover
            raise RuntimeError(errormsg)
        return

    def check_method_calls(self, die=None, warn=None, verbose=False):
        """
        Check if any required methods were not called.

        Typically called automatically by `sim.run()`; default behavior is to warn
        (see `options.check_method_calls`).

        Args:
            die (bool): whether to raise an exception if missing methods were found (default False)
            warn (bool): whether to raise a warning if missing methods were found (default False)
            verbose (bool): whether to print the number of times each method was called (default False)

        Returns:
            A list of missing method calls by module
        """
        # Handle arguments, or fall back to options
        valid = ['die', 'warn', False]
        if die is not None or warn is not None:
            check = 'die' if die else 'warn' if warn else False
        else:
            check = ss.options.check_method_calls
        if check not in valid:
            errormsg = f'Could not understand {check=}, must be one of {valid}'
            raise ValueError(errormsg)

        if check:
            if verbose:
                sc.pp(self._call_required)

            missing = []
            for mod in self.module_list:
                modmissing = mod.check_method_calls()
                if modmissing:
                    missing.append([type(mod), modmissing])
            if missing:
                errormsg = 'The following methods are required, but were not called.\n'
                errormsg += 'Did you mistype a method name, forget a super() call,\n'
                errormsg += 'or did part of the sim not run (e.g. zero infections)?\n'
                for modtype,modmissing in missing:
                    errormsg += f'{modtype}: {sc.strjoin(modmissing)}\n'
                if check == 'die':
                    raise RuntimeError(errormsg)
                elif check == 'warn':
                    ss.warn(errormsg)
        return missing

    def to_df(self, sep='_', **kwargs):
        """
        Export results as a Pandas dataframe
        Args:
            sep (str): separator for the keys
            kwargs: passed to results.to_df()
        """
        self.check_results_ready('Please run the sim before exporting the results')
        df = self.results.to_df(sep=sep, descend=True, **kwargs)
        return df

    def profile(self, line=True, do_run=True, plot=True, follow=None, **kwargs):
        """
        Profile the performance of the simulation using a line profiler (`sc.profile()`)

        Args:
            do_run (bool): whether to immediately run the sim
            plot (bool): whether to plot time spent per module step
            follow (func/list): a list of functions/methods to follow in detail
            **kwargs (dict): passed to `sc.profile()`

        **Example**:

            import starsim as ss

            net = ss.RandomNet()
            sis = ss.SIS()
            sim = ss.Sim(networks=net, diseases=sis)
            prof = sim.profile(follow=[net.add_pairs, sis.infect])
        """
        prof = ss.Profile(self, follow=follow, do_run=do_run, plot=plot, **kwargs)
        return prof

    def cprofile(self, sort='cumtime', mintime=1e-3, **kwargs):
        """
        Profile the performance of the simulation using a function profiler (`sc.cprofile()`)

        Args:
            sort (str): default sort column; common choices are 'cumtime' (total time spent in a function, includin subfunctions) and 'selftime' (excluding subfunctions)
            mintime (float): exclude function calls less than this time in seconds
            **kwargs (dict): passed to `sc.cprofile()`

        **Example**:

            import starsim as ss

            net = ss.RandomNet()
            sis = ss.SIS()
            sim = ss.Sim(networks=net, diseases=sis)
            prof = sim.cprofile()
        """
        cprof = sc.cprofile(sort=sort, mintime=mintime, **kwargs)
        with cprof:
            self.run()
        return cprof

    def save(self, filename=None, shrink=None, **kwargs):
        """
        Save to disk as a gzipped pickle.

        Args:
            filename (str or None): the name or path of the file to save to; if None, uses stored
            shrink (bool or None): whether to shrink the sim prior to saving (reduces size by ~99%)
            kwargs: passed to sc.makefilepath()

        Returns:
            filename (str): the validated absolute path to the saved file

        **Example**:

            sim.save() # Saves to a .sim file
        """
        # Set shrink based on whether we're in the middle of a run
        if shrink is None:
            shrink = False if self.initialized and not self.results_ready else True

        # Handle the filename
        if filename is None:
            filename = self.simfile
        filename = sc.makefilepath(filename=filename, **kwargs)
        self.filename = filename  # Store the actual saved filename

        # Handle the shrinkage and save
        sim = self.shrink(inplace=False) if shrink else self
        sc.save(filename=filename, obj=sim)
        return filename

    def to_json(self, filename=None, keys=None, tostring=False, indent=2, verbose=False, **kwargs):
        """
        Export results and parameters as JSON.

        Args:
            filename (str): if None, return string; else, write to file
            keys (str/list): attributes to write to json (choices: 'pars', 'summary', and/or 'results')
            verbose (bool): detail to print
            **kwargs (dict): passed to `sc.jsonify()`

        Returns:
            A dictionary representation of the parameters and/or summary results
            (or write that dictionary to a file)

        **Examples**:

            json = sim.to_json() # Convert to a dict
            sim.to_json('sim.json') # Write everything
            sim.to_json('summary.json', keys='summary') # Just write the summary
        """
        # Handle keys
        if keys is None:
            keys = ['pars', 'summary', 'results']
        keys = sc.promotetolist(keys)

        # Convert to JSON-compatible format
        d = sc.objdict()
        for key in keys:
            if key in ['pars', 'parameters']:
                pardict = self.pars.to_json()
                d.pars = pardict
            elif key == 'summary':
                if self.results_ready:
                    d.summary = dict(sc.dcp(self.summary))
                else:
                    d.summary = 'Summary not available (Sim has not yet been run)'
            elif key == 'results':
                d.results = self.to_df().to_dict()
            else:  # pragma: no cover
                warnmsg = f'Could not convert "{key}" to JSON; continuing...'
                ss.warn(warnmsg)

        # Final conversion
        d = sc.jsonify(d, **kwargs)
        if filename is not None:
            sc.savejson(filename=filename, obj=d)
        return d

    def to_yaml(self, filename=None, sort_keys=False, **kwargs):
        """
        Export results and parameters as YAML.

        Args:
            filename (str): the name of the file to write to (default `{sim.label}.yaml`)
            kwargs (dict): passed to `sim.to_json()`

        **Example**:

            sim = ss.Sim(diseases='sis', networks='random').run()
            sim.to_yaml('results.yaml', keys='results')
        """
        if filename is None:
            if self.label:
                filename = f'{self.label}.yaml'
            else:
                filename = 'sim.yaml'
        json = self.to_json(filename=None, **kwargs)
        return sc.saveyaml(filename=filename, obj=json, sort_keys=sort_keys)

    def plot(self, key=None, fig=None, show_data=True, show_skipped=False, show_module=None,
             show_label=False, n_ticks=None, **kwargs):
        """
        Plot all results in the Sim object

        Args:
            key (str/list): the results key to plot (by default, all); if a list, plot exactly those keys
            fig (Figure): if provided, plot results into an existing figure
            style (str): the plotting style to use
            show_data (bool): plot the data, if available
            show_skipped (bool): show even results that are skipped by default
            show_module (int): whether to show the module as well as the result name; if an int, show if the label is less than that length (default, 26); if -1, use a newline
            show_label (str): if 'fig', reset the fignum; if 'title', set the figure suptitle
            n_ticks (tuple of ints): if provided, specify how many x-axis ticks to use (default: `(2,5)`, i.e. minimum of 2 and maximum of 5)
            fig_kw (dict): passed to `sc.getrowscols()`, then `plt.subplots()` and `plt.figure()`
            plot_kw (dict): passed to `plt.plot()`
            data_kw (dict): passed to `plt.scatter()`, for plotting the data
            style_kw (dict): passed to `ss.style()`, for controlling the detailed plotting style (default "starsim"; other options are "simple", None, or any Matplotlib style)
            **kwargs (dict): known arguments (e.g. figsize, font) split between the above dicts; see `ss.plot_args()` for all valid options

        **Examples**:

            sim = ss.Sim(diseases='sis', networks='random').run()

            # Basic usage
            sim.plot()

            # Plot a single result
            sim.plot('sis.prevalence')

            # Plot with a custom figure size, font, and style
            sim.plot(figsize=(12,16), font='Raleway', style='fancy')
        """
        self.check_results_ready('Please run the sim before plotting')

        # Figure out the flat structure of results to plot
        flat = ss.utils.match_result_keys(self.results, key, show_skipped=(show_skipped or key)) # Always show skipped with a custom key

        # Set figure defaults
        n_cols,_ = sc.getrowscols(len(flat)) # Number of columns of axes
        default_figsize = np.array([8, 6])
        figsize_factor = np.clip((n_cols-3)/6+1, 1, 1.5) # Scale the default figure size based on the number of rows and columns
        figsize = default_figsize*figsize_factor

        # Set plotting defaults
        kw = ss.plot_args(kwargs, figsize=figsize, alpha=0.8, data_alpha=0.3, data_color='k')

        # Do the plotting
        with ss.style(**kw.style):

            # Get the figure
            if fig is None:
                if show_label in ['fig', 'fignum'] and self.label:
                    plotlabel = self.label
                    figlist = sc.autolist()
                    while plt.fignum_exists(plotlabel):
                        figlist += plotlabel
                        plotlabel = sc.uniquename(self.label, figlist, human=True)
                    kw.fig['num'] = plotlabel
                fig, axs = sc.getrowscols(len(flat), make=True, **kw.fig)
                if isinstance(axs, np.ndarray):
                    axs = axs.flatten()
            else:
                axs = fig.axes
            if not sc.isiterable(axs):
                axs = [axs]

            # Do the plotting
            df = self.data if show_data else None # For plotting the data
            for ax, (key, res) in zip(axs, flat.items()):

                # Plot data
                if df is not None:
                    mod = res.module
                    name = res.name
                    found = False
                    for dfkey in [f'{mod}.{name}', f'{mod}_{name}']: # Allow dot or underscore
                        if dfkey in df.cols:
                            found = True
                            break
                    if found:
                        ax.scatter(df.index.values, df[dfkey].values, **kw.data)

                # Plot results
                ax.plot(res.timevec, res.values, **kw.plot, label=self.label)
                ss.utils.format_axes(ax, res, n_ticks, show_module)

        if show_label in ['title', 'suptitle'] and self.label:
            fig.suptitle(self.label, weight='bold')

        sc.figlayout(fig=fig)

        return ss.return_fig(fig, **kw.return_fig)


class AlreadyRunError(RuntimeError):
    """ Raised if trying to re-run an already-run sim without re-initializing """
    pass


def demo(run=True, plot=True, summary=True, show=True, **kwargs):
    """
    Create a simple demo simulation for Starsim

    Defaults to using the SIR model with a random network, but these can be configured.

    Args:
        run (bool): whether to run the sim
        plot (bool): whether to plot the results
        summary (bool): whether to print a summary of the results
        kwargs (dict): passed to `ss.Sim()`

    **Examples**:

        ss.demo() # Run, plot, and show results
        ss.demo(diseases='hiv', networks='mf') # Run with different defaults
    """
    pars = sc.mergedicts(dict(diseases='sir', networks='random'), kwargs)
    sim = Sim(pars)
    if run:
        sc.heading('Running demo:')
        sim.run()
        if summary:
            sc.heading('Results:')
            print(sim.summary)
            if plot:
                sim.plot()
                if show:
                    plt.show()
    return sim


def diff_sims(sim1, sim2, skip_key_diffs=False, skip=None, full=False, output=False, die=False):
    '''
    Compute the difference of the summaries of two simulations, and print any
    values which differ.

    Args:
        sim1 (Sim/MultiSim/dict): either a simulation/MultiSim object or the sim.summary dictionary
        sim2 (im/dict): ditto
        skip_key_diffs (bool): whether to skip keys that don't match between sims
        skip (list): a list of values to skip
        full (bool): whether to print out all values (not just those that differ)
        output (bool): whether to return the output as a string (otherwise print)
        die (bool): whether to raise an exception if the sims don't match
        require_run (bool): require that the simulations have been run

    **Example**:

        s1 = ss.Sim(rand_seed=1).run()
        s2 = ss.Sim(rand_seed=2).run()
        ss.diff_sims(s1, s2)
    '''

    # Convert to dict
    if isinstance(sim1, (Sim, ss.MultiSim)):
        sim1 = sim1.summarize()
    if isinstance(sim2, (Sim, ss.MultiSim)):
        sim2 = sim2.summarize()
    for sim in [sim1, sim2]:
        if not isinstance(sim, dict):  # pragma: no cover
            errormsg = f'Cannot compare object of type {type(sim)}, must be a sim or a sim.summary dict'
            raise TypeError(errormsg)


    # Check if it's a multisim
    sim1 = sc.objdict(sim1)
    sim2 = sc.objdict(sim2)
    multi = isinstance(sim1[0], dict) # If it's a dict, then it's a multisim
    if multi:
        for sim in [sim1, sim2]:
            assert 'mean' in sim[0], f"Can only compare multisims with summarize(method='mean'), but you have keys {sim[0].keys()}"

    # Compare keys
    keymatchmsg = ''
    sim1_keys = set(sim1.keys())
    sim2_keys = set(sim2.keys())
    if sim1_keys != sim2_keys and not skip_key_diffs:  # pragma: no cover
        keymatchmsg = "Keys don't match!\n"
        missing = list(sim1_keys - sim2_keys)
        extra = list(sim2_keys - sim1_keys)
        if missing:
            keymatchmsg += f'  Missing sim1 keys: {missing}\n'
        if extra:
            keymatchmsg += f'  Extra sim2 keys: {extra}\n'

    # Compare values
    valmatchmsg = ''
    mismatches = {}
    n_mismatch = 0
    skip = sc.tolist(skip)
    for key in sim2.keys():  # To ensure order
        if key in sim1_keys and key not in skip:  # If a key is missing, don't count it as a mismatch
            d = sc.objdict()
            if multi:
                d.sim1 = sim1[key]['mean']
                d.sim2 = sim2[key]['mean']
                d.sim1_sem = sim1[key]['sem']
                d.sim2_sem = sim2[key]['sem']
            else:
                d.sim1 = sim1[key]
                d.sim2 = sim2[key]
            mm = not np.isclose(d.sim1, d.sim2, equal_nan=True)
            n_mismatch += mm
            if mm or full:
                mismatches[key] = d

    df = sc.dataframe() # Preallocate in case there were no mismatches
    if len(mismatches):
        valmatchmsg = '\nThe following values differ between the two simulations:\n' if not full else ''
        df = sc.dataframe.from_dict(mismatches).transpose()
        diff = []
        ratio = []
        change = []
        zscore = []
        small_change = 1 + 1e-3  # Define a small change, e.g. a rounding error
        for d in mismatches.values():
            old = d.sim1
            new = d.sim2
            if multi:
                old_sem = d.sim1_sem
                new_sem = d.sim1_sem
                sem = old_sem + new_sem
                small_change = 1.96 # 95% CI, roughly speaking

            numeric = sc.isnumber(old) and sc.isnumber(new) # Should all be numeric, but just in case
            if numeric and old > 0:
                this_diff = new - old
                this_ratio = new / old
                if multi:
                    abs_ratio = abs(this_diff)/sem
                    this_zscore = abs_ratio
                else:
                    abs_ratio = max(this_ratio, 1.0/this_ratio)
                    this_zscore = np.nan

                # Set the character to use
                approx_eq = abs_ratio < small_change
                if approx_eq:    change_char = '≈'
                elif new > old:  change_char = '↑'
                elif new < old:  change_char = '↓'
                elif new == old: change_char = '='
                else:
                    errormsg = f'Could not determine relationship between sim1={old} and sim2={new}'
                    raise ValueError(errormsg)

                # Set how many repeats it should have
                repeats = 1
                if abs_ratio == 0:   repeats = 0
                if abs_ratio >= 1.1: repeats = 2
                if abs_ratio >= 2:   repeats = 3
                if abs_ratio >= 10:  repeats = 4
                this_change = change_char * repeats
            else:  # pragma: no cover
                this_diff =  np.nan
                this_ratio  = np.nan
                this_change = 'N/A'
                this_zscore = np.nan

            diff.append(this_diff)
            ratio.append(this_ratio)
            change.append(this_change)
            zscore.append(this_zscore)

        df['diff'] = diff
        df['ratio'] = ratio
        numeric_cols = ['sim1', 'sim2', 'diff', 'ratio']
        if multi:
            numeric_cols += ['sim1_sem', 'sim2_sem']
        for col in numeric_cols:
            df[col] = df[col].round(decimals=3)
        df['change'] = change
        if multi:
            df['zscore'] = zscore
            df['statsig'] = df.zscore > small_change
        valmatchmsg += str(df)

    # Raise an error if mismatches were found
    mismatchmsg = keymatchmsg + valmatchmsg
    if mismatchmsg:  # pragma: no cover
        if die and n_mismatch: # To catch full=True case
            raise ValueError(mismatchmsg)
        elif output:
            return df
        else:
            print(mismatchmsg)
            return True
    else:
        if not output:
            print('Sims match')
        return False


def check_sims_match(*args, full=False):
    """
    Shortcut to using ss.diff_sims() to check if multiple sims match

    Args:
        args (list): a list of 2 or more sims to compare
        full (bool): if True, return whether each sim matches the first

    **Example**:

        s1 = ss.Sim(diseases='sir', networks='random')
        s2 = ss.Sim(pars=dict(diseases='sir', networks='random'))
        s3 = ss.Sim(diseases=ss.SIR(), networks=ss.RandomNet())
        assert ss.check_sims_match(s1, s2, s3)
    """
    if len(args) < 2:
        errormsg = 'Must compare at least 2 sims'
        raise ValueError(errormsg)
    base = args[0]
    matches = []
    for other in args[1:]:
        diff = diff_sims(base, other, full=False, output=False, die=False)
        matches.append(not(diff)) # Return the opposite of the diff
    if full:
        return matches
    else:
        return all(matches)

