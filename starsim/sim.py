"""
Define core Sim classes
"""

# Imports
import numpy as np
import sciris as sc
import starsim as ss
import itertools
import matplotlib.pyplot as pl

__all__ = ['Sim', 'AlreadyRunError', 'demo', 'diff_sims']


class Sim(sc.prettyobj):

    def __init__(self, pars=None, label=None, people=None, demographics=None, diseases=None, networks=None,
                 connectors=None, interventions=None, analyzers=None, **kwargs):

        # Set attributes
        self.label = label  # The label/name of the simulation
        self.created = None  # The datetime the sim was created
        self.people = people  # People object
        self.results = ss.Results(module='sim')  # For storing results
        self.summary = None  # For storing a summary of the results
        self.initialized = False  # Whether initialization is complete
        self.complete = False  # Whether a simulation has completed running # TODO: replace with finalized?
        self.results_ready = False  # Whether results are ready
        self.filename = None

        # Time indexing
        self.ti = None  # The time index, e.g. 0, 1, 2 # TODO: do we need all of these?
        self.yearvec = None
        self.tivec = None
        self.npts = None

        # Make default parameters (using values from parameters.py)
        self.pars = ss.make_pars()  # Start with default pars
        self.pars.update_pars(sc.mergedicts(pars, kwargs))  # Update the parameters

        # Placeholders for plug-ins: demographics, diseases, connectors, analyzers, and interventions
        # Products are not here because they are stored within interventions
        if demographics == True: demographics = [ss.Births(), ss.Deaths()]  # Use default assumptions for demographics
        self.demographics  = ss.ndict(demographics, type=ss.Demographics)
        self.diseases      = ss.ndict(diseases, type=ss.Disease)
        self.networks      = ss.ndict(networks, type=ss.Network)
        self.connectors    = ss.ndict(connectors, type=ss.Connector)
        self.interventions = ss.ndict(interventions, type=ss.Intervention, strict=False) # strict=False since can be a function
        self.analyzers     = ss.ndict(analyzers, type=ss.Analyzer, strict=False)

        # Initialize the random number generator container
        self.dists = ss.Dists(obj=self)

        return

    @property
    def dt(self):
        if 'dt' in self.pars:
            return self.pars['dt']
        else:
            return np.nan

    @property
    def year(self):
        try:
            return self.yearvec[self.ti]
        except:
            return np.nan

    @property
    def modules(self):
        # Return iterator over all Module instances (stored in standard places) in the Sim
        products = [intv.product for intv in self.interventions.values() if
                    hasattr(intv, 'product') and isinstance(intv.product, ss.Product)]
        return itertools.chain(
            self.demographics.values(),
            self.networks.values(),
            self.diseases.values(),
            self.connectors.values(),
            self.interventions.values(),
            products,
            self.analyzers.values(),
        )

    def initialize(self, reset=False, **kwargs):
        """
        Perform all initializations on the sim.
        """
        # Validation and initialization
        self.ti = 0  # The current time index
        self.validate_pars()  # Ensure parameters have valid values
        self.validate_dt()
        self.init_time_vecs()  # Initialize time vecs
        ss.set_seed(self.pars.rand_seed)  # Reset the seed before the population is created -- shouldn't matter if only using Dist objects

        # Initialize the core sim components
        self.init_people(reset=reset, **kwargs)  # Create all the people (the heaviest step)

        # Initialize plug-ins
        self.init_demographics()
        self.init_networks()
        self.init_diseases()
        self.init_connectors()
        self.init_interventions()
        self.init_analyzers()

        # Initialize all distributions now that everything else is in place
        self.dists.initialize(obj=self, base_seed=self.pars.rand_seed, force=True)

        # Final steps
        self.initialized = True
        self.complete = False
        self.results_ready = False

        return self

    def validate_dt(self):
        """
        Check that 1/dt is an integer value, otherwise results and time vectors will have mismatching shapes.
        init_results explicitly makes this assumption by casting resfrequency = int(1/dt).
        """
        dt = self.dt
        reciprocal = 1.0 / dt  # Compute the reciprocal of dt
        if not reciprocal.is_integer():  # Check if reciprocal is not a whole (integer) number
            # Round the reciprocal
            reciprocal = int(reciprocal)
            rounded_dt = 1.0 / reciprocal
            self.pars.dt = rounded_dt
            if self.pars.verbose:
                warnmsg = f"Warning: Provided time step dt: {dt} resulted in a non-integer number of steps per year. Rounded to {rounded_dt}."
                print(warnmsg)
        return

    def validate_pars(self):
        """
        Some parameters can take multiple types; this makes them consistent.
        """
        # Handle n_agents
        if self.people is not None:
            self.pars.n_agents = len(self.people)
        # elif self.popdict is not None: # Starsim does not currenlty support self.popdict
        # self.pars.n_agents = len(self.popdict)
        elif self.pars.n_agents is not None:
            self.pars.n_agents = int(self.pars.n_agents)
        else:
            errormsg = 'Must supply n_agents, a people object, or a popdict'
            raise ValueError(errormsg)

        # Handle end and n_years
        if self.pars.end:
            self.pars.n_years = self.pars.end - self.pars.start
            if self.pars.n_years <= 0:
                errormsg = f"Number of years must be >0, but you supplied start={str(self.pars.start)} and " \
                           f"end={str(self.pars.end)}, which gives n_years={self.pars.n_years}"
                raise ValueError(errormsg)
        else:
            if self.pars.n_years:
                self.pars.end = self.pars.start + self.pars.n_years
            else:
                errormsg = 'You must supply one of n_years and end."'
                raise ValueError(errormsg)

        # Handle verbose
        if self.pars.verbose == 'brief':
            self.pars.verbose = -1
        if not sc.isnumber(self.pars.verbose):  # pragma: no cover
            errormsg = f'Verbose argument should be either "brief", -1, or a float, not {type(self.par.verbose)} "{self.par.verbose}"'
            raise ValueError(errormsg)

        return

    def init_time_vecs(self):
        """
        Construct vectors things that keep track of time
        """
        self.yearvec = np.arange(start=self.pars.start, stop=self.pars.end + self.pars.dt, step=self.pars.dt)
        self.npts = len(self.yearvec)
        self.tivec = np.arange(self.npts)
        return

    def init_people(self, reset=False, verbose=None, **kwargs):
        """
        Initialize people within the sim
        Sometimes the people are provided, in which case this just adds a few sim properties to them.
        Other time people are not provided and this method makes them.
        Args:
            reset           (bool): whether to regenerate the people even if they already exist
            verbose         (int):  detail to print
            kwargs          (dict): passed to ss.make_people()
        """

        # Handle inputs
        if verbose is None:
            verbose = self.pars.verbose
        if verbose > 0:
            resetstr = ''
            if self.people and reset:
                resetstr = ' (resetting people)'
            print(f'Initializing sim{resetstr} with {self.pars["n_agents"]:0n} agents')

        # If people have not been supplied, make them
        if self.people is None or reset:
            self.people = ss.People(n_agents=self.pars.n_agents, **kwargs)  # This just assigns UIDs and length

        # If a popdict has not been supplied, we can make one from location data
        if self.pars.location is not None:
            # Check where to get total_pop from
            if self.pars.total_pop is not None:  # If no pop_scale has been provided, try to get it from the location
                errormsg = 'You can either define total_pop explicitly or via the location, but not both'
                raise ValueError(errormsg)

        else:
            if self.pars.total_pop is not None:  # If no pop_scale has been provided, try to get it from the location
                total_pop = self.pars.total_pop
            else:
                if self.pars.pop_scale is not None:
                    total_pop = self.pars.pop_scale * self.pars.n_agents
                else:
                    total_pop = self.pars.n_agents

        self.pars.total_pop = total_pop
        if self.pars.pop_scale is None:
            self.pars.pop_scale = total_pop / self.pars.n_agents

        # Any other initialization
        if not self.people.initialized:
            self.people.initialize(self)

        # Set time attributes
        self.people.ti = self.ti
        self.people.dt = self.dt
        self.people.year = self.year
        self.people.init_results(self)
        return self

    def convert_plugins(self, plugin_class, plugin_name=None):
        """
        Common logic for converting plug-ins to a standard format
        Used for networks, demographics, diseases, connectors, analyzers, and interventions
        Args:
            plugin: class
        """

        if plugin_name is None: plugin_name = plugin_class.__name__.lower()

        # Get lower-case names of all subclasses
        known_plugins = {n.__name__.lower():n for n in ss.all_subclasses(plugin_class)}
        if plugin_name == 'networks': # Allow "msm" or "msmnet"
            known_plugins.update({k.removesuffix('net'):v for k,v in known_plugins.items()})

        # Figure out if it's in the sim pars or provided directly
        attr_plugins = getattr(self, plugin_name)  # Get any plugins that have been provided directly

        # See if they've been provided in the pars dict
        if self.pars.get(plugin_name):

            par_plug = self.pars[plugin_name]

            # String: convert to ndict
            if isinstance(par_plug, str):
                plugins = ss.ndict(dict(name=par_plug))

            # List or dict: convert to ndict
            elif sc.isiterable(par_plug) and len(par_plug):
                if isinstance(par_plug, dict) and 'type' in par_plug and 'name' not in par_plug:
                    par_plug['name'] = par_plug['type'] # TODO: simplify/remove this
                plugins = ss.ndict(par_plug)

        else:  # Not provided directly or in pars
            plugins = {}

        # Check that we don't have two copies
        for attr_key in attr_plugins.keys():
            if plugins.get(attr_key):
                errormsg = f'Sim was created with {attr_key} module, cannot create another through the pars dict.'
                raise ValueError(errormsg)

        plugins = sc.mergedicts(plugins, attr_plugins)

        # Process
        processed_plugins = sc.autolist()
        for plugin in plugins.values():

            if not isinstance(plugin, plugin_class):

                if isinstance(plugin, dict):
                    ptype = (plugin.get('type') or plugin.get('name') or '').lower()
                    name = plugin.get('name') or ptype
                    if ptype in known_plugins:
                        # Make an instance of the requested plugin
                        plugin_pars = {k: v for k, v in plugin.items() if k not in ['type', 'name']}
                        pclass = known_plugins[ptype]
                        plugin = pclass(name=name, pars=plugin_pars) # TODO: does this handle par_dists, etc?
                    else:
                        errormsg = (f'Could not convert {plugin} to an instance of class {plugin_name}.'
                                    f'Try specifying it directly rather than as a dictionary.')
                        raise ValueError(errormsg)
                elif plugin_name in ['analyzers', 'interventions'] and callable(plugin):
                    pass # This is ok, it's a function instead of an Intervention object
                else:
                    errormsg = (
                        f'{plugin_name.capitalize()} must be provided as either class instances or dictionaries with a '
                        f'"name" key corresponding to one of these known subclasses: {known_plugins}.')
                    raise ValueError(errormsg)

            processed_plugins += plugin

        return processed_plugins

    def init_demographics(self):
        """ Initialize demographics """

        # Demographics can be provided via sim.demographics or sim.pars - this methods reconciles them
        demographics = self.convert_plugins(ss.Demographics, plugin_name='demographics')

        # We also allow users to add vital dynamics by entering birth_rate and death_rate parameters directly to the sim
        if self.pars.birth_rate is not None:
            births = ss.Births(pars={'birth_rate': self.pars.birth_rate})
            demographics += births
        if self.pars.death_rate is not None:
            background_deaths = ss.Deaths(pars={'death_rate': self.pars.death_rate})
            demographics += background_deaths

        # Iterate over demographic modules and initialize them
        for dem_mod in demographics:
            dem_mod.initialize(self)
            self.results[dem_mod.name] = dem_mod.results

        # Count how many of each kind of demographic module we have
        demdict = {'births': ss.Births, 'pregnancy': ss.Pregnancy, 'deaths': ss.Deaths}
        mod_names = dict()
        for demname, demtype in demdict.items():
            mod_names[demname] = [d.name for d in demographics if isinstance(d, demtype)]

            # Validation
            if len(mod_names[demname]) > 1:
                if len(mod_names[demname]) == len(set(mod_names[demname])):  # No duplicate names, raise warning
                    ss.warn(f'Two instances of {demname} module added to the sim; was this intentional?')
                else:
                    errormsg = (f'Cannot add two identically-named {demname} modules to a sim.\n '
                                f'Demographic modules are: \n{sc.newlinejoin(mod_names[demname])}.\n'
                                f'Tip: if using demographic modules, do not use birth and death rates in the sim pars.')
                    raise ValueError(errormsg)

        # Ensure they're stored at the sim level
        self.demographics = ss.ndict(*demographics)

    def init_diseases(self):
        """ Initialize diseases """

        # Diseases can be provided in sim.demographics or sim.pars
        diseases = self.convert_plugins(ss.Disease, plugin_name='diseases')

        # Interate over diseases and initialize them
        for disease in diseases:
            disease.initialize(self)

            # Add the disease's parameters and results into the Sim's dicts
            self.pars[disease.name] = disease.pars
            self.results[disease.name] = disease.results

            # Add disease states to the People's dicts
            self.people.add_module(disease)

        # Store diseases in the sim
        self.diseases = ss.ndict(*diseases)

        return

    def init_connectors(self):
        for connector in self.connectors.values():
            connector.initialize(self)

    def init_networks(self):
        """ Initialize networks if these have been provided separately from the people """

        processed_networks = self.convert_plugins(ss.Network, plugin_name='networks')

        # Now store the networks in a Networks object, which also allows for connectors between networks
        if not isinstance(processed_networks, ss.Networks):
            self.networks = ss.Networks(*processed_networks)
        self.networks.initialize(self)

        return

    def init_interventions(self):
        """ Initialize and validate the interventions """

        interventions = self.convert_plugins(ss.Intervention, plugin_name='interventions')

        # Translate the intervention specs into actual interventions
        for i, intervention in enumerate(interventions):
            if isinstance(intervention, type) and issubclass(intervention, ss.Intervention):
                intervention = intervention()  # Convert from a class to an instance of a class
            if isinstance(intervention, ss.Intervention):
                intervention.initialize(self)
            elif callable(intervention):
                intv_func = intervention
                intervention = ss.Intervention(name=f'intervention_func_{i}')
                intervention.apply = intv_func # Monkey-patch together an intervention from a function
            else:
                errormsg = f'Intervention {intervention} does not seem to be a valid intervention: must be a function or Intervention subclass'
                raise TypeError(errormsg)
            
            if intervention.name not in self.interventions:
                self.interventions += intervention

            # Add the intervention parameters and results into the Sim's dicts
            self.pars[intervention.name] = intervention.pars
            self.results[intervention.name] = intervention.results

            # Add intervention states to the People's dicts
            self.people.add_module(intervention)

            # If there's a product module present, initialize and add it
            if hasattr(intervention, 'product') and isinstance(intervention.product, ss.Product):
                intervention.product.initialize(self)

                self.people.add_module(intervention.product)
        
        # TODO: combine this with the code above
        for k,intervention in self.interventions.items():
            if not isinstance(intervention, ss.Intervention):
                intv_func = intervention
                intervention = ss.Intervention(name=f'intervention_func_{k}')
                intervention.apply = intv_func # Monkey-patch together an intervention from a function
                self.interventions[k] = intervention

        return

    def init_analyzers(self):
        """ Initialize the analyzers """
        
        analyzers = self.pars.analyzers
        if not np.iterable(analyzers):
            analyzers = sc.tolist(analyzers)

        # Interpret analyzers
        for ai, analyzer in enumerate(analyzers):
            if isinstance(analyzer, type) and issubclass(analyzer, ss.Analyzer):
                analyzer = analyzer()  # Convert from a class to an instance of a class
            if not (isinstance(analyzer, ss.Analyzer) or callable(analyzer)):
                errormsg = f'Analyzer {analyzer} does not seem to be a valid analyzer: must be a function or Analyzer subclass'
                raise TypeError(errormsg)
            self.analyzers += analyzer  # Add it in

        # TODO: should tidy/remove this code
        for k,analyzer in self.analyzers.items():
            if not isinstance(analyzer, ss.Analyzer) and callable(analyzer):
                ana_func = analyzer
                analyzer = ss.Analyzer(name=f'analyzer_func_{k}')
                analyzer.apply = ana_func # Monkey-patch together an intervention from a function
                self.analyzers[k] = analyzer
            if isinstance(analyzer, ss.Analyzer):
                analyzer.initialize(self)

        return

    def step(self):
        """ Step through time and update values """

        # Set the time and if we have reached the end of the simulation, then do nothing
        if self.complete:
            raise AlreadyRunError('Simulation already complete (call sim.initialize() to re-run)')

        # Advance random number generators forward to prepare for any random number calls that may be necessary on this step
        self.dists.jump(to=self.ti+1)  # +1 offset because ti=0 is used on initialization

        # Clean up dead agents, if removing agents is enabled
        if self.pars.remove_dead and (self.ti % self.pars.remove_dead == 0):
            self.people.remove_dead(self)

        # Update demographic modules (create new agents from births/immigration, schedule non-disease deaths and emigration)
        for dem_mod in self.demographics.values():
            dem_mod.update(self)

        # Carry out autonomous state changes in the disease modules. This allows autonomous state changes/initializations
        # to be applied to newly created agents
        for disease in self.diseases.values():
            disease.update_pre(self)

        # Update connectors -- TBC where this appears in the ordering
        for connector in self.connectors.values():
            connector.update(self)

        # Update networks - this takes place here in case autonomous state changes at this timestep
        # affect eligibility for contacts
        self.networks.update(self.people)

        # Apply interventions - new changes to contacts will be visible and so the final networks can be customized by
        # interventions, by running them at this point
        for intervention in self.interventions.values():
            intervention(self)

        # Carry out transmission/new cases
        for disease in self.diseases.values():
            disease.make_new_cases(self)

        # Execute deaths that took place this timestep (i.e., changing the `alive` state of the agents). This is executed
        # before analyzers have run so that analyzers are able to inspect and record outcomes for agents that died this timestep
        uids = self.people.resolve_deaths()
        for disease in self.diseases.values():
            disease.update_death(self, uids)

        # Update results
        self.people.update_results(self)

        for disease in self.diseases.values():
            disease.update_results(self)

        for analyzer in self.analyzers.values():
            analyzer(self)

        # Tidy up
        self.ti += 1
        self.people.ti = self.ti
        self.people.update_post(self)

        if self.ti == self.npts:
            self.complete = True

        return

    def run(self, until=None, verbose=None):
        """ Run the model once """

        # Initialization steps
        T = sc.timer()
        if not self.initialized:
            self.initialize()
            self._orig_pars = sc.dcp(self.pars)  # Create a copy of the parameters to restore after the run

        if verbose is None:
            verbose = self.pars.verbose

        # Check for AlreadyRun errors
        errormsg = None
        if until is None: until = self.npts
        if until > self.npts:
            errormsg = f'Requested to run until t={until} but the simulation end is ti={self.npts}'
        if self.ti >= until:  # NB. At the start, self.t is None so this check must occur after initialization
            errormsg = f'Simulation is currently at t={self.ti}, requested to run until ti={until} which has already been reached'
        if self.complete:
            errormsg = 'Simulation is already complete (call sim.initialize() to re-run)'
        if errormsg:
            raise AlreadyRunError(errormsg)

        # Main simulation loop
        while self.ti < until:

            # Check if we were asked to stop
            elapsed = T.toc(output=True)

            # Print progress
            if verbose:
                simlabel = f'"{self.label}": ' if self.label else ''
                string = f'  Running {simlabel}{self.yearvec[self.ti]:0.1f} ({self.ti:2.0f}/{self.npts}) ({elapsed:0.2f} s) '
                if verbose >= 2:
                    sc.heading(string)
                elif verbose > 0:
                    if not (self.ti % int(1.0 / verbose)):
                        sc.progressbar(self.ti + 1, self.npts, label=string, length=20, newline=True)

            # Actually run the model
            self.step()

        # If simulation reached the end, finalize the results
        if self.complete:
            self.finalize(verbose=verbose)
            sc.printv(f'Run finished after {elapsed:0.2f} s.\n', 1, verbose)

        return self

    def finalize(self, verbose=None):
        """ Compute final results """

        if self.results_ready:
            # Because the results are rescaled in-place, finalizing the sim cannot be run more than once or
            # otherwise the scale factor will be applied multiple times
            raise AlreadyRunError('Simulation has already been finalized')

        # Scale the results
        for reskey, res in self.results.items():
            if isinstance(res, ss.Result) and res.scale:
                self.results[reskey] = self.results[reskey] * self.pars.pop_scale

        for module in self.modules:
            module.finalize(self)

        self.summarize()
        self.results_ready = True  # Set this first so self.summary() knows to print the results
        self.ti -= 1  # During the run, this keeps track of the next step; restore this be the final day of the sim
        return

    def summarize(self, how='default'):
        """
        Provide a quick summary of the sim
        
        Args:
            how (str): how to summarize: can be 'mean', 'median', 'last', or a mapping of result keys to those
        
        Returns the last entry for count and cumulative results, and the mean otherwise
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
            how = {'n_':'mean', 'new_':'mean', 'cum_':'last', '':'mean'}
        elif isinstance(how, str):
            how = {'':how} # Match everything
        
        summary = sc.objdict()
        flat = sc.flattendict(self.results, sep='_')
        for key, res in flat.items():
            try:
                func = get_func(key, how)
                entry = get_result(res, func)
            except Exception as E:
                entry = f'N/A {E}'
            summary[key] = entry
        self.summary = summary
        return summary
    
    def disp(self):
        print(self.summary)
    
    def shrink(self, skip_attrs=None, in_place=True):
        """
        "Shrinks" the simulation by removing the people and other memory-intensive
        attributes (e.g., some interventions and analyzers), and returns a copy of
        the "shrunken" simulation. Used to reduce the memory required for RAM or
        for saved files.

        Args:
            skip_attrs (list): a list of attributes to skip (remove) in order to perform the shrinking; default "people"
            in_place (bool): whether to perform the shrinking in place (default), or return a shrunken copy instead

        Returns:
            shrunken (Sim): a Sim object with the listed attributes removed
        """
        # By default, skip people (~90% of memory), popdict, and _orig_pars (which is just a backup)
        if skip_attrs is None:
            skip_attrs = ['people']

        # Create the new object, and copy original dict, skipping the skipped attributes
        if in_place:
            shrunken = self
            for attr in skip_attrs:
                setattr(self, attr, None)
        else:
            shrunken = object.__new__(self.__class__)
            shrunken.__dict__ = {k: (v if k not in skip_attrs else None) for k, v in self.__dict__.items()}

        # Don't return if in place
        if in_place:
            return
        else:
            return shrunken

    def _get_ia(self, which, label=None, partial=False, as_list=False, as_inds=False, die=True, first=False):
        """ Helper method for get_interventions() and get_analyzers(); see get_interventions() docstring """

        # Handle inputs
        if which not in ['interventions', 'analyzers']:  # pragma: no cover
            errormsg = f'This method is only defined for interventions and analyzers, not "{which}"'
            raise ValueError(errormsg)

        ia_ndict = self.analyzers if which == 'analyzers' else self.interventions  # List of interventions or analyzers
        n_ia = len(ia_ndict)  # Number of interventions/analyzers

        position = 0 if first else -1  # Choose either the first or last element
        if label is None:  # Get all interventions if no label is supplied, e.g. sim.get_interventions()
            label = np.arange(n_ia)
        if isinstance(label, np.ndarray):  # Allow arrays to be provided
            label = label.tolist()
        labels = sc.promotetolist(label)

        # Calculate the matches
        matches = []
        match_inds = []

        for label in labels:
            if sc.isnumber(label):
                matches.append(ia_ndict[label])
                label = n_ia + label if label < 0 else label  # Convert to a positive number
                match_inds.append(label)
            elif sc.isstring(label) or isinstance(label, type):
                for ind, ia_key, ia_obj in ia_ndict.enumitems():
                    if sc.isstring(label) and ia_obj.label == label or (partial and (label in str(ia_obj.label))):
                        matches.append(ia_obj)
                        match_inds.append(ind)
                    elif isinstance(label, type) and isinstance(ia_obj, label):
                        matches.append(ia_obj)
                        match_inds.append(ind)
            else:  # pragma: no cover
                errormsg = f'Could not interpret label type "{type(label)}": should be str, int, list, or {which} class'
                raise TypeError(errormsg)

        # Parse the output options
        if as_inds:
            output = match_inds
        elif as_list:  # Used by get_interventions()
            output = matches
        else:
            if len(matches) == 0:  # pragma: no cover
                if die:
                    errormsg = f'No {which} matching "{label}" were found'
                    raise ValueError(errormsg)
                else:
                    output = None
            else:
                output = matches[
                    position]  # Return either the first or last match (usually), used by get_intervention()

        return output

    def get_interventions(self, label=None, partial=False, as_inds=False):
        """
        Find the matching intervention(s) by label, index, or type. If None, return
        all interventions. If the label provided is "summary", then print a summary
        of the interventions (index, label, type).

        Args:
            label (str, int, Intervention, list): the label, index, or type of intervention to get; if a list, iterate over one of those types
            partial (bool): if true, return partial matches (e.g. 'beta' will match all beta interventions)
            as_inds (bool): if true, return matching indices instead of the actual interventions
        """
        return self._get_ia('interventions', label=label, partial=partial, as_inds=as_inds, as_list=True)

    def get_intervention(self, label=None, partial=False, first=False, die=True):
        """
        Find the matching intervention(s) by label, index, or type.
        If more than one intervention matches, return the last by default.
        If no label is provided, return the last intervention in the list.

        Args:
            label (str, int, Intervention, list): the label, index, or type of intervention to get; if a list, iterate over one of those types
            partial (bool): if true, return partial matches
            first (bool): if true, return first matching intervention (otherwise, return last)
            die (bool): whether to raise an exception if no intervention is found
        """
        return self._get_ia('interventions', label=label, partial=partial, first=first, die=die, as_inds=False,
                            as_list=False)

    def export_df(self):
        """
        Export results as a Pandas dataframe

        :return:

        """

        if not self.results_ready:  # pragma: no cover
            errormsg = 'Please run the sim before exporting the results'
            raise RuntimeError(errormsg)

        def flatten_results(d, prefix=''):
            flat = {}
            for key, val in d.items():
                if isinstance(val, dict):
                    flat.update(flatten_results(val, prefix=prefix+key+'.'))
                else:
                    flat[prefix+key] = val
            return flat

        resdict = flatten_results(self.results)
        resdict['t'] = self.yearvec

        df = sc.dataframe.from_dict(resdict).set_index('t')
        return df

    def save(self, filename=None, keep_people=None, skip_attrs=None, **kwargs):
        """
        Save to disk as a gzipped pickle.

        Args:
            filename (str or None): the name or path of the file to save to; if None, uses stored
            keep_people (bool or None): whether to keep the people
            skip_attrs (list): attributes to skip saving
            kwargs: passed to sc.makefilepath()

        Returns:
            filename (str): the validated absolute path to the saved file

        **Example**::

            sim.save() # Saves to a .sim file
        """

        # Set keep_people based on whether we're in the middle of a run
        if keep_people is None:
            if self.initialized and not self.results_ready:
                keep_people = True
            else:
                keep_people = False

        # Handle the filename
        if filename is None:
            filename = self.simfile
        filename = sc.makefilepath(filename=filename, **kwargs)
        self.filename = filename  # Store the actual saved filename

        # Handle the shrinkage and save
        if skip_attrs or not keep_people:
            obj = self.shrink(skip_attrs=skip_attrs, in_place=False)
        else:
            obj = self
        sc.save(filename=filename, obj=obj)

        return filename

    @staticmethod
    def load(filename, *args, **kwargs):
        """ Load from disk from a gzipped pickle.  """

        sim = sc.load(filename, *args, **kwargs)
        if not isinstance(sim, Sim):  # pragma: no cover
            errormsg = f'Cannot load object of {type(sim)} as a Sim object'
            raise TypeError(errormsg)
        return sim

    def export_pars(self, filename=None, indent=2, *args, **kwargs):
        '''
        Return parameters for JSON export -- see also to_json().

        This method is required so that interventions can specify
        their JSON-friendly representation.

        Args:
            filename (str): filename to save to; if None, do not save
            indent (int): indent (int): if writing to file, how many indents to use per nested level
            args (list): passed to savejson()
            kwargs (dict): passed to savejson()

        Returns:
            pardict (dict): a dictionary containing all the parameter values
        '''
        pardict = {}
        for key in self.pars.keys():
            if key == 'interventions':
                pardict[key] = [intervention.to_json() for intervention in self.pars[key]]
            elif key == 'start_day':
                pardict[key] = str(self.pars[key])
            else:
                pardict[key] = self.pars[key]
        if filename is not None:
            sc.savejson(filename=filename, obj=pardict, indent=indent, *args, **kwargs)
        return pardict

    def to_json(self, filename=None, keys=None, tostring=False, indent=2, verbose=False, *args, **kwargs):
        '''
        Export results and parameters as JSON.

        Args:
            filename (str): if None, return string; else, write to file
            keys (str or list): attributes to write to json (default: results, parameters, and summary)
            tostring (bool): if not writing to file, whether to write to string (alternative is sanitized dictionary)
            indent (int): if writing to file, how many indents to use per nested level
            verbose (bool): detail to print
            args (list): passed to savejson()
            kwargs (dict): passed to savejson()

        Returns:
            A unicode string containing a JSON representation of the results,
            or writes the JSON file to disk

        **Examples**::

            json = sim.to_json()
            sim.to_json('results.json')
            sim.to_json('summary.json', keys='summary')
        '''

        # Handle keys
        if keys is None:
            keys = ['results', 'pars', 'summary', 'short_summary']
        keys = sc.promotetolist(keys)

        # Convert to JSON-compatible format
        d = {}
        for key in keys:
            if key == 'results':
                if self.results_ready:
                    resdict = self.export_results(for_json=True)
                    d['results'] = resdict
                else:
                    d['results'] = 'Results not available (Sim has not yet been run)'
            elif key in ['pars', 'parameters']:
                pardict = self.export_pars()
                d['parameters'] = pardict
            elif key == 'summary':
                if self.results_ready:
                    d['summary'] = dict(sc.dcp(self.summary))
                else:
                    d['summary'] = 'Summary not available (Sim has not yet been run)'
            elif key == 'short_summary':
                if self.results_ready:
                    d['short_summary'] = dict(sc.dcp(self.short_summary))
                else:
                    d['short_summary'] = 'Full summary not available (Sim has not yet been run)'
            else:  # pragma: no cover
                try:
                    d[key] = sc.sanitizejson(getattr(self, key))
                except Exception as E:
                    errormsg = f'Could not convert "{key}" to JSON: {str(E)}; continuing...'
                    print(errormsg)

        if filename is None:
            output = sc.jsonify(d, tostring=tostring, indent=indent, verbose=verbose, *args, **kwargs)
        else:
            output = sc.savejson(filename=filename, obj=d, indent=indent, *args, **kwargs)

        return output

    def plot(self):
        with sc.options.with_style('fancy'):
            flat = sc.flattendict(self.results, sep=': ')
            fig, axs = sc.getrowscols(len(flat), make=True)
            for ax, (k, v) in zip(axs.flatten(), flat.items()):
                ax.plot(self.yearvec, v)
                ax.set_title(k)
                ax.set_xlabel('Year')
        return fig


class AlreadyRunError(RuntimeError):
    """
    This error is raised if a simulation is run in such a way that no timesteps
    will be taken. This error is a distinct type so that it can be safely caught
    and ignored if required, but it is anticipated that most of the time, calling
    :py:func:`Sim.run` and not taking any timesteps, would be an inadvertent error.
    """
    pass


def demo(run=True, plot=True, summary=True, show=True, **kwargs):
    """
    Create a simple demo simulation for Starsim
    
    Defaults to using the SIR model with a random network, but these can be configured.
    
    Args:
        run (bool): whether to run the sim
        plot (bool): whether to plot the results
        summary (bool): whether to print a summary of the results
        kwargs (dict): passed to ``ss.Sim()``
    
    **Examples**::
        
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
                    pl.show()
    return sim


def diff_sims(sim1, sim2, skip_key_diffs=False, skip=None, full=False, output=False, die=False):
    '''
    Compute the difference of the summaries of two simulations, and print any
    values which differ.

    Args:
        sim1 (sim/dict): either a simulation object or the sim.summary dictionary
        sim2 (sim/dict): ditto
        skip_key_diffs (bool): whether to skip keys that don't match between sims
        skip (list): a list of values to skip
        full (bool): whether to print out all values (not just those that differ)
        output (bool): whether to return the output as a string (otherwise print)
        die (bool): whether to raise an exception if the sims don't match
        require_run (bool): require that the simulations have been run

    **Example**::

        s1 = ss.Sim(rand_seed=1).run()
        s2 = ss.Sim(rand_seed=2).run()
        ss.diff_sims(s1, s2)
    '''

    if isinstance(sim1, Sim):
        sim1 = sim1.summarize()
    if isinstance(sim2, Sim):
        sim2 = sim2.summarize()
    for sim in [sim1, sim2]:
        if not isinstance(sim, dict):  # pragma: no cover
            errormsg = f'Cannot compare object of type {type(sim)}, must be a sim or a sim.summary dict'
            raise TypeError(errormsg)

    # Compare keys
    keymatchmsg = ''
    sim1_keys = set(sim1.keys())
    sim2_keys = set(sim2.keys())
    if sim1_keys != sim2_keys and not skip_key_diffs:  # pragma: no cover
        keymatchmsg = "Keys don't match!\n"
        missing = list(sim1_keys - sim2_keys)
        extra = list(sim2_keys - sim1_keys)
        if missing:
            keymatchmsg += f'  Missing sim1 keys: {missing}\ns'
        if extra:
            keymatchmsg += f'  Extra sim2 keys: {extra}\n'

    # Compare values
    valmatchmsg = ''
    mismatches = {}
    n_mismatch = 0
    skip = sc.tolist(skip)
    for key in sim2.keys():  # To ensure order
        if key in sim1_keys and key not in skip:  # If a key is missing, don't count it as a mismatch
            sim1_val = sim1[key] if key in sim1 else 'not present'
            sim2_val = sim2[key] if key in sim2 else 'not present'
            mm = not np.isclose(sim1_val, sim2_val, equal_nan=True)
            n_mismatch += mm
            if mm or full:
                mismatches[key] = {'sim1': sim1_val, 'sim2': sim2_val}

    if len(mismatches):
        valmatchmsg = '\nThe following values differ between the two simulations:\n' if not full else ''
        df = sc.dataframe.from_dict(mismatches).transpose()
        diff = []
        ratio = []
        change = []
        small_change = 1e-3  # Define a small change, e.g. a rounding error
        for mdict in mismatches.values():
            old = mdict['sim1']
            new = mdict['sim2']
            numeric = sc.isnumber(sim1_val) and sc.isnumber(sim2_val)
            if numeric and old > 0:
                this_diff = new - old
                this_ratio = new / old
                abs_ratio = max(this_ratio, 1.0 / this_ratio)

                # Set the character to use
                if abs_ratio < small_change:
                    change_char = '≈'
                elif new > old:
                    change_char = '↑'
                elif new < old:
                    change_char = '↓'
                elif new == old:
                    change_char = '='
                else:
                    errormsg = f'Could not determine relationship between sim1={old} and sim2={new}'
                    raise ValueError(errormsg)

                # Set how many repeats it should have
                repeats = 1
                if abs_ratio == 0:
                    repeats = 0
                if abs_ratio >= 1.1:
                    repeats = 2
                if abs_ratio >= 2:
                    repeats = 3
                if abs_ratio >= 10:
                    repeats = 4

                this_change = change_char * repeats
            else:  # pragma: no cover
                this_diff = np.nan
                this_ratio = np.nan
                this_change = 'N/A'

            diff.append(this_diff)
            ratio.append(this_ratio)
            change.append(this_change)

        df['diff'] = diff
        df['ratio'] = ratio
        for col in ['sim1', 'sim2', 'diff', 'ratio']:
            df[col] = df[col].round(decimals=3)
        df['change'] = change
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
    else:
        if not output:
            print('Sims match')
    return
