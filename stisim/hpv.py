import sciris as sc
import numpy as np
import stisim as ss


class HPV(ss.Disease):

    states = [
        ss.State('sev', float, np.nan),  # Severity of infection, taking values between 0-1
        ss.State('rel_sev', float, 1.0),  # Individual relative risk for rate severe disease growth
        ss.State('rel_sus', float, 1.0),  # Individual relative risk for acquiring infection (does not vary by genotype)
        ss.State('rel_imm', float, 1.0),  # Individual relative level of immunity acquired from infection/vaccination
    ]

    alive_states = [
        ss.State('dead_cancer', bool, False),  # Dead from cancer
    ]

    viral_states = [
        # States related to whether virus is present
        ss.State('susceptible', bool, True),
        # Allowable dysp states: no_dysp
        ss.State('infectious', bool, False),
        # Allowable dysp states: no_dysp, cin1, cin2, cin3
        ss.State('inactive', bool, False),
        # Allowable dysp states: no_dysp, cancer in at least one genotype
    ]

    cell_states = [
        # States related to the cellular changes present in the cervix.
        ss.State('normal', bool, True),
        # Allowable viral states: susceptible, infectious, and inactive
        ss.State('episomal', bool, False),
        # Allowable viral states: susceptible, infectious, and inactive
        ss.State('transformed', bool, False),
        # Allowable viral states: susceptible, infectious, and inactive
        ss.State('cancerous', bool, False),
        # Allowable viral states: inactive
    ]

    derived_states = [
        # From the viral states, cell states, and severity markers, we derive the following additional states:
        ss.State('infected', bool, False),
        # Union of infectious and inactive. Includes people with cancer, latent infections, and active infections
        ss.State('abnormal', bool, False),
        # Union of episomal, transformed, and cancerous. Allowable viral states: infectious
        ss.State('latent', bool, False),
        # Intersection of normal and inactive.
        ss.State('precin', bool, False),
        # Defined as those with sev < clinical_cuttoff[0]
        ss.State('cin1', bool, False),
        # Defined as those with clinical_cuttoff[0] < sev < clinical_cuttoff[1]
        ss.State('cin2', bool, False),
        # Defined as those with clinical_cuttoff[1] < sev < clinical_cuttoff[2]
        ss.State('cin3', bool, False),
        # Defined as those with clinical_cuttoff[2] < sev < clinical_cuttoff[3]
        ss.State('cin', bool, False),
        # Union of CIN1, CIN3, and CIN3
    ]

    # Immune states, by genotype/vaccine
    imm_states = [
        ss.State('sus_imm', float, 0),  # Float, by genotype
        ss.State('sev_imm', float, 0),  # Float, by genotype
        ss.State('peak_imm', float, 0),  # Float, peak level of immunity
        ss.State('nab_imm', float, 0),  # Float, current immunity level
        ss.State('t_imm_event', ss.INT_NAN, 0),  # Int, time since immunity event
        ss.State('cell_imm', float, 0),
    ]

    # Duration of different states: these are floats per person -- used in people.py
    durs = [
        ss.State('dur_infection', float, np.nan),
        # Length of time that a person has any HPV present. For females, dur_infection = dur_episomal + dur_transformed.
        # For males, it's taken from a separate distribution
        ss.State('dur_precin', float, np.nan),
        # Length of time that a person has HPV prior to precancerous changes
        ss.State('dur_cin', float, np.nan),
        # Length of time that a person has precancerous changes
        ss.State('dur_episomal', float, np.nan),
        # Length of time that a person has episomal HPV
        ss.State('dur_transformed', float, np.nan),
        # Length of time that a person has transformed HPV
        ss.State('dur_cancer', float, np.nan),  # Duration of cancer
    ]

    date_states = [state for state in alive_states + viral_states + cell_states + derived_states if not state.fill_value]

    dates = [ss.State(f'ti_{state.name}', ss.INT_NAN, np.nan) for state in date_states]
    dates += [
        ss.State('ti_clearance', ss.INT_NAN, np.nan),
        ss.State('ti_exposed', ss.INT_NAN, np.nan),
        ss.State('ti_reactivated', ss.INT_NAN, np.nan),
    ]

    states_to_set = states + alive_states + viral_states + cell_states + imm_states + durs + dates

    default_init_prev = {
        'age_brackets': np.array([12, 17, 24, 34, 44, 64, 80, 150]),
        'm': np.array([0.0, 0.25, 0.6, 0.25, 0.05, 0.01, 0.0005, 0]),
        'f': np.array([0.0, 0.35, 0.7, 0.25, 0.05, 0.01, 0.0005, 0]),
    }

    pars = sc.objdict()

    # Initialization parameters
    pars.init_hpv_prev = sc.dcp(default_init_prev)  # Initial prevalence
    pars.init_hpv_dist = None  # Initial type distribution
    pars.rel_init_prev = 1.0  # Initial prevalence scale factor

    # Basic disease transmission parameters
    pars.beta                = 0.35   # Per-act transmission probability; absolute value, calibrated
    pars.transf2m            = 1.0   # Relative transmissibility of receptive partners in penile-vaginal intercourse; baseline value
    pars.transm2f            = 3.69  # Relative transmissibility of insertive partners in penile-vaginal intercourse; based on https://doi.org/10.1038/srep10986: "For vaccination types, the risk of male-to-female transmission was higher than that of female-to-male transmission"
    pars.eff_condoms         = 0.7   # The efficacy of condoms; https://www.nejm.org/doi/10.1056/NEJMoa053284?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200www.ncbi.nlm.nih.gov

    # Parameters for disease progression
    pars.hpv_control_prob    = 0.0 # Probability that HPV is controlled latently vs. cleared
    pars.hpv_reactivation    = 0.025 # Placeholder; unused unless hpv_control_prob>0
    pars.dur_cancer          = dict(dist='lognormal', par1=8.0, par2=3.0)  # Duration of untreated invasive cerival cancer before death (years)
    pars.dur_infection_male  = dict(dist='lognormal', par1=1, par2=1) # Duration of infection for men
    pars.clinical_cutoffs    = dict(cin1=0.33, cin2=0.676, cin3=0.8) # Parameters used to map disease severity onto cytological grades
    pars.sev_dist            = dict(dist='normal_pos', par1=1.0, par2=0.25) # Distribution to draw individual level severity scale factors
    pars.age_risk            = dict(age=30, risk=1)

    # Parameters used to calculate immunity
    pars.use_waning      = False  # Whether or not to use waning immunity. If set to False, immunity from infection and vaccination is assumed to stay at the same level permanently
    pars.imm_init        = dict(dist='beta_mean', par1=0.35, par2=0.025)  # beta distribution for initial level of immunity following infection clearance. Parameters are mean and variance from https://doi.org/10.1093/infdis/jiv753
    pars.imm_decay       = dict(form=None)  # decay rate, with half life in years
    pars.cell_imm_init   = dict(dist='beta_mean', par1=0.25, par2=0.025) # beta distribution for level of immunity against persistence/progression of infection following infection clearance and seroconversion
    pars.imm_boost       = []  # Multiplicative factor applied to a person's immunity levels if they get reinfected. No data on this, assumption.
    pars.cross_immunity_sus = None  # Matrix of susceptibility cross-immunity factors, set by init_immunity() in immunity.py
    pars.cross_immunity_sev = None  # Matrix of severity cross-immunity factors, set by init_immunity() in immunity.py
    pars.cross_imm_sus_med   = 0.3
    pars.cross_imm_sus_high  = 0.5
    pars.cross_imm_sev_med   = 0.5
    pars.cross_imm_sev_high  = 0.7
    pars.own_imm_hr = 0.9

    # Genotype parameters
    pars.genotypes       = [16, 18, 'hi5']  # Genotypes to model
    pars.genotype_pars   = sc.objdict()  # Can be directly modified by passing in arguments listed in get_genotype_pars

    # The following variables are stored within the pars dict for ease of access, but should not be directly specified.
    # Rather, they are automatically constructed during sim initialization.
    pars.immunity_map    = None  # dictionary mapping the index of immune source to the type of immunity (vaccine vs natural)
    pars.imm_kin         = None  # Constructed during sim initialization using the nab_decay parameters
    pars.genotype_map    = dict()  # Reverse mapping from number to genotype key
    pars.n_genotypes     = 3 # The number of genotypes circulating in the population
    pars.n_imm_sources   = 3 # The number of immunity sources circulating in the population
    pars.vaccine_pars    = dict()  # Vaccines that are being used; populated during initialization
    pars.vaccine_map     = dict()  # Reverse mapping from number to vaccine key
    pars.cumdysp         = dict()

    def update_states_pre(self, sim):
        """ Update states """

        # Perform updates that are genotype-specific
        ng = self.pars.n_genotypes
        for g in range(ng):
            self.check_clearance(sim, g) # check for clearance (need to do this first)
            self.update_severity(sim, g) # update severity values
            self.check_transformation(sim, g)  # check for new transformations

            for key in ['cin1s', 'cin2s', 'cin3s', 'cancers']:  # update flows
                self.check_progress(sim, key, g)

        # Perform updates that are not genotype specific
        self.check_cancer_deaths(sim)

        sim.people[self.name].nab_imm[:] = sim.people[self.name].peak_imm
        self.check_immunity(sim.people)

    def initialize(self, sim):

        self.init_genotypes(sim)
        self.init_immunity(sim)
        self.init_results(sim)
        self.init_states(sim)
        return

    def init_states(self, sim):
        """
        Initialize prior immunity and seed infections
        """

        # Shorten key variables
        people = sim.people
        ng = self.pars.n_genotypes
        init_hpv_prev = self.pars.init_hpv_prev
        age_brackets = init_hpv_prev['age_brackets']

        # Assign people to age buckets
        age_inds = np.digitize(people.age, age_brackets)

        # Assign probabilities of having HPV to each age/sex group
        hpv_probs = np.full(len(people), np.nan, dtype=float)
        hpv_probs[people.f_inds] = init_hpv_prev['f'][age_inds[people.f_inds]] * self.pars.rel_init_prev
        hpv_probs[people.m_inds] = init_hpv_prev['m'][age_inds[people.m_inds]] * self.pars.rel_init_prev
        hpv_probs[~people.is_active] = 0  # Blank out people who are not yet sexually active

        # Get indices of people who have HPV
        hpv_inds = ss.true(ss.binomial_arr(hpv_probs))

        # Determine which genotype people are infected with
        if self.pars.init_hpv_dist is None:  # No type distribution provided, assume even split
            genotypes = np.random.randint(0, ng, len(hpv_inds))
        else:
            # Error checking
            if not sc.checktype(self.pars.init_hpv_dist, dict):
                errormsg = 'Please provide initial HPV type distribution as a dictionary keyed by genotype.'
                raise ValueError(errormsg)
            if set(self.pars.init_hpv_dist.keys()) != set(self.pars.genotype_map.values()):
                errormsg = 'The initial HPV type distribution does not match the HPV types being simulated.'
                raise ValueError(errormsg)

            type_dist = np.array(list(self.pars.init_hpv_dist.values()))
            genotypes = ss.choose_w(type_dist, len(hpv_inds), unique=False)

        for g in range(ng):
            self.make_new_cases(sim, inds=hpv_inds[genotypes == g], g=g, layer='seed_infection')

    def init_immunity(self, sim, create=True):
        """ Initialize immunity matrices with all genotypes and vaccines in the sim """
        pass

    def update_results(self, sim):
        pass

    def make_new_cases(self, sim):
        pass

    def set_prognoses(self, sim, inds, g):
        """
        Assigns prognoses for all infected women on day of infection.
        """
        pass


    @classmethod
    def _get_from_pars(cls, pars, default=False, key=None, defaultkey='default'):
        ''' Helper function to get the right output from genotype functions '''

        # If a string was provided, interpret it as a key and swap
        if isinstance(default, str):
            key, default = default, key

        # Handle output
        if key is not None:
            try:
                return pars[key]
            except Exception as E:
                errormsg = f'Key "{key}" not found; choices are: {sc.strjoin(pars.keys())}'
                raise sc.KeyNotFoundError(errormsg) from E
        elif default:
            return pars[defaultkey]
        else:
            return pars

    def get_genotype_pars(self, default=False, genotype=None):
        """
        Define the default parameters for the different genotypes
        """
        pass

    def init_genotypes(self, sim, upper_dysp_lim=200):
        pass

    def get_cross_immunity(self, cross_imm_med=None, cross_imm_high=None, own_imm_hr=None, default=False, genotype=None):
        """
        Get the cross immunity between each genotype in a sim
        """
        pars = dict(
            # All values based roughly on https://academic.oup.com/jnci/article/112/10/1030/5753954 or assumptions
            hpv16=dict(
                hpv16=1.0,  # Default for own-immunity
                hpv18=cross_imm_high,
                hi5=cross_imm_med,
                ohr=cross_imm_med,
                hr=cross_imm_med,
                lr=cross_imm_med,
            ),

            hpv18=dict(
                hpv16=cross_imm_high,
                hpv18=1.0,  # Default for own-immunity
                hi5=cross_imm_med,
                ohr=cross_imm_med,
                hr=cross_imm_med,
                lr=cross_imm_med,
            ),

            hi5=dict(
                hpv16=cross_imm_med,
                hpv18=cross_imm_med,
                hi5=own_imm_hr,
                ohr=cross_imm_med,
                hr=cross_imm_med,
                lr=cross_imm_med,
            ),

            ohr=dict(
                hpv16=cross_imm_med,
                hpv18=cross_imm_med,
                hi5=cross_imm_med,
                ohr=own_imm_hr,
                hr=cross_imm_med,
                lr=cross_imm_med,
            ),

            lr=dict(
                hpv16=cross_imm_med,
                hpv18=cross_imm_med,
                hi5=cross_imm_med,
                ohr=cross_imm_med,
                hr=cross_imm_med,
                lr=own_imm_hr,
            ),

        )

        return self._get_from_pars(pars, default, key=genotype, defaultkey='hpv16')

    # %% Methods for computing severity
    def compute_severity(self, t, rel_sev=None, pars=None):
        """
        Process functional form and parameters into values:
        """

        pars = sc.dcp(pars)
        form = pars.pop('form')
        choices = [
            'logf2',
            'logf3',
        ]

        # Scale t
        if rel_sev is not None:
            t = rel_sev * t

        # Process inputs
        if form is None or form == 'logf2':
            output = ss.logf2(t, **pars)

        elif form == 'logf3':
            output = ss.logf3(t, **pars)

        elif callable(form):
            output = form(t, **pars)

        else:
            errormsg = f'The selected functional form "{form}" is not implemented; choices are: {sc.strjoin(choices)}'
            raise NotImplementedError(errormsg)

        return output

    def compute_inv_severity(self, sev_vals, rel_sev=None, pars=None):
        """
        Compute time to given severity level given input parameters
        """

        pars = sc.dcp(pars)
        form = pars.pop('form')
        choices = [
            'logf2',
            'logf3',
        ]

        # Process inputs
        if form is None or form == 'logf2':
            output = ss.invlogf2(sev_vals, **pars)

        elif form == 'logf3':
            output = ss.invlogf3(sev_vals, **pars)

        elif callable(form):
            output = form(sev_vals, **pars)

        else:
            errormsg = f'The selected functional form "{form}" is not implemented; choices are: {sc.strjoin(choices)}'
            raise NotImplementedError(errormsg)

        # Scale by relative severity
        if rel_sev is not None:
            output = output / rel_sev

        return output

    def compute_severity_integral(self, t, rel_sev=None, pars=None):
        """
        Process functional form and parameters into values:
        """

        pars = sc.dcp(pars)
        form = pars.pop('form')
        choices = [
            'logf2',
            'logf3 with s=1',
        ]

        # Scale t
        if rel_sev is not None:
            t = rel_sev * t

        # Process inputs
        if form is None or form == 'logf2':
            output = ss.intlogf2(t, **pars)

        elif form == 'logf3':
            s = pars.pop('s')
            if s == 1:
                output = ss.intlogf2(t, **pars)
            else:
                errormsg = f'Analytic integral for logf3 only implemented for s=1. Select integral=numeric.'
                raise NotImplementedError(errormsg)

        else:
            errormsg = f'Analytic integral for the selected functional form "{form}" is not implemented; choices are: {sc.strjoin(choices)}, or select integral=numeric.'
            raise NotImplementedError(errormsg)

        return output

    def check_immunity(self, people):
        """
        Calculate people's immunity on this timestep from prior infections.
        As an example, suppose HPV16 and 18 are in the sim, and the cross-immunity matrix is:

            pars['immunity'] = np.array([[1., 0.5],
                                         [0.3, 1.]])

        This indicates that people who've had HPV18 have 50% protection against getting 16, and
        people who've had 16 have 30% protection against getting 18.
        Now suppose we have 3 people, whose immunity levels are

            people.nab_imm = np.array([[0.9, 0.0, 0.0],
                                   [0.0, 0.7, 0.0]])

        This indicates that person 1 has a prior HPV16 infection, person 2 has a prior HPV18
        infection, and person 3 has no history of infection.

        In this function, we take the dot product of pars['immunity'] and people.nab_imm to get:

            people.sus_imm = np.array([[0.9 , 0.35, 0.  ],
                                       [0.27, 0.7 , 0.  ]])

        This indicates that the person 1 has high protection against reinfection with HPV16, and
        some (30% of 90%) protection against infection with HPV18, and so on.

        """
        cross_immunity_sus = self.pars.cross_immunity_sus  # scalars to be applied to sus immunity level
        cross_immunity_sev = self.pars.cross_immunity_sev  # scalars to be applied to sev immunity level
        sus_imm = np.dot(cross_immunity_sus, self.nab_imm)  # Dot product gives immunity to all genotypes
        sev_imm = np.dot(cross_immunity_sev, self.cell_imm)  # Dot product gives immunity to all genotypes
        self.sus_imm[:] = np.minimum(sus_imm, np.ones_like(sus_imm))  # Don't let this be above 1
        self.sev_imm[:] = np.minimum(sev_imm, np.ones_like(sev_imm))  # Don't let this be above 1
        return

    def check_clearance(self, sim, gtype):
        """
        Check for HPV clearance.
        """
        f_filter = (sim.people.female & self.infectious[gtype, :]).nonzero()[-1]
        m_filter = (sim.people.male & self.infectious[gtype, :]).nonzero()[-1]
        f_inds = self.check_inds_true(self.infectious[gtype, :], self.ti_clearance[gtype, :], filter_inds=f_filter)
        m_inds = self.check_inds_true(self.infectious[gtype, :], self.ti_clearance[gtype, :], filter_inds=m_filter)
        m_cleared_inds = m_inds  # All males clear

        # For females, determine who clears and who controls
        if self.pars.hpv_control_prob > 0:
            latent_probs = np.full(len(f_inds), self.pars.hpv_control_prob, dtype=float)
            latent_bools = ss.binomial_arr(latent_probs)
            latent_inds = f_inds[latent_bools]

            if len(latent_inds):
                self.susceptible[gtype, latent_inds] = False  # should already be false
                self.infectious[gtype, latent_inds] = False
                self.inactive[gtype, latent_inds] = True
                self.ti_clearance[gtype, latent_inds] = ss.INT_NAN

            f_cleared_inds = f_inds[~latent_bools]

        else:
            f_cleared_inds = f_inds

        cleared_inds = np.array(m_cleared_inds.tolist() + f_cleared_inds.tolist())

        # Now reset disease states
        if len(cleared_inds):
            self.susceptible[gtype, cleared_inds] = True
            self.infectious[gtype, cleared_inds] = False
            self.inactive[gtype, cleared_inds] = False  # should already be false

        if len(f_cleared_inds):
            # female_cleared_inds = np.intersect1d(cleared_inds, self.f_inds) # Only give natural immunity to females
            self.update_peak_immunity(f_cleared_inds, imm_pars=sim.pars, imm_source=gtype)  # update immunity
            self.ti_reactivated[gtype, f_cleared_inds] = ss.INT_NAN

        # Whether infection is controlled on not, clear all cell changes and severity markeres
        self.episomal[gtype, f_inds] = False
        self.transformed[gtype, f_inds] = False
        self.sev[gtype, f_inds] = np.nan
        self.ti_cin1[gtype, f_inds] = ss.INT_NAN
        self.ti_cin2[gtype, f_inds] = ss.INT_NAN
        self.ti_cin3[gtype, f_inds] = ss.INT_NAN

    def update_severity(self, sim, gtype):
        """
        Update disease severity for women with infection and update their current severity
        """
        gpars = self.pars.genotype_pars[gtype]

        # Only need to update severity for people who with dysplasia underway
        fg_cin_inds = ss.true(sim.people.female & ~np.isnan(self.sev[gtype, :]) & self.infectious[gtype, :])
        fg_cin_underway_inds = fg_cin_inds[(sim.t >= self.ti_cin1[gtype, fg_cin_inds])]  # Dysplasia is underway

        time_with_dysplasia = (sim.ti - self.date_cin1[gtype, fg_cin_underway_inds]) * sim.dt
        rel_sevs = self.rel_sev[fg_cin_underway_inds]
        if (time_with_dysplasia < 0).any() or (np.isnan(time_with_dysplasia)).any():
            errormsg = 'Time with dysplasia cannot be less than zero or NaN.'
            raise ValueError(errormsg)
        if (np.isnan(self.ti_exposed[gtype, fg_cin_inds])).any():
            errormsg = f'No date of exposure defined for {ss.iundefined(self.ti_exposed[gtype, fg_cin_inds], fg_cin_inds)} on timestep {sim.ti}'
            raise ValueError(errormsg)
        if (np.isnan(self.ti_cin1[gtype, fg_cin_inds])).any():
            errormsg = f'No date of dysplasia onset defined for {ss.iundefined(self.ti_cin1[gtype, fg_cin_inds], fg_cin_inds)} on timestep {sim.ti}'
            raise ValueError(errormsg)

        self.sev[gtype, fg_cin_underway_inds] = self.compute_severity(time_with_dysplasia, rel_sev=rel_sevs, pars=gpars['sev_fn'])

        if (np.isnan(self.sev[gtype, fg_cin_underway_inds])).any():
            errormsg = 'Invalid severity values.'
            raise ValueError(errormsg)

    def check_transformation(self, sim, gtype):
        """ Check for new transformations """
        # Only include infectious, episomal females who haven't already cleared infection
        filter_inds = self.true_by_genotype('episomal', gtype)
        inds = self.check_inds(self.transformed[gtype, :], self.ti_transformed[gtype, :], filter_inds=filter_inds)
        self.transformed[gtype, inds] = True  # Now transformed, cannot clear
        self.ti_clearance[gtype, inds] = ss.INT_NAN  # Remove their clearance dates

    def check_progress(self, sim, what, gtype):
        """ Wrapper function for all the new progression checks """
        if what == 'cin1s':
            self.check_cin1(sim, gtype)
        elif what == 'cin2s':
            self.check_cin2(sim, gtype)
        elif what == 'cin3s':
            self.check_cin3(sim, gtype)
        elif what == 'cancers':
            self.check_cancer(sim, gtype)
        return

    def check_cancer_deaths(self, sim):
        pass

    def set_severity(self, sim, param, g):
        pass

    def update_peak_immunity(self, sim, f_cleared_inds, imm_pars, imm_source):
        pass

    def check_cin1(self, sim, gtype):
        """ Check for new progressions to CIN1 """
        # Only include infectious females who haven't already cleared CIN1 or progressed to CIN2
        filters = (self.infectious[gtype, :] * sim.people.female * ~(self.ti_clearance[gtype, :] <= sim.ti)
                   * (self.ti_cin2[gtype, :] >= sim.t))
        filter_inds = filters.nonzero()[0]
        inds = self.check_inds(self.cin1[gtype, :], self.ti_cin1[gtype, :], filter_inds=filter_inds)
        self.cin1[gtype, inds] = True
        return

    def check_cin2(self, gtype):
        """ Check for new progressions to CIN2 """
        filter_inds = self.true_by_genotype('cin1', gtype)
        inds = self.check_inds(self.cin2[gtype,:], self.ti_cin2[gtype,:], filter_inds=filter_inds)
        self.cin2[gtype, inds] = True
        self.cin1[gtype, inds] = False # No longer counted as CIN1
        return

    def check_cin3(self, gtype):
        """ Check for new progressions to CIN3 """
        filter_inds = self.true_by_genotype('cin2', gtype)
        inds = self.check_inds(self.cin3[gtype, :], self.ti_cin3[gtype, :], filter_inds=filter_inds)
        self.cin3[gtype, inds] = True
        self.cin2[gtype, inds] = False  # No longer counted as CIN2
        return

    def check_cancer(self, gtype):
        """ Check for new progressions to cancer """
        filter_inds = ss.true(self.transformed)
        inds = self.check_inds(self.cancerous[gtype, :], self.ti_cancerous[gtype, :], filter_inds=filter_inds)

        # Set infectious states
        self.susceptible[:, inds] = False  # No longer susceptible to any genotype
        self.infectious[:, inds] = False  # No longer counted as infectious with any genotype
        self.inactive[:, inds] = True  # Set any other infections from any other genotypes to inactive

        self.ti_clearance[:, inds] = ss.INT_NAN  # Remove their clearance dates for all genotypes

        # Deal with dysplasia states and dates
        for g in range(self.pars.ng):
            if g != gtype:
                self.ti_cancerous[g, inds] = ss.INT_NAN  # Remove date of cancer for all types but the one causing cancer
                self.ti_cin1[g, inds] = ss.INT_NAN
                self.ti_cin2[g, inds] = ss.INT_NAN
                self.ti_cin3[g, inds] = ss.INT_NAN

        # Set the properties related to cell changes and disease severity markers
        self.cancerous[gtype, inds] = True
        self.episomal[:, inds] = False  # No longer counted as episomal with any genotype
        self.transformed[:, inds] = False  # No longer counted as transformed with any genotype
        self.sev[:, inds] = np.nan  # NOTE: setting this to nan means this people no longer counts as CIN1/2/3, since those categories are based on this value
        return

    