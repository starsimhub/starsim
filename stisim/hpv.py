import sciris as sc
import numpy as np
from .people import State
from .results import Result
from .modules import Module
from . import settings as sss
from . import utils as ssu

class HPV(Module):

    states = [
        State('sev', sss.default_float, np.nan, shape='n_genotypes'),  # Severity of infection, taking values between 0-1
        State('rel_sev', sss.default_float, 1.0),  # Individual relative risk for rate severe disease growth
        State('rel_sus', sss.default_float, 1.0),  # Individual relative risk for acquiring infection (does not vary by genotype)
        State('rel_imm', sss.default_float, 1.0),  # Individual relative level of immunity acquired from infection clearance/vaccination
    ]

    alive_states = [
        State('dead_cancer', bool, False, label='Cumulative cancer deaths'),  # Dead from cancer
    ]

    viral_states = [
        # States related to whether virus is present
        State('susceptible', bool, True, 'n_genotypes', label='Number susceptible', color='#4d771e'),
        # Allowable dysp states: no_dysp
        State('infectious', bool, False, 'n_genotypes', label='Number infectious', color='#c78f65'),
        # Allowable dysp states: no_dysp, cin1, cin2, cin3
        State('inactive', bool, False, 'n_genotypes', label='Number with inactive infection', color='#9e1149'),
        # Allowable dysp states: no_dysp, cancer in at least one genotype
    ]

    cell_states = [
        # States related to the cellular changes present in the cervix.
        State('normal', bool, True, 'n_genotypes', label='Number with no cellular changes', color='#9e1149'),
        # Allowable viral states: susceptible, infectious, and inactive
        State('episomal', bool, False, 'n_genotypes', label='Number with episomal infection', color='#9e1149'),
        # Allowable viral states: susceptible, infectious, and inactive
        State('transformed', bool, False, 'n_genotypes', label='Number with transformation', color='#9e1149'),
        # Allowable viral states: susceptible, infectious, and inactive
        State('cancerous', bool, False, 'n_genotypes', label='Number with cancer', color='#5f5cd2'),
        # Allowable viral states: inactive
    ]

    derived_states = [
        # From the viral states, cell states, and severity markers, we derive the following additional states:
        State('infected', bool, False, 'n_genotypes', label='Number infected', color='#c78f65'),
        # Union of infectious and inactive. Includes people with cancer, people with latent infections, and people with active infections
        State('abnormal', bool, False, 'n_genotypes', label='Number with abnormal cells', color='#9e1149'),
        # Union of episomal, transformed, and cancerous. Allowable viral states: infectious
        State('latent', bool, False, 'n_genotypes', label='Number with latent infection', color='#5f5cd2'),
        # Intersection of normal and inactive.
        State('precin', bool, False, 'n_genotypes', label='Number with precin', color='#9e1149'),
        # Defined as those with sev < clinical_cuttoff[0]
        State('cin1', bool, False, 'n_genotypes', label='Number with cin1', color='#9e1149'),
        # Defined as those with clinical_cuttoff[0] < sev < clinical_cuttoff[1]
        State('cin2', bool, False, 'n_genotypes', label='Number with cin2', color='#9e1149'),
        # Defined as those with clinical_cuttoff[1] < sev < clinical_cuttoff[2]
        State('cin3', bool, False, 'n_genotypes', label='Number with cin3', color='#5f5cd2'),
        # Defined as those with clinical_cuttoff[2] < sev < clinical_cuttoff[3]
        State('cin', bool, False, 'n_genotypes', label='Number with detectable dysplasia', color='#5f5cd2'),
        # Union of CIN1, CIN3, and CIN3
    ]

    # Immune states, by genotype/vaccine
    imm_states = [
        State('sus_imm', ssd.default_float, 0, 'n_imm_sources'),  # Float, by genotype
        State('sev_imm', ssd.default_float, 0, 'n_imm_sources'),  # Float, by genotype
        State('peak_imm', ssd.default_float, 0, 'n_imm_sources'),  # Float, peak level of immunity
        State('nab_imm', ssd.default_float, 0, 'n_imm_sources'),  # Float, current immunity level
        State('t_imm_event', ssd.default_int, 0, 'n_imm_sources'),  # Int, time since immunity event
        State('cell_imm', ssd.default_float, 0, 'n_imm_sources'),
    ]

    # Duration of different states: these are floats per person -- used in people.py
    durs = [
        State('dur_infection', ssd.default_float, np.nan, shape='n_genotypes'),
        # Length of time that a person has any HPV present. Defined for males and females. For females, dur_infection = dur_episomal + dur_transformed. For males, it's taken from a separate distribution
        State('dur_precin', ssd.default_float, np.nan, shape='n_genotypes'),
        # Length of time that a person has HPV prior to precancerous changes
        State('dur_cin', ssd.default_float, np.nan, shape='n_genotypes'),
        # Length of time that a person has precancerous changes
        State('dur_episomal', ssd.default_float, np.nan, shape='n_genotypes'),
        # Length of time that a person has episomal HPV
        State('dur_transformed', ssd.default_float, np.nan, shape='n_genotypes'),
        # Length of time that a person has transformed HPV
        State('dur_cancer', ssd.default_float, np.nan, shape='n_genotypes'),  # Duration of cancer
    ]

    date_states = [state for state in alive_states + viral_states + cell_states + derived_states if not state.fill_value]

    dates = [State(f'date_{state.name}', ssd.default_float, np.nan, shape=state.shape) for state in date_states]
    dates += [
        State('date_clearance', ssd.default_float, np.nan, shape='n_genotypes'),
        State('date_exposed', ssd.default_float, np.nan, shape='n_genotypes'),
        State('date_reactivated', ssd.default_float, np.nan, shape='n_genotypes'),
    ]

    states_to_set = states + alive_states + viral_states + cell_states + imm_states + durs + dates

    default_init_prev = {
        'age_brackets': np.array([12, 17, 24, 34, 44, 64, 80, 150]),
        'm': np.array([0.0, 0.25, 0.6, 0.25, 0.05, 0.01, 0.0005, 0]),
        'f': np.array([0.0, 0.35, 0.7, 0.25, 0.05, 0.01, 0.0005, 0]),
    }

    pars = {}

    # Initialization parameters
    pars['init_hpv_prev'] = sc.dcp(default_init_prev)  # Initial prevalence
    pars['init_hpv_dist'] = None  # Initial type distribution
    pars['rel_init_prev'] = 1.0  # Initial prevalence scale factor

    # Basic disease transmission parameters
    pars['beta']                = 0.35   # Per-act transmission probability; absolute value, calibrated
    pars['transf2m']            = 1.0   # Relative transmissibility of receptive partners in penile-vaginal intercourse; baseline value
    pars['transm2f']            = 3.69  # Relative transmissibility of insertive partners in penile-vaginal intercourse; based on https://doi.org/10.1038/srep10986: "For vaccination types, the risk of male-to-female transmission was higher than that of female-to-male transmission"
    pars['eff_condoms']         = 0.7   # The efficacy of condoms; https://www.nejm.org/doi/10.1056/NEJMoa053284?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200www.ncbi.nlm.nih.gov

    # Parameters for disease progression
    pars['hpv_control_prob']    = 0.0 # Probability that HPV is controlled latently vs. cleared
    pars['hpv_reactivation']    = 0.025 # Placeholder; unused unless hpv_control_prob>0
    pars['dur_cancer']          = dict(dist='lognormal', par1=8.0, par2=3.0)  # Duration of untreated invasive cerival cancer before death (years)
    pars['dur_infection_male']  = dict(dist='lognormal', par1=1, par2=1) # Duration of infection for men
    pars['clinical_cutoffs']    = dict(cin1=0.33, cin2=0.676, cin3=0.8) # Parameters used to map disease severity onto cytological grades
    pars['sev_dist']            = dict(dist='normal_pos', par1=1.0, par2=0.25) # Distribution to draw individual level severity scale factors
    pars['age_risk']            = dict(age=30, risk=1)

    # Parameters used to calculate immunity
    pars['use_waning']      = False  # Whether or not to use waning immunity. If set to False, immunity from infection and vaccination is assumed to stay at the same level permanently
    pars['imm_init']        = dict(dist='beta_mean', par1=0.35, par2=0.025)  # beta distribution for initial level of immunity following infection clearance. Parameters are mean and variance from https://doi.org/10.1093/infdis/jiv753
    pars['imm_decay']       = dict(form=None)  # decay rate, with half life in years
    pars['cell_imm_init']   = dict(dist='beta_mean', par1=0.25, par2=0.025) # beta distribution for level of immunity against persistence/progression of infection following infection clearance and seroconversion
    pars['imm_boost']       = []  # Multiplicative factor applied to a person's immunity levels if they get reinfected. No data on this, assumption.
    pars['cross_immunity_sus'] = None  # Matrix of susceptibility cross-immunity factors, set by init_immunity() in immunity.py
    pars['cross_immunity_sev'] = None  # Matrix of severity cross-immunity factors, set by init_immunity() in immunity.py
    pars['cross_imm_sus_med']   = 0.3
    pars['cross_imm_sus_high']  = 0.5
    pars['cross_imm_sev_med']   = 0.5
    pars['cross_imm_sev_high']  = 0.7
    pars['own_imm_hr'] = 0.9

    # Genotype parameters
    pars['genotypes']       = [16, 18, 'hi5']  # Genotypes to model
    pars['genotype_pars']   = sc.objdict()  # Can be directly modified by passing in arguments listed in get_genotype_pars

    # The following variables are stored within the pars dict for ease of access, but should not be directly specified.
    # Rather, they are automatically constructed during sim initialization.
    pars['immunity_map']    = None  # dictionary mapping the index of immune source to the type of immunity (vaccine vs natural)
    pars['imm_kin']         = None  # Constructed during sim initialization using the nab_decay parameters
    pars['genotype_map']    = dict()  # Reverse mapping from number to genotype key
    pars['n_genotypes']     = 3 # The number of genotypes circulating in the population
    pars['n_imm_sources']   = 3 # The number of immunity sources circulating in the population
    pars['vaccine_pars']    = dict()  # Vaccines that are being used; populated during initialization
    pars['vaccine_map']     = dict()  # Reverse mapping from number to vaccine key
    pars['cumdysp']         = dict()

    @classmethod
    def update_states(cls, sim):

        # Perform updates that are genotype-specific
        ng = sim.pars[cls.name]['n_genotypes']
        for g in range(ng):
            cls.check_clearance(sim, g) # check for clearance (need to do this first)
            cls.update_severity(sim, g) # update severity values
            cls.check_transformation(sim, g)  # check for new transformations

            for key in ['cin1s','cin2s','cin3s','cancers']:  # update flows
                cls.check_progress(sim, key, g)

        # Perform updates that are not genotype specific
        cls.check_cancer_deaths(sim)

        sim.people[cls.name].nab_imm[:] = sim.people[cls.name].peak_imm
        cls.check_immunity(sim.people)
    @classmethod
    def initialize(cls, sim):

        cls.init_genotypes(sim)
        cls.init_immunity(sim)
        cls.init_pars(sim)
        cls.init_results(sim)
        sim.people.add_module(cls)
        cls.init_states(sim)
        return

    @classmethod
    def init_states(cls, sim):
        '''
        Initialize prior immunity and seed infections
        '''

        # Shorten key variables
        people = sim.people
        ng = people.pars[cls.name]['n_genotypes']
        init_hpv_prev = people.pars[cls.name]['init_hpv_prev']
        age_brackets = init_hpv_prev['age_brackets']

        # Assign people to age buckets
        age_inds = np.digitize(people.age, age_brackets)

        # Assign probabilities of having HPV to each age/sex group
        hpv_probs = np.full(len(people), np.nan, dtype=ssd.default_float)
        hpv_probs[people.f_inds] = init_hpv_prev['f'][age_inds[people.f_inds]] * people.pars[cls.name]['rel_init_prev']
        hpv_probs[people.m_inds] = init_hpv_prev['m'][age_inds[people.m_inds]] * people.pars[cls.name]['rel_init_prev']
        hpv_probs[~people.is_active] = 0  # Blank out people who are not yet sexually active

        # Get indices of people who have HPV
        hpv_inds = ssu.true(ssu.binomial_arr(hpv_probs))

        # Determine which genotype people are infected with
        if people.pars[cls.name]['init_hpv_dist'] is None:  # No type distribution provided, assume even split
            genotypes = np.random.randint(0, ng, len(hpv_inds))
        else:
            # Error checking
            if not sc.checktype(people.pars[cls.name]['init_hpv_dist'], dict):
                errormsg = f'Please provide initial HPV type distribution as a dictionary keyed by genotype, not {people.pars[cls.name]["init_hpv_dist"]}'
                raise ValueError(errormsg)
            if set(people.pars[cls.name]['init_hpv_dist'].keys()) != set(people.pars[cls.name]['genotype_map'].values()):
                errormsg = f'The HPV types provided in the initial HPV type distribution are not the same as the HPV types being simulated: {people.pars[cls.name]["init_hpv_dist"].keys()} vs {people.pars[cls.name]["genotype_map"].values()}.'
                raise ValueError(errormsg)

            type_dist = np.array(list(people.pars[cls.name]['init_hpv_dist'].values()))
            genotypes = ssu.choose_w(type_dist, len(hpv_inds), unique=False)

        for g in range(ng):
            cls.make_new_cases(sim, inds=hpv_inds[genotypes == g], g=g, layer='seed_infection')
    @classmethod
    def init_pars(cls, sim):
        # Merge parameters
        if cls.name not in sim.pars:
            sim.pars[cls.name] = sc.objdict(sc.dcp(cls.pars))
        else:
            if ~isinstance(sim.pars[cls.name], sc.objdict):
                sim.pars[cls.name] = sc.objdict(sim.pars[cls.name])
            for k, v in cls.pars.items():
                if k not in sim.pars[cls.name]:
                    sim.pars[cls.name][k] = v

    @classmethod
    def init_results(cls, sim):
        sim.results[cls.name] = sc.objdict()
        sim.results[cls.name]['n_susceptible'] = Result(cls.name, 'n_susceptible', sim.npts, dtype=int)
        sim.results[cls.name]['n_infected'] = Result(cls.name, 'n_infected', sim.npts, dtype=int)
        sim.results[cls.name]['prevalence'] = Result(cls.name, 'prevalence', sim.npts, dtype=float)
        sim.results[cls.name]['new_infections'] = Result(cls.name, 'n_infected', sim.npts, dtype=int)
        sim.results[cls.name]['n_art'] = Result(cls.name, 'n_art', sim.npts, dtype=int)

    @classmethod
    def init_immunity(cls, sim, create=True):
        ''' Initialize immunity matrices with all genotypes and vaccines in the sim'''

        # Pull out all of the circulating genotypes for cross-immunity
        ng = cls.pars['n_genotypes']

        # # Pull out all the unique vaccine products
        # vx_intvs = [x for x in sim['interventions'] if isinstance(x, hpi.BaseVaccination)]
        # vx_intv_prods = [x.product.genotype_pars['name'].values[0] for x in vx_intvs]
        # unique_vx_prods, unique_vx_prod_inds = np.unique(vx_intv_prods, return_index=True)
        # unique_vx_prod_dict = dict()
        # for unique_vx_prod, unique_vx_prod_ind in zip(unique_vx_prods, unique_vx_prod_inds):
        #     unique_vx_prod_dict[unique_vx_prod] = unique_vx_prod_ind
        # nv = len(unique_vx_prods) if len(vx_intvs) else 0
        #
        # unique_vx_intvs = sc.autolist()
        # for ind in unique_vx_prod_inds:
        #     unique_vx_intvs += vx_intvs[ind]
        #
        # for iv, vx_intv in enumerate(vx_intvs):
        #     vx_intv.product.imm_source = unique_vx_prod_dict[vx_intv_prods[iv]] + ng
        #
        # txv_intvs = [x for x in sim['interventions'] if isinstance(x, hpi.BaseTxVx)]
        # txv_intv_prods = [x.product.name.replace('2', '1') for x in txv_intvs]
        # unique_txv_prods, unique_txv_prod_inds = np.unique(txv_intv_prods, return_index=True)
        # unique_txv_prod_dict = dict()
        # for unique_txv_prod, unique_txv_prod_ind in zip(unique_txv_prods, unique_txv_prod_inds):
        #     unique_txv_prod_dict[unique_txv_prod] = unique_txv_prod_ind
        # ntxv = len(unique_txv_prods) if len(txv_intvs) else 0
        #
        # unique_txv_intvs = sc.autolist()
        # for ind in unique_txv_prod_inds:
        #     unique_txv_intvs += txv_intvs[ind]
        #
        # for itxv, txv_intv in enumerate(txv_intvs):
        #     txv_intv.product.imm_source = unique_txv_prod_dict[txv_intv_prods[itxv]] + ng + nv
        #
        # all_vx_intvs = unique_vx_intvs + unique_txv_intvs
        #
        # Dimension for immunity matrix
        ndim = ng #+ nv + ntxv

        # If cross-immunity values have been provided, process them
        if cls.pars['cross_immunity_sus'] is None or create:

            # Precompute waning - same for all genotypes
            if cls.pars['use_waning']:
                imm_decay = sc.dcp(cls.pars['imm_decay'])
                if 'half_life' in imm_decay.keys():
                    imm_decay['half_life'] /= sim['dt']
                cls.pars['imm_kin'] = cls.precompute_waning(t=sim.tvec, pars=imm_decay)

            cls.pars['immunity_map'] = dict()
            # Firstly, initialize immunity matrix with defaults. These are then overwitten with specific values below
            immunity = np.ones((ng, ng), dtype=ssd.default_float)  # Fill with defaults

            # Next, overwrite these defaults with any known immunity values about specific genotypes
            default_cross_immunity = cls.get_cross_immunity(cross_imm_med=cls.pars['cross_imm_sus_med'],
                                                              cross_imm_high=cls.pars['cross_imm_sus_high'],
                                                              own_imm_hr=cls.pars['own_imm_hr'])
            for i in range(ng):
                cls.pars['immunity_map'][i] = 'infection'
                label_i = cls.pars['genotype_map'][i]
                for j in range(ng):
                    label_j = cls.pars['genotype_map'][j]
                    if label_i in default_cross_immunity and label_j in default_cross_immunity:
                        immunity[j][i] = default_cross_immunity[label_j][label_i]

            # for vi, vx_intv in enumerate(all_vx_intvs):
            #     genotype_pars_df = vx_intv.product.genotype_pars[
            #         vx_intv.product.genotype_pars.genotype.isin(sim['genotype_map'].values())]  # TODO fix this
            #     vacc_mapping = [genotype_pars_df[genotype_pars_df.genotype == gtype].rel_imm.values[0] for gtype in
            #                     sim['genotype_map'].values()]
            #     vacc_mapping += [1] * (vi + 1)  # Add on some ones to pad out the matrix
            #     vacc_mapping = np.reshape(vacc_mapping, (len(immunity) + 1, 1)).astype(hpd.default_float)  # Reshape
            #     immunity = np.hstack((immunity, vacc_mapping[0:len(immunity), ]))
            #     immunity = np.vstack((immunity, np.transpose(vacc_mapping)))

            cls.pars['cross_immunity_sus'] = immunity

        # If cross-immunity values have been provided, process them
        if cls.pars['cross_immunity_sev'] is None or create:

            # # Precompute waning - same for all genotypes
            # if cls.pars['use_waning']:
            #     imm_decay = sc.dcp(cls.pars['imm_decay'])
            #     if 'half_life' in imm_decay.keys():
            #         imm_decay['half_life'] /= sim['dt']
            #     cls.pars['imm_kin'] = cls.precompute_waning(t=sim.tvec, pars=imm_decay)

            cls.pars['immunity_map'] = dict()
            # Firstly, initialize immunity matrix with defaults. These are then overwitten with specific values below
            immunity = np.ones((ng, ng), dtype=ssd.default_float)  # Fill with defaults

            # Next, overwrite these defaults with any known immunity values about specific genotypes
            default_cross_immunity = cls.get_cross_immunity(cross_imm_med=cls.pars['cross_imm_sev_med'],
                                                              cross_imm_high=cls.pars['cross_imm_sev_high'],
                                                              own_imm_hr=cls.pars['own_imm_hr'])
            for i in range(ng):
                cls.pars['immunity_map'][i] = 'infection'
                label_i = cls.pars['genotype_map'][i]
                for j in range(ng):
                    label_j = cls.pars['genotype_map'][j]
                    if label_i in default_cross_immunity and label_j in default_cross_immunity:
                        immunity[j][i] = default_cross_immunity[label_j][label_i]

            # for vi, vx_intv in enumerate(all_vx_intvs):
            #     genotype_pars_df = vx_intv.product.genotype_pars[
            #         vx_intv.product.genotype_pars.genotype.isin(sim['genotype_map'].values())]  # TODO fix this
            #     vacc_mapping = [genotype_pars_df[genotype_pars_df.genotype == gtype].rel_imm.values[0] for gtype in
            #                     sim['genotype_map'].values()]
            #     vacc_mapping += [1] * (vi + 1)  # Add on some ones to pad out the matrix
            #     vacc_mapping = np.reshape(vacc_mapping, (len(immunity) + 1, 1)).astype(hpd.default_float)  # Reshape
            #     immunity = np.hstack((immunity, vacc_mapping[0:len(immunity), ]))
            #     immunity = np.vstack((immunity, np.transpose(vacc_mapping)))

            cls.pars['cross_immunity_sev'] = immunity

        cls.pars['cross_immunity_sus'] = cls.pars['cross_immunity_sus'].astype('float32')
        cls.pars['cross_immunity_sev'] = cls.pars['cross_immunity_sev'].astype('float32')
        cls.pars['n_imm_sources'] = ndim

        pass

    @classmethod
    def update_results(cls, sim):
        pass
    @classmethod
    def make_new_cases(cls, sim):
        eff_condoms = sim.pars[cls.name]['eff_condoms']

        for k, layer in sim.people.contacts.items():
            condoms = layer.pars['condoms']
            effective_condoms = ssd.default_float(condoms * eff_condoms)
            if k in sim.pars[cls.name]['beta']:
                for g in range(sim.pars[cls.name]['n_genotypes']):
                    rel_trans = (sim.people[cls.name].infectious[g,:] & sim.people.alive).astype(float)
                    rel_sus = (sim.people[cls.name].susceptible[g,:] & sim.people.alive).astype(float)
                    for a,b in [[layer['p1'],layer['p2']],[layer['p2'],layer['p1']]]:
                        # probability of a->b transmission
                        p_transmit = rel_trans[a]*sim.people[cls.name].rel_trans[a]*rel_sus[b]*sim.people[cls.name].rel_sus[b]*sim.pars[cls.name]['beta'][k]*(1-effective_condoms)
                        cls.make_new_cases(sim, b[np.random.random(len(a))<p_transmit], g)

    @classmethod
    def make_new_cases(cls, sim, inds, g, layer=None):
        '''
                Infect people and determine their eventual outcomes.
                Method also deduplicates input arrays in case one agent is infected many times
                and stores who infected whom in infection_log list.

                Args:
                    inds      (array): array of people to infect
                    g         (int):   int of genotype to infect people with
                    layer     (str):   contact layer this infection was transmitted on

                Returns:
                    count (int): number of people infected
                '''

        if len(inds) == 0:
            return 0

        # Check whether anyone is already infected with genotype - this should not happen because we only
        # infect susceptible people
        if len(ssu.true(sim.people[cls.name].infectious[g, inds])):
            errormsg = f'Attempting to reinfect the following agents who are already infected with genotype {g}: {ssu.itruei(sim.people[cls.name].infectious[g, :], inds)}'
            raise ValueError(errormsg)

        dt = sim['dt']

        # Set date of infection and exposure
        base_t = sim.t
        sim.people[cls.name].date_infectious[g, inds] = base_t
        if layer != 'reactivation':
            sim.people[cls.name].date_exposed[g, inds] = base_t

        # # Count reinfections and remove any previous dates
        # self.genotype_flows['reinfections'][g] += self.scale_flows(
        #     (~np.isnan(self.date_clearance[g, inds])).nonzero()[-1])
        # self.flows['reinfections'] += self.scale_flows((~np.isnan(self.date_clearance[g, inds])).nonzero()[-1])
        # for key in ['date_clearance', 'date_transformed']:
        #     self[key][g, inds] = np.nan
        #
        # # Count reactivations and adjust latency status
        # if layer == 'reactivation':
        #     self.genotype_flows['reactivations'][g] += self.scale_flows(inds)
        #     self.flows['reactivations'] += self.scale_flows(inds)
        #     self.age_flows['reactivations'] += \
        #     np.histogram(self.age[inds], bins=self.age_bin_edges, weights=self.scale[inds])[0]
        #     self.latent[g, inds] = False  # Adjust states -- no longer latent
        #     self.date_reactivated[g, inds] = base_t

        # Update states, genotype info, and flows
        sim.people[cls.name].susceptible[g, inds] = False  # no longer susceptible
        sim.people[cls.name].infectious[g, inds] = True  # now infectious
        sim.people[cls.name].episomal[g, inds] = True  # now episomal
        sim.people[cls.name].inactive[g, inds] = False  # no longer inactive

        # # Add to flow results. Note, we only count these infectious in the results if they happened at this timestep
        # if layer != 'seed_infection':
        #     # Create overall flows
        #     self.flows['infections'] += self.scale_flows(inds)  # Add the total count to the total flow data
        #     self.genotype_flows['infections'][g] += self.scale_flows(inds)  # Add the count by genotype to the flow data
        #     self.age_flows['infections'][:] += \
        #     np.histogram(self.age[inds], bins=self.age_bin_edges, weights=self.scale[inds])[0]
        #
        #     # Create by-sex flows
        #     infs_female = self.scale_flows(hpu.true(self.is_female[inds]))
        #     infs_male = self.scale_flows(hpu.true(self.is_male[inds]))
        #     self.sex_flows['infections_by_sex'][0] += infs_female
        #     self.sex_flows['infections_by_sex'][1] += infs_male

        # Now use genotype-specific prognosis probabilities to determine what happens.
        # Only women can progress beyond infection.
        f_inds = ssu.itruei(sim.people.is_female, inds)
        m_inds = ssu.itruei(sim.people.is_male, inds)

        # Compute disease progression for females
        if len(f_inds) > 0:

            cls.set_prognoses(sim, f_inds, g)

        # Compute infection clearance for males
        if len(m_inds) > 0:
            dur_infection = ssu.sample(**sim.pars[cls.name]['dur_infection_male'], size=len(m_inds))
            sim.people[cls.name].date_clearance[g, m_inds] = sim.people[cls.name].date_infectious[g, m_inds] + np.ceil(
                dur_infection / dt)  # Date they clear HPV infection (interpreted as the timestep on which they recover)

        return

    @classmethod
    def set_prognoses(cls, sim, inds, g):
        '''
        Assigns prognoses for all infected women on day of infection.
        '''

        gpars = sim.pars[cls.name]['genotype_pars'][g]
        dt = sim.pars['dt']

        # Set length of infection, which is moderated by any prior cell-level immunity
        sev_imm = sim.people[cls.name].sev_imm[g, inds]
        age_mod = np.ones(len(inds))
        age_mod[sim.people.age[inds]>= sim.pars[cls.name]['age_risk']['age']] = sim.pars[cls.name]['age_risk']['risk']
        sim.people[cls.name].dur_episomal[g, inds]  = ssu.sample(**gpars['dur_episomal'], size=len(inds))*(1-sev_imm)*age_mod
        sim.people[cls.name].dur_infection[g, inds]  = sim.people[cls.name].dur_episomal[g, inds]

        # Determine how long before precancerous cell changes
        dur_precin = ssu.sample(**gpars['dur_precin'], size=len(inds)) # Sample from distribution
        cin_bools = sim.people[cls.name].dur_episomal[g, inds] > dur_precin # Pull out those whose infection is long enough for precancer
        cin_inds = inds[cin_bools]
        nocin_inds = inds[~cin_bools]
        sim.people[cls.name].dur_precin[g, inds] = np.minimum(sim.people[cls.name].dur_episomal[g, inds], dur_precin)
        sim.people[cls.name].dur_cin[g, cin_inds] = sim.people[cls.name].dur_episomal[g, cin_inds] - sim.people[cls.name].dur_precin[g, cin_inds]

        # Set date of clearance for those who don't develop precancer
        sim.people[cls.name].date_clearance[g, nocin_inds] = sim.t + sc.randround(sim.people[cls.name].dur_precin[g, nocin_inds]/dt)

        # Set date of onset of precancer and eventual severity outcomes for those who develop precancer
        sim.people[cls.name].date_cin1[g, cin_inds] = sim.t + sc.randround(sim.people[cls.name].dur_precin[g, cin_inds]/dt)

        # Set infection severity and outcomes
        cls.set_severity(sim, inds[cin_bools], g)

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

    @classmethod
    def get_genotype_pars(cls, default=False, genotype=None):
        '''
        Define the default parameters for the different genotypes
        '''

        pars = sc.objdict()

        pars.hpv16 = sc.objdict()
        pars.hpv16.dur_precin = dict(dist='normal_pos', par1=0.5, par2=0.25)  # Duration of infection prior to precancer
        pars.hpv16.dur_episomal = dict(dist='lognormal', par1=2,
                                       par2=5)  # Duration of episomal infection prior to cancer
        pars.hpv16.sev_fn = dict(form='logf2', k=0.175, x_infl=0,
                                 ttc=30)  # Function mapping duration of infection to severity
        pars.hpv16.rel_beta = 1.0  # Baseline relative transmissibility, other genotypes are relative to this
        pars.hpv16.transform_prob = 1.3e-9  # Annual rate of transformed cell invading
        pars.hpv16.sev_integral = 'analytic'  # Type of integral used for translating severity to transformation probability. Accepts numeric, analytic, or None
        pars.hpv16.sero_prob = 0.75  # https://www.sciencedirect.com/science/article/pii/S2666679022000027#fig1

        pars.hpv18 = sc.objdict()
        pars.hpv18.dur_precin = dict(dist='normal_pos', par1=0.5, par2=0.25)  # Duration of infection prior to precancer
        pars.hpv18.dur_episomal = dict(dist='lognormal', par1=2, par2=5)  # Duration of infection prior to cancer
        pars.hpv18.sev_fn = dict(form='logf2', k=0.15, x_infl=0,
                                 ttc=30)  # Function mapping duration of infection to severity
        pars.hpv18.rel_beta = 0.75  # Relative transmissibility, current estimate from Harvard model calibration of m2f tx
        pars.hpv18.transform_prob = 1.0e-9  # Annual rate of transformed cell invading
        pars.hpv18.sev_integral = 'analytic'  # Type of integral used for translating severity to transformation probability. Accepts numeric, analytic, or None
        pars.hpv18.sero_prob = 0.56  # https://www.sciencedirect.com/science/article/pii/S2666679022000027#fig1

        # High-risk oncogenic types included in 9valent vaccine: 31, 33, 45, 52, 58
        pars.hi5 = sc.objdict()
        pars.hi5.dur_precin = dict(dist='normal_pos', par1=0.5, par2=0.25)  # Duration of infection prior to precancer
        pars.hi5.dur_episomal = dict(dist='lognormal', par1=2, par2=4)  # Duration of infection prior to cancer
        pars.hi5.sev_fn = dict(form='logf2', k=0.125, x_infl=0,
                               ttc=30)  # Function mapping duration of infection to severity
        pars.hi5.rel_beta = 0.9  # placeholder
        pars.hi5.transform_prob = 3e-10  # Annual rate of transformed cell invading
        pars.hi5.sev_integral = 'analytic'  # Type of integral used for translating severity to transformation probability. Accepts numeric, analytic, or None
        pars.hi5.sero_prob = 0.60  # placeholder

        # Other high-risk: oncogenic but not covered in 9valent vaccine: 35, 39, 51, 56, 59
        pars.ohr = sc.objdict()
        pars.ohr.dur_precin = dict(dist='normal_pos', par1=0.5, par2=0.25)  # Duration of infection prior to precancer
        pars.ohr.dur_episomal = dict(dist='lognormal', par1=2, par2=6)  # Duration of infection prior to cancer
        pars.ohr.sev_fn = dict(form='logf2', k=0.125, x_infl=0,
                               ttc=30)  # Function mapping duration of infection to severity
        pars.ohr.rel_beta = 0.9  # placeholder
        pars.ohr.transform_prob = 3e-10  # Annual rate of transformed cell invading
        pars.ohr.sev_integral = 'analytic'  # Type of integral used for translating severity to transformation probability. Accepts numeric, analytic, or None
        pars.ohr.sero_prob = 0.60  # placeholder

        # All other high-risk types: 31, 33, 35, 39, 45, 51, 52, 56, 58, 59
        # Warning: this should not be used in conjuction with hi5 or ohr
        pars.hr = sc.objdict()
        pars.hr.dur_precin = dict(dist='normal_pos', par1=0.5, par2=0.25)  # Duration of infection prior to precancer
        pars.hr.dur_episomal = dict(dist='lognormal', par1=2, par2=4)  # Duration of infection prior to cancer
        pars.hr.sev_fn = dict(form='logf2', k=0.125, x_infl=0,
                              ttc=30)  # Function mapping duration of infection to severity
        pars.hr.rel_beta = 0.9  # placeholder
        pars.hr.transform_prob = 3e-10  # Annual rate of transformed cell invading
        pars.hr.sev_integral = 'analytic'  # Type of integral used for translating severity to transformation probability. Accepts numeric, analytic, or None
        pars.hr.sero_prob = 0.60  # placeholder

        # Low-risk
        pars.lr = sc.objdict()
        pars.lr.dur_precin = dict(dist='normal_pos', par1=0.5, par2=0.25)  # Duration of infection prior to precancer
        pars.lr.dur_episomal = dict(dist='lognormal', par1=2, par2=4)  # Duration of infection prior to cancer
        pars.lr.sev_fn = dict(form='logf2', k=0.0, x_infl=0,
                              ttc=30)  # Function mapping duration of infection to severity
        pars.lr.rel_beta = 0.9  # placeholder
        pars.lr.transform_prob = 0  # Annual rate of transformed cell invading
        pars.lr.sev_integral = 'analytic'  # Type of integral used for translating severity to transformation probability. Accepts numeric, analytic, or None
        pars.lr.sero_prob = 0.60  # placeholder

        return cls._get_from_pars(pars, default, key=genotype, defaultkey='hpv16')

    @classmethod
    def init_genotypes(cls, sim, upper_dysp_lim=200):
        default_gpars   = cls.get_genotype_pars()
        user_gpars      = sc.dcp(cls.pars['genotype_pars'])
        cls.pars['genotype_pars'] = sc.objdict()


        # Handle special input cases
        if cls.pars['genotypes'] == 'all':
            cls.pars['genotypes'] = default_gpars.keys()
        if not len(cls.pars['genotypes']):
            print('No genotypes provided: simulating 16, 18, and 5 other pooled HR types (31, 33, 45, 52, 58).')
            cls.pars['genotypes'] = [16,18,'hi5']

        # Loop over genotypes
        for i, g in enumerate(cls.pars['genotypes']):

            # Standardize format of genotype inputs
            if sc.isnumber(g): g = f'hpv{g}' # Convert e.g. 16 to hpv16
            if sc.checktype(g,str):
                if not g in default_gpars.keys():
                    errormsg = f'Genotype {i} ({g}) is not one of the inbuilt options.'
                    raise ValueError(errormsg)
            else:
                errormsg = f'Format {type(g)} is not understood.'
                raise ValueError(errormsg)

            # Add to genotype_par dict
            cls.pars['genotype_pars'][g] = default_gpars[g]
            cls.pars['genotype_map'][i] = g

        # Loop over user-supplied genotype parameters that can overwrite values
        if len(user_gpars):
            for g,gpars in user_gpars.items():

                # Standardize format of genotype inputs
                if sc.isnumber(g): g = f'hpv{g}'  # Convert e.g. 16 to hpv16
                if sc.checktype(g, str):
                    if not g in cls.pars['genotype_pars'].keys():
                        errormsg = f'Parameters provided for genotype {g}, but it is not in the sim.'
                        raise ValueError(errormsg)
                    else:
                        for gparname,gparval in gpars.items():
                            if gparname in cls.pars['genotype_pars'][g].keys():
                                printmsg = f"Resetting parameter '{gparname}' from {cls.pars['genotype_pars'][g][gparname]} to {gparval} for genotype {g}"
                                cls.pars['genotype_pars'][g][gparname] = gparval
                            else:
                                errormsg = f"Parameter {gparname} does not exist for genotype {g}"
                                raise ValueError(errormsg)

        cls.pars['n_genotypes'] = len(cls.pars['genotype_pars'])  # Each genotype has an entry in genotype_pars

        # Set the number of immunity sources
        cls.pars['n_imm_sources'] = len(cls.pars['genotypes'])

        # Do any precomputations for the genotype transformation functions
        t_step = sim['dt']
        t_sequence = np.arange(0, upper_dysp_lim, t_step)
        cumdysp = dict()
        for g in range(cls.pars['n_genotypes']):
            sev_fn = cls.pars['genotype_pars'][g]['sev_fn']
            sev_integral = cls.pars['genotype_pars'][g]['sev_integral']
            if sev_integral=='numeric':
                glabel = cls.pars['genotype_map'][g]
                dysp_arr = cls.compute_severity(t_sequence, rel_sev=None, pars=sev_fn)
                cumdysp[glabel] = np.cumsum(dysp_arr) * t_step

        cls.pars['cumdysp'] = cumdysp  # Store

        return

    @classmethod
    def get_cross_immunity(cls, cross_imm_med=None, cross_imm_high=None, own_imm_hr=None, default=False, genotype=None):
        '''
        Get the cross immunity between each genotype in a sim
        '''
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

        return cls._get_from_pars(pars, default, key=genotype, defaultkey='hpv16')

    # %% Methods for computing severity
    @classmethod
    def compute_severity(cls, t, rel_sev=None, pars=None):
        '''
        Process functional form and parameters into values:
        '''

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
            output = ssu.logf2(t, **pars)

        elif form == 'logf3':
            output = ssu.logf3(t, **pars)

        elif callable(form):
            output = form(t, **pars)

        else:
            errormsg = f'The selected functional form "{form}" is not implemented; choices are: {sc.strjoin(choices)}'
            raise NotImplementedError(errormsg)

        return output

    @classmethod
    def compute_inv_severity(cls, sev_vals, rel_sev=None, pars=None):
        '''
        Compute time to given severity level given input parameters
        '''

        pars = sc.dcp(pars)
        form = pars.pop('form')
        choices = [
            'logf2',
            'logf3',
        ]

        # Process inputs
        if form is None or form == 'logf2':
            output = ssu.invlogf2(sev_vals, **pars)

        elif form == 'logf3':
            output = ssu.invlogf3(sev_vals, **pars)

        elif callable(form):
            output = form(sev_vals, **pars)

        else:
            errormsg = f'The selected functional form "{form}" is not implemented; choices are: {sc.strjoin(choices)}'
            raise NotImplementedError(errormsg)

        # Scale by relative severity
        if rel_sev is not None:
            output = output / rel_sev

        return output

    @classmethod
    def compute_severity_integral(cls, t, rel_sev=None, pars=None):
        '''
        Process functional form and parameters into values:
        '''

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
            output = ssu.intlogf2(t, **pars)

        elif form == 'logf3':
            s = pars.pop('s')
            if s == 1:
                output = ssu.intlogf2(t, **pars)
            else:
                errormsg = f'Analytic integral for logf3 only implemented for s=1. Select integral=numeric.'

        else:
            errormsg = f'Analytic integral for the selected functional form "{form}" is not implemented; choices are: {sc.strjoin(choices)}, or select integral=numeric.'
            raise NotImplementedError(errormsg)

        return output

    @classmethod
    def check_immunity(cls, people):
        '''
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

        '''
        cross_immunity_sus = people.pars[cls.name]['cross_immunity_sus']  # cross-immunity/own-immunity scalars to be applied to sus immunity level
        cross_immunity_sev = people.pars[cls.name]['cross_immunity_sev']  # cross-immunity/own-immunity scalars to be applied to sev immunity level
        sus_imm = np.dot(cross_immunity_sus, people[cls.name].nab_imm)  # Dot product gives immunity to all genotypes
        sev_imm = np.dot(cross_immunity_sev, people[cls.name].cell_imm)  # Dot product gives immunity to all genotypes
        people[cls.name].sus_imm[:] = np.minimum(sus_imm, np.ones_like(sus_imm))  # Don't let this be above 1
        people[cls.name].sev_imm[:] = np.minimum(sev_imm, np.ones_like(sev_imm))  # Don't let this be above 1
        return

    @classmethod
    def check_clearance(cls, sim, genotype):
        '''
        Check for HPV clearance.
        '''
        f_filter_inds = (sim.people[cls.name].is_female_alive & sim.people[cls.name].infectious[genotype, :]).nonzero()[-1]
        m_filter_inds = (sim.people[cls.name].is_male_alive & sim.people[cls.name].infectious[genotype, :]).nonzero()[-1]
        f_inds = sim.people[cls.name].check_inds_true(sim.people[cls.name].infectious[genotype, :], sim.people[cls.name].date_clearance[genotype, :],
                                      filter_inds=f_filter_inds)
        m_inds = sim.people[cls.name].check_inds_true(sim.people[cls.name].infectious[genotype, :], sim.people[cls.name].date_clearance[genotype, :],
                                      filter_inds=m_filter_inds)
        m_cleared_inds = m_inds  # All males clear

        # For females, determine who clears and who controls
        if sim.pars[cls.name]['hpv_control_prob'] > 0:
            latent_probs = np.full(len(f_inds), sim.pars[cls.name]['hpv_control_prob'], dtype=ssd.default_float)
            latent_bools = ssu.binomial_arr(latent_probs)
            latent_inds = f_inds[latent_bools]

            if len(latent_inds):
                sim.people[cls.name].susceptible[genotype, latent_inds] = False  # should already be false
                sim.people[cls.name].infectious[genotype, latent_inds] = False
                sim.people[cls.name].inactive[genotype, latent_inds] = True
                sim.people[cls.name].date_clearance[genotype, latent_inds] = np.nan

            f_cleared_inds = f_inds[~latent_bools]

        else:
            f_cleared_inds = f_inds

        cleared_inds = np.array(m_cleared_inds.tolist() + f_cleared_inds.tolist())

        # Now reset disease states
        if len(cleared_inds):
            sim.people[cls.name].susceptible[genotype, cleared_inds] = True
            sim.people[cls.name].infectious[genotype, cleared_inds] = False
            sim.people[cls.name].inactive[genotype, cleared_inds] = False  # should already be false

        if len(f_cleared_inds):
            # female_cleared_inds = np.intersect1d(cleared_inds, self.f_inds) # Only give natural immunity to females
            cls.update_peak_immunity(f_cleared_inds, imm_pars=sim.pars, imm_source=genotype)  # update immunity
            sim.people[cls.name].date_reactivated[genotype, f_cleared_inds] = np.nan

        # Whether infection is controlled on not, clear all cell changes and severity markeres
        sim.people[cls.name].episomal[genotype, f_inds] = False
        sim.people[cls.name].transformed[genotype, f_inds] = False
        sim.people[cls.name].sev[genotype, f_inds] = np.nan
        sim.people[cls.name].date_cin1[genotype, f_inds] = np.nan
        sim.people[cls.name].date_cin2[genotype, f_inds] = np.nan
        sim.people[cls.name].date_cin3[genotype, f_inds] = np.nan

    @classmethod
    def update_severity(cls, sim, genotype):
        '''
        Update disease severity for women with infection and update their current severity
        '''
        gpars = sim.pars[cls.name]['genotype_pars'][genotype]

        # Only need to update severity for people who with dysplasia underway
        fg_cin_inds = ssu.true(sim.people[cls.name].is_female & ~np.isnan(sim.people[cls.name].sev[genotype, :]) & sim.people[cls.name].infectious[genotype,
                                                                                   :])  # Indices of women infected with this genotype who will develop CIN1
        fg_cin_underway_inds = fg_cin_inds[
            (sim.t >= sim.people[cls.name].date_cin1[genotype, fg_cin_inds])]  # Indices of women for whom dysplasia is underway

        time_with_dysplasia = (sim.t - sim.people[cls.name].date_cin1[genotype, fg_cin_underway_inds]) * sim.dt
        rel_sevs = sim.people[cls.name].rel_sev[fg_cin_underway_inds]
        if (time_with_dysplasia < 0).any() or (np.isnan(time_with_dysplasia)).any():
            errormsg = 'Time with dysplasia cannot be less than zero or NaN.'
            raise ValueError(errormsg)
        if (np.isnan(sim.people[cls.name].date_exposed[genotype, fg_cin_inds])).any():
            errormsg = f'No date of exposure defined for {ssu.iundefined(sim.people[cls.name].date_exposed[genotype, fg_cin_inds], fg_cin_inds)} on timestep {sim.t}'
            raise ValueError(errormsg)
        if (np.isnan(sim.people[cls.name].date_cin1[genotype, fg_cin_inds])).any():
            errormsg = f'No date of dysplasia onset defined for {ssu.iundefined(sim.people[cls.name].date_cin1[genotype, fg_cin_inds], fg_cin_inds)} on timestep {sim.t}'
            raise ValueError(errormsg)

        sim.people[cls.name].sev[genotype, fg_cin_underway_inds] = cls.compute_severity(time_with_dysplasia, rel_sev=rel_sevs,
                                                                          pars=gpars['sev_fn'])

        if (np.isnan(sim.people[cls.name].sev[genotype, fg_cin_underway_inds])).any():
            errormsg = 'Invalid severity values.'
            raise ValueError(errormsg)

    @classmethod
    def check_transformation(cls, sim, genotype):
        ''' Check for new transformations '''
        # Only include infectious, episomal females who haven't already cleared infection
        filter_inds = sim.people[cls.name].true_by_genotype('episomal', genotype)
        inds = sim.people[cls.name].check_inds(sim.people[cls.name].transformed[genotype, :], sim.people[cls.name].date_transformed[genotype, :],
                               filter_inds=filter_inds)
        sim.people[cls.name].transformed[genotype, inds] = True  # Now transformed, cannot clear
        sim.people[cls.name].date_clearance[genotype, inds] = np.nan  # Remove their clearance dates

    @classmethod
    def check_progress(cls, sim, what, genotype):
        ''' Wrapper function for all the new progression checks '''
        if what == 'cin1s':
            cls.check_cin1(sim, genotype)
        elif what == 'cin2s':
            cls.check_cin2(sim, genotype)
        elif what == 'cin3s':
            cls.check_cin3(sim, genotype)
        elif what == 'cancers':
            cls.check_cancer(sim, genotype)
        return

    @classmethod
    def check_cancer_deaths(cls, sim):
        pass

    @classmethod
    def set_severity(cls, sim, param, g):
        pass

    @classmethod
    def update_peak_immunity(cls, sim, f_cleared_inds, imm_pars, imm_source):
        pass

    @classmethod
    def check_cin1(cls, sim, genotype):
        ''' Check for new progressions to CIN1 '''
        # Only include infectious females who haven't already cleared CIN1 or progressed to CIN2
        filters = sim.people[cls.name].infectious[genotype, :] * sim.people[cls.name].is_female * ~(sim.people[cls.name].date_clearance[genotype, :] <= sim.t) * (
                    sim.people[cls.name].date_cin2[genotype, :] >= sim.t)
        filter_inds = filters.nonzero()[0]
        inds = sim.people[cls.name].check_inds(sim.people[cls.name].cin1[genotype, :], sim.people[cls.name].date_cin1[genotype, :], filter_inds=filter_inds)
        sim.people[cls.name].cin1[genotype, inds] = True
        return


    @classmethod
    def check_cin2(cls, sim, genotype):
        ''' Check for new progressions to CIN2 '''
        filter_inds = sim.people[cls.name].true_by_genotype('cin1', genotype)
        inds = sim.people[cls.name].check_inds(sim.people[cls.name].cin2[genotype,:], sim.people[cls.name].date_cin2[genotype,:], filter_inds=filter_inds)
        sim.people[cls.name].cin2[genotype, inds] = True
        sim.people[cls.name].cin1[genotype, inds] = False # No longer counted as CIN1
        return


    @classmethod
    def check_cin3(cls, sim, genotype):
        ''' Check for new progressions to CIN3 '''
        filter_inds = sim.people[cls.name].true_by_genotype('cin2', genotype)
        inds = sim.people[cls.name].check_inds(sim.people[cls.name].cin3[genotype, :],
                                               sim.people[cls.name].date_cin3[genotype, :], filter_inds=filter_inds)
        sim.people[cls.name].cin3[genotype, inds] = True
        sim.people[cls.name].cin2[genotype, inds] = False  # No longer counted as CIN2
        return

    @classmethod
    def check_cancer(cls, sim, genotype):
        ''' Check for new progressions to cancer '''
        filter_inds = sim.people[cls.name].true('transformed')
        inds = sim.people[cls.name].check_inds(sim.people[cls.name].cancerous[genotype, :], sim.people[cls.name].date_cancerous[genotype, :], filter_inds=filter_inds)

        # Set infectious states
        sim.people[cls.name].susceptible[:, inds] = False  # No longer susceptible to any genotype
        sim.people[cls.name].infectious[:, inds] = False  # No longer counted as infectious with any genotype
        sim.people[cls.name].inactive[:,inds] = True  # If this person has any other infections from any other genotypes, set them to inactive

        sim.people[cls.name].date_clearance[:, inds] = np.nan  # Remove their clearance dates for all genotypes

        # Deal with dysplasia states and dates
        for g in range(sim.pars[cls.name].ng):
            if g != genotype:
                sim.people[cls.name].date_cancerous[g, inds] = np.nan  # Remove their date of cancer for all genotypes but the one currently causing cancer
                sim.people[cls.name].date_cin1[g, inds] = np.nan
                sim.people[cls.name].date_cin2[g, inds] = np.nan
                sim.people[cls.name].date_cin3[g, inds] = np.nan

        # Set the properties related to cell changes and disease severity markers
        sim.people[cls.name].cancerous[genotype, inds] = True
        sim.people[cls.name].episomal[:, inds] = False  # No longer counted as episomal with any genotype
        sim.people[cls.name].transformed[:, inds] = False  # No longer counted as transformed with any genotype
        sim.people[cls.name].sev[:,inds] = np.nan  # NOTE: setting this to nan means this people no longer counts as CIN1/2/3, since those categories are based on this value
        return