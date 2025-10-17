"""
Age-stratified SEIR population class for metapopulation modeling

Each "agent" represents an entire community with internal age structure.
Supports within-community age-structured transmission and between-community
dose-based transmission via MetapopCompatible interface.
"""

import numpy as np
from pandas.core.internals.blocks import new_block
import starsim as ss
from metapop_interface import MetapopCompatible


class SEIRPopulation(ss.Infection, MetapopCompatible):
    """
    Age-stratified SEIR model where each agent represents a community
    
    Features:
    - 3 age groups: 0-4, 5-19, 20+
    - Erlang-staged E and I compartments (kE=3, kI=3 by default)
    - Within-community age-structured transmission
    - Between-community transmission via MetapopCompatible interface
    - Measles-like parameterization
    """
    
    def __init__(self, pars=None, communities=None, contact_matrix=None, **kwargs):
        super().__init__()  # Inherits rel_sus, rel_trans, infectious
        
        # Store communities for disease-specific initialization
        self.communities = communities
        
        # Validate and store required contact matrix
        if contact_matrix is None:
            raise ValueError("contact_matrix is required. Expected 3x3 array for age groups ['0_4', '5_19', '20p']")
        
        contact_matrix = np.array(contact_matrix)
        if contact_matrix.shape != (3, 3):
            raise ValueError(f"contact_matrix must be 3x3, got shape {contact_matrix.shape}")
        
        self.contact_matrix = contact_matrix # Number of effective contacts per person per day
        
        self.define_pars(
            R0=15.0,                    # Basic reproduction number (user-configurable)
            kE=3,                       # Number of E stages (Erlang)
            kI=3,                       # Number of I stages (Erlang)
            dur_latent=ss.days(9),      # Mean latent period
            dur_infectious=ss.days(8),  # Mean infectious period
        )
        self.update_pars(pars, **kwargs)
        
        # Calculate beta from R0, contact matrix, and infectious duration
        self.pars.beta = self._calculate_beta_from_R0()
        
        # Age groups
        self.age_groups = ['0_4', '5_19', '20p']
        
        # Setup all states and distributions
        self._setup_states()
    
    def _calculate_beta_from_R0(self):
        """
        Calculate transmission rate beta from R0, contact matrix, and infectious duration
        
        For age-structured models: R0 = dominant eigenvalue of (beta × C × D)
        where C is contact matrix and D is infectious duration
        """
        # Mean infectious duration (days)
        dur_infectious_days = float(self.pars.dur_infectious)
        
        # For simplicity, use the dominant eigenvalue approach
        # The next generation matrix is K = beta × C × D
        # We want dominant eigenvalue of K to equal R0
        
        # Calculate the dominant eigenvalue of the contact matrix
        eigenvalues = np.linalg.eigvals(self.contact_matrix)
        dominant_contact_eigenvalue = np.max(np.real(eigenvalues))
        
        # Solve: R0 = beta × dominant_contact_eigenvalue × dur_infectious_days
        beta_value = self.pars.R0 / (dominant_contact_eigenvalue * dur_infectious_days)
        
        return ss.perday(beta_value)
        
    def _setup_states(self):
        """Define all age-stratified SEIR states"""
        states_to_define = []
        
        # S and R compartments by age
        for age in self.age_groups:
            states_to_define.extend([
                ss.FloatArr(f'S_{age}', default=0.0, label=f'Susceptible {age}'),
                ss.FloatArr(f'R_{age}', default=0.0, label=f'Recovered {age}')
            ])
        
        # E stages by age (kE stages each)
        for age in self.age_groups:
            for stage in range(1, self.pars.kE + 1):
                states_to_define.append(
                    ss.FloatArr(f'E{stage}_{age}', default=0.0, label=f'Exposed stage {stage} age {age}')
                )
        
        # I stages by age (kI stages each)  
        for age in self.age_groups:
            for stage in range(1, self.pars.kI + 1):
                states_to_define.append(
                    ss.FloatArr(f'I{stage}_{age}', default=0.0, label=f'Infectious stage {stage} age {age}')
                )
        
        # Age population tracking (for initial conditions and demographics)
        states_to_define.extend([
            ss.FloatArr('N_0_4', default=0.0, label='Population 0-4'),
            ss.FloatArr('N_5_19', default=0.0, label='Population 5-19'), 
            ss.FloatArr('N_20p', default=0.0, label='Population 20+'),
        ])
        
        # Shedding field (export intensity)
        states_to_define.append(ss.FloatArr('shed', default=0.0, label='Export shedding'))
        
        self.define_states(*states_to_define)
    
    def init_pre(self, sim):
        super().init_pre(sim)
        
        # Set up distributions for Erlang progressions
        self._setup_progression_distributions()
        
    def init_post(self):
        """Initialize after states are allocated"""
        super().init_post()
        
        # Initialize from communities data if available
        if self.communities is not None:
            self._initialize_from_communities(self.communities)
    
    def _initialize_from_communities(self, communities):
        """Initialize SEIR states from communities DataFrame"""
        if len(communities) != len(self.S_0_4):
            raise ValueError(f"Communities length {len(communities)} != number of agents {len(self.S_0_4)}")
        
        # Initialize age-stratified compartments from communities
        for age in self.age_groups:
            # Population totals by age
            N_col = f'N_{age}'
            if N_col in communities.columns:
                getattr(self, N_col)[:] = communities[N_col].values
            
            # Initial susceptible by age (handle initial prevalence later)
            S_col = f'S_{age}'
            if S_col in communities.columns:
                getattr(self, S_col)[:] = communities[S_col].values
        
        # Handle initial prevalence if specified
        if 'init_prev' in communities.columns:
            self._apply_initial_prevalence(communities['init_prev'].values)
        
        # Set rel_sus and rel_trans from communities if available
        if 'rel_sus' in communities.columns:
            self.rel_sus[:] = communities['rel_sus'].values
        if 'rel_trans' in communities.columns:
            self.rel_trans[:] = communities['rel_trans'].values

    def _apply_initial_prevalence(self, init_prev):
        """Apply initial prevalence by moving susceptibles to infectious"""
        for i, prev in enumerate(init_prev):
            if prev > 0:
                for age in self.age_groups:
                    S_state = getattr(self, f'S_{age}')
                    current_S = int(S_state[i])
                    
                    if current_S > 0:
                        # Move fraction to first infectious stage
                        initial_I = int(current_S * prev)
                        S_state[i] -= initial_I
                        getattr(self, f'I1_{age}')[i] += initial_I

    def _setup_progression_distributions(self):
        """Setup pre-defined distributions for Erlang progressions"""
        # Pre-calculate fixed progression probabilities
        dt = float(self.sim.t.dt)
        rate_E = self.pars.kE / self.pars.dur_latent  # stages per day
        rate_I = self.pars.kI / self.pars.dur_infectious  # stages per day
        
        self.p_E = np.clip(1.0 - np.exp(-rate_E * dt), 0.0, 1.0)
        self.p_I = np.clip(1.0 - np.exp(-rate_I * dt), 0.0, 1.0)
        
        # Create single distributions for E and I progressions
        self.E_dist = ss.bernoulli(p=self.p_E)
        self.I_dist = ss.bernoulli(p=self.p_I)
    
    
    def _get_erlang_prob_E(self, uids, state_name):
        """Get E stage progression probability"""
        dt = float(self.sim.t.dt)
        rate_E = self.pars.kE / self.pars.dur_latent  # stages per day
        return np.clip(1.0 - np.exp(-rate_E * dt), 0.0, 1.0)
    
    def _get_erlang_prob_I(self, uids, state_name):
        """Get I stage progression probability"""
        dt = float(self.sim.t.dt)
        rate_I = self.pars.kI / self.pars.dur_infectious  # stages per day
        return np.clip(1.0 - np.exp(-rate_I * dt), 0.0, 1.0)
    
    def step_state(self):
        """Handle Erlang progressions and compute export shedding"""
        
        # Erlang stage progressions
        self.progress_erlang_stages()
        
        # Compute export field at end of step_state
        self._compute_export_shedding()
    
    def progress_erlang_stages(self):
        """Progress through E and I stages using pre-defined distributions"""
        
        # E stage progressions (reverse order to avoid conflicts)
        for age in self.age_groups:
            for stage in range(self.pars.kE, 0, -1):
                state_name = f'E{stage}_{age}'
                current_counts = getattr(self, state_name).copy().astype(int)
                
                if np.any(current_counts > 0):
                    transitions = np.random.binomial(current_counts, self.p_E)
                    
                    if stage == self.pars.kE:
                        # Last E stage → first I stage
                        getattr(self, state_name)[:] -= transitions
                        getattr(self, f'I1_{age}')[:] += transitions
                    else:
                        # E stage → next E stage
                        getattr(self, state_name)[:] -= transitions
                        getattr(self, f'E{stage+1}_{age}')[:] += transitions
        
        # I stage progressions (reverse order)
        for age in self.age_groups:
            for stage in range(self.pars.kI, 0, -1):
                state_name = f'I{stage}_{age}'
                current_counts = getattr(self, state_name).copy().astype(int)
                
                if np.any(current_counts > 0):
                    transitions = np.random.binomial(current_counts, self.p_I)
                    
                    if stage == self.pars.kI:
                        # Last I stage → recovered
                        getattr(self, state_name)[:] -= transitions
                        getattr(self, f'R_{age}')[:] += transitions
                    else:
                        # I stage → next I stage
                        getattr(self, state_name)[:] -= transitions
                        getattr(self, f'I{stage+1}_{age}')[:] += transitions
    
    def _compute_export_shedding(self):
        """Compute export shedding intensity for between-community transmission"""
        # Total infectious across all age groups
        Itot = np.zeros(len(self.S_0_4))
        for age in self.age_groups:
            for stage in range(1, self.pars.kI + 1):
                I_arr = getattr(self, f'I{stage}_{age}')
                Itot = Itot + np.array(I_arr)
        
        # Total population across all age groups  
        Ntot = np.zeros(len(self.S_0_4))
        for age in self.age_groups:
            N_age = self.get_age_total(age)
            Ntot = Ntot + np.array(N_age)
        
        # Export shedding = infectiousness scaled by rel_trans
        rel_trans_arr = np.array(self.rel_trans)
        self.shed[:] = rel_trans_arr * (Itot / np.maximum(Ntot, 1e-12))
        
        # Keep .infectious boolean for compatibility (node has any infection)
        self.infectious[:] = (Itot > 0)
    
    def get_age_total(self, age):
        """Get total population in age group (all compartments)"""
        total = np.array(getattr(self, f'S_{age}')) + np.array(getattr(self, f'R_{age}'))
        
        # Add all E stages
        for stage in range(1, self.pars.kE + 1):
            E_arr = getattr(self, f'E{stage}_{age}')
            total = total + np.array(E_arr)
        
        # Add all I stages
        for stage in range(1, self.pars.kI + 1):
            I_arr = getattr(self, f'I{stage}_{age}')
            total = total + np.array(I_arr)
        
        return total
    
    def get_total_E_by_age(self, age):
        """Get total exposed (all E stages) for given age group"""
        total_E = np.zeros(len(getattr(self, f'S_{age}')), dtype=float)
        
        for stage in range(1, self.pars.kE + 1):
            E_arr = getattr(self, f'E{stage}_{age}')
            total_E += np.array(E_arr)
        
        return total_E
    
    def get_total_I_by_age(self, age):
        """Get total infectious (all I stages) for given age group"""
        total_I = np.zeros(len(getattr(self, f'S_{age}')), dtype=float)
        
        for stage in range(1, self.pars.kI + 1):
            I_arr = getattr(self, f'I{stage}_{age}')
            total_I += np.array(I_arr)
        
        return total_I
    
    def get_seir_summary(self):
        """Get organized SEIR data for all communities and age groups"""
        summary = {}
        
        for age in self.age_groups:
            summary[age] = {
                'S': np.array(getattr(self, f'S_{age}')),
                'E': self.get_total_E_by_age(age),
                'I': self.get_total_I_by_age(age), 
                'R': np.array(getattr(self, f'R_{age}'))
            }
        
        return summary
    
    def step(self):
        """Handle transmission in two phases: within-community then between-community"""
        
        # 1. Within-community age-structured transmission FIRST
        self.apply_within_transmission()
        
        # 2. Between-community transmission handled by routes via callbacks
        # Routes are called automatically by Starsim disease.infect()
        return super().step()
    
    def apply_within_transmission(self):
        """Within-community age-structured transmission with Starsim RNG"""
        lambda_times_dt_within = self.compute_within_rates()  # Shape: (nC, 3)
        
        # Now convert to probabilities
        p_within = 1.0 - np.exp(-self.rel_sus[:, None] * lambda_times_dt_within)

        S = np.array([getattr(self, f'S_{age}') for age in self.age_groups]).T.astype(int)
        new_infections = np.random.binomial(S, p_within)

        # Apply the infections to the states
        for i, age in enumerate(self.age_groups):
            getattr(self, f'S_{age}')[:] -= new_infections[:, i]
            getattr(self, f'E1_{age}')[:] += new_infections[:, i]

        return new_infections.sum()
    
    def compute_within_rates(self):
        """
        Vectorized age-structured transmission rates within communities
        
        Returns:
            np.ndarray: Shape (nC, 3) transmission rates per age group per community
        """
        nC = len(self.S_0_4)
        
        # Total infectious by age group (nC, 3)
        I_totals = np.zeros((nC, 3))
        for i, age in enumerate(self.age_groups):
            for stage in range(1, self.pars.kI + 1):
                I_totals[:, i] += getattr(self, f'I{stage}_{age}')
        
        # Population totals by age (nC, 3)
        N_totals = np.array([
            self.get_age_total('0_4'),
            self.get_age_total('5_19'), 
            self.get_age_total('20p')
        ]).T
        
        # Infectious density (nC, 3)
        rho = I_totals / np.maximum(N_totals, 1.0)
        
        # Vectorized transmission rates: (nC, 3)
        beta_dt = self.pars.beta.to_events(self.sim.t.dt)  # Convert rate to events per timestep
        beta_dt_matrix = beta_dt * self.contact_matrix.T  # (3, 3) - transposed so [i,j] = contacts from j to i
        lambda_dt = rho @ beta_dt_matrix  # FOI (lambda) times the time step, dt. Shape is (nC, 3).

        return lambda_dt


    # MetapopCompatible interface implementation
    def _apply_binomial_by_age(self, p_between: np.ndarray, route_name: str) -> None:
        """Apply probability to all age groups using Starsim RNG"""

        S = np.array([getattr(self, f'S_{age}') for age in self.age_groups]).T.astype(int)
        new_infections = np.random.binomial(S, p_between[:, None])

        for i, age in enumerate(self.age_groups):
            getattr(self, f'S_{age}')[:] -= new_infections[:, i]
            getattr(self, f'E1_{age}')[:] += new_infections[:, i]

        return new_infections.sum()