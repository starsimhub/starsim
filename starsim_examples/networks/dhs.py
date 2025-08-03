"""
Networks based on DHS household data. One is static and the other evolves over
time, with females forming new households as they become pregnant.
"""
import numpy as np
import starsim as ss

# This has a significant impact on runtime, surprisingly
ss_float = ss.dtypes.float
_ = None

# Specify all externally visible functions this file defines; see also more definitions below
__all__ = ['HouseholdDHSNet', 'EvolvingHouseholdDHSNet']


class HouseholdDHSNet(ss.Network):
    """ Static DHS network

    A network class of static households derived from DHS data. There's no formation of new households. The network
    requires a DHS dataset. When this network is initialized it overrides the age and sex of all agents in the sim and
    assigns each agent a household ID. Use with caution if other modules depend upon or alter age and sex too. Because
    this network is static, it can have unexpected side effects such as growing or shrinking household sizes when used
    with aging enabled or with other demographics modules like Pregnancy.

    This network creates households by selecting a random household from the DHS data and setting the age and sex of
    agents to match, repeating until all agents have been assigned to a household. DHS data record ages in integer years
    but we typically don't want all agents to have exactly the same age, so we add a random fractional age to each
    agent.

    This network assumes only one mother per household. Births are automatically added to their mother's household network.

    Args:
        dhs_data (dataframe): A pandas dataframe containing DHS data with columns 'hh_id' and 'ages'

    Note: the dataframe should look something like this:

            hh_id                ages
       0        0          72, 17, 30
       1        1                  37
       2        2          13, 55, 36
       3        3  52, 13, 12, 64, 53
       4        4              30, 66

    **Example**:

        import numpy as np
        import sciris as sc
        import starsim as ss
        import starsim_examples as sse

        # Construct DHS data
        n = 1000
        hhid_id = np.arange(n)
        age_strings = []
        for i in range(n):
            household_size = np.random.randint(1,6)
            ages = np.random.randint(0, 80, household_size)
            age_strings.append(sc.strjoin(ages))
        dhs_data = sc.dataframe(hh_id=np.arange(n), ages=age_strings) # If real data are available: dhs_data = pd.read_csv(dhs_data_path)

        household_dhs = sse.HouseholdDHSNet(dhs_data=dhs_data)

        sim = ss.Sim(diseases='sis', networks=household_dhs)
        sim.run()
        sim.plot()
    """
    def __init__(self, pars=None, dhs_data=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars()
        self.update_pars(pars, **kwargs)
        self.dhs_data = dhs_data
        self.define_states(
            ss.FloatArr('household_ids'),
        )
        self.p_fractional_age = ss.uniform()
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.dhs_data is None:
            raise ValueError("Please provide DHS data to the HouseholdDHSNetwork.")
        return

    def init_post(self, add_pairs=True):
        super().init_post(add_pairs)

        # DHS age data is in integer years. This doesn't make sense for a simulation, so we add a random fractional age
        self.sim.people.age[:] = self.sim.people.age + self.p_fractional_age.rvs(self.sim.people.auids)
        return

    def add_pairs(self):
        """
        Generate contacts
        """
        ppl = self.sim.people
        pop_size = len(ppl)
        dhs = self.dhs_data

        # Initialize
        n_remaining = len(ppl)

        # Loop over the clusters
        cluster_id = -1
        p1 = []
        p2 = []
        while n_remaining > 0:
            cluster_id += 1 # Assign cluster id

            # Sample a household from the DHS data
            rand_row = np.random.choice(len(dhs))
            household_data = dhs.iloc[rand_row]
            age_data = household_data['ages']
            sex_data = None
            if 'sexes' in household_data.keys():
                sex_data = household_data['sexes']

            age_data = np.array([float(x) for x in age_data.split(', ')], dtype=float)  # Parse the ages
            cluster_size = len(age_data)  # Sample the cluster size

            if cluster_size > n_remaining:
                cluster_size = n_remaining

            # UIDS of people in this cluster
            cluster_uids = ss.uids((pop_size-n_remaining)+np.arange(cluster_size))

            # Set the ages and household_ids
            ppl.age[cluster_uids] = age_data[0:cluster_size]
            if sex_data is not None:
                ppl.sex[cluster_uids] = sex_data[0:cluster_size]
            self.household_ids[cluster_uids] = cluster_id

            # Add symmetric pairwise contacts in each cluster
            for i in cluster_uids:
                for j in cluster_uids:
                    if j > i:
                        p1.append(i)
                        p2.append(j)
            n_remaining -= cluster_size
        beta = np.ones(len(p1), dtype=ss_float)
        self.append(p1=p1, p2=p2, beta=beta)
        return

    # No updates on step, so pass
    def step(self):
        return


class EvolvingHouseholdDHSNet(HouseholdDHSNet):
    """
    Extends the HouseholdDHSNetwork by:

        1. Assigning one random female as head of each household
        2. Allowing females who are not current heads of household to move out and start their own household with a random male partner
        3. Adding new births into the household of the mother

    Args:
        dhs_data (dataframe): A pandas dataframe containing DHS data with columns 'hh_id' and 'ages'
        prob_move_out (float): The probability a female non-head of household moves out to start her own household with a randomly selected partner. Evaluated once at the beginning of each pregnancy.
        update_freq (int): The frequency of updates in timesteps, e.g. 7 for weekly if using daily timesteps, 1 if using weekly timesteps.
    """
    def __init__(self, pars=None, dhs_data=None, prob_move_out=_, update_freq=_, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            prob_move_out = ss.bernoulli(p=0.7), # Probability that female moves out of her household when pregnant and not a head of household
            update_freq = 1,
        )
        self.update_pars(pars, **kwargs)
        self.dhs_data = dhs_data

        self.define_states(
            ss.BoolArr('fhoh', default='False'),  # An array for tracking the female head of household
            ss.FloatArr('ti_move_out_check', default='-inf'), # A time index for checking when to move out. Used to prevent multiple moveout checks per pregnancy.
        )
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        ss.check_requires(self.sim, ['pregnancy'])
        return

    def add_pairs(self):
        # Generate the pairs, then select the fhoh for each household
        super().add_pairs()

        ppl = self.sim.people

        # Find a female head of household if one exists between the ages of 15 and 50
        # We just created new households, so the last cluster id is the total number of clusters
        for cluster_id in range(int(self.household_ids[-1])):
            cluster_uids = ss.uids(self.household_ids == cluster_id)

            female_uids = cluster_uids[
                ppl.female[cluster_uids] & (ppl.age[cluster_uids] >= 15) & (ppl.age[cluster_uids] <= 50)]
            if len(female_uids) > 0:
                fhoh = np.random.choice(a=female_uids)
                self.fhoh[ss.uids(fhoh)] = True


    def step(self):
        super().step()

        if np.mod(self.ti, self.pars.update_freq):
            return

        self.add_births()
        self.create_new_households()
        return

    def add_births(self):
        sim = self.sim
        birth_uids = ss.uids((sim.people.age >= 0) & (sim.people.age < self.pars.update_freq * self.sim.t.dt_year))
        if len(birth_uids) == 0:
            return 0

        mat_uids = sim.people.parent[birth_uids]
        # Could be very young people in the initial population or mother could have died
        keep = mat_uids != sim.people.parent.nan
        birth_uids = birth_uids[keep]
        mat_uids = mat_uids[keep]
        if len(birth_uids) == 0:
            return 0

        p1 = []
        p2 = []
        for new_uid, mat_uid in zip(birth_uids, mat_uids):
            hh_contacts = ss.uids(self.household_ids == self.household_ids[mat_uid])
            p1.append(hh_contacts)
            p2.append([new_uid] * len(hh_contacts))

        p1 = ss.uids.cat(p1)
        p2 = ss.uids.cat(p2)

        beta = np.ones(len(p1), dtype=ss.dtypes.float)
        self.append(p1=p1, p2=p2, beta=beta)

        self.household_ids[birth_uids] = self.household_ids[mat_uids]  # Assign HHID (after adding network edges)

        return len(birth_uids)

    def create_new_households(self):
        """
        Create new households. Find females that are pregnant and not a head of household.
        Move them and a randomly sampled male partner to a new household.
        Then fully connect those contacts.
        """
        ppl = self.sim.people
        potential_movers = ss.uids(~self.fhoh & ppl.pregnancy.pregnant & (self.ti_move_out_check <= self.sim.ti))
        moving_out = self.pars['prob_move_out'].filter(potential_movers)
        if len(moving_out) > 0:
            self.fhoh[moving_out] = True
            potential_partners = ss.uids(ppl.male & (ppl.age > 15) & (ppl.age < 50))
            partner_inds = np.random.permutation(len(potential_partners))[:len(moving_out)]
            partners = potential_partners[partner_inds]  # Faster than np.random.shuffle
            to_remove = ss.uids.cat([moving_out, partners])
            self.remove_uids(to_remove)  # Remove their old contacts from the HH network
            beta = np.ones(len(moving_out), dtype=ss.dtypes.float)
            self.append(p1=moving_out, p2=partners, beta=beta)  # Create a new household

            # Create new household IDs
            new_cids = np.nanmax(self.household_ids) + 1 + np.arange(len(moving_out))
            self.household_ids[moving_out] = new_cids
            self.household_ids[partners] = new_cids

        # Only consider move out once per pregnancy
        self.ti_move_out_check[potential_movers] = ppl.pregnancy.ti_postpartum[
            potential_movers]  # Update the time index for checking when to move out
        return