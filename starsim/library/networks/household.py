"""
Household contact networks built from DHS-style survey data
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

__all__ = ['HouseholdNet']

_ = None # Sentinel for "not provided" default arguments


class HouseholdNet(ss.Network):
    """
    A household contact network built from DHS-style survey data.

    When initialized, this network overrides the age (and optionally sex) of
    all agents in the sim and assigns each agent a household ID. Use with
    caution if other modules depend upon or alter age and sex.

    Households are created by selecting a random household from the provided data
    and setting the age and sex of agents to match, repeating until all agents
    have been assigned to a household. Ages in the data are typically in integer
    years; a random fractional year is added so agents don't share exact ages.

    This network assumes only one mother per household. Births are automatically
    added to their mother's household network.

    Args:
        dhs_data (DataFrame): A pandas or Sciris dataframe with columns ``hh_id``
            and ``ages``. Optionally also ``sexes``. The ``ages`` column should
            contain comma-separated age strings (e.g. ``"72, 17, 30"``). If
            ``sexes`` is included, it should contain comma-separated values
            using DHS convention (1 = male, 2 = female) with the same number
            of entries as ``ages``.
        dynamic (bool): If ``True`` (default), households evolve over time:
            one female is assigned as head of each household, pregnant non-head
            females may move out to form new households, and births are added to
            the mother's household. Requires the ``Pregnancy`` module. If
            ``False``, the network is static and ``step()`` is a no-op.
        prob_move_out (float): Probability a non-head female moves out to start
            her own household, evaluated once at the start of each pregnancy.
            Default 0.7. Only used when ``dynamic=True``.
        update_freq (int): How often (in timesteps) to update the network.
            Default 1. Only used when ``dynamic=True``.

    The expected dataframe format is::

            hh_id                ages          sexes
        0       0          72, 17, 30        1, 1, 2
        1       1                  37              2
        2       2          13, 55, 36        2, 1, 2
        3       3  52, 13, 12, 64, 53     1, 2, 1, 2
        4       4              30, 66           1, 1

    Data in this format can be obtained from the `DHS Program
    <https://dhsprogram.com>`_. To prepare a DHS household dataset:

    1. Register and request access at https://dhsprogram.com
    2. Download a Household Recode (HR) dataset in Stata format
       (e.g. ``XXHR7xDT.zip``)
    3. Use ``HouseholdNet.load_dhs()`` to extract the data::

        import starsim as ss; import starsim.library as ssl
        dhs_data = ssl.networks.HouseholdNet.load_dhs('XXHR7xDT/XXHR7xFL.DTA')
        sim = ss.Sim(networks=ssl.networks.HouseholdNet(dhs_data=dhs_data))
        sim.run()

    If real data are not available, synthetic data can be constructed::

        import numpy as np
        import sciris as sc
        import starsim as ss; import starsim.library as ssl

        n = 1000
        age_strings = []
        for i in range(n):
            household_size = np.random.randint(1, 6)
            ages = np.random.randint(0, 80, household_size)
            age_strings.append(sc.strjoin(ages))
        dhs_data = sc.dataframe(hh_id=np.arange(n), ages=age_strings)

        household = ssl.networks.HouseholdNet(dhs_data=dhs_data)
        sim = ss.Sim(diseases='sis', networks=household)
        sim.run()
        sim.plot()
    """
    def __init__(self, pars=None, dhs_data=None, dynamic=True, prob_move_out=_, update_freq=_, **kwargs):
        super().__init__()
        self.define_pars(
            prob_move_out = ss.bernoulli(p=0.7),
            update_freq = 1,
        )
        self.update_pars(pars, **kwargs)
        self.dhs_data = dhs_data
        self.dynamic = dynamic

        states = [ss.FloatArr('household_ids')]
        if self.dynamic:
            states += [
                ss.BoolArr('fhoh', default=False),
                ss.FloatArr('ti_move_out_check', default='-inf'),
            ]
        self.define_states(*states)
        self.p_fractional_age = ss.uniform()
        self.n_households = 0 
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.dhs_data is None:
            raise ValueError("Please provide household data via the dhs_data argument.")
        if self.dynamic:
            ss.check_requires(self.sim, ['pregnancy'])
        return

    def init_post(self, add_pairs=True):
        super().init_post(add_pairs)
        # DHS age data is in integer years; add a random fractional age for realism
        self.sim.people.age[:] = self.sim.people.age + self.p_fractional_age.rvs(self.sim.people.auids)
        return

    def add_pairs(self):
        """ Generate contacts by assigning agents to households from the data """
        ppl = self.sim.people
        pop_size = len(ppl)
        dhs = self.dhs_data

        n_remaining = len(ppl)
        p1 = []
        p2 = []
        while n_remaining > 0:
            self.n_households += 1

            # Sample a household from the data
            rand_row = np.random.choice(len(dhs)) # TODO: make CRN-safe
            household_data = dhs.iloc[rand_row]
            age_data = household_data['ages']
            sex_data = None
            if 'sexes' in household_data.keys():
                sex_data = household_data['sexes']

            age_data = np.array([float(x) for x in age_data.split(', ')], dtype=float)
            cluster_size = len(age_data)

            if cluster_size > n_remaining:
                cluster_size = n_remaining

            cluster_uids = ss.uids((pop_size - n_remaining) + np.arange(cluster_size))

            ppl.age[cluster_uids] = age_data[0:cluster_size]
            if sex_data is not None:
                sex_data = np.array([int(x) for x in sex_data.split(', ')])
                ppl.female[cluster_uids] = (sex_data[0:cluster_size] == 2)
            self.household_ids[cluster_uids] = self.n_households - 1 # Zero based indexing for actual IDs

            # Add symmetric pairwise contacts in each cluster
            for i in cluster_uids:
                for j in cluster_uids:
                    if j > i:
                        p1.append(i)
                        p2.append(j)
            n_remaining -= cluster_size

        beta = np.ones(len(p1), dtype=ss.dtypes.float)
        self.append(p1=p1, p2=p2, beta=beta)

        if self.dynamic:
            # Find a female head of household between ages 15 and 50
            for cid in range(self.n_households):
                cluster_uids = ss.uids(self.household_ids == cid)
                female_uids = cluster_uids[
                    ppl.female[cluster_uids] & (ppl.age[cluster_uids] >= 15) & (ppl.age[cluster_uids] <= 50)]
                if len(female_uids) > 0:
                    fhoh = np.random.choice(a=female_uids) # TODO: make CRN-safe
                    self.fhoh[ss.uids(fhoh)] = True
        return

    def step(self):
        if not self.dynamic:
            return

        self.add_births()

        if np.mod(self.ti, self.pars.update_freq): # Skip all but 0
            return

        self.create_new_households()
        return

    def add_births(self):
        sim = self.sim
        ppl = sim.people

        # Find agents born during the sim (have a parent), already delivered
        # (age >= 0), and not yet assigned to a household (household_ids is NaN).
        # The isnan guard ensures each newborn is processed exactly once.
        candidates = ss.uids(ppl.parent.notnan & (ppl.age >= 0))
        if len(candidates) == 0:
            return 0
        birth_uids = candidates[np.isnan(self.household_ids[candidates])]
        if len(birth_uids) == 0:
            return 0

        mat_uids = ppl.parent[birth_uids]

        # Assign household IDs before creating edges so the newborn is
        # included when looking up household members
        self.household_ids[birth_uids] = self.household_ids[mat_uids]

        p1 = []
        p2 = []
        for new_uid, mat_uid in zip(birth_uids, mat_uids):
            hh_contacts = ss.uids(self.household_ids == self.household_ids[mat_uid])
            hh_contacts = hh_contacts[hh_contacts != new_uid]  # Exclude self-loops
            p1.append(hh_contacts)
            p2.append([new_uid] * len(hh_contacts))

        if p1:
            p1 = ss.uids.concatenate(p1)
            p2 = ss.uids.concatenate(p2)
            beta = np.ones(len(p1), dtype=ss.dtypes.float)
            self.append(p1=p1, p2=p2, beta=beta)

        return len(birth_uids)

    def create_new_households(self):
        """
        Find females that are pregnant and not a head of household.
        Move them and a randomly sampled male partner to a new household.
        """
        ppl = self.sim.people
        potential_movers = ss.uids(~self.fhoh & ppl.pregnancy.pregnant & (self.ti_move_out_check <= self.sim.ti))
        moving_out = self.pars['prob_move_out'].filter(potential_movers)
        if len(moving_out) > 0:
            self.fhoh[moving_out] = True
            potential_partners = ss.uids(ppl.male & (ppl.age > 15) & (ppl.age < 50))
            partner_inds = np.random.permutation(len(potential_partners))[:len(moving_out)] # TODO: make CRN-safe
            partners = potential_partners[partner_inds]
            to_remove = ss.uids.concatenate([moving_out, partners])
            self.remove_uids(to_remove)
            beta = np.ones(len(moving_out), dtype=ss.dtypes.float)
            self.append(p1=moving_out, p2=partners, beta=beta)

            n_moving_out = len(moving_out)
            new_cids = self.n_households + np.arange(n_moving_out)
            self.n_households += n_moving_out
            self.household_ids[moving_out] = new_cids
            self.household_ids[partners] = new_cids

        self.ti_move_out_check[potential_movers] = ppl.pregnancy.ti_delivery[potential_movers]
        return

    @staticmethod
    def load_dhs(path):
        """
        Load a DHS Household Recode (HR) Stata file and return a dataframe
        suitable for use with ``HouseholdNet``.

        Reads the wide-format HR file, extracts per-member age (``HV105``)
        and sex (``HV104``) columns, filters to valid entries (age <= 95 and
        sex in [1, 2]), and returns a dataframe with columns ``hh_id``,
        ``ages``, and ``sexes``.

        Args:
            path (str/Path): Path to a DHS Household Recode Stata file
                (e.g. ``XXHR7xFL.DTA``).

        Returns:
            sc.dataframe: A dataframe with columns ``hh_id``, ``ages``, and
            ``sexes`` ready for use with ``HouseholdNet(dhs_data=...)``.

        Examples:
            ```python
            import starsim as ss; import starsim.library as ssl
            dhs_data = ssl.networks.HouseholdNet.load_dhs('ZZHR62FL.DTA')
            sim = ss.Sim(networks=ssl.networks.HouseholdNet(dhs_data=dhs_data))
            sim.run()
            ```
        """
        import pandas as pd
        hr = pd.read_stata(str(path), convert_categoricals=False)

        rows = []
        for _, hh in hr.iterrows():
            n_members = int(hh['hv009'])
            ages, sexes = [], []
            for i in range(1, n_members + 1):
                idx = f'{i:02d}'
                age = hh.get(f'hv105_{idx}', np.nan)
                sex = hh.get(f'hv104_{idx}', np.nan)
                if not np.isnan(age) and age <= 95 and sex in [1, 2]:
                    ages.append(int(age))
                    sexes.append(int(sex))
            if ages:
                rows.append(dict(hh_id=hh['hhid'].strip(), ages=sc.strjoin(ages), sexes=sc.strjoin(sexes)))

        return sc.dataframe(rows)
