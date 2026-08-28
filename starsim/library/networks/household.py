"""
Household contact networks built from DHS-style survey data
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

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
        dhs_data (DataFrame/str): A pandas or Sciris dataframe with columns `hh_id`
            and `ages`. Optionally also `sexes`. The `ages` column should
            contain comma-separated age strings (e.g. `"72, 17, 30"`). If
            `sexes` is included, it should contain comma-separated values
            using DHS convention (1 = male, 2 = female) with the same number
            of entries as `ages`. Pass `'default'` to use synthetic data (see
            `make_default_data()`), e.g. for demos and testing.
        dynamic (bool): If `True` (default), households evolve over time:
            one female is assigned as head of each household, pregnant non-head
            females may move out to form new households, and births are added to
            the mother's household. Requires the `Pregnancy` module. If
            `False`, the network is static and `step()` is a no-op.
        prob_move_out (float): Probability a non-head female moves out to start
            her own household, evaluated once at the start of each pregnancy.
            Default 0.7. Only used when `dynamic=True`.
        update_freq (int): How often (in timesteps) to update the network.
            Default 1. Only used when `dynamic=True`.

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
       (e.g. `XXHR7xDT.zip`)
    3. Use `HouseholdNet.load_dhs()` to extract the data::

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
        if isinstance(dhs_data, str) and dhs_data == 'default':
            dhs_data = self.make_default_data()  # Populate with synthetic data for demos and testing
        if dhs_data is None:
            raise ValueError("Please provide household data via the dhs_data argument, or use dhs_data='default' for synthetic data.")
        self.dhs_data = dhs_data
        self.dynamic = dynamic

        states = [ss.FloatArr('household_ids')]
        if self.dynamic:
            states += [
                ss.BoolArr('fhoh', default=False),
                ss.FloatArr('ti_move_out_check', default='-inf'),
            ]
        self.define_states(*states)
        self.rng_fractional_age = ss.uniform()
        self.rng_household = ss.randint()            # Which DHS household to draw when building the network (high set once data is parsed)
        self.rng_head = ss.random()                  # Per-agent score for selecting the female head of each household
        self.rng_partner = ss.choice(replace=False)  # Which male partner moves out with a pregnant non-head female
        self.n_households = 0
        self._dhs_parsed = None  # Cached (sizes, ages_flat, sexes_flat, offsets, has_sex) from the DHS data
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.dynamic:
            ss.check_requires(self.sim, ['pregnancy'])
        return

    def init_post(self, add_pairs=True):
        super().init_post(add_pairs)
        ppl = self.sim.people
        # DHS age data is in integer years; add a random fractional age for realism
        ppl.age[:] = ppl.age + self.rng_fractional_age.rvs(ppl.auids)

        # Women already pregnant at initialization were captured by the DHS survey in
        # their current household, so treat that as their post-move-out-decision state:
        # mark them as already evaluated for their current pregnancy. Without this,
        # `ti_move_out_check` defaults to -inf, so every pregnant non-head becomes eligible
        # on the first step and ~prob_move_out of them move into new households simultaneously,
        # distorting the input household size distribution. (Pregnancy is a demographics module,
        # so it is initialized before networks and its states are populated by this point.)
        if self.dynamic:
            preg = ppl.pregnancy
            preg_uids = preg.pregnant.uids
            self.ti_move_out_check[preg_uids] = preg.ti_pregnant[preg_uids]
        return

    def _parse_dhs(self):
        """ Parse the DHS age/sex strings into flat arrays once, and cache the result.

        Returns (sizes, ages_flat, sexes_flat, offsets, has_sex) where ages_flat/sexes_flat are
        the members of every household concatenated end to end, offsets[r] gives the start of
        household r within them, and sizes[r] its member count.
        """
        if self._dhs_parsed is None:
            dhs = self.dhs_data
            ages_list = [np.array(a.split(', '), dtype=float) for a in dhs['ages']]
            sizes = np.array([len(a) for a in ages_list])
            ages_flat = np.concatenate(ages_list)
            offsets = np.concatenate([[0], np.cumsum(sizes)])  # length len(dhs)+1
            has_sex = 'sexes' in dhs.columns
            sexes_flat = np.concatenate([np.array(s.split(', '), dtype=int) for s in dhs['sexes']]) if has_sex else None
            self._dhs_parsed = (sizes, ages_flat, sexes_flat, offsets, has_sex)
        return self._dhs_parsed

    def add_pairs(self):
        """ Generate contacts by assigning agents to households sampled from the data.

        Households are drawn uniformly at random (with replacement) until they cover the whole
        population, exactly as the reference algorithm, but sampling, age/sex assignment, edge
        creation, and head-of-household selection are all vectorized rather than looped per
        household. Results are statistically equivalent but not bit-identical to the loop version
        (the random draws differ).
        """
        ppl = self.sim.people
        pop_size = len(ppl)
        sizes, ages_flat, sexes_flat, offsets, has_sex = self._parse_dhs()
        n_dhs = len(sizes)

        # Sample household rows (uniform in [0, n_dhs), with replacement) until their members cover the
        # population. Draws use the network's own RNG stream rather than the global np.random, so they are
        # reproducible and don't perturb other modules; scalar-size draws aren't slot-based, so not CRN-safe.
        self.rng_household.set(high=n_dhs)  # high isn't known until the data is parsed, so set it here
        n_est = int(pop_size/sizes.mean()*1.25) + 16
        rows = self.rng_household.rvs(n_est)
        while sizes[rows].sum() < pop_size:
            rows = np.concatenate([rows, self.rng_household.rvs(n_est)])
        n_hh = int(np.searchsorted(np.cumsum(sizes[rows]), pop_size)) + 1
        rows = rows[:n_hh]
        hsize = sizes[rows].copy()
        hsize[-1] -= int(hsize.sum() - pop_size)  # Truncate the last household so the total is exactly pop_size
        self.n_households = n_hh

        # Assign contiguous agent blocks to households
        all_uids = ss.uids(np.arange(pop_size))
        hh_ids = np.repeat(np.arange(n_hh), hsize)
        self.household_ids[all_uids] = hh_ids

        # Gather each member's age (and sex) via a vectorized ragged gather into ages_flat
        seg_start = np.cumsum(hsize) - hsize                          # output position where each household starts
        intra = np.arange(pop_size) - np.repeat(seg_start, hsize)     # 0..size-1 within each household
        gather = np.repeat(offsets[rows], hsize) + intra
        ppl.age[all_uids] = ages_flat[gather]
        if has_sex:
            ppl.female[all_uids] = (sexes_flat[gather] == 2)

        # Build all within-household edges (every i<j pair), batched by household size
        p1_list, p2_list = [], []
        for s in np.unique(hsize):
            if s < 2:
                continue
            starts = seg_start[hsize == s]
            ii, jj = np.triu_indices(s, k=1)
            p1_list.append((starts[:, None] + ii[None, :]).ravel())
            p2_list.append((starts[:, None] + jj[None, :]).ravel())
        p1 = np.concatenate(p1_list) if p1_list else np.empty(0, dtype=ss.dtypes.int)
        p2 = np.concatenate(p2_list) if p2_list else np.empty(0, dtype=ss.dtypes.int)
        beta = np.ones(len(p1), dtype=ss.dtypes.float)
        self.append(p1=p1, p2=p2, beta=beta)

        if self.dynamic:
            # Assign one random eligible female (age 15-50) as head of each household: give every
            # eligible agent a random score and pick the highest-scoring one within each household.
            ages = ppl.age[all_uids]
            elig = ppl.female[all_uids] & (ages >= 15) & (ages <= 50)
            score = self.rng_head.rvs(all_uids)  # Per-agent CRN-safe (slot-based) random score
            score[~elig] = -1.0
            best = np.full(n_hh, -1.0)
            np.maximum.at(best, hh_ids, score)
            is_head = elig & (score == best[hh_ids])
            self.fhoh[all_uids[is_head]] = True
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
        ppl = self.sim.people

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

        # Sort alive agents by household id so each household's members form a contiguous slice
        # of `sorted_uids` (ascending uid within a household, since argsort is stable).
        auids = ppl.auids
        hvals = self.household_ids[auids]
        valid = ~np.isnan(hvals)
        auids = auids[valid]
        hvals = hvals[valid]
        order = np.argsort(hvals, kind='stable')
        sorted_uids = auids[order]
        sorted_h = hvals[order]

        # For every newborn, locate its household's member slice [lo, hi) in one vectorized pass,
        # then gather all birth edges at once (no per-household split, no per-birth Python loop).
        mat_hids = self.household_ids[mat_uids]
        lo = np.searchsorted(sorted_h, mat_hids, side='left')
        hi = np.searchsorted(sorted_h, mat_hids, side='right')
        counts = hi - lo
        counts[np.isnan(mat_hids)] = 0  # Mother has no household; no contacts to add (matches prior behavior)

        # Ragged gather: newborn i connects to sorted_uids[lo_i : lo_i+counts_i]
        total = int(counts.sum())
        seg_start = np.repeat(np.cumsum(counts) - counts, counts)  # output start of each newborn's block
        gather = np.repeat(lo, counts) + (np.arange(total) - seg_start)
        p1 = sorted_uids[gather]                          # household members (contacts)
        p2 = np.repeat(np.asarray(birth_uids), counts)    # the newborn for each contact
        keep = p1 != p2                                   # exclude self-loops
        p1 = p1[keep]
        p2 = p2[keep]

        if len(p1):
            beta = np.ones(len(p1), dtype=ss.dtypes.float)
            self.append(p1=ss.uids(p1), p2=ss.uids(p2), beta=beta)

        return len(birth_uids)

    def create_new_households(self):
        """
        Find females that are pregnant and not a head of household.
        Move them and a randomly sampled male partner to a new household.
        """
        ppl = self.sim.people
        # Evaluate each pregnancy for move-out exactly once (at its start). `ti_move_out_check`
        # stores the `ti_pregnant` of the pregnancy a woman was last evaluated for; she is eligible
        # only when her current `ti_pregnant` is newer than that. This is dt-agnostic: comparing
        # against `ti_pregnant` (rather than the sim timestep vs. `ti_delivery`) avoids mis-timed,
        # repeated move-outs when the pregnancy module runs on a coarser dt than the sim (e.g.
        # ss.Pregnancy(dt=ss.months(3))), which otherwise over-fragments households. Bit-identical
        # to the previous logic when pregnancy dt == sim dt.
        potential_movers = ss.uids(~self.fhoh & ppl.pregnancy.pregnant & (self.ti_move_out_check < ppl.pregnancy.ti_pregnant))
        moving_out = self.pars['prob_move_out'].filter(potential_movers)
        if len(moving_out) > 0:
            self.fhoh[moving_out] = True
            potential_partners = ss.uids(ppl.male & (ppl.age > 15) & (ppl.age < 50))
            self.rng_partner.set(a=potential_partners)  # Choose distinct partners from the eligible males
            partners = ss.uids(self.rng_partner.rvs(len(moving_out)))
            to_remove = ss.uids.concatenate([moving_out, partners])
            self.remove_uids(to_remove)
            beta = np.ones(len(moving_out), dtype=ss.dtypes.float)
            self.append(p1=moving_out, p2=partners, beta=beta)

            n_moving_out = len(moving_out)
            new_cids = self.n_households + np.arange(n_moving_out)
            self.n_households += n_moving_out
            self.household_ids[moving_out] = new_cids
            self.household_ids[partners] = new_cids

        self.ti_move_out_check[potential_movers] = ppl.pregnancy.ti_pregnant[potential_movers]
        return

    @staticmethod
    def make_default_data(n=1000, seed=1):
        """
        Generate synthetic household data, used when `dhs_data='default'`.

        Creates `n` households of 1-5 members with ages uniformly distributed
        between 0 and 80. Intended for demos and testing when real DHS data are
        not available; see `load_dhs()` for loading actual survey data.

        Args:
            n (int): number of synthetic households to generate
            seed (int): random seed for reproducibility

        Returns:
            sc.dataframe: A dataframe with columns `hh_id` and `ages` ready for
            use with `HouseholdNet(dhs_data=...)`.
        """
        rng = np.random.default_rng(seed)
        age_strings = []
        for i in range(n):
            household_size = rng.integers(1, 6)
            ages = rng.integers(0, 80, household_size)
            age_strings.append(sc.strjoin(ages))
        return sc.dataframe(hh_id=np.arange(n), ages=age_strings)

    @staticmethod
    def load_dhs(path):
        """
        Load a DHS Household Recode (HR) Stata file and return a dataframe
        suitable for use with `HouseholdNet`.

        Reads the wide-format HR file, extracts per-member age (`HV105`)
        and sex (`HV104`) columns, filters to valid entries (age <= 95 and
        sex in [1, 2]), and returns a dataframe with columns `hh_id`,
        `ages`, and `sexes`.

        Args:
            path (str/Path): Path to a DHS Household Recode Stata file
                (e.g. `XXHR7xFL.DTA`).

        Returns:
            sc.dataframe: A dataframe with columns `hh_id`, `ages`, and
            `sexes` ready for use with `HouseholdNet(dhs_data=...)`.

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
