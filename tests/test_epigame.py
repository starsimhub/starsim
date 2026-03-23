"""
Epigame LLM intervention study.

This study simulates an epidemic game in which autonomous LLM-driven agents
decide each day whether to quarantine or remain active. 

Agents interact via real-world proximity contact networks derived 
from the Epigames dataset and are infected according to a SEIR disease model. 

The LLM intervention uses the Health Belief Model (HBM) to inform each agent's decision, 
weighting perceived susceptibility, severity, self-efficacy, and benefits. 

Research Questions: 

1. How does LLM-guided quarantine behavior, conditioned on 
individual health status and reward incentives, alter epidemic curve trajectories 
and cumulative scoring outcomes across distinct payoff structures?

2. Results are benchmarked against epigames how are these scores different to what 
was done in practice?

Usage:

# Run and auto-save
OPENROUTER_API_KEY=... uv run python tests/test_epigame.py

# Load saved sim without re-running
uv run python tests/test_epigame.py --load

# Load from a specific file
uv run python tests/test_epigame.py --load my_run.sim
"""
import os
import sys
import numpy as np
import pandas as pd
import starsim as ss

SIM_FILE = 'epigame_sim.sim'  # default save/load path

def clean_data(df: pd.DataFrame, beta: float = 1.0) -> pd.DataFrame:
    df = df[df["type"] == "contact"].copy()
    df = df.dropna(subset=["user_id", "peer_id", "time"])
    df = df.drop_duplicates(subset=["user_id", "peer_id", "time"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["beta"] = beta
    return df

def remap_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, int, dict]:
    all_ids = pd.Index(pd.unique(pd.concat([df["user_id"], df["peer_id"]], ignore_index=True)))
    id_map = {old: new for new, old in enumerate(all_ids)}  # original user_id -> sequential sim uid

    out = df.copy()
    out["p1"] = out["user_id"].map(id_map).astype("int64")
    out["p2"] = out["peer_id"].map(id_map).astype("int64")
    return out, len(all_ids), id_map

def add_subdaily_timeline(
    df: pd.DataFrame,
    time_col: str = "time",
    duration_col: str = "contact_length",
    tick: pd.Timedelta = pd.Timedelta(milliseconds=1),
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Convert timestamps to a fixed subdaily simulation timeline.

    The default tick is 1 ms (thousandths of a second), which preserves the
    full resolution in contact timestamps.

    Assumes `duration_col` is stored in milliseconds.
    """
    out = df.copy()
    start_date = out[time_col].min().floor(tick)
    stop_date = out[time_col].max().ceil(tick) + tick

    out["start_offset_ti"] = ((out[time_col] - start_date) / tick).round().astype("int64")
    tick_ms = tick / pd.Timedelta(milliseconds=1)
    out["dur_ti"] = np.ceil(out[duration_col].fillna(0) / tick_ms).astype("int64").clip(lower=1)

    return out, start_date, stop_date

class EpigamesNet(ss.DynamicNetwork):
    def __init__(self, df, start_col='start_offset_ti', dur_col='dur_ti', **kwargs):
        super().__init__(**kwargs)
        self.df = df.sort_values(start_col).reset_index(drop=True)
        self.start_col = start_col
        self.dur_col = dur_col 
        self._next = 0

    def init_pre(self, sim): 
        super().init_pre(sim)
        self._starts = self.df[self.start_col].to_numpy()
        return
    
    def add_pairs(self):
        added = 0
        while self._next < len(self.df) and self._starts[self._next] == self.ti:
            row = self.df.iloc[self._next]

            if added < 5:
                print(f"[ti={self.ti}] adding edge ({row.p1}, {row.p2}) for {row[self.dur_col]} ms")

            self.append(
                p1=np.array([int(row.p1)], dtype=ss.dtypes.int),
                p2=np.array([int(row.p2)], dtype=ss.dtypes.int),
                beta=np.array([float(row.beta)], dtype=ss.dtypes.float),
                dur=np.array([float(row[self.dur_col])], dtype=ss.dtypes.float),
            )
            self._next += 1
            added += 1

        return added
    

def build_network(csv_path: str):
    df = pd.read_csv(csv_path, index_col=0)
    df = clean_data(df)
    df, n_agents, id_map = remap_ids(df)
    df, start_date, stop_date = add_subdaily_timeline(df)
    net = EpigamesNet(df, label="Epigames")
    start_date = ss.date(start_date)
    stop_date = ss.date(stop_date)
    return net, n_agents, id_map, start_date, stop_date



net, n_agents, id_map, start_date, stop_date = build_network("data_ingestion/histories.csv")


class SEIR(ss.SIR):
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(
            dur_exp = ss.lognorm_ex(0.5),
        )
        self.update_pars(pars, **kwargs)

        # Additional states beyond the SIR ones 
        self.define_states(
            ss.BoolState('exposed', label='Exposed'),
            ss.FloatArr('ti_exposed', label='TIme of exposure'),
        )
        return

    @property
    def infectious(self):
        return self.infected | self.exposed

    def step_state(self):
        """ Make all the updates from the SIR model """
        # Perform SIR updates
        super().step_state()

        # Additional updates: progress exposed -> infected
        infected = self.exposed & (self.ti_infected <= self.ti)
        self.exposed[infected] = False
        self.infected[infected] = True
        return

    def step_die(self, uids):
        super().step_die(uids)
        self.exposed[uids] = False
        return

    def set_prognoses(self, uids, sources=None):
        """ Carry out state changes associated with infection """
        super().set_prognoses(uids, sources)
        ti = self.ti
        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.ti_exposed[uids] = ti

        # Calculate and schedule future outcomes
        p = self.pars # Shorten for convenience
        dur_exp = p.dur_exp.rvs(uids)
        self.ti_infected[uids] = ti + dur_exp
        dur_inf = p.dur_inf.rvs(uids)
        will_die = p.p_death.rvs(uids)        
        self.ti_recovered[uids[~will_die]] = ti + dur_inf[~will_die]
        self.ti_dead[uids[will_die]] = ti + dur_inf[will_die]
        return

# TODO: we can try different models here
_MODEL = 'nvidia/nemotron-3-super-120b-a12b:free'

# TODO: we can create better prompts for info given to the LLM.
# we need to add game rules to the llm for context
def default_agent_prompt(mod, uid, disease):
    """
    Build the per-agent quarantine prompt for LLMIntervention.

    Args:
        mod (LLMIntervention): The calling module (gives access to ``ti``,
            ``low_reward``, ``high_reward``, and HBM belief states).
        uid (int): Agent UID.
        disease: The target disease module (used to determine health status).

    Returns:
        str: Prompt text sent to the LLM. Must end with a question that the
            LLM answers with only ``'yes'`` or ``'no'``.
    """
    status         = mod._agent_status(uid, disease)
    local_prev     = mod._local_prevalence(uid, disease)
    return (
        f"Epidemic game. Time {mod.ti}. Local prevalence {local_prev:.0%}.\n"
        f"Quarantine={mod.low_reward}pts no-risk. Active={mod.high_reward}pts risk-infection.\n"
        f"Agent: {status} pts={mod.points[uid]:.0f}",
        f"susceptibility={mod.susceptibility[uid]:.0f} "
        f"severity={mod.severity[uid]:.0f} "
        f"self_efficacy={mod.self_efficacy[uid]:.0f} "
        f"benefits={mod.benefits[uid]:.0f}\n"
        f"Should you quarantine? Reply with only 'yes' or 'no'."
    )

# TODO: we can update the beliefs with the data we have from the pre survey.
def default_init_beliefs(mod):
    """
    Default HBM belief initializer for LLMIntervention.

    The Health Belief Model (HBM) is a psychological framework developed in the 
    1950s by social psychologists at the U.S. Public Health Service to explain and 
    predict health-related behaviors, particularly why individuals do or do not
    engage in preventive actions like screenings or vaccinations.

    Assigns each agent four belief scores sampled from a truncated normal
    distribution (mean=3.5, sd=1.2, clipped to [1, 6]) using the sim's
    ``rand_seed`` for reproducibility.

    Args:
        mod (LLMIntervention): The calling module. Populates ``mod.susceptibility``,
            ``mod.severity``, ``mod.self_efficacy``, and ``mod.benefits`` in-place.

    To supply a custom initializer, write a function with the same signature and
    pass it as ``init_beliefs=my_fn`` to ``LLMIntervention``.
    """
    rng = np.random.default_rng(seed=int(mod.sim.pars.rand_seed))
    n   = len(mod.sim.people)
    for state_name in ('susceptibility', 'severity', 'self_efficacy', 'benefits'):
        raw     = rng.normal(loc=3.5, scale=1.2, size=n)
        clipped = np.clip(raw, 1.0, 6.0).astype(float)
        getattr(mod, state_name)[:] = clipped
    return

def main():
    """
    Test for LLMIntervention. Each timestep runs the following loop:

        start_step()       -- restore transmission from last step's quarantine
          |
        intv.step()        -- LLM decides per agent; zeros rel_sus/rel_trans for quarantined agents
          |
        disease.step()     -- transmission runs; quarantined agents are invisible to it
          |
        people.step_die()  -- deaths processed
          |
        update_results()
          |
        finish_step()      -- zero points for dead agents, increment ti
    """
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError('Set OPENROUTER_API_KEY env var before running')

    # A/B group assignment: map original user_ids -> sequential sim uids via id_map.
    # Replace GROUP_A_USER_IDS / GROUP_B_USER_IDS with real lists from the study when available.
    # For now: 50/50 split by sequential uid order as a placeholder.
    all_uids    = list(range(n_agents))
    mid         = n_agents // 2
    group_a_uids = all_uids[:mid]       # high_reward = 10
    group_b_uids = all_uids[mid:]       # high_reward = 15
    # To use real original user_ids instead:
    #   group_a_uids = [id_map[uid] for uid in GROUP_A_USER_IDS]
    #   group_b_uids = [id_map[uid] for uid in GROUP_B_USER_IDS]

    def make_intervention(high_reward, agent_uids, label):
        return ss.LLMIntervention(
            low_reward    = 5,
            high_reward   = high_reward,
            agent_uids    = agent_uids,
            model         = _MODEL,
            api_key       = api_key,
            interval      = 1,
            decision_hour = 9.5,      # 09:30 AM; respected when dt < 1 day
            build_prompt  = default_agent_prompt,
            init_beliefs  = default_init_beliefs,
            verbose       = True,
            label         = label,
        )

    # TODO: ask andres for actual parameters
    seir = SEIR(
        init_prev = ss.bernoulli(p=0.01),
        beta      = ss.perday(0.0907 * 24),
        dur_inf   = ss.lognorm_ex(mean=ss.days(77 / 24), std=ss.days(0.5)),
        p_death   = ss.bernoulli(p=0.6 * 0.25 + 0.4 * 0.7),
        dur_exp   = ss.lognorm_ex(mean=ss.days(10 / 24), std=ss.days(0.2)),
    )

    sim = ss.Sim(
        n_agents      = n_agents,
        start         = start_date,
        stop          = stop_date,
        # TODO: update resolution
        dt            = ss.days(1),
        diseases      = seir,
        networks      = net,
        interventions = [
            make_intervention(high_reward=10, agent_uids=group_a_uids, label='group_a'),
            make_intervention(high_reward=15, agent_uids=group_b_uids, label='group_b'),
        ],
        rand_seed     = 42,
    )

    # Run or load from saved file
    do_load = '--load' in sys.argv
    idx = sys.argv.index('--load') if do_load else -1
    sim_path = sys.argv[idx + 1] if do_load and idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith('-') else SIM_FILE

    if do_load and os.path.exists(sim_path):
        print(f'Loading sim from {sim_path} ...')
        sim = ss.load(sim_path)
    else:
        sim.run()
        saved = sim.save(sim_path)
        print(f'Sim saved to {saved}')

    for label in ('group_a', 'group_b'):
        mod = sim.interventions[label]
        print(f'\n{"=" * 50}')
        print(f'Group: {label}  (low=5, high={mod.high_reward}, n_agents={len(mod.agent_uids)})')
        print('=' * 50)

        print('\n--- Step log ---')
        for e in mod.log:
            status = f"{e.n_quarantined}/{e.n_agents} quarantined"
            err    = f"  ERROR: {e.error}" if e.error else ''
            print(f"  t={e.t}: {status}{err}")

        print('\n--- Per-agent quarantine decisions ---')
        dl = mod.decision_log if not isinstance(mod.decision_log, list) else pd.DataFrame(mod.decision_log)
        if len(dl):
            print(dl.to_string(index=False))

        print('\n--- Per-agent summary ---')
        print(mod.agent_summary.to_string())

        print('\n--- Quarantine rate over time ---')
        print(sim.results[label].quarantine_rate)


if __name__ == '__main__':
    main()
