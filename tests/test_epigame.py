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

OPENROUTER_API_KEY=... uv run python tests/test_epigame.py
"""
import os
import pandas as pd
import starsim as ss

def main():
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError('Set OPENROUTER_API_KEY env var before running')

    # change models here: https://openrouter.ai/models
    MODEL = 'nvidia/nemotron-3-super-120b-a12b:free'
    net, n_agents, id_map, start_date, stop_date = ss.build_network("data_ingestion/histories.csv")

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

    seir = ss.SEIR(
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
        dt            = ss.days(1/1440),
        rand_seed     = 42,
        diseases      = seir,
        networks      = net,
        interventions = [
            ss.make_intervention(high_reward=10, agent_uids=[0,1,2], name='group_a', model=MODEL, api_key=api_key),
            ss.make_intervention(high_reward=15, agent_uids=[3,4,5], name='group_b', model=MODEL, api_key=api_key),
        ],
    )
    sim.run()

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
        print(mod.agent_summary)

        print('\n--- Quarantine rate over time ---')
        print(sim.results[label].quarantine_rate)


if __name__ == '__main__':
    main()
