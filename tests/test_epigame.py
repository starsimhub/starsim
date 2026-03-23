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
import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import starsim as ss

def main():
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError('Set OPENROUTER_API_KEY env var before running')

    run_dir = Path("run_outputs") / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    # change models here: https://openrouter.ai/models
    MODEL = 'nvidia/nemotron-3-super-120b-a12b:free'
    net, n_agents, start_date, stop_date, id_map = ss.build_network("data_ingestion/histories.csv")
    # To use real original user_ids instead:
    group_a_uids, group_b_uids = ss.group_split("data_ingestion/participants.csv", id_map)

    seir = ss.SEIR_AMS(
        init_prev = ss.bernoulli(p=0.01),
        beta = ss.perday(0.0907*24),
        dur_exp = ss.lognorm_ex(mean=ss.days(10/24), std=ss.days(0.2)),
        dur_inf = ss.lognorm_ex(mean=ss.days(77/24), std=ss.days(0.5)),
        p_symp = ss.choice(a=3, p=[0.30, 0.42, 0.28]),  # 0=asymptomatic, 1=mild, 2=severe
        p_death_mild = ss.bernoulli(p=0.25),
        p_death_severe = ss.bernoulli(p=0.70),
    )

    sim = ss.Sim(
        n_agents      = n_agents,
        start         = start_date,
        stop          = stop_date,
        dt            = ss.days(1/8640),
        rand_seed     = 42,
        diseases      = seir,
        networks      = net,
        interventions = [
            ss.make_intervention(high_reward=10, agent_uids=group_a_uids, name='group_a', model=MODEL, api_key=api_key, id_map=id_map),
            ss.make_intervention(high_reward=15, agent_uids=group_b_uids, name='group_b', model=MODEL, api_key=api_key, id_map=id_map),
        ],
    )
    sim.run()

    # Save a high-level plot of the run.
    fig = sim.plot()
    fig.savefig(run_dir / "sim_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save a manifest of the run configuration.
    run_metadata = {
        "model": MODEL,
        "rand_seed": 42,
        "n_agents": n_agents,
        "start_date": str(start_date),
        "stop_date": str(stop_date),
        "group_a_uids": group_a_uids,
        "group_b_uids": group_b_uids,
        "id_map": {str(k): int(v) for k, v in id_map.items()},
    }
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, default=str)

    # Save the full simulation object if possible.
    try:
        with open(run_dir / "sim.pkl", "wb") as f:
            pickle.dump(sim, f)
    except Exception as e:
        with open(run_dir / "sim_pickle_error.txt", "w", encoding="utf-8") as f:
            f.write(repr(e))

    # Save sim.results in a best-effort way.
    results = getattr(sim, "results", None)
    if results is not None:
        try:
            if isinstance(results, pd.DataFrame):
                results.to_csv(run_dir / "sim_results.csv")
            elif isinstance(results, pd.Series):
                results.to_csv(run_dir / "sim_results.csv")
            elif isinstance(results, dict):
                for key, value in results.items():
                    safe_key = str(key).replace("/", "_")
                    if isinstance(value, pd.DataFrame):
                        value.to_csv(run_dir / f"results_{safe_key}.csv")
                    elif isinstance(value, pd.Series):
                        value.to_csv(run_dir / f"results_{safe_key}.csv")
                    else:
                        with open(run_dir / f"results_{safe_key}.json", "w", encoding="utf-8") as f:
                            json.dump(value, f, indent=2, default=str)
            else:
                with open(run_dir / "sim_results.json", "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, default=str)
        except Exception as e:
            with open(run_dir / "sim_results_error.txt", "w", encoding="utf-8") as f:
                f.write(repr(e))

    for label in ('group_a', 'group_b'):
        mod = sim.interventions[label]
        # Save intervention-specific logs and summaries.
        try:
            pd.DataFrame(mod.log).to_csv(run_dir / f"{label}_step_log.csv", index=False)
        except Exception:
            pass

        try:
            decision_log = mod.decision_log
            if isinstance(decision_log, list):
                decision_log = pd.DataFrame(decision_log)
            decision_log.to_csv(run_dir / f"{label}_decision_log.csv", index=False)
        except Exception:
            pass

        try:
            if isinstance(mod.agent_summary, pd.DataFrame):
                mod.agent_summary.to_csv(run_dir / f"{label}_agent_summary.csv")
            else:
                pd.DataFrame(mod.agent_summary).to_csv(run_dir / f"{label}_agent_summary.csv")
        except Exception:
            pass

        try:
            quarantine_rate = sim.results[label].quarantine_rate
            if hasattr(quarantine_rate, "to_csv"):
                quarantine_rate.to_csv(run_dir / f"{label}_quarantine_rate.csv")
            else:
                pd.DataFrame(quarantine_rate).to_csv(run_dir / f"{label}_quarantine_rate.csv")
        except Exception:
            pass
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

    print(f"\nSaved run artifacts to: {run_dir.resolve()}")


if __name__ == '__main__':
    main()
