"""
Run with:
    OPENROUTER_API_KEY=<your-key> uv run python tests/test_epigame.py
"""
import os
import numpy as np
import starsim as ss

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
        f"Agent: {status} pts={mod.points[uid]:.0f} sus={mod.susceptibility[uid]:.0f} "
        f"sev={mod.severity[uid]:.0f} "
        f"eff={mod.self_efficacy[uid]:.0f} "
        f"ben={mod.benefits[uid]:.0f}\n"
        f"Should this agent quarantine? Reply with only 'yes' or 'no'."
    )

# TODO: we can update the beliefs with the data we have from the pre survey.
def default_init_beliefs(mod):
    """
    Default HBM belief initializer for LLMIntervention.

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

    llmintervention = ss.LLMIntervention(
        low_reward   = 5,
        high_reward  = 10,
        model        = _MODEL,
        api_key      = api_key,
        interval     = 1,   # decide every step
        build_prompt = default_agent_prompt,
        init_beliefs = default_init_beliefs,
        verbose      = True,
    )

    sim = ss.Sim(
        n_agents      = 2,               # small so API calls are fast
        dur           = 3,               # 3 days
        dt            = 1,
        # TODO add epigame model here
        diseases      = ss.SIR(init_prev=ss.bernoulli(p=0.2)), 
        # TODO add real network here
        networks      = 'random', 
        interventions = llmintervention,
        rand_seed     = 42,
    )
    sim.run()

    mod = sim.interventions['llmintervention']

    print('\n--- Decision log ---')
    for e in mod.log:
        status = f"{e.n_quarantined}/{e.n_agents} quarantined"
        err    = f"  ERROR: {e.error}" if e.error else ''
        print(f"  t={e.t}: {status}{err}")

    print('\n--- Points per agent ---')
    for uid in sim.people.auids:
        print(f"  agent {int(uid)}: {mod.points[uid]:.0f} pts")

    print('\n--- Quarantine rate over time ---')
    print(sim.results['llmintervention'].quarantine_rate)


if __name__ == '__main__':
    main()
