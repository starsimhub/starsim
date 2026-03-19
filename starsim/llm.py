"""
LLM-powered modules for integrating large language model decisions into simulations.

Uses the OpenRouter API to make LLM calls during the simulation loop.
"""
import json
import ssl
import certifi
import urllib.request
import sciris as sc
import starsim as ss


__all__ = ['LLMIntervention']

def _call_openrouter(prompt, model, api_key, max_tokens, timeout, seed=None):
    """
    Send a prompt to OpenRouter and return the response text.

    Pass ``seed`` (int) to enable deterministic outputs across runs (requires
    ``temperature=0``; not all models honour this field).
    """
    _OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
    payload = json.dumps({
        'model':       model,
        'messages':    [{'role': 'user', 'content': prompt}],
        'max_tokens':  max_tokens,
        'temperature': 0,
        **({'seed': seed} if seed is not None else {}),
    }).encode('utf-8')

    req = urllib.request.Request(
        _OPENROUTER_URL,
        data    = payload,
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type':  'application/json',
        },
        method = 'POST',
    )
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=timeout) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    return data['choices'][0]['message']['content']


class LLMIntervention(ss.Intervention):
    """
    At each timestep one LLM call is made per active agent for reproducibility.
    Each call asks whether that agent should quarantine (true = quarantine). Quarantined
    agents have their ``rel_sus`` and ``rel_trans`` zeroed on the target disease
    before ``disease.step()`` runs, effectively removing them from the contact
    network for that timestep. Beliefs are restored at the start of the next step.

    Each agent carries a Health Belief Model (HBM) profile sampled at init:
    perceived susceptibility, severity, self-efficacy, and benefits (Likert 1-6).

    Args:
        low_reward (float): Points awarded per step for quarantining. Default 5.
        high_reward (float): Points awarded per step for NOT quarantining (if uninfected). Default 10.
        model (str): OpenRouter model identifier. Defaults to a free model.
        api_key (str): OpenRouter API key (or set ``OPENROUTER_API_KEY`` env var).
        interval (int): Timesteps between decision rounds. Default 1 (every step).
        build_prompt (callable): Optional ``fn(mod, uid, disease) -> str`` that
            builds the per-agent prompt. Defaults to the built-in HBM prompt.
        init_beliefs (callable): Optional ``fn(mod) -> None`` that populates per-agent
            HBM belief states in-place. Defaults to ``default_init_beliefs``.
        max_tokens (int): Max LLM response tokens. Default 2000.
        timeout (int): HTTP request timeout in seconds. Default 30.
        verbose (bool): Print prompts and responses. Default False.
    """

    def __init__(self, low_reward=5, high_reward=10,
                 model=None, api_key=None, interval=1, build_prompt=None,
                 init_beliefs=None, max_tokens=2000, timeout=30, verbose=False, **kwargs):
        super().__init__(**kwargs)

        self.low_reward       = low_reward
        self.high_reward      = high_reward
        self.model            = model
        self.api_key          = api_key
        self.interval         = interval
        self.build_prompt_fn  = build_prompt
        self.init_beliefs_fn  = init_beliefs
        self.max_tokens       = max_tokens
        self.timeout          = timeout
        self.verbose          = verbose

        # Per-agent states
        self.define_states(
            ss.BoolState('quarantined',   label='Quarantined'),
            ss.FloatArr('susceptibility', default=3.0, label='HBM perceived susceptibility (1-6)'),
            ss.FloatArr('severity',       default=3.0, label='HBM perceived severity (1-6)'),
            ss.FloatArr('self_efficacy',  default=3.0, label='HBM perceived self-efficacy (1-6)'),
            ss.FloatArr('benefits',       default=3.0, label='HBM perceived benefits (1-6)'),
            ss.FloatArr('points',         default=0.0, label='Accumulated game points'),
        )

        self.log = []  # Per-step sc.objdict: {t, ti, n_agents, n_quarantined, error}
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('quarantine_rate', dtype=float, scale=False, label='Quarantine rate'),
            ss.Result('mean_points',     dtype=float, scale=False, label='Mean points'),
        )
        return

    def init_post(self):
        """ Initialize per-agent HBM belief scores via ``init_beliefs_fn`` """
        super().init_post()
        self.init_beliefs_fn(self)
        return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _target_disease(self):
        """ Return the first disease module in the sim, or None """
        vals = list(self.sim.diseases.values())
        return vals[0] if vals else None

    def _local_prevalence(self, uid, disease):
        """
        Fraction of uid's direct network contacts that were infected at the
        start of this step (i.e. before disease.step() runs).

        Since intv.step() executes before disease.step(), disease.infected
        reflects the previous step's state — matching the AUIB game design
        where agents see local prevalence from the prior timestep.
        """
        if disease is None or not hasattr(disease, 'infected'):
            return 0.0
        contacts = set()
        for net in self.sim.networks.values():
            contacts.update(net.find_contacts(ss.uids([uid])))
        contacts.discard(int(uid))  # exclude self if present
        if not contacts:
            return 0.0
        contact_uids = ss.uids(list(contacts))
        return float(disease.infected[contact_uids].mean())

    def _agent_status(self, uid, disease):
        """ Return a brief health status string for one agent """
        if disease is None:
            return 'unknown'
        if hasattr(disease, 'infected') and disease.infected[uid]:
            return 'infected'
        if hasattr(disease, 'susceptible') and disease.susceptible[uid]:
            return 'susceptible'
        return 'recovered'

    def _call_llm_agent(self, uid, disease):
        """ Ask the LLM whether this agent should quarantine. Returns bool. """
        fn     = self.build_prompt_fn
        prompt = fn(self, uid, disease)

        if self.verbose:
            print(f'\n[LLMIntervention t={self.ti}] Agent {uid}:\n{prompt}')

        seed    = int(self.sim.pars.rand_seed)
        content = _call_openrouter(prompt, self.model, self.api_key, self.max_tokens, self.timeout, seed=seed)
        content = (content or '').strip().lower()

        if self.verbose:
            print(f'[LLMIntervention t={self.ti}] Agent {uid} response: {content}')

        return 'yes' in content

    def _zero_transmission(self, q_uids, disease):
        """ Set rel_sus and rel_trans to 0 for quarantined agents """
        if disease is None:
            return
        if hasattr(disease, 'rel_sus'):
            disease.rel_sus[q_uids]   = 0.0
        if hasattr(disease, 'rel_trans'):
            disease.rel_trans[q_uids] = 0.0
        return

    def _restore_transmission(self, q_uids, disease):
        """ Restore rel_sus and rel_trans to 1 for previously quarantined agents """
        if disease is None or len(q_uids) == 0:
            return
        if hasattr(disease, 'rel_sus'):
            disease.rel_sus[q_uids]   = 1.0
        if hasattr(disease, 'rel_trans'):
            disease.rel_trans[q_uids] = 1.0
        return

    # ------------------------------------------------------------------
    # Module lifecycle
    # ------------------------------------------------------------------

    def start_step(self):
        """
        Restore transmission for agents that quarantined last step, then
        reset quarantine flags so each step starts clean.
        """
        super().start_step()
        if self.ti == 0:
            return
        disease = self._target_disease()
        self._restore_transmission(self.quarantined.uids, disease)
        self.quarantined[:] = False
        return

    def step(self):
        """
        For each active agent:
        1. Ask the LLM whether they quarantine.
        2. Zero transmission for quarantined agents.
        3. Award points.
        """
        if self.ti % self.interval != 0:
            return

        uids    = self.sim.people.auids
        disease = self._target_disease()
        entry   = sc.objdict(t=self.ti, ti=self.ti,
                             n_agents=len(uids), n_quarantined=0, error=None)

        if len(uids) == 0:
            self.log.append(entry)
            return

        # One LLM call per agent
        decisions = {}
        errors    = []
        for uid in uids:
            try:
                decisions[int(uid)] = self._call_llm_agent(uid, disease)
            except Exception as e:
                decisions[int(uid)] = False
                errors.append(f'agent {uid}: {e}')

        if errors:
            entry.error = '; '.join(errors)
            if self.verbose:
                print(f'[LLMIntervention t={self.ti}] LLM errors: {entry.error}')

        q_list      = ss.uids([uid for uid, q in decisions.items() if     q])
        active_list = ss.uids([uid for uid, q in decisions.items() if not q])

        if len(q_list):
            self.quarantined[q_list] = True
            self._zero_transmission(q_list, disease)
            self.points[q_list] += self.low_reward

        if len(active_list):
            if disease is not None and hasattr(disease, 'infected'):
                not_infected = active_list[~disease.infected[active_list]]
            else:
                not_infected = active_list
            self.points[not_infected] += self.high_reward

        entry.n_quarantined = int(len(q_list))
        self.log.append(entry)
        return

    def finish_step(self):
        """ Zero points for agents who died this timestep """
        dead_uids = self.sim.people.dead.uids
        if len(dead_uids):
            self.points[dead_uids] = 0.0
        super().finish_step()
        return

    def update_results(self):
        super().update_results()
        uids = self.sim.people.auids
        n    = max(len(uids), 1)
        n_q  = int(self.quarantined.sum())
        pts  = float(self.points[uids].mean()) if len(uids) else 0.0
        self.results['quarantine_rate'][self.ti] = n_q / n
        self.results['mean_points'][self.ti]     = pts
        return

    def finalize(self):
        super().finalize()
        n_calls  = len(self.log)
        n_errors = sum(1 for e in self.log if e.error)
        if self.verbose or n_errors:
            total_q = sum(e.n_quarantined for e in self.log)
            total_a = sum(e.n_agents      for e in self.log)
            rate    = total_q / max(total_a, 1)
            # Across all decisions made, what fraction were quarantine?
            print(f'[LLMIntervention] {n_calls} steps | overall quarantine rate: {rate:.1%} | {n_errors} errors')
        return
