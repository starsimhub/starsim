"""
LLM-powered modules for integrating large language model decisions into simulations.

Uses the OpenRouter API to make LLM calls during the simulation loop.
"""
import json
import ssl
import certifi
import urllib.request
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss


__all__ = ['LLMIntervention', 'make_intervention']

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
        f"You are playing an epidemic decision game where your goal is to maximise your total points.\n"
        f"\n"
        f"Game mechanics:\n"
        f"- A disease spreads through a contact network: interacting with others exposes you to infection risk.\n"
        f"- Your infection risk increases with local prevalence ({local_prev:.0%}) and your contacts.\n"
        f"- If infected, you may lose points (reduced rewards, possible large penalties).\n"
        f"- You move through health states (susceptible → infected → recovered).\n"
        f"\n"
        f"Decision each round:\n"
        f"- Quarantine: {mod.low_reward} pts. No infection risk this round.\n"
        f"- Stay active: {mod.high_reward} pts. Risk infection from contacts.\n"
        f"\n"
        f"This is a trade-off between:\n"
        f"- Short-term reward (staying active)\n"
        f"- Long-term risk (infection causing point losses)\n"
        f"\n"
        f"Your objective:\n"
        f"Maximise your total points over time. Consider expected future losses from infection, not just immediate reward.\n"
        f"\n"
        f"Health decision framework (Health Belief Model):\n"
        f"The Health Belief Model (HBM) is a psychological framework used to understand health-related decisions. "
        f"It explains behavior based on six main factors:\n"
        f"Perceived susceptibility: your personal assessment of risk (how likely you are to get infected).\n"
        f"Perceived severity: how serious infection would be for you (possible point losses, complications).\n"
        f"Perceived benefits: the advantages of taking preventive action (quarantining protects you).\n"
        f"Perceived barriers: the costs or downsides of action (losing high reward points by quarantining).\n"
        f"Self-efficacy: your confidence in successfully performing the preventive action (ability to quarantine effectively).\n"
        f"Cues to action: triggers from your environment or game that prompt a decision (local prevalence, infected contacts).\n"
        f"\n"
        f"Your current state:\n"
        f"- Time: {mod.ti}\n"
        f"- Status: {status}\n"
        f"- Points: {mod.points[uid]:.0f}\n"
        f"- Perceived infection risk: {mod.percieved_infection_risk[uid]:.2f}\n"
        f"- Perceived health severity: {mod.percieved_health_severity[uid]:.2f}\n"
        f"- Quarantine self-efficacy: {mod.quarantine_self_efficacy[uid]:.2f}\n"
        f"- Quarantine response efficacy: {mod.quarantine_response_efficacy[uid]:.2f}\n"
        f"\n"
        f"Use this framework to guide your decision. Should you quarantine this round? Reply with only 'yes' or 'no'."
    )

CHOICE_TO_SCORE = {c: i + 1 for i, c in enumerate(["a", "b", "c", "d", "e", "f"])}

QUESTION_TO_STATE = {
    35: "percieved_infection_risk",
    41: "percieved_infection_risk",

    36: "percieved_health_severity",

    37: "quarantine_self_efficacy",
    43: "quarantine_self_efficacy",

    38: "quarantine_response_efficacy",
    44: "quarantine_response_efficacy",
}

CORE_STATES = [
    "percieved_infection_risk",
    "percieved_health_severity",
    "quarantine_self_efficacy",
    "quarantine_response_efficacy",
]

# TODO: we can update the beliefs with the data we have from the pre survey.
def build_pregame_beliefs(
    answers_path: str,
    user_id_map: dict,
    default_value: float = 3.0,
) -> pd.DataFrame:
    """
    Return a user-indexed dataframe with one column per core belief/state.
    """

    answers = pd.read_csv(answers_path)

    answers = answers[answers["survey_id"].isin([3, 4])].copy()
    answers["score"] = (
        answers["value"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(CHOICE_TO_SCORE)
    )

    answers["dimension"] = answers["question_id"].map(QUESTION_TO_STATE)
    answers = answers.dropna(subset=["score", "dimension"])
    beliefs = (
        answers
        .groupby(["user_id", "dimension"])["score"]
        .mean()
        .unstack()
    )

    beliefs = beliefs.reindex(columns=CORE_STATES)

    # Fill missing values with the column median, then default_value if needed
    for col in CORE_STATES:
        if col in beliefs.columns:
            beliefs[col] = beliefs[col].fillna(beliefs[col].median())
        else:
            beliefs[col] = np.nan

    beliefs = beliefs.fillna(default_value)

    # Remap user_id -> sim uid
    beliefs = beliefs.reset_index()
    beliefs["uid"] = beliefs["user_id"].map(user_id_map)
    beliefs = beliefs.dropna(subset=["uid"]).copy()
    beliefs["uid"] = beliefs["uid"].astype(int)

    beliefs_final = beliefs.set_index("uid")[CORE_STATES]
    return beliefs.set_index("uid")[CORE_STATES]


def init_beliefs_from_survey(mod, answers_path: str, user_id_map: dict):
    beliefs = build_pregame_beliefs(answers_path, user_id_map)
    n = len(mod.sim.people)


    # Overwrite with survey-derived beliefs
    for uid, row in beliefs.iterrows():
        if uid >= n:
            continue
        
        mod.percieved_infection_risk[uid] = float(row["percieved_infection_risk"])
        mod.percieved_health_severity[uid] = float(row["percieved_health_severity"])
        mod.quarantine_self_efficacy[uid] = float(row["quarantine_self_efficacy"])
        mod.quarantine_response_efficacy[uid] = float(row["quarantine_response_efficacy"])

def make_intervention(high_reward, agent_uids, name, model, api_key, id_map):
    return ss.LLMIntervention(
        low_reward    = 5,
        high_reward   = high_reward,
        agent_uids    = agent_uids,
        model         = model,
        api_key       = api_key,
        interval      = 1,
        decision_hour = 9.5,      # 09:30 AM; respected when dt < 1 day
        build_prompt  = default_agent_prompt,
        init_beliefs  = lambda mod: init_beliefs_from_survey(
            mod,
            answers_path="data_ingestion/survey-answers.csv",
            user_id_map=id_map,
        ),
        verbose       = True,
        name          = name,
    )

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
        interval (int): Timesteps between decision rounds (daily or coarser dt). Default 1.
        decision_hour (float): Hour of day to make decisions for sub-daily sims (9.5 = 09:30). Default 9.5.
        build_prompt (callable): Optional ``fn(mod, uid, disease) -> str`` that
            builds the per-agent prompt. Defaults to the built-in HBM prompt.
        init_beliefs (callable): Optional ``fn(mod) -> None`` that populates per-agent
            HBM belief states in-place. Defaults to ``default_init_beliefs``.
        max_tokens (int): Max LLM response tokens. Default 2000.
        timeout (int): HTTP request timeout in seconds. Default 30.
        verbose (bool): Print prompts and responses. Default False.
    """

    def __init__(self, low_reward=5, high_reward=10,
                 model=None, api_key=None, interval=1, decision_hour=9.5,
                 build_prompt=None, init_beliefs=None,
                 max_tokens=2000, timeout=30, verbose=False,
                 agent_uids=None, **kwargs):
        super().__init__(**kwargs)

        self.low_reward       = low_reward
        self.high_reward      = high_reward
        self.model            = model
        self.api_key          = api_key
        self.interval         = interval
        self.decision_hour    = decision_hour   # Hour of day to make decisions (9.5 = 09:30)
        self.build_prompt_fn  = build_prompt
        self.init_beliefs_fn  = init_beliefs
        self.max_tokens       = max_tokens
        self.timeout          = timeout
        self.verbose          = verbose
        self.agent_uids       = np.asarray(agent_uids, dtype=int) if agent_uids is not None else None

        self._last_decision_date = None  # Track last day decisions were made (sub-daily sims)

        # Per-agent states
        self.define_states(
            ss.BoolState('quarantined',   label='Quarantined'),
            ss.FloatArr('percieved_infection_risk', default=3.0, label='HBM perceived susceptibility (1-6)'),
            ss.FloatArr('percieved_health_severity',       default=3.0, label='HBM perceived severity (1-6)'),
            ss.FloatArr('quarantine_self_efficacy',  default=3.0, label='HBM perceived self-efficacy (1-6)'),
            ss.FloatArr('quarantine_response_efficacy',       default=3.0, label='HBM perceived response-efficacy (1-6)'),
            ss.FloatArr('points',         default=0.0, label='Accumulated game points'),
            ss.FloatArr('n_quarantine_steps', default=0.0, label='Number of steps agent quarantined'),
        )

        self.log              = []   # Per-step sc.objdict: {t, ti, n_agents, n_quarantined, error}
        self.decision_log     = []   # Per-agent per-day: {date, uid, quarantined}
        self.agent_summary    = None  # Populated by finalize()
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
        if hasattr(disease, 'recovered') and disease.recovered[uid]:
            return 'recovered'
        if hasattr(disease, 'exposed') and disease.exposed[uid]:
            return 'exposed'
        if hasattr(disease, 'dead') and disease.dead[uid]:
            return 'dead'
        return 'susceptible'

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

    def _is_decision_time(self):
        """
        Return True if a decision round should run this timestep.

        With daily (or coarser) resolution the ``interval`` parameter governs
        frequency exactly as before.  With sub-daily resolution the intervention
        fires exactly once per calendar day, on the first timestep whose
        wall-clock hour is >= ``self.decision_hour`` (default 9.5 = 09:30).
        """
        dt_days = float(self.sim.t.dt)
        if dt_days >= 1.0:
            # Daily or coarser: use interval as usual
            return self.ti % self.interval == 0

        # Sub-daily: fire once per day at decision_hour
        now  = pd.Timestamp(self.now)
        hour = now.hour + now.minute / 60.0 + now.second / 3600.0
        if hour < self.decision_hour:
            return False
        today = now.date()
        if self._last_decision_date == today:
            return False
        self._last_decision_date = today
        return True

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
        1. Ask the LLM whether they quarantine (once per day at decision_hour).
        2. Zero transmission for quarantined agents.
        3. Award points.
        """
        if not self._is_decision_time():
            return

        all_uids = self.sim.people.auids
        if self.agent_uids is not None:
            uids = ss.uids(np.intersect1d(all_uids, self.agent_uids))
        else:
            uids = all_uids
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

        # Record per-agent decision for this day
        current_date = str(self.now)
        for uid, did_quarantine in decisions.items():
            self.decision_log.append(
                dict(
                    date=current_date, 
                    uid=uid, 
                    quarantined=did_quarantine, 
                    status=self._agent_status(uid, disease), 
                    points=self.points[uid]
                )
            )

        if len(q_list):
            self.quarantined[q_list] = True
            self._zero_transmission(q_list, disease)
            self.points[q_list]             += self.low_reward
            self.n_quarantine_steps[q_list] += 1

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
            print(f'[LLMIntervention] {n_calls} steps | overall quarantine rate: {rate:.1%} | {n_errors} errors')

        self._build_agent_summary()
        self.decision_log = pd.DataFrame(self.decision_log)  # date | uid | quarantined
        return

    def _build_agent_summary(self):
        """
        Build a per-agent summary DataFrame stored as ``self.agent_summary``.

        Columns: uid, status, points, n_quarantine_steps, quarantine_rate,
                 susceptibility, severity, self_efficacy, benefits.
        """
        disease    = self._target_disease()
        n_decisions = max(len(self.log), 1)
        rows = []
        for uid in self.sim.people.auids:
            uid_int = int(uid)
            rows.append(dict(
                uid                = uid_int,
                status             = self._agent_status(uid, disease),
                points             = float(self.points[uid]),
                n_quarantine_steps = int(self.n_quarantine_steps[uid]),
                quarantine_rate    = float(self.n_quarantine_steps[uid]) / n_decisions,
                susceptibility     = float(self.percieved_infection_risk[uid]),
                severity           = float(self.percieved_health_severity[uid]),
                self_efficacy      = float(self.quarantine_self_efficacy[uid]),
                benefits           = float(self.quarantine_response_efficacy[uid]),
            ))
        self.agent_summary = pd.DataFrame(rows).set_index('uid')
        return
