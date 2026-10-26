# Backlog from the untracked design notes

On 2026-10-25, 27 untracked files and folders in `v4_design/` (October 20–24: simplification surveys, JAX and performance work, the October 22 assessments, port reviews, killer features, docs feedback, and spec gaps) were checked against the code at `1b4b0f51` plus the working tree, mostly by reading rather than running. Nearly all of it is done, superseded, or already in [deferred.md](../deferred.md). This page lists what is left, judged against the [constitution](../constitution.md): bugs and simplifications first, few new features, and nothing that adds concepts for one model. Once the steps under "Before deleting" are done, every untracked file can go.

[CK: i agree with all except where otherwise noted]

Details, with a verdict for each source file and the evidence for each item:

- [backlog/oct20_simplify.md](backlog/oct20_simplify.md): the October 20 simplification surveys (A–H) and their scripts
- [backlog/perf_jax.md](backlog/perf_jax.md): Numba against JAX, the JAX track, the step clock, and the chain binomial's bias, plus measured findings worth keeping
- [backlog/oct22_reviews.md](backlog/oct22_reviews.md): the Claude, Codex, and skeptic assessments, with the status of deferred T1–T13
- [backlog/ports.md](backlog/ports.md): the FPsim, STIsim, HPVsim, Covasim, and tbsim port reviews, the stale lists, and rstarsim, with what is pending in each other repo
- [backlog/features_docs.md](backlog/features_docs.md): killer features, docs feedback, and He et al. 2010
- [backlog/decisions.md](backlog/decisions.md): 16 built decisions waiting for a yes or no, copied from `plan_feedback3.md` and `docs_proposals.md`

## Do: bugs

Each is small (S) and was reproduced on 2026-10-25.

- **`Disease.infect(uids)` is shadowed by `Infection.infect()`** (ML3): `ssl.AgentSIR().infect(ss.uids([1, 2]))` raises `TypeError`. The method needs a new name (`infect_now()`?), which is Cliff's call. [oct22_reviews](backlog/oct22_reviews.md) [CK: i don't get why Disease.infect() and Infection.infect() are doing different things! seems like there's a deeper issue here]
- **A network with no contacts crashes**: `ss.RandomNet(n_contacts=0)` raises `ZeroDivisionError` (`networks.py:950`). Likewise `ss.Sim(n_agents=0, ...)` (`parameters.py:454`): zero should mean none, or give a clear error. [ports](backlog/ports.md), [features_docs](backlog/features_docs.md)
- **A one-destination split is misread** (ST48): three `ss.Transition(I, [F], p=[1.0], rate=...)` from one state raise `BadBranchSplit: ... sum to 3`. [ports](backlog/ports.md)
- **`ss.by()` with a NumPy array**: `ss.perday(ss.by(herd=np.array(...)))` raises `ShapeError`, though a list works. [features_docs](backlog/features_docs.md)
- **A distribution in a rate after `@`**: `S >> I @ ss.gamma(...)*I` raises `WrongDimension`, though the keyword form works. [features_docs](backlog/features_docs.md)
- **Several `Mobility`s on one dimension**: a list gives a bare `TypeError`; it should be a clear error. [features_docs](backlog/features_docs.md)

## Do: simplifications and small fixes

- **Delete the refusals that the trial layout already makes**: `init_capabilities()` has 29 `'no'` records (`choose.py:660–850`), and some of them repeat refusals the layout makes itself. Audit them and delete the duplicates. S–M. [oct22_reviews](backlog/oct22_reviews.md)
- **Ask for lactation explicitly**: `ss.Pregnancy(lactation='auto')` silently adds `ss.Lactation` when a `BreastfeedingNet` is present, which is a hidden rule. S, if Cliff agrees. [oct20_simplify](backlog/oct20_simplify.md) [CK: ok]
- **One JAX run cache**: crossed ODE runs keep their own cache on the crossing, where single engines share the module-level `runs` dict (`gradients.py:568–599`). S. [perf_jax](backlog/perf_jax.md)
- **`describe()` under an ODE** shows step-timing lines meant for agents (`presentation.py:328`), and the `data=` docstring says "used for plotting only" (`sim.py:98`). S each. [oct22_reviews](backlog/oct22_reviews.md)
- **Small output fixes**: dates print as `2000.01.07` (`options.date_sep`) [CK: hm, i like that format :) but i agree - is much more common than . ok, let's change ss.date() and elsewhere to use - instead of ., if this is is a one-line change to options.date_sep then let's make - the default instead of .]; `describe()` before a run prints the run's start line; notebooks print `Figure(640x480)` [CK: think this is a jupyter limitation no?] (check rstarsim first); there is no reciprocity note as agents. S each. [features_docs](backlog/features_docs.md)
- **Write up Numba's Poisson rounding bug** for Numba (`int()` where NumPy floors; still in 0.68.0). Starsim's own draws are not affected. S. [perf_jax](backlog/perf_jax.md)

## Do: worth a look

- **Density-dependent transmission (`beta*I.n`)** isn't recognized as a transmission: there is no R0, and `describe()` counts R as infected. It is correctness [CK: what?] on a textbook model. M. [features_docs](backlog/features_docs.md)
- **`describe()` should describe the whole model**: parameter bounds, each dimension's shares and adjustments, outcome definitions, and distributions as they were given. Also, `diagram()` should draw in a notebook, with values as labels. S–M. These are `[PROPOSAL]` callouts on the docs pages. [features_docs](backlog/features_docs.md)
- **tbsim's "every solver" example**: 299 people with active TB as an ODE, 400 under the chain binomial, and 469 as agents. It may be the settings, but a 57% gap needs explaining. S to diagnose. [ports](backlog/ports.md)
- **Profile many small stratified stochastic runs**: 6.7 µs a step against odin's 1.4 µs. Also measure LASER against threaded Starsim on the full machine. S each. [features_docs](backlog/features_docs.md) [CK: as with the he et al model, can also explore porting laser more explicitly -- reusing more of the original optimized/jitted code, for example]
- **Common random numbers showed no variance reduction** in a quick measles check (SD of infections averted 31 with them, 22 without). This goes with deferred.md's "CRN for declared diseases as agents". [features_docs](backlog/features_docs.md) [CK: we wouldn't expect CRN to reduce variance within a simulation, only between simulations]
- **The migration guide** should say that coarse steps differ from v3, which let people pass through several states in one step (tbsim: latent TB +36% at `dt=0.1`). S. [ports](backlog/ports.md)

## For Cliff to decide

- The 16 built decisions in [backlog/decisions.md](backlog/decisions.md). [CK: TODO]
- ML3's new name (above), and whether to ask for lactation explicitly (above).
- Birth rates per person (`ss.birth`) against per 1,000 for a rate above 1 (`ss.Births`): two conventions, one of them guessed from the number's size. M. [CK: i think this is fine]
- A clearer name for `res.summarize()`. [CK: i think this ok, but open to suggestions]
- What is left from the killer-features builds: a parameters table, spending and coverage as results, seeds for posterior draws, and seeds in a `pars=` sweep. [features_docs](backlog/features_docs.md) [CK: TODO]
- Two governance points for the release: a stability policy for the declared language and the model spec, and review of every PR by someone other than its author. [oct22_reviews](backlog/oct22_reviews.md) [CK: no to both]

## Defer (to add to deferred.md)

These are worth doing later. Each is at least M, or waits for a reason. The details are in the linked pages.

- Moving the declared build steps out of `Sim` (2,621 lines) and `choose_solver()` into `choose.py` and `presentation.py`: a pure move, for the next time `sim.py` is reworked ([oct20_simplify](backlog/oct20_simplify.md), [oct22_reviews](backlog/oct22_reviews.md)). [CK: you can do that now]
- One table of moves in the kernels, in place of four mechanisms, needing a perf A/B first ([oct20_simplify](backlog/oct20_simplify.md)). [CK: you can do this too]
- One internal form of an event, as part of T3 ([oct20_simplify](backlog/oct20_simplify.md)). [CK: can do this if it's not larger than M]
- `moments()` on `ss.Dist` in place of checks on class names; lazy `scipy.stats` (0.37 s of a 1.0 s import) ([oct22_reviews](backlog/oct22_reviews.md)). [CK: you can do this, at least the lazy scipy.stats]
- Not removing the dead from `dur=0` networks (21 of 72 ms a step at 1M agents) ([perf_jax](backlog/perf_jax.md)). [CK: can do]
- Filtered trajectories, and projecting from the filter's state; movement bans by place status (into B9); commuting that depends on state; several `Mobility`s as a list; the larger docs proposals ([features_docs](backlog/features_docs.md)). [CK: agree, put in deferred]
- Five untraced differences from v3 that plan_b I2 meant to record (Covasim VE, STIsim HIV-6 and #35, HPVsim's age-gap SD), and HP58 ([ports](backlog/ports.md)). [CK: yes can put this in deferred]

Also, deferred.md's "rest of the gap to Ghana measles" is out of date: JAX now runs Ghana in 0.58 s, against 1.0 s for the hand-written NumPy loop, so the gap applies only to the Numba kernels ([perf_jax](backlog/perf_jax.md)). [CK: agree, can remove]

## Other repos

The pending changes for STIsim, HPVsim, FPsim, Covasim, tbsim, rstarsim, and Sciris are listed in [backlog/ports.md](backlog/ports.md), under "Pending in other repos". That list replaces `oct22/stale_other_repos.md`, which was the only write-up. These are to be written up for each repo, not made from here. The `v4` branches of STIsim, HPVsim, FPsim, and Covasim still equal `main` (T13).

## Before deleting

1. **Carry over the defer list** into deferred.md (the section above).
2. **Fix the links.** Tracked files link to the untracked ones:
   - Code comments: `kernels.py:154`, `perflib/_draws.py:13`, `perflib/_memory.py:7`, and `perflib/_passes.py:10` cite `oct22/jax/*.md`.
   - The spec: `design/11-random.md:21`, `design/12-backends.md:337`, and `design/10-execution.md:118`.
   - The tracked plans: `oct21/jax_plan.md`, `oct21/numba_vs_jax.md`, `oct22/plan.md`, `oct22/plan_b.md`, `oct22/measure.md:57`, and `oct24/spec_gaps.md:75`.
   - `deferred.md`, and `README.md:10`.

   Point the `spec_gaps_response.md` links at `docs/tutorials/g8_extending.qmd`, and the rest at the backlog pages or plain text. The full lists are at the end of each detail page.
3. **Decide on the five October 22 port reviews.** `/v4.review` compares each run with the latest earlier report (`ai/README.md:36`), so deleting them leaves the next run with no baseline. Either track `oct22/reviews/` or accept that. [CK: i don't think we need a baseline -- no baggage remember!]
4. **Optionally track the measurement notes** that the code and the spec cite: `oct22/jax/{README,draws,agents,jit}.md` and `oct24/step_bias.md` (TC1's evidence). Their key numbers are also in [backlog/perf_jax.md](backlog/perf_jax.md), so deleting them loses only the detail.

These links were broken before this review: deferred.md writes its links as `../oct22/...` (from when it was in `admin/`), and `README.md:11` points to `admin/`, which no longer exists. `oct22/worklogs_c/` is also gone. Nothing else needs keeping: the scripts are superseded by `perflib/` and the tests, and the He 2010 data and code are already in tutorial G8.
