# Starsim v4 conversion review — shared brief

Goal: port Robyn's real models (STIsim, HPVsim) to the Starsim v4 API to find out what looks ugly, what is missing, and what breaks once you leave SIR. This is a stress test of the v4 *design*, not a production port.

## Sources
- v4 code: `/Users/robynstuart/gf/starsim/starsim/` (branch `v4`, editable install in conda env `starsim`; `conda run -n starsim python ...`). DO NOT modify, commit, or switch branches in this repo.
- v4 design spec: `/Users/robynstuart/gf/starsim/v4_design/design/` — start with `00-overview.md`, `03-states-and-transitions.md`, `07-effects.md`, `08-interventions.md`; also `05-dimensions.md` (strains, stratification), `06-routes.md` (networks), `09-results.md`, `19-open-questions.md`, `admin/implementation.md` (what is actually implemented), `admin/questions-oct12.md`.
- Working v4 usage examples (the ground truth for what runs): `/Users/robynstuart/gf/starsim/tests/test_transitions.py`, `test_deliveries.py`, `test_dimensions.py`, `test_networks.py`. Grep `starsim/transitions.py`, `interventions.py`, `products.py`, `diseases.py`, `networks.py` for signatures.
- STIsim v3 (current, on starsim 3): `/Users/robynstuart/gf/stisim/stisim/`
- HPVsim v3 (current, on starsim 3): `/Users/robynstuart/gf/hpvsim/hpvsim/`
- Do not modify stisim or hpvsim either.

## What to produce (in your assigned subfolder of `/Users/robynstuart/gf/v4_review/`)
1. `port.py` (or several files): the model written in **idiomatic v4**, using the declarative forms (`ss.states`, `>>`/`@`, `ss.transmission`/`ss.progression`, `ss.modify`, `ss.Test`/`ss.Treatment`/`ss.screen`/`ss.treat`, `stratify`, `capacity=` etc.) wherever they can express the biology, and falling back to `step()` code only where they cannot. Aim for faithful biology, not a toy. It MUST run: `conda run -n starsim python port.py` with a small sim (e.g. 2–5k agents, a few decades or less). Print a few key results to show it does something sensible. If a feature is impossible or crashes, keep the attempted code (commented, or in a `try`) and record the error.
2. `NOTES.md`: the review. For each feature of the original model:
   - the v3 code (brief, file:line) and the v4 code side by side;
   - whether it was expressible declaratively, partially, or only via `step()`/subclass;
   - an ugliness rating (clean / awkward / ugly / impossible) and *why* (concretely: what you had to write that a modeler shouldn't have to);
   - bugs or spec/implementation mismatches found (with the error text);
   - a concrete API suggestion where something is ugly.
   End with a short ranked list: top problems for this model, and what was surprisingly nice.
   Be honest and specific. Don't pad. Terse style, no filler.

## Cross-cutting questions every port should answer where relevant
- How do latency, recurrence/reactivation, clearance, reinfection, and treatment generalize beyond `I -> R`? E.g. treatment that moves people from *any* infected state, with failure; latent states that reactivate; clearance competing with progression.
- Count-based delivery: "the program can treat/test 100 people per week" (`capacity=`), vs rate/coverage. Does it work, with queues/backlogs carrying over? With priority?
- Per-agent continuous state (CD4, viral load, dysplasia severity, time since infection) — how is it written, and does it force everything into `step()`?
- Sex-specific / age-specific parameters (e.g. different durations or symptom probabilities for men/women).
- Multiple diseases + connectors (`ss.modify` on another disease's `rel_sus`/`rel_trans`).
- Results: are the results you'd want (new infections by sex/age, prevalence, diagnoses, treatments) automatic or a pain?
- Does mixing declarative transitions with a custom `step()` work cleanly in one class?
- Line counts: v3 vs v4 for comparable functionality.

Use the time units of the originals (STIsim/HPVsim are annual or monthly timesteps). Keep runs small and fast. Don't run anything big.
