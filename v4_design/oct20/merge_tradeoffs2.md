# Merging v4-rs-port into v4-rs-ck3: tradeoffs

Written 2026-10-20 while merging `origin/v4-rs-port` (tip `d5239361`) into `v4-rs-ck3` (`9ed8bb27`). The merge is resolved and staged (`git add -u`), not committed. The first merge's choices are in [../oct19/merge_tradeoffs.md](../oct19/merge_tradeoffs.md); this file covers only what is new.

## What the two branches had

Both sides had already merged the same two parents: v4 at `92902b20` (Cliff's delivery redesign) and Robyn's replay of v4-rs at `60519d39`. They did it independently:

- **Cliff's merge, `77362e62`**, is the base of `v4-rs-ck3`. This session's work (`9ed8bb27`) is on top of it.
- **Robyn's merge, `0d709392`**, is followed by 55 commits on her branch. Some re-port v4-rs features onto v4's deliveries: treatment `on_fail=`, products over several diseases, memory, and `target=`/`remove=`. The rest are new: the declared `ss.Pregnancy` with fetuses, lactation and burn-in; `ss.attr`/`ss.func`; `ss.birth(per=)`; `ss.spread` (B5); triggers as compartments (B2); synchronized flows (B3); save/load; and migration renames.

So the merge has two layers:

1. The two merges of the same parents, which resolved the same conflicts differently.
2. Robyn's work since hers.

## How it was merged

- **The base: Cliff's resolutions.** Each file is a three-way merge: ours = `HEAD`, base = Robyn's merge `0d709392`, theirs = her tip. That keeps Cliff's merge wherever the two merges differ, and adds everything Robyn did after hers.
- **Her delivery re-ports are left out.** Eight of her commits re-port features that Cliff's merge had already ported, in the spec's spelling:
  - `5ac0fb8e` (numbers are people)
  - `bc650909` (`on_fail=`)
  - `bade156e` (several diseases)
  - `878e0b31` (memory)
  - `d2e5048b` (`target=`/`remove=`)
  - `afa0934b` (`test_rescreen`)
  - `48d7dd9e` (TL52 in `tails()`)
  - `ae12efcc` (HPV acceptance xfails)

  These were reverted from her tip before the merge. Each is weighed below; in each case Cliff's version is kept.
- **Her other small delivery commits are brought over:**
  - `81549280`: coverage as a function giving a typed rate.
  - `26283a09`: cascade steps given in `pars=`.
  - `a66434e6`: `undriven=True`, as `tr.intended`.
  - `9dbfbb10`: `ss.state('mod.X').enter/.exit`.
  - `eb4f67e4`: save and load mid-run.
- **The pregnancy rename is Robyn's** (see below). Before the merge, Cliff's side was renamed to match her naming, so the rename itself causes no conflicts.
- **About 20 hunks were resolved by hand:** listed in the last section.

## Decisions

### `ss.Pregnancy`: the declared module (Robyn's naming)

- **Cliff's merge:** `ss.Pregnancy` stays the module written as code, and the declared one is `ss.pregnancy.Pregnancy`. The first tradeoffs file gives the reason: the declared module lacked unborn agents, the maternal networks and burn-in, "the rename can be made when the declared module has the rest".
- **Robyn's:** `ss.Pregnancy` is declared, and the coded module is `ss.LegacyPregnancy`. Since her merge, the declared module has the rest as agents:
  - fetuses as agents (`fetus='auto'`), `ss.mother()` and `ss.child`;
  - `ss.Lactation`, and `BreastfeedingNet`/`MaternalNet` on it;
  - burn-in, the tfr and asfr results;
  - Starsim 3 attributes, results and callbacks.
- **Pros of Robyn's:**
  - The condition Cliff set is met for agents.
  - About 20 of her commits are written against it, and renaming them back would touch every one.
  - It is the v4 direction: the declared module is the main one, and the coded one is kept for compatibility.
- **Cons:**
  - As compartments, fetal levels, prenatal transmission, lactation and breastfeeding are still not built (B7, as strict xfails).
  - Burn-in as compartments starts with nobody pregnant, with a note (B4/B5).
  - Code that relied on `ss.Pregnancy` meaning the coded module now gets the declared one. In this repo that code (the MNCH library, the household network, the v3-style tests) was renamed to `ss.LegacyPregnancy`. Downstream code would need `python -m starsim.migrate` to learn the rename.
- **Recommendation:** Robyn's naming, as merged. Add a rule to `starsim.migrate` saying when `ss.Pregnancy` should become `ss.LegacyPregnancy` (code that reads unborn agents or the coded module's arrays). The changelog has a line for the rename.

### Treatment failure: a triggered transition, not `on_fail=` (Cliff's)

- **Robyn:** `ss.Treatment(moves=ng.cure, on_fail=ng.fail)`, where `on_fail=` can be a transition, a delivery, or a function.
- **Cliff:** the failure is a transition triggered by the outcome, `fail = ss.Transition(P + Y + C, A, on=drug.failed, p=p_fail)`, or a peer step `who=clinic.failed`. `on_fail=` raises a `TypeError` that gives the new spelling.
- **Pros of Cliff's:**
  - It matches the spec's rule of no nested follow-ups (care cascade decision 14).
  - It allows `p=` and a split at failure.
  - It works for `cure=` treatments too.
  - There is one cure code path.
  - As compartments, `cured`, `failed` and `unneeded` are all events. Robyn's `moves=` path fires only `failed` as compartments, so a step on `cured` silently gets nobody (found by reading the code, not run).
- **Cons of Cliff's:**
  - The product must exist before the disease, so `moves=` is given by name (`'ng.cure'`).
  - For a treatment of several diseases tracked jointly, its outcomes are not events as compartments: reading one raises. Worse, the check misses `on=kit.failed` on a transition, which then looks like a silent no-op.
- **Pros of Robyn's:** failure is written next to the treatment, and `failed` works across a crossing.
- **Cons of Robyn's:**
  - It nests a step inside a product.
  - `Sim._add_steps` rewires the delivery and renames it behind the user's back.
  - It works only with `moves=`, and failure is all-or-nothing.
- **Recommendation:** Cliff's, as merged. Still to do:
  - Bring over Robyn's firing of `failed` across a crossing (`force_part` and `cure_groups` in her `interventions.py`).
  - Extend the read check to transitions' `on=`.
  - Add tests for both, and for the `on_fail=` `TypeError`.

### Products and tests over several diseases (Cliff's)

- **What both sides share:** the core is identical: efficacy keyed by disease, `vx.held`, sensitivity keyed by states of several diseases with `combine=`, `ss.any()`, a delay per result, and the same draws.
- **Cliff's adds:**
  - `results=` (most severe first, `normal=None` last), against Robyn's `outcomes=` in the reverse order.
  - `disease=[...]` or an `ss.Strains` on a delivery, through `Product.key_by()` (HP7).
  - A guard against results named `ever`, `since`, `count` or `event`, which the memory properties would hide.
  - Diseases written as code.
- **Robyn's adds:** `positive` and `negative` on every test, including tests with several results. This is handy: a triage step works the same after a two-valued or a categorical test.
- **Recommendation:** Cliff's, as merged. Two follow-ups:
  - Whether to add `positive`/`negative` to tests with several results is Cliff's call. It needs a spec line and the clash check on those names.
  - `key_by()` changes the shared product in place, so one product given to a delivery with a list and to one with a single disease could be expanded twice. Copy it, or key it per delivery.

### Stock targets: `above=` rather than `remove=` (Cliff's)

Both came from the same v4-rs commit, so `ss.share`, `strata_targets` and `record_target` are nearly line for line the same.

- **Cliff's covers all of Robyn's:** `remove=tx.dropout` becomes `above=tx.dropout`, and `remove=True` becomes `above='stop'`.
- **Cliff's adds:**
  - a rate above the target, so the stock drifts down;
  - moving people back to the level a `move=` came from;
  - stopping the lowest priority first, and serving the gap in `priority=` order;
  - errors for `target=` with rounds, an event, or `deliver=`. Robyn's accepts these, and with `deliver=` the target is recorded but never acted on.
- **Robyn's adds:** only `of=` (count something other than the product's holders).
- **Tests:** Cliff's `test_target`, `test_target_stop` and `test_hiv_art` all pass. Robyn's `test_hiv_art` is a strict xfail.
- **Recommendation:** Cliff's, as merged. Restore `of=` only if a port needs it. Two bugs exist on both sides:
  - a target with `increment=` counts wrongly (it compares `('+', k)` with `==`);
  - when a target is binding, `lost_capacity` is too high.

  Separately, `care_cascade.md:350` says "a share of the eligible (below 1)", but neither side reads a bare number below 1 as a share; both use `ss.share`. Fix the doc.

### `priority=`, `queue=`, `ss.Resource` and doses (Cliff's)

- **Robyn ruled them out until a shared-resource design existed.** Her reasons: they overlap `ss.Resource`, and "doses have no v4 form". Both were true of the v4 she ported onto. Neither is true now: Cliff's side has `ss.Resource` with `uses=`, `priority=` (tiers or a score), `queue=` (with `waiting`, `waited`, `left` and `never_ran`, as agents and as compartments), and doses as levels of the product's dimension (`Vaccine(doses=)`, next dose by `vx.since`).
- **Robyn's earlier ports of these** used nearly the same algorithms (water-filling, a `waiting_` dimension, `ti_queued`), so nothing of hers is lost.
- **Tests:** Cliff has eight live tests. All of Robyn's are strict xfails in the v4-rs spelling (`ss.Capacity`, `on_positive=`, `dose=`, `ss.desc`).
- **Recommendation:** Cliff's, as merged.
  - Rewrite Robyn's three HPV acceptance tests (`test_hpv_vaccines`, `test_hpv_screen_treat`, `test_hpv_colposcopy`; left out with `ae12efcc`) in Cliff's spelling. They are the only tests of doses and queues across genotypes tracked jointly.
  - Robyn's `v4_design/admin/implementation.md` and `V4_PORT_PLAN.md` still list these as waiting on a design. They are her files, merged as she left them, so she should update them.

### Delivery memory and same-visit steps (Cliff's)

- **Cliff's has all of Robyn's `878e0b31`:** spells (`ss.Delivery.spells`), typed coverage keeping no record, `.ever`/`.since` in place of `repeat=`, steps triggered by a delivery in place of `then=`, and the crossed-treatment event fix.
- **Cliff's is cleaner:**
  - Memory is a condition in `who=` at construction (`who=sis.I & ~drug.ever`), where Robyn's tests assign `clinic.eligible` afterwards.
  - It raises errors that give the new spelling for `then=`, `repeat=`, `dose=`, `remove=` and `of=`.
  - People a step takes out of eligibility start a new spell in the same step.
- **Recommendation:** Cliff's, as merged; there is nothing to bring over.

### Numbers are people

The two sides are byte-identical (`ss.results.people_now`, `in_people()`, `test_people`), so there is nothing to choose. Neither tests `people_now = False`; add a test so the revert path keeps working until the switch is decided.

### Triggers as compartments: both, combined

- **Cliff's side** (this session's ST18 work) moves a trigger's people at the end of each step, in Python (`step_triggers()`). It covers triggers on conditions, on the module's own events, and `dur=0` with `where=`. Outcomes of deliveries move as they happen (`trigger_moves()`).
- **Robyn's B2 and B3:**
  - B2, triggers on states and strata: people pass straight through on arrival, in the kernels' layout (a track with no wait, masked by group). People who come to meet the trigger otherwise (aging, a transfer, another disease) are swept at each step. A trigger on another disease's state tracks the two jointly.
  - B3: `ss.on(other.tr)` is a synchronized flow, moved in the same flow as the other module's transition.
  - Triggers on the module's own events stay agents only.
- **Pros of Robyn's:** it matches agents' timing (no wait of a step), it handles other diseases and synchronized flows, and it is tested against agents.
- **Pros of Cliff's:** it handles the module's own events and outcome triggers, which hers does not.
- **Merged:**
  - Robyn's mechanism for conditions and synchronized flows.
  - Cliff's `step_triggers()` kept only for events of the module's own transitions (`event_triggers`).
  - Outcome triggers skipped in the layout, so they still move through `trigger_moves()`.
  - In `choose.py`, one row for each case:

    | Trigger | Runs as compartments? |
    |---|---|
    | An outcome | yes |
    | A synchronized flow | yes |
    | An own-module event | yes, at the end of each step |
    | Another module's or a coded event | no |
    | A function, a child, or entering or leaving a state | no |
    | A condition | yes, passed straight through |

  Cliff's old row sent a trigger on another module's state to agents; that case now runs as compartments (Robyn's).
- **Two fixes made where they meet:**
  - Robyn's sweep runs at the start of each step. It now also runs at the end, after the step's flows, so that someone a transfer moves into a trigger's condition moves in the same step, as agents do. Without it, `test_immediate` (`dur=0` with `where=`) lagged a step under the ODE: 0.150 left against e^-2 = 0.135.
  - Cliff's test for "the outcome of a delivery" (`tr.trigger.outcome`) matched any event leaf, so a trigger on an event that a module written as code records (`ss.on(preg.conception)` with `ss.LegacyPregnancy`) was taken for an outcome and never fired (`test_legacy`). It now matches only events of deliveries and products.
- **One test changed:** Robyn's `test_on_event` expected an event of the module's own transition to raise as compartments. It now runs, and the test checks that its flow equals the event's.
- **Recommendation:** keep the combination.

### An immediate remainder (`exits.rest`): kept

- **Robyn's merge dropped v4-rs's `rest`:** a transition with no rate, duration or `p=` takes what the branches leave over.
- **Cliff's merge kept it,** and this session's `dur=0` (care cascade step 5) is built on it.
- **Recommendation:** keep it, as merged.

### Small resolutions

- `test_route_modifiers`: Robyn's 12 agent seeds and `rtol=0.1` (her `17083547`), against 8 and 0.15.
- `init_prev` for a member of an `ss.Strains` family (Cliff's) and `ss.spread` seeding (Robyn's B5): both kept, the family's share first.
- The standard prevalence as a ratio by group (Cliff's ST20) and denominators with fewer dimensions (Robyn's `94bbe0ee`): both kept.
- `Agents._leaving` is `None` when no one is leaving (Cliff's), alongside Robyn's `_uids` and `_for_uids`. A derived attribute's level in `by_agent()` is Robyn's.
- `test_spec`'s list of public names takes the union.

## Where the two merges differ, kept as Cliff's

This layer takes Cliff's merge wherever the two merges of the same parents differ. The first tradeoffs file gives his reasons. The differences that change behavior:

| Where | Cliff's merge (kept) | Robyn's merge | Note |
|---|---|---|---|
| `Strains.protection()` | none before the first step (0 as compartments) | worked out when first asked for, e.g. by a dwell drawn as people are placed | Robyn's handles a dwell time protected by a family at placement. Worth bringing over if a test needs it |
| `ss.conditions.Event` | op `'move'`, a leaf of its own | op `'event'`, shared with deliveries' events | Robyn's `9dbfbb10` is merged on top of Cliff's form |
| A dwell time with a distribution for each level | `by_dist()` (the port's, from the first merge) | v4's `dwell_by()` | As the first file says |
| `ss.stay` in a split | a `stay=` argument of `branch_shares()` | the rest worked out before | Same numbers |
| A parameter named after its slot in a subclass | renamed again (`slot_named`) | not | Cliff's fixes a `DuplicateDeclaration` |
| Branch remainder | `exits.rest` | dropped | See above |

## Not done

- **Merge-level follow-ups:**
  - Robyn's `failed` event across a crossing.
  - Robyn's HPV acceptance tests, in Cliff's spelling.
  - `positive`/`negative` on tests with several results (Cliff's call).
  - `of=`.
  - The `increment=` and `lost_capacity` bugs.
  - `key_by()` changing the product in place.
  - A test with `people_now = False`.
- **Docs and records:**
  - The v4 docs (tutorials and user guide) that build a Starsim 3 pregnancy now say `ss.LegacyPregnancy`, as on Robyn's branch. They are not otherwise rewritten, since the docs wait for the final implementation.
- **The per-act round 2** (the partnership approximation with several partners at a time) is still only in its worktree, `scratchpad/wt_peract`. It was not part of this merge.

## Tests

The first full run gave 3 failed, 577 passed, 11 skipped and 18 xfailed: `test_legacy`, `test_immediate` and `test_on_event`, all three from where the two trigger designs meet, fixed as described above.

After the fixes: 580 passed, 11 skipped and 18 xfailed (`pytest test_*.py -n 4`). The skips are the same as before the merge. The xfails are Robyn's strict xfails for her Phase B (B1, B3, B6 and B7: fetal levels, lactation and clocks as compartments, and the like) and v4's own.
