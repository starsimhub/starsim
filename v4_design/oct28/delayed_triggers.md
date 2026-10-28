# Delayed triggers without polling

*Status: proposal, 2026-10-28. Branch `rs-decl-perf` (PR #1492).*

## Summary

A transition triggered by an event after a delay (`ss.Transition(X, Y, on=event, dur=d)`) is checked by polling: every time the same-step cascade reaches it, `triggered()` asks which events are new, and `delayed()` scans the whole population for pending times. In a declared STIsim gonorrhea this costs 70 ms of a 0.80 s run. 26 ms of that is polls that move nobody, and much of the rest is whole-population scans.

The proposal keeps each pending time where it is now (`ti_after_<tr>`), but finds who is due from a queue keyed by step, and finds who has left the source from a list of those who left since the last poll. Nothing is drawn differently, so results are bit-for-bit identical. Expected saving: about 35 ms at 5,000 agents (about 5% of the run), more at larger populations, since the scans it removes are over every agent.

This is the first of three changes to bring declared diseases run as agents close to coded ones (see [Context](#context)). On its own it closes about a sixth of the gap for STIsim's gonorrhea.

## Context

STIsim's diseases were ported to v4 in the declared form and then rewritten as code, because the declared form was 20–50% slower (Cliff, 2026-10-28). Measured on `v4` (1ce69a13), 5,000 agents, 20 years at monthly steps, `StructuredSexual`, best of 5:

| | Run | Without the network (0.35 s) |
|---|---|---|
| Coded gonorrhea (stisim `f4f8a72`) | 0.58 s | 0.24 s |
| Declared gonorrhea (stisim `7fab784`) | 0.80 s | 0.46 s |

The random draws are the same in number (about 3,000) and cost (about 11 µs each) in both. The difference is per-batch Python in the agent engine:

| Part of the declared run | Time | Calls |
|---|---|---|
| `finish_moves()` | 208 ms | 481 |
| ↳ `send()` (→ `enter()` → `schedule()`) | 109 ms | 2,060 |
| ↳ `triggered()` | 70 ms | 1,924 |
| `transmit()` | 107 ms | 480 |

Three changes would address it, in order of how contained they are:

1. **This document:** delayed triggers without polling.
2. **Batching arrivals in `finish_moves()`:** enter and schedule everyone arriving in a state during a sweep together, not once per transition and destination (about 5 agents per `send()` today).
3. **A fast path for simple modules:** for a module with no dimensions, strains or per-agent modifiers, work out the per-state checks once at build time.

## How delayed triggers work now

`SEIS` has four: `seek_care` and `clear_symp` (after `symptoms`), `pid_onset` (after `onset`), and `seek_care_pid` (after `pid_onset`), each with a `dur=`.

For each delayed trigger `tr`, the engine keeps `ti_after_<tr>` (an `ss.TimeArr`, NaN if nothing is pending) and a random stream `after_<tr>`. `finish_moves()` takes the transitions in sweep order, once per sweep. It runs twice a step (from `step_state()`, and after transmission in `step()`) and again whenever code moves people (`force()`). Each time it reaches `tr`, it calls `triggered([tr])`:

1. **New events:** `new_events(tr)` gives those the event has happened to since `tr` last looked (with a shortcut when its transition has recorded nothing since), kept to those in the source.
2. **`delayed()`:**
   - **Forget:** `pending = np.flatnonzero(~np.isnan(ti_after))`, a scan of every slot. Pending agents no longer in the source have their time cleared.
   - **Draw:** for those with new events and no pending time, draw from `after_<tr>` in the current round, and set `ti_after = ti + whole_steps(dur)`. A pending time is never redrawn.
   - **Due:** those pending with `ti_after <= ti`, in UID order, less those `taken` by an earlier trigger in this call (who keep their time). Their time is cleared.
3. **`where=`** is applied to those due; anyone it drops has lost their time and does not move.
4. **Move:** those left are moved from whichever source state they are in.

What a person sees, which must not change:

- **The time is drawn** at the trigger's turn in the sweep after the event, in that round (`draw()` keys by step and round), and only if nothing is pending.
- **It is forgotten** at any poll that finds the person outside the source. Someone who leaves and comes back between two polls keeps it.
- **It is taken** at the first poll at or after it is due, if the person is in the source and passes `where=`. A delay that rounds to 0 steps moves them in the same poll.
- **Taken by another trigger** in the same call: the time is kept and taken at a later poll.

## Where the time goes

Measured with `perf_counter` wrappers (unprofiled), 5,000 agents:

| | Time | Calls |
|---|---|---|
| Polls that moved nobody | 26.1 ms | 869 |
| Polls that moved someone | 44.0 ms | 1,055 |
| ↳ `new_events()` (all polls) | 12.6 ms | 1,924 |
| ↳ draws in `delayed()` | 14.0 ms | 821 |
| ↳ `whole_steps()` in `delayed()` | 4.4 ms | 821 |

The draws and rounding (18 ms) are needed in any design. The rest is mostly the per-poll scans in `delayed()` (`flatnonzero` over every slot, `in_states_for` over everyone pending, `np.sort` of the union), paid even when nothing is due.

## Proposal

Per delayed trigger, keep three small structures next to `ti_after_<tr>`, which stays the record of each person's pending time:

- **`due_at`: a queue of pending people by step**, `{step: [uids, ...]}`. When `delayed()` sets a time, it appends the people to their step's entry.
- **`left`: who has left any source state since the last poll.** It is appended to at the two places people are taken out of a state: the engine's own moves (`put(name, uids, False)`, through `move()`, `move_states()` and `shift_states()`), and writes by code (`dis.symptomatic[uids] = False`), which `ss.BoolState` reports through `wrote(name, entered, left)`. Only states that are the source of a delayed trigger are watched. Writes by code are reported only while the engine keeps counts (`live` is not None; `regroup()` sets it to None until the counts are taken again), so while it doesn't, `left` may be incomplete. That is marked with a flag, and the next poll forgets by the full scan, as today.
- **`carry`: people due but taken by another trigger**, kept for the next poll.

Each poll of `tr` then does:

1. **Forget**, among `left` only: those with a pending time who are not in the source now have it cleared. This is the same set the scan finds, since being pending-and-outside changes only by leaving (pending-and-inside people are kept either way). `left` is emptied.
2. **New events and draws**, unchanged: the same people, in the same round, with the same key.
3. **Due:** pop every `due_at` entry with step `<= ti`, add `carry`, and keep those whose `ti_after` still equals the step they were queued for and is `<= ti`. That drops anyone whose time was cleared or redrawn since. Sort by UID, as now. Then `taken` (into `carry`), clearing, and `where=` as now.
4. **Skip:** if `due_at` has nothing at or before `ti`, `carry` and `left` are empty, and `new_events()`'s shortcut says nothing happened, return at once. This is most of the 869 empty polls.

Each poll then costs in proportion to the people who moved, entered or left, not the population. The draws, their order, and who moves are unchanged, so results are bit-for-bit identical.

### What does not change

- Triggers with no delay, triggers on a condition, and triggers on another module's event or a delivery's outcome: still polled as now.
- `ti_after_<tr>` stays as it is: code and `describe()` that read it see the same values.
- Compartmental solvers: delayed triggers are agents only already.

## Scope and implementation

All in `starsim/solvers/agents.py`, about 60 lines:

- `__init__`/`init_pre`: `self.delays` is set already. Add a `state → [delayed triggers]` map for the sources, and the three structures per trigger.
- `put()` (on `value=False`) and `wrote()` (its `left`): append the UIDs to `left` of each delayed trigger with that source. This is one dict lookup per write for modules with no delayed triggers.
- `regroup()`: set the flag that makes the next poll of each delayed trigger forget by the full scan.
- `delayed()`: replace the scan with the forget, queue and pop steps above, and push to `due_at` when a time is set.
- `triggered()`: the skip check.
- Copying (`sim.new()`, `ss.parallel`) and saving: the structures are plain dicts and lists of UIDs, made in `init_pre`, so a new sim starts with empty ones.

## Alternatives considered

- **Draw the time at the event**, when the event's transition records people, instead of at the trigger's turn. It saves no more (the draws are the same), changes the round each draw is keyed by (so results differ, though not in distribution), and draws for people who leave the source before the trigger's turn.
- **Fold the trigger into the source state's scheduled exit** (`ti_exit_<state>`, `exit_<state>`) so the existing due check takes it. A state has one scheduled-exit slot, drawn on entry, and a trigger's time is drawn later. Overwriting the slot loses the original exit, which a split with `ss.stay` (`seek_care` here) must give back to those who stay (`_kept_exits`).
- **A heap instead of a dict of steps.** Times are whole steps and few are pending at once, so a dict is simpler. A heap would only help with many distinct future steps.

## Testing

- **Bit-for-bit:** the declared gonorrhea's 224 results unchanged (`/tmp/bench/ng.py check`), plus `test_triggers.py` (which has delayed triggers shaped like `seek_care` and `clear_symp`) and `test_transitions.py`.
- **New test** (one, in `test_triggers.py`), with the queue and the scan both run and compared for each case:
  - an agent who leaves the source and comes back between polls (keeps the time) and across a poll (loses it);
  - a second event while one is pending (not redrawn);
  - a delay that rounds to 0 steps (moved in the same poll);
  - `where=` failing when due (time lost);
  - two triggers due for one agent in the same call (the second taken later);
  - a move by code (`force()`) between polls.
- **Timing:** the declared gonorrhea at 5,000 and 50,000 agents, against the coded one.

## Open questions

- **Writes through the NumPy array** (`dis.symptomatic.raw[uids] = False`) are not reported to the engine, and its counts don't see them either (`STARSIM_CHECK_COUNTS=1` checks). Today the full scan catches them for delayed triggers; with this proposal it would not, unless the flag is also set when the counts are found to be off. Since the counts and results are already wrong for such writes, the proposal treats them as unsupported, as `01-objects.md` does.
- **Is 5% worth 60 lines?** On its own, perhaps not; it is proposed as the first and smallest of the three changes in [Context](#context), and because it scales: the scans it removes grow with the population.
