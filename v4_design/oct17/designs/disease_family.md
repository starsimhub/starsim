# A family of diseases

Design pass for Phase H, second bullet of [implementation_oct18.md](../implementation_oct18.md) (decision D-F). Findings: HP22, HP23, HP15 (its joint part), HP29, and HP7 as the interface with the care cascade. Author: design agent H2, October 19.

## Revision, 2026-10-19

After Cliff's `[CK:]` notes (H0 in [oct19/implementation.md](../../oct19/implementation.md)). His notes are kept where their lines stayed.

- The family is **`ss.Strains`**, not `ss.Diseases`.
- `name` comes first: `ss.Strains('hpv', members, ...)`.
- No list expansion in user code (open: D10). `hpvsim.genotypes('hpv16 hpv18 hi5 ohr')` returns the members (a list of declared diseases), and `ss.Strains` takes that list as is.
- `of=` is renamed. The proposal is `protected=` (who is protected: the people in each member's `nab`), with `on=` renamed `reduces=`, since the care cascade uses `on=` for events. Open: decisions D8 and D9.
- `combine=` also takes a function (D2's note). It is exact as agents, an approximation under the compartmental solvers, and has no gradients (open: D11, D12).
- A new section on how `ss.Strains` relates to the strain dimension (`stratify(strain=..., immunity=...)`), with one story for users (open: D13).
- An answer to "i thought the port was faster?" (HP29's table).
- The open decisions are in a table at the end.

## Summary

**The problem.** HPV genotypes are co-circulating infections, each with its own natural history, that a person can carry several of at once, and that protect against each other. In v4 today each genotype is its own `ss.Disease`, and everything that ties them together is missing or slow. Cross-immunity at 4 genotypes takes 88 `ss.modify` calls, and v4's combination rule (`base × Πmultiply + Σadd`) gives 0.2 where HPVsim's `(1 − clip(Σ)) × (1 − vax)` gives 0.35; since Phase A, `add=` on `rel_sus` raises, so the review's port fails at `init()` (HP22). "Infected with any genotype" has no form: a sum counts the co-infected (4 to 7% of women) twice, and the workaround is a 41-line analyzer (HP23). There is no joint seeding: HPVsim seeds one genotype per infected person, and the port does it by hand after `sim.init()`, outside common random numbers (HP15). As an ODE, the cross-genotype conditions cross the genotypes' compartments: about 20 compartments per genotype, so 8,000 at 2 genotypes and 160,000 at 3 (9.4 s now, from 383 s); 4 would be 3.2 million, and the 9 genotypes of the nonavalent vaccine or the 14 high-risk ones cannot be built at all (HP29). And a vaccine delivered for two genotypes protects against only the first (10,009 and 4,977 delivered; the only effect is on `hpv16`) (HP7). HPVsim's Starsim 3 code for all this is 402 lines; the v4 port's workarounds are 95.

**The proposal.** One container, `ss.Strains` [CK: i prefer `ss.Strains`] (renamed from `ss.Diseases`), whose members are ordinary declared diseases, plus one small object, `ss.immunity`, for protection with a combination rule. The family resolves references between its members by name (so they need not exist when each member is written), gives union results and conditions (`hpv.infected` is "infected with any genotype"), seeds jointly, and is the dimension along which values differ by member (`ss.by(hpv=...)`, `res.by('hpv')`). As agents everything is exact. Under the compartmental solvers the members run side by side, not crossed, and the couplings use each member's shares by group (an independence approximation, marked `approx`); the crossing stays available, exact, for anyone who writes the couplings as explicit `ss.modify(..., where=)`.

What HPVsim provides (in `hpvsim`, not Starsim): one function that makes one genotype's natural history, and one that makes several by name.

```python
# In hpvsim
def genotype(name, beta, p_cin, ...):                      # One natural history: an ordinary declared disease
    S, M, In, Ic, Qa, Qn, Ca, R, Rs, X = ss.states('S M In Ic Qa Qn Ca R Rs X')
    return ss.Disease(
        name     = name,
        inf_f    = S >> (p_cin*Ic + (1 - p_cin)*In) @ beta*(M + In + Ic + Qa + Qn),
        ...,
        infected = M + In + Ic + Qa + Qn,                  # A named sum of states, as now
        nab      = Rs,                                      # Who has neutralizing antibodies to this genotype
        cleared  = R + Rs,                                  # Who has cleared it (cell-mediated immunity)
    ) # [CK: i think generally the name should go first] (done)

def genotypes(names):                                      # 'hpv16 hpv18' -> [genotype('hpv16', ...), genotype('hpv18', ...)]
    return [genotype(name, **genotype_pars[name]) for name in names.split()]
```

What a user writes:

```python
import starsim as ss
import hpvsim

# [CK: don't love this, where does this come from? in general i don't like magical list expansions, except for defining states. also, why can't this whole line just be hpvsim.genotypes('hpv16 hpv18 hi5 ohr'), and the ss.Diseases/ss.Strains class does the expansion?] (done: the members are one list)
hpv = ss.Strains('hpv', hpvsim.genotypes('hpv16 hpv18 hi5 ohr'),
    immunity = [
        ss.immunity(m_sus, protected='nab', combine='sum'),                           # rel_sus of each genotype × (1 − clip(Σ_k m_sus[k, g]·[in k's nab]))
        ss.immunity(m_sev, protected='cleared', reduces=['In', 'Ic'], combine='sum'), # Time in In and Ic × (1 − clip(Σ ...)): severity immunity
    ], # [CK: what does 'of' mean? not sure i understand this block] (renamed: see below and D8)
    init_prev = hpvsim.total_prev,                         # A function of age and sex: the share with any genotype
    split     = [0.3, 0.2, 0.3, 0.2],                      # Each seeded person gets one genotype, in these shares
    exclusive = 'Ca',                                      # Cancer from one genotype ends the person's course in the others
)
vx  = ss.Vaccine(efficacy=ss.by(hpv=dict(hpv16=0.95, hpv18=0.95, hi5=0.3, ohr=0)))
sim = ss.Sim(diseases=hpv, interventions=ss.vaccinate(vx, disease=hpv, ...), ...)
sim.results.hpv.infected            # Infected with any genotype (a union)
sim.results.hpv.cancer              # New cancers, any genotype (a sum of flows)
sim.results.hpv16.infected          # Each member, as now
# [CK: this last block is nice]
```

**How to read the immunity block.** Each genotype has some states whose holders are protected: `nab` (antibodies) and `cleared` (cell-mediated immunity) are named sums of states in each member. `ss.immunity(m_sus, protected='nab', combine='sum')` says: for each genotype `g`, a person's protection against `g` is the sum, over the genotypes `k` whose `nab` they are in, of `m_sus[k, g]`, clipped at 1; it reduces `rel_sus` (the default) by that much. The second line does the same from `cleared`, but reduces the time spent in `In` and `Ic`. So `protected=` names who is protected, `reduces=` names what the protection lowers, the matrix says by how much from which genotype against which, and `combine=` says how protections from several genotypes add up.

**`ss.Strains` takes its members as one list.** The signature is `ss.Strains(name, members, immunity=None, init_prev=None, split=None, exclusive=None)`. `members` is a list of declared diseases (or an `ss.ndict` of them), whatever made it: a downstream function like `hpvsim.genotypes()`, a list written out (`[hpv16, hpv18]`), or a comprehension in the user's own code. There is no `*members` form, so there is one way to pass them. The members keep their own names (`hpv16`, ...), which are the levels of the family's dimension (`ss.by(hpv=dict(hpv16=...))`). `ss.Strains` does not make diseases from names itself, since what a genotype is (its states and parameters) belongs to HPVsim, not Starsim.

**Decisions for Cliff.**

| | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | What a family is | (a) `ss.Strains` (was `ss.Diseases`), a container of ordinary declared diseases; (b) a "replicated" dimension (`stratify(genotype=[...], independent=True)`), each person holding a state in every level; (c) no new object: fix the combination rule and add `ss.any()`, leaving 88 modifiers | (a). A dimension gives each person one coordinate, and every dimension feature (split, transfers, `move=`, mixing by level) would need an exception for one that does not; (c) leaves HP15, HP29, and the mutual references unsolved [CK: ok, but as above, i think Strains rather than Diseases] |
| D2 | The combination rule for protection from several sources | `combine='max'` (the strongest; the strain dimension's rule today), `'sum'` (added, clipped at 1; HPVsim), `'multiply'` (independent: escapes multiply, `1 − Π(1 − p)`); default `'max'` or no default | Offer all three, default `'max'`, the same in the strain dimension's `immunity=`, so the two become one mechanism (19-open-questions asks for this) [CK: ok, plus allow a custom lambda function for more complex options, obvs not differentiable] (done: see "A function for `combine=`" below) |
| D3 | What the compartmental solvers do with a family's couplings | (a) always side by side, couplings from shares (independence; `approx`); (b) the exact crossing up to a size, side by side beyond it; (c) the crossing only, refusing beyond a size | (a): results never change in kind because a genotype was added. The crossing stays available, exact, by writing the couplings as `ss.modify(where=)`, and it prints its size before building [CK: ok] |
| D4 | Joint seeding | (a) the family's `init_prev=` (share with any member) and `split=` (shares by member), one member per seeded person; (b) also allow independent seeding by each member's own `init_prev` (today's behavior) | (a) when given; (b) is what happens when the family has no `init_prev=`. Giving both raises [CK: ok] |
| D5 | One course at a time in some state (HPVsim's one cancer per person) | (a) `exclusive='Ca'` on the family; (b) leave it to each member's transition with `where=hpv.others.Ca` and an immediate move (ST26) | (a): one word, and it needs no "as soon as" rate, which hangs an ODE today [CK: ok] |
| D6 | Where immunity comes from (`of=`, now `protected=`) | (a) a state or a named sum of states of each member only; (b) also a per-person value a member keeps (HPVsim's running maximum of draws), agents only | (a) now; (b) when the behavior vocabulary (h4) gives a declared per-person value, which `protected=` then takes as is [CK: ok, but i don't like 'of', i don't know what it means] (renamed; the name is D8, still open) |
| D7 | Fix now, independent of this design | The compartmental solvers freeze an expression-valued `multiply=` at its value at the start (found here, below) | Fix in Phase A style (S): it is a silent wrong answer, and the side-by-side solver depends on it [CK: ok] |

## The use cases

### HP22: cross-immunity with HPVsim's combination rule

**Needs.** Protection against genotype `g` from having cleared each genotype `k` (including `g` itself), weighted by a matrix, summed and clipped, then combined with the vaccine as an independent path: `rel_sus_g = (1 − clip(Σ_k M_sus[g, k]·nab_k)) × (1 − vax_g)`. The same for severity, where it shortens the time before CIN or clearance: `sev_g = clip(Σ_k M_sev[g, k]·cell_k)`.

**Starsim 3.** `hpvsim.CrossImmunity`, a connector, 99 lines plus 37 for own immunity on clearance: each step it stacks every genotype's `nab_imm`, `cell_imm`, and `vax_imm` into matrices, multiplies, clips, and writes `rel_sus` and `sev_imm` of each.

**v4 today.** `cross_modifiers()` in `v4port/genotypes/genotypes_v4.py`: one `ss.modify(dg.rel_sus, add=..., where=nabpos)` per pair of genotypes and one `ss.modify(transition, multiply=1/(1 − m·CELL), where=cleared)` per pair and per precancer exit: 88 modifiers at 4 genotypes. The severity part is multiplicative where HPVsim sums (it never saturates), the additive part now raises (`ValueError: add= on A.rel_sus would replace each person's own value; use multiply=`), and no combination of `add=` and `multiply=` reproduces HPVsim (0.2 against 0.35; `probe_vocab.py` e).

**Proposed.** Two lines on the family (the summary's example). `ss.immunity(matrix, protected='nab', combine='sum')` is one multiplier on each member's `rel_sus`: `1 − clip(Σ_k matrix[k, g]·[the person is in member k's nab])`. Modifiers on the same quantity multiply as now, so a vaccine's `(1 − efficacy)` multiplies it, which is HPVsim's rule exactly. The diagonal is own-genotype immunity, so the port's `own_sus_*` modifiers go too. The matrix is rows from, columns against, as the strain dimension's `immunity=` (HPVsim's is rows against: the migration skill says to transpose it). Simpler than Starsim 3 (2 lines against 136) and than the port (2 against 19, and right).

### HP23: union results

**Needs.** Prevalence of any genotype, new cancers from any genotype, people infected with two or more; age-standardized rates per 100,000 women.

**Starsim 3.** `hpvsim.HPVTotal`, 180 lines: a boolean OR across each module's state arrays for stocks, an element-wise sum for flows, and the WHO 2000 standardization.

**v4 today.** `ss.outcome(A.cancer + B.cancer)`, `A.I + B.I`, and `A.I | B.I` as sim keywords all raise `KeyNotFoundError` (`probe_vocab.py` b), and the spec has no outcome over several modules (09-results, R17). The port's `AnyHPV` analyzer is 41 lines.

**Proposed.** The family's results are made from its members' results of the same name: a stock (a state, or a named sum of states such as `infected`) is the union, people in it in at least one member; a flow (a transition, such as `cancer`) is the sum, since each event belongs to one member. `sim.results.hpv.infected` and `sim.results.hpv.cancer` need no code; `.by('hpv')` gives each member's own. The family's attributes are the same unions as conditions, so `where=hpv.infected` (on a delivery or a modifier) means infected with any genotype. Co-infection, "in two or more", is `hpv.infected.count() >= 2` (one more method on the union, S). Standardized rates are not part of this design (out of scope, below), so of the 41 lines, 33 go and the standardization stays until the results design has it.

### HP15: joint seeding

**Needs.** One draw for "infected with any genotype" by age and sex (gated on sexual debut), then one genotype for each person infected, in given shares, entering that genotype's states as a new infection would (men to `M`; women to `Ic` or `In` by `p_cin`).

**Starsim 3.** `_ExclusiveSeeder`, `_live_seeder`, and `_make_init_prev_fn`: 70 lines.

**v4 today.** `init_prev` is per disease. `init_prev=ss.bernoulli(p=fn)` now works, but `init_prev=lambda sim: ...` and `init_prev=ss.by(age=...)` raise `TypeError`, and `init=dict(I=ss.by(age=...))` raises `KeyError` (HP15's single-disease part: S, separate from this design). The port's `seed_exclusive()` (19 lines) runs after `sim.init()`, sets the states by hand, draws from NumPy outside common random numbers, and calls `engine.record_stocks()`.

**Proposed.** `ss.Strains('hpv', members, init_prev=fn, split=[...])`. `init_prev=` takes what a member's does (a number, `ss.by`, a function per agent, or an `ss.bernoulli`); `split=` is the share of each member among those seeded, as `split=` is the share of each level in `stratify()` (default equal, normalized, printed as a guess). Each seeded person is infected by exactly one member, entering the destinations of that member's transmission from its first state, with its branch shares, as each member's own `init_prev` does now ("seeded through the transmission"). One random stream on the family, so common random numbers hold. A member with its own `init_prev` or `init` in a family that has one raises, naming both. Simpler: 2 keywords against 70 lines or 19.

### HP29: the size of the crossing

**Needs.** HPVsim's own genotypes (4) as compartments for calibration and gradients; 9 (the nonavalent vaccine's) or 14 (every high-risk genotype) for vaccine-impact work, at least as agents.

**v4 today.** Conditions on another disease's states cross the diseases' compartments (07-effects, `starsim/solvers/crossing.py`): one program whose compartments are the product of each disease's. Measured for this design (`scratch-h2/probe_scale.py`, a 5-state toy genotype, homogeneous mixing, 10,000 people, 1990 to 2030 at `dt=0.25`, `ss.modify(rel_sus, multiply=0.7, where=other.R)`):

| Genotypes | Crossed compartments | ODE run | Agents run |
|---|---|---|---|
| 1 | 5 | 0.06 s | 0.23 s |
| 2 | 25 | 0.10 s | |
| 3 | 125 | 0.17 s | 0.68 s |
| 4 | 625 | 0.36 s | |
| 5 | 3,125 | 1.26 s | |
| 6 | 15,625 | 6.8 s | |
| 9 | 1.95 million (not built) | | 1.88 s |

The run grows about fivefold for each genotype added, in step with the compartments. With lognormal dwells the toy has 14 compartments per genotype (2,744 at 3; 0.86 s). The HPV port has about 20 per genotype, so 8,000 at 2 and 160,000 at 3 (9.4 s, after Phase G's fix); 4 genotypes would be 3.2 million compartments and, at the same growth, of order 200 s; 9 would be about 10^13 and 14 about 10^19, which no machine holds.

**Proposed.** Under the compartmental solvers a family is not crossed (D3). Its members run side by side in one program (as a crossing's diseases already run in one program, but with the sum of their compartments, not the product), and each coupling is evaluated from the members' shares, by group, at each evaluation of the right-hand side: a susceptible of `g` gets `1 − clip(Σ_k M[k, g]·f_k)` with `combine='sum'`, `Π_k (1 − M[k, g]·f_k)` with `'multiply'`, where `f_k` is the share of the group in member `k`'s source states. That is exact if membership of each genotype's states is independent across genotypes, and it is not: having had one genotype protects against the others, and (over a network) the same people are exposed to all of them. The size is linear: 4 genotypes about 80 compartments, 14 about 280.

How far off it is, on the toy family, against the exact crossing with the same integrator (`scratch-h2/ode_mf.py`, RK4 at `h=0.05`, cancers absorbing; relative error at 40 years):

| Genotypes | Protection per other genotype | Error in prevalence | Error in cumulative cancers |
|---|---|---|---|
| 2 | 0.3 | +1.9% | +1.7% |
| 3 | 0.3 | +7.6 to +7.9% | +4.8 to +5.2% |
| 4 | 0.3 | +8.2 to +9.2% | +3.8 to +4.5% |
| 5 | 0.3 | −2.3 to −4.9% | +0.7 to +2.1% |
| 3 | 0.6 | −9 to −25% (prevalence near zero) | +1.2 to +5.2% |
| 3 | 0.9 | −51 to +39% (prevalence near zero) | −0.4 to +11.7% |

So the approximation is within about 5% on cancers and 10% on prevalence at HPVsim's protections (0.3 to 0.5 susceptibility, 0.5 to 0.7 severity), and poor where a genotype is nearly excluded. As agents (the same toy, cross-immunity by shares against by person: 25,423 against 24,190 cancers for the first genotype, means of 3 and 5 seeds) it is +5%. That is what `sim.solvers()` says: `ode`, `binomial`, `gillespie`: `approx`, "the members of hpv are independent given their shares"; `abm`: exact; `auto` picks agents, saying why, as it does for a strain dimension with partial immunity.

The alternatives, for the record, with their sizes at `c` compartments per genotype (about 20 for HPV) and `k` genotypes:

| Form | Compartments | 4 | 9 | 14 | Exact? |
|---|---|---|---|---|---|
| Crossing (today) | `c^k` | 160,000 to 3.2 million | about 10^13 | about 10^19 | Yes |
| Each genotype crossed with the others' immunity, one bit each (D13's history dimension) | `k·c·2^(k−1)` | 640 | 46,000 | 2.3 million | No: the others' states other than the bit are independent |
| The same with HPV's two kinds (nab, cleared: three values) | `k·c·3^(k−1)` | 2,160 | 1.2 million | 450 million | No |
| Side by side (proposed) | `k·c` | 80 | 180 | 280 | No: independent given shares |
| Agents | linear in `k` and people | 0.2 s per genotype per 10,000 people for 160 steps (toy); the fully declared HPV port 13.5 s at 4 genotypes against HPVsim's 8.7 s (see below) [CK: i thought the port was faster?] | | | Yes |

**"I thought the port was faster?"** It is, unchanged. HPVsim's own code run on v4 is 1.3 to 2.4 times faster than on Starsim 3 (one genotype: 4.0 s against 9.6 s). What is slower is the fully declared port. With per-person immunity it takes 8.9 s against HPVsim's 8.7 s. With history states and 88 modifiers it takes 13.5 s, because each lazy condition rebuilt a module lookup on every leaf (HP30). HP30's fix (Stage 2, done) caches the lookup (0.55 to 0.43 s for the 88 modifiers alone); what is left is each modifier's own work, which `ss.immunity` removes by replacing the 88 modifiers with one matrix product a step.

The history form is the next step if the side-by-side error matters for someone's calibration; it needs no new user vocabulary (the same family, a solver choice), so it can come later.

### HP7: one product over several diseases (the interface with the cascade)

**Needs.** One bivalent or nonavalent vaccine dose that protects against each genotype it covers, with an efficacy per genotype; one test that is positive for any high-risk genotype, or reports which.

**v4 today.** A product's `disease` is set by its first delivery and later `disease=` are ignored; the two deliveries share `has_<product>`, so the second skips the first's recipients; `disease=[hpv16, hpv18]` raises `AttributeError`; `efficacy=dict(hpv16=, hpv18=)` raises `E506`, since `efficacy=` keys are quantities (`acquisition`, ...). Nobody is protected against `hpv18`.

**Proposed.** `disease=` of a product or a delivery may be a family; the product then has one effect on each member it covers, and holding it is still one property of people (`has_vx`), which is right for one dose. A value per member is `ss.by(hpv=dict(...))`, the family being the dimension along which values differ by member, so `efficacy=` keeps its keys for quantities (`efficacy=dict(acquisition=ss.by(hpv=...))`). Members not named get 0, printed. The quick part of HP7 (raise when a product bound to one disease is delivered for another) is S and goes first, independent of this design.

## Semantics under every solver

**What a family is.** `ss.Strains(name, members, immunity=None, init_prev=None, split=None, exclusive=None)` is a module of its own kind in `sim.diseases`, under its name, beside its members under theirs (`sim.diseases.hpv`, `sim.diseases.hpv16`). It has no states and no step of its own; its members are stepped as now, in their order, and code that loops over diseases (deaths by cause, a migration's checks) skips families. Members must be declared diseases (a coded `ss.Disease` raises, naming the member: its states are not visible to the family), share one timestep (as a crossing's must), and their names must be unique. Members may have different natural histories; nothing requires them to have the same states, except that a name the family uses (`protected='nab'`, `exclusive='Ca'`, a union result) must exist in each member it applies to, or the family raises listing which members lack it (a member that should be left out of an immunity is left out of its matrix with a zero row and column).

**References by name.** `protected=`, `reduces=`, `exclusive=`, and union results name states or named sums of states in each member, and are resolved when the sim is built, so no member needs another to exist when it is written. That is the root of the port's functions of the sim (`other_cancer`): today a rate or condition that names another disease's state object made before that disease is built raises `KeyNotFoundError: odict key "Rb" not found` (`scratch-h2/probe_adopt.py`), so a symmetric relation cannot be written without a function, and a function makes the model agents-only.

**`ss.immunity(matrix, protected, reduces='rel_sus', combine='max')`.** `matrix[k][g]` is the protection that being in member `k`'s `protected` states gives against member `g` (1 complete). For each member `g` and each person, the protection is the combination over `k` of `matrix[k][g]·[in k's protected]`: the largest (`'max'`), the sum clipped at 1 (`'sum'`), `1 − Π(1 − ·)` (`'multiply'`), or a function (below). It multiplies `reduces` by `1 − protection`: `reduces='rel_sus'` (default) or `'rel_trans'`, or a state or list of states, whose time spent is multiplied (so every exit of `In` is faster by `1/(1 − protection)`: HP22's "a modifier on all of a state's exits"). A protection of 1 on a time spent raises (an infinite rate); `describe()` gives the largest the matrix can reach. Several `ss.immunity` objects, and any `ss.modify` or product effect on the same quantity, multiply. In a strain dimension, `stratify(strain=..., immunity=ss.immunity(matrix, combine='max'))` is the same object, with the levels for the members (no `protected=`: in a dimension it is the strains a person has had); a bare matrix stays the matrix with the default rule.

**A function for `combine=`.** For a rule the three names do not cover, `combine=` takes a function of one argument, `p`, the protections against one member from each source (`matrix[k, g]` for the sources the person or group is in, 0 for the others), as an array with the sources along the first axis, and returns the protection, between 0 and 1. For example, the strongest source plus half the rest: `combine=lambda p: np.minimum(1, p.max(0) + 0.5*(p.sum(0) - p.max(0)))`. It is called once per member and step, with every person at once (agents) or every group (compartmental), so it is vectorized NumPy, not a loop. What it costs:

- **Agents:** exact, as for the named rules. One Python call per member and step.
- **ODE, binomial, gillespie:** it runs, but it is applied to the shares (`p` holds `matrix[k, g]·f_k`, the protection from each source times the share of the group in it), which is the side-by-side approximation (D3) again; the named rules have their own forms over shares (`'sum'` and `'multiply'` above; `'max'` takes the largest `matrix[k, g]·f_k`), and a function is treated as `'max'` is. Marked `approx`, as any coupling of a family is. The ODE calls it from Python at each evaluation of the right-hand side, so the kernels that run a family without a function do not apply (a note says so, with the rough slowdown).
- **Gradients (`sim.gradient()`, NUTS, `jax`):** none. A function is NumPy, so `sim.gradient()` and a NUTS fit raise `ss.CapabilityError`, naming the function and the three named rules. Fits without gradients (Optuna, Nelder-Mead) work.
- **The saved form:** the function is code, marked as such in `to_spec()` (as a function-valued `init_prev` is), so two models that differ only in it hash the same, with the usual note.

**Agents.** Exact, per person: each step, before transmission, each coupling is worked out from the members' state arrays (one matrix product over the people, as HPVsim's connector does), union results and conditions are an OR over members, the joint seeding is one draw on the family's stream, and `exclusive='Ca'` moves a person who enters `Ca` of one member into each other member's last state with no exits from which they cannot be infected (`X` in the port; the family raises if a member has no such state, naming what to add), within the same step and without an "as soon as" rate. Over networks this works as any `ss.modify` does (unlike the strain dimension's `immunity=`, HP14).

**ODE, binomial, gillespie.** Side by side (D3): one program, the members' compartments one after another, each flow of a member a flow of the program, and each coupling a factor worked out from the members' shares by group at each evaluation of the rates (the ODE's right-hand side; the stochastic solvers' rates at the start of each step). Union stocks are `1 − Π_k (1 − f_k)` of each group (independence), flows are sums, and joint seeding gives each member `init_prev × split_k` of each group (so the seeded are not exclusive: at 10% seeded and 4 genotypes, about 0.4% of people are counted in two). `exclusive='Ca'` is a flow from each of a member's other states into `X` at the rate that the others' cancers arrive among the group's people. The crossing is not used for a family's couplings; if a user also writes an `ss.modify(..., where=)` between two members, those two are crossed as now (so the exact form stays available), and the crossing prints its size before it is built (HP29's first ask). `jax` and the gradient: the factors are smooth functions of the shares, except at the clip of `'sum'`, which uses the clip as the ODE does (its gradient is 0 beyond it).

**Which solvers cannot run it.** None refuse a family. A member that is agents-only (a function `where=`, a network) makes the family agents-only, with the reason, as for any declared disease. `sim.solvers()` marks the compartmental solvers `approx` ("members independent given their shares") when the family has couplings, and runs a family with none (union results and seeding only) under every solver with the same flags as its members (the union and the seeding are the only approximations, and both are named).

**`describe()`.** One block for the family after its members' blocks:

```
Family hpv: 4 members (hpv16, hpv18, hi5, ohr), stepped in this order
  immunity: rel_sus of each member × (1 − min(1, Σ_k m[k, g]·nab_k)), from hpv16.nab, hpv18.nab, hi5.nab, ohr.nab (4×4 matrix, largest protection 0.95)
  immunity: time in In, Ic of each member × (1 − min(1, Σ_k m[k, g]·cleared_k)) (largest protection 0.9)
  seeding: infected with any member p(age, sex) (a function), one member each, in shares 0.3, 0.2, 0.3, 0.2
  exclusive: entering Ca of one member moves the person to X in the others
  results: infected (union of the members' infected), cancer (sum of the members' cancer), ...
  compartmental solvers: members side by side; couplings, unions, and seeding from each member's shares (independent)
```

**`to_spec()`.** A node of kind `strains` with its members' specs in order, and its options as data: `{"kind": "strains", "name": "hpv", "members": [...], "immunity": [{"matrix": [[...]], "protected": "nab", "reduces": "rel_sus", "combine": "sum"}, ...], "init_prev": ..., "split": [0.3, 0.2, 0.3, 0.2], "exclusive": "Ca"}`. A function-valued `init_prev` is marked as code, as everywhere else in the spec. `ss.diff()` compares families member by member.

**`to_code()`.** Each member as its own `ss.Disease(...)` assignment, then `hpv = ss.Strains('hpv', [hpv16, hpv18, ...], immunity=[ss.immunity(...)], ...)`, and the sim with `diseases=hpv`. The round-trip test of F1 covers it.

## Fit with what exists

**Reused.** Members are ordinary `ss.Disease` objects with ordinary states, transitions, named sums of states, and results; a family changes nothing about them. `split=` means what it means in `stratify()`. `ss.by(<family>=...)` and `res.by('<family>')` treat the members as the levels of a dimension named after the family, so values per member and results per member use the vocabulary they use for any dimension. Unions are conditions wherever a condition is taken (`where=` on deliveries, modifiers, and transitions). Products take `disease=` a family (HP7). Modifier effects combine as now (multiply), and `ss.immunity` is one more multiplier among them. The side-by-side program reuses the crossing's machinery for one program over several engines (`Crossing.init_layout()`, `scatter()`, `gather()`), with the members' compartments summed instead of multiplied.

**New public names, two.**

- `ss.Strains` (CamelCase: a big thing, a module holding modules). Justified against extending `ss.Disease` (a disease that is several diseases would need every member's natural history in one graph, and the crossing again) and against a dimension (D1). D-F called it `ss.Diseases`; Cliff chose `ss.Strains`. Its arguments: `name`, `members`, `immunity`, `init_prev`, `split`, `exclusive`.
- `ss.immunity` (lowercase: a small thing, a value). Justified against extending `ss.modify` (a modifier has one target and one condition; this is a matrix of conditions, one per pair, combined by a rule before it multiplies, which is the part `ss.modify` cannot do), and against a keyword taking a bare matrix (HPVsim needs two, on different quantities). It also replaces the strain dimension's bare `immunity=` rule, making cross-immunity within a dimension and between diseases one mechanism.

**Replaced or removed.** Nothing in Starsim is removed. The refusal of `add=` on `rel_sus` stays: nobody needs it once the rule is in `ss.immunity`. The guidance in 05-dimensions ("As a second disease ... connect them with `ss.modify`") changes to "as a family". The crossing stays as the exact form for explicit cross-disease conditions (HIV and syphilis, where there are two diseases and one condition, and the size is small).

**`ss.Strains` and the strain dimension.** Starsim already has strains as a dimension of one disease: `sir.stratify(strain=['A', 'B'], immunity=[[1, 0.5], [0.5, 1]])`, which makes a dimension of kind `'strain'` (`dimensions.py`: the kind is guessed from `immunity=` or a name such as `strain`, `variant`, `genotype`, `serotype`, or `resistance`). With `ss.Strains` there are two things called strains, so they need one story. The story for users:

- **One disease, one strain at a time: a strain dimension.** The strains share one natural history, and a person is in one level (or a level that names a set, `'A+B'`). Dengue serotypes, influenza strains, TB drug resistance. Written `stratify(strain=..., immunity=...)`.
- **Several infections a person can carry at once, each with its own natural history: `ss.Strains`.** HPV genotypes. Written `ss.Strains(name, members, immunity=...)`.
- **The same `immunity=` in both.** Both take `ss.immunity(...)` (or a bare matrix, with the default rule), with the same matrix layout (rows from, columns against) and the same `combine=` rules, so cross-immunity is one mechanism. That is what makes the two read as one idea: "strains, with immunity between them", held either as levels of one disease or as diseases in a family.
- **The user guide's section "Several strains or genotypes"** opens with this choice, as a two-row table (one at a time and shared history: a dimension; at once or separate histories: `ss.Strains`).

What keeps them from colliding:

- The dimension's kind stays `'strain'`, an internal word users rarely write (it is guessed from `immunity=`). Nothing in the family is called a kind.
- A family is a dimension for `ss.by(<family>=...)` and `res.by('<family>')`, named after the family (`hpv`), never `strain`. A family whose name is a dimension's name (`ss.Strains('strain', ...)` beside `stratify(strain=...)`) raises, naming both.
- A member of a family may itself have a strain dimension (each genotype with variants); the two are different axes (`res.by('hpv')`, `res.hpv16.by('strain')`).
- The name-based guess (`genotype=` suggests kind `'strain'`) stays a note: `stratify(genotype=[...])` on one disease for HPV gets a note pointing to `ss.Strains` when no `immunity=` is given, since co-infection is the usual reason someone says "genotype".

A strain dimension is right when a person has one strain at a time and the strains share a natural history: dengue serotypes (sequential infections; severity on a second infection depends on the first), influenza strains, TB drug resistance (tbsim's `resistance` dimension, whose levels are the sets of strains carried, with `move=` for acquired resistance). It is wrong for HPV: co-infection needs a level for every set of genotypes carried (15 at 4, 511 at 9, 16,383 at 14; `probe_strain.py` 3), and one clearance clears every genotype a person carries, so "16 in CIN, 18 just acquired" has no form. Under the compartmental solvers the strain dimension keeps only the last strain each person has had (noted, `approx`); for dengue the exact form is a history of strains had, `2^k` levels (16 at 4 serotypes), which is D13's follow-on and out of scope here. The family adds nothing to the strain dimension except `ss.immunity` for its rule.

**Against the crossing.** The crossing is the exact compartmental form of cross-disease conditions, and it stays. It is the right tool for two or three diseases with a condition between them. For a family it is exact but grows as `c^k` (HP29), it needs one modifier per pair and transition (HP22), and its members cannot refer to each other without functions. The family gives the same model, exact as agents, at linear size as compartments.

**Other frameworks.** Multi-strain compartmental models in the literature use the same three forms: full history (`2^k` per state, exact for sequential infection), status-based reductions, and independent strains coupled through their prevalences; the side-by-side form is the third, and most compartmental HPV models go further and ignore cross-immunity entirely. Agent-based HPV models (HPVsim, HPV-ADVISE) keep one state per genotype per person, as a family does as agents. None of the frameworks in `lingua_franca_design/examples/` has a multi-strain example to follow.

## A bug found while designing (D7)

Under the compartmental solvers, a modifier whose `multiply=` is an expression is worked out once, at the start, and kept (`scratch-h2/probe_frac*.py`). Two SIRS diseases, `a` and `b`, 5,000 people, 10 years, `ss.modify(a.rel_sus, multiply=1 - 0.7*b.R.frac)`: `a.n_I` at the end is 1,605 under the ODE, the same as with no modifier, since `b.R` is 0 at the start; with half of `b` starting in `R`, it is 136, the same as `multiply=0.65`. Agents work it out each step (414). The same holds for `a.R.frac` (a disease's own share), for `b.R.n/5000`, for a modifier on a transition, and under `binomial`. Nothing is printed. It is a silent wrong answer independent of this design (S: the effect is `tabulated` because it is not callable; an expression of current shares must be worked out at each evaluation, or refused under the compartmental solvers), and the side-by-side solver depends on the fix.

## Implementation plan

| Step | What | Size | Files |
|---|---|---|---|
| 0 | D7's fix: expression multipliers worked out each step under the compartmental solvers, or refused; HP7's quick error; HP15's single-disease part (function and `ss.by` `init_prev`, or `E804`) | S | `connectors.py`, `products.py`, `solvers/compartmental.py`, `graph.py` |
| 1 | `ss.Strains`: members, registration in `sim.diseases`, references by name, the family as a dimension for `ss.by` and `res.by`, `describe()` | M | new `starsim/families.py` (or in `diseases.py`), `sim.py`, `dimensions.py`, `presentation.py` |
| 2 | Union results and conditions, as agents and from shares; `.count()` | S | `families.py`, `results.py`, `conditions.py` |
| 3 | `ss.immunity` as agents (one matrix product a step, before transmission), on `rel_sus`, `rel_trans`, and time in states; `combine=` a function; in the strain dimension too | M | `families.py`, `solvers/agents.py`, `solvers/engine.py`, `dimensions.py` |
| 4 | Joint seeding (`init_prev`, `split`) on the family's stream; `exclusive=` as agents | S | `families.py`, `solvers/agents.py` |
| 5 | The side-by-side program for the compartmental solvers: the crossing's layout with summed compartments, couplings as factors from shares (a `combine=` function called from Python, with its note), unions, seeding, `exclusive`; `sim.solvers()` rows; `CapabilityError` for a function under `sim.gradient()`; the crossing printing its size | M to L | `solvers/crossing.py` (or a sibling), `solvers/compartmental.py`, `kernels.py`, `solvers/choose.py` |
| 6 | Products over a family (`disease=` a family, `ss.by(<family>=...)` in `efficacy=`), with h1 | M | `products.py`, `interventions.py` |
| 7 | `to_spec()`, `to_code()`, `ss.diff()`; docs (a user guide section, "Several strains or genotypes", opening with the table of when to use the strain dimension and when `ss.Strains`; the note on `stratify(genotype=...)` pointing to `ss.Strains`) and the migration skill (transpose HPVsim's matrices) | S to M | `spec.py`, `presentation.py`, `docs/user_guide/`, `skills/migrate-starsim3/` |

About two weeks of agent work in all, steps 1 to 4 first (agents exact: what HPVsim needs to port), then 5.

**Tests**, on numbers a user sees:

- Against HPVsim: the port's `compare.py` measures (prevalence of each genotype and of any genotype among women 15 to 49, co-infection, cancers, age-standardized incidence) with the family at 2 and 4 genotypes, as agents, within the tolerance the port already meets with its 95 lines of workarounds.
- The combination rule: one person in `hpv16.nab` and `hpv18.nab` and vaccinated, `rel_sus` of `hpv16` is `(1 − min(1, m[16,16] + m[18,16]))·(1 − 0.95)` for `'sum'`, and the `'max'` and `'multiply'` values; 0.35 for HP22's case.
- Unions: as agents, `sim.results.hpv.infected` equals the OR of the members' arrays counted with `people.scale` at every step; under the ODE, `1 − Π(1 − f)`; flows equal the sum.
- Seeding: as agents, nobody is seeded in two members, the shares by member match `split=` within sampling error, and two runs with the same seed seed the same people.
- Side by side against the crossing: the toy family at 2 to 4 genotypes, protection 0.3, cancers within 6% and prevalence within 10% (the numbers above), and marked `approx`.
- A contract across solvers (`tests/test_contracts.py`): a family with no couplings equals its members run alone under every solver, bit for bit; a family with couplings agrees among `ode`, `binomial`, `gillespie` (same approximation) within the usual tolerance, and `abm` agrees with the exact crossing.
- Errors: a coded member, a missing `protected=` state, a `combine=` function under `sim.gradient()`, a family named like a dimension, a member with its own `init_prev` in a seeded family, a protection of 1 on a time spent, `exclusive=` without a state to move to.
- Size: 9 and 14 toy genotypes build and run under the ODE in under 5 s, and as agents at 10,000 people in under 5 s.

**What downstream deletes.** HPVsim: `CrossImmunity` and `_build_cross_matrix` (99 lines, the matrices stay as data), own immunity on clearance (37), the union and sum parts of `HPVTotal` (most of 180; the standardization stays until the results design has it), `_ExclusiveSeeder`, `_live_seeder`, `_make_init_prev_fn` (70), `_cancel_other_genotype_progression_for` (16): about 300 of the 402 lines. The v4 port: `cross_modifiers`, `seed_exclusive`, `other_cancer` and the cancel transition, and most of `AnyHPV`: 95 lines to about 15. STIsim and FPsim: nothing (no families). tbsim: nothing (its resistance stays a strain dimension), unless it takes `ss.immunity` for a rule.

## Spec sections that would change

- 05-dimensions, "Strain": the two ways become three (a dimension; a family; a second disease for unrelated pathogens), with the table of when to use which; `immunity=` takes `ss.immunity`.
- 07-effects: cross-disease conditions between members of a family are couplings, not crossings; `ss.immunity` and its rules; the paragraph on crossing gives the size rule and the print before building.
- 08-interventions: `disease=` a family; values per member with `ss.by(<family>=...)`.
- 09-results: family results (union of stocks, sum of flows); R17's "no outcome over several modules" stays for unrelated diseases.
- 12-backends: the solver table's row for a family; the side-by-side program.
- 14-model-spec: the `strains` node.
- 17-vocabulary: `ss.Strains`, `ss.immunity` (and its keywords `protected=`, `reduces=`, `combine=`).
- 19-open-questions: "Is strain a dimension or a second disease?" becomes three answers; the note at line 129 (one mechanism for cross-immunity) is resolved.

## Out of scope, and the other Phase H documents

**Out of scope.** Age-standardized rates and rates per 100,000 women (HP23's second half: a results design, and a denominator of women by age, D15). Per-person continuous immunity (HPVsim's running maximum of draws: D6, after h4). The exact history form for strain dimensions (dengue; D13's follow-on). Families of coded diseases. Unions across unrelated diseases (R17). HPVsim's per-act transmission over its network and newborns in no state (ST1 to ST4: h3 and Phase C).

**The care cascade (h1).** I assume: any product or delivery's `disease=` may be a family; values per member are `ss.by(<family>=...)`; a test's `detects=` on a family names a state or named sum of states, and is positive when the person is in it in any member (the union); a genotyping test, whose result says which members, is h1's remembered result with a value per member (h1 decides its form; this design gives it the members as levels of a dimension); a cure on a family cures each member by that member's own rule (ST5, HP25). Under the compartmental solvers, a delivery that depends on a member's state (screening the HPV-positive) sees the union share by group, and the correlation between members is lost as for the couplings.

**Per-act transmission (h3).** Members' transmissions over a sexual network are each member's own; a family adds nothing to them. Per-act probabilities (ST2, the port's `peract()`) and named directions (ST3) apply to each member.

**Behavior vocabulary and dimensions over people's properties (h4).** `init_prev=` as a function of age and sex, and `split=` by group, use whatever h4 gives for a value by people's existing properties (D15, FP19). A declared per-person value, kept across states and set on an event, would let `protected=` take HPVsim's continuous immunity as is (D6).

## Decisions still open

For Cliff's `[CK:]`. D1 to D7 are answered above; these are new or reopened by the revision.

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D8 | The name for `of=`: the states whose holders are protected | (a) `protected='nab'`; (b) `when_in='nab'`; (c) `source='nab'` (or `from_state=`) | (a). It reads as "who is protected", which is what the states are: `ss.immunity(m_sus, protected='nab')`. (b) reads well but is a phrase, not a noun; (c) is short but says nothing about immunity, and `source` already means the infecting side in transmission [CK: i'm ok with a, but not sure i like this structure over all] **Decided:** (a), within the structure of D9 |
| D9 | The name for `on=`, what the protection lowers | (a) `reduces='rel_sus'` (or a list of states); (b) keep `on=`; (c) `target=` | (a). The care cascade uses `on=` for events (`on=hpv.cin.X`), and `target=` for stock targets, so both would mean two things. `reduces=` says what happens to it [CK: why do we need a new name here? what about `modifies`, like `ss.modify()`? is there a downside to that? will it always be a reduction? but why have `reduces='rel_sus'` and not `rel_sus=0.5`?] **Revised:** the quantity is the keyword and the protection its value, as Cliff suggests: `ss.immunity(protected='nab', rel_sus=m_sus, combine='sum')`, `rel_trans=` likewise, and `dur=dict(In=m_sev, Ic=m_sev)` for time spent in states. No `reduces=`. A value is a matrix (rows from, columns against) or a scalar (the same protection from any member against every member). It is always a reduction (the value is a protection, 1 complete), which is why it is `ss.immunity` and not `ss.modify`: a modifier says what a value is multiplied by, immunity says how much protection people in `protected` have, and `combine=` says how protections from several members add up, which `ss.modify` has no use for. Built this way; Cliff to confirm |
| D10 | How `ss.Strains` takes its members | (a) one list (or `ss.ndict`), as the second argument: `ss.Strains('hpv', hpvsim.genotypes(...))`; (b) also `*members`; (c) also names, made by a function the family is given (`ss.Strains('hpv', 'hpv16 hpv18', make=hpvsim.genotype)`) | (a). One way, no expansion. (c) puts a downstream factory inside Starsim's signature for no gain over (a) [CK: ok, but it should have a kwarg name too as well, of course] **Decided:** (a), with the members given positionally or as `members=` |
| D11 | The arguments of a `combine=` function | (a) one array of protections by source (sources on the first axis); (b) the matrix and the memberships separately; (c) a function of two protections, folded over the sources | (a). It is vectorized and is what the three named rules are written as. (c) is easy to write but cannot express a rule over all sources at once (such as "strongest plus half the rest") [CK: not sure but ok] **Decided:** (a) |
| D12 | A function under the compartmental solvers | (a) run, applied to the shares, `approx` (as `'max'`); (b) refuse with `CapabilityError`, agents only | (a), since the named rules are approximate there too, and it is marked. (b) if Cliff prefers that a custom rule never runs where its meaning is changed [CK: ok] **Decided:** (a) |
| D13 | The strain story | (a) as above: two forms, one `immunity=`; (b) rename the dimension's kind to `'variant'` so "strain" means only the family; (c) make `stratify(strain=...)` build an `ss.Strains` when the strains overlap | (a). The kind is internal; users see `stratify(strain=...)` and `ss.Strains`, and the guide's table says which. (b) renames a word users rarely type; (c) mixes two structures behind one call [CK: ok] **Decided:** (a) |

## Implementation notes: steps 0 to 4 (agents), where the code departs

Cliff's answers to D8 to D13 are applied as given (the quantity lowered is the keyword of `ss.immunity`: `rel_sus=`, `rel_trans=`, `dur=dict(In=m)`, a matrix or a number; no `reduces=`). Where the code otherwise departs from the text above:

- **Agents only.** A family under a compartmental solver raises `ss.CapabilityError` (the capability row is `no`, not `approx`), and `auto` chooses agents. Step 5 (side by side), step 6 (products over a family, `ss.by(<family>=...)`), and step 7 (`to_spec()`, `to_code()`, `ss.diff()`, docs) are not built; the family's options are not in the spec or the hash yet.
- **KI15.** A strain dimension with partial immunity and people who can be infected again raises `CapabilityError` under a compartmental solver (it was a note), rather than keeping the last strain; the exact `2^k` history form stays deferred. A `combine=` function on a dimension also raises there.
- **When protection is worked out.** From the members' states at the start of each step (the family's `start_step()`), before any member moves, so it does not depend on the order of the members.
- **A protection of 1 on a time spent** raises when the sim is built if the matrix and a named rule can reach it, and during the run for a function. HPVsim's severity matrices with `combine='sum'` can reach 1 at four genotypes; the port's multiplicative modifiers are `combine='multiply'` exactly.
- **Seeding** draws each person on their own (a Bernoulli share, on the family's random numbers), not an exact number per group as a disease's own `init_prev` does. A seeded person enters the member's first transmission's destinations, in its shares; a member with one transmission per direction (the port's ST3 workaround) would seed women into the men's state, so a port of HPVsim with those transmissions still seeds with its own `seed_exclusive()` until per-act transmission gives one transmission per member.
- **`exclusive=`** moves people to the member's last state of people with no way out but death (other than the exclusive state itself); such a state is declared as `X=ss.State(people=True)` (a state in no transition is otherwise a count). If two members' exclusive states are entered in the same step, the earlier member keeps it.
- **Union results** are made for every state of people and every plain sum of states that all members have (`n_I`, `infected`), and sums for every transition and outcome they all have; they are not kept by group, so `select()` does not apply to them; `by('<family>')` does.
- **Members' names.** A member built without a name keeps its default (e.g. `sir`) if no variable names it; names must differ from each other and from the family.
- **The module** is `starsim/strains.py` (not `families.py`).
