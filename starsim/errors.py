"""
Errors, warnings, and notes.

Each error is an exception class, e.g. `ss.errors.UnknownState`, and each warning a
warning class, e.g. `ss.errors.ImportIncomplete`. A note is a line about something that
was assumed or approximated: it is kept in `sim.notes` and printed when `verbose`
is 1 or more (see `ss.note()`). The docstring of each class says when it is raised,
and is what the catalog in the specification (`15-errors.md`) is made from: see
`ss.errors.catalog()`.
"""
import re
import sciris as sc
import starsim as ss

__all__ = ['CodedError', 'StarsimWarning', 'Note', 'note', 'CapabilityError', 'AlreadyRunError'] # The base classes, the error for what a solver cannot run, and the error for a sim that has already run are also ss.<Name>; the rest are ss.errors.<Name>


#%% Base classes

class CodedError(ValueError):
    """ Base class for Starsim's named errors, e.g. `ss.errors.UnknownState`; `except ss.CodedError` catches any of them """
    __str__ = Exception.__str__ # Not KeyError's, which would quote the message


class StarsimWarning(RuntimeWarning):
    """ Base class for Starsim's named warnings, e.g. `ss.errors.ImportIncomplete` """
    pass


class Note(StarsimWarning):
    """ Base class for Starsim's named notes, e.g. `ss.errors.UnitInferred`; a note is said with `ss.note()` """
    pass


def note(msg, category=None, sim=None, verbose=None):
    """
    Say something that was assumed or approximated, in one line

    The note is kept in `sim.notes` (if a sim is supplied), whatever the verbosity,
    and is printed if `verbose` (by default the sim's) is 1 or more.

    Args:
        msg (str): what was assumed or approximated
        category (class): the kind of note, if it has a name, e.g. `ss.errors.LargeStep`
        sim (Sim): the sim to keep the note in
        verbose (int): whether to print it

    **Example**:

        ss.note("unit='day' (from start and stop dates)", ss.errors.UnitInferred, sim=sim)
    """
    if category is not None and not (isinstance(category, type) and issubclass(category, Note)):
        errormsg = f'The category of a note must be a subclass of ss.Note (e.g. ss.errors.LargeStep), not {category}'
        raise TypeError(errormsg)
    if sim is not None and not hasattr(sim, 'notes'): # A stand-in, e.g. for a disease's equations before it is in a sim
        sim = None
    if sim is not None:
        if msg in sim.notes: # Only say it once
            return
        sim.notes.append(msg)
        if verbose == 0: # Kept, but not said
            return
        if getattr(sim, '_held', None) is not None: # Said after the line that starts the run (see ss.Sim.init()), once, though the notes may be kept again as the sim is laid out again
            if msg not in sim._held:
                sim._held.append(msg)
            return
        verbose = sc.ifelse(verbose, sim.pars.verbose)
    verbose = sc.ifelse(verbose, ss.options.verbose)
    if verbose > 0:
        print(f'  Note: {msg}')
    return


#%% Errors

class IncompatibleTypes(CodedError, TypeError):
    """ Typed quantities are combined in a way that has no meaning, such as adding an `ss.per` to an `ss.prob` or `ss.freq`, or a number to a rate; names both types """

class WrongDimension(CodedError, TypeError):
    """
    Typed arithmetic whose result has no type (a rate times a rate, a duration times a duration), or typed values that give a
    slot the wrong dimension: a rate times a duration, or a duration, in `rate=` or after `@`; a rate in `dur=`; both `beta`
    and `n_contacts` carry a time unit; or a typed rate as `beta` on a network with acts (acts per unit time times a rate per
    unit time is not a hazard), or inside `ss.peract`
    """

class ReservedName(CodedError):
    """ A state, transition, parameter, or result is named like an attribute of `ss.Module` (`step`, `results`, `pars`, `describe`, `stratify`, `name`, `dt`, ...) """

class UnknownState(CodedError, KeyError):
    """ A state is named (in `init=`, `first=`, `states=`, a product, or a delivery) that the module does not have. Lists the closest names """

class BadBranchSplit(CodedError):
    """ Constant branch probabilities out of one state do not sum to 1, or `ss.null` is one of several destinations (a branch out of the model is a transition of its own) """

class DuplicateDeclaration(CodedError):
    """ Two members of a module are given the same explicit name, or a state is an endpoint of transitions in two modules """

class NonlinearExposure(CodedError):
    """ A rate is not linear in its bare states (`I ** 2`, `I / (I + K)`, or a term containing two bare states). Says to use `I.n` if a literal count was meant """

class UndrivenTransition(CodedError):
    """ A transition reaches build time with no rate, no duration, and no `p=` """

class UnnamedState(CodedError):
    """
    A state made by `ss.states(3)` or `ss.State()` cannot be matched to a variable name when the module is built (two names for
    one object, a state built inline). Suggests `ss.states('S I R')` or `ss.State('S')`
    """

class ExposureToCount(CodedError):
    """ A bare state that does not hold people appears in a rate (`beta * W`). There is no contact with a reservoir to work out. Suggests `W.n` """

class StrainsByCount(CodedError):
    """ In a disease with strains, an acquisition's rate counts its own states (`beta*I.frac`), which carry no strain. Suggests the transmission `beta*I` """

class UnknownTransition(CodedError, KeyError):
    """ `adjust=` names a transition that does not exist. Lists the ones that do """

class BadSplit(CodedError):
    """ `split=` has the wrong number of entries. (A split that does not sum to 1 is normalized, and `describe()` says so) """

class DuplicateStructuralDimension(CodedError):
    """
    A model or sim is stratified by a dimension it already has, in another way. `stratify` works in place, so a notebook cell that
    is run twice stratifies it in the same way again, which does nothing (with a note) and is not this error. (Stratifying a disease by a people property the sim already has, with the
    same levels, only adds the adjustments and is not an error)
    """

class InvalidAdjustment(CodedError):
    """ A negative multiplier in `adjust=`. (A multiplier that pushes a probability above 1 is clipped, with a note) """

class PartialStratificationMixing(CodedError):
    """
    Contacts are given by a dimension that some of the people who mix do not have: a state the dimension does not apply to
    (`states=`) is the origin or the source of a transmission, or the dimension has `where=`. There is no coordinate for those
    people to mix as
    """

class InconsistentDimension(CodedError):
    """
    Two modules in one sim are stratified by the same people property (age, sex, place) with different groups. Names both
    modules and their groups, and suggests stratifying the sim: `sim.stratify(age=[...])`
    """

class NoGeography(CodedError):
    """ A movement model such as `ss.geo.gravity()` is used on a dimension with no geography to supply distances """

class RouteMismatch(CodedError):
    """
    A key of a per-layer `beta` dict matches no network or layer, or a network or layer has no key; a route has a `beta` of its own in a
    declared model; a `beta` is a positional pair (directions are named: `ss.peract(m2f=..., f2m=...)`); `ss.peract` is used on a route
    without acts, or names a direction the network does not have (or leaves one out, or has keys on a network with no sides), listing
    its directions; or a modifier's `route=` names no route
    """

class UnmodifiableTarget(CodedError):
    """
    The target of `ss.modify` or `ss.change` cannot be resolved to a named numeric quantity (a plain number, a boolean state,
    a name that does not exist). Lists the closest names
    """

class Unapplied(CodedError):
    """
    A modifier, change, or efficacy of a product on a quantity that no solver would apply: a plain parameter of a declared module
    (read only by its own code), a value over "to" in a `p=` table, the `rel_sus` or `rel_trans` of a disease that is not
    transmitted, a condition on people on an inflow or a transition of counts, or an efficacy on a transition with several
    destinations (rather than on one of them); or a stratification of a sim in which nothing uses it. Names it and why
    """

class UnknownEffect(CodedError, KeyError):
    """
    A product names something it cannot act on: a key of a product's `efficacy` that is not something it can act on in the
    disease, a `transition=` of a treatment that the disease does not have (or a cure with no single transition to leave by),
    or a sensitivity of a test for a state the disease does not have. Lists the valid names
    """

class UnknownDisease(CodedError, KeyError):
    """ An intervention needs to know which disease it is for: `disease=` names one that is not a declared disease in the sim, or is not given and the sim has more than one """

class UnknownParameter(CodedError, KeyError):
    """ A parameter name the module does not have is set, directly or through `ss.Sim(pars=...)`. Lists the closest names """

class ShapeError(CodedError):
    """ A value's length does not match its stratification (four values for three age groups). Names the parameter, the transition, and both lengths """

class UnvaluedParameter(CodedError):
    """ An `ss.par` has no value, no bounds to start from, and nothing supplied through `pars=` """

class UnknownLocation(CodedError, KeyError):
    """ A place name cannot be resolved by the data provider (`starsim_data`), or a source has no contact matrix for it. Says where it can be found, if anywhere """

class OutOfBounds(CodedError):
    """ A parameter's value is outside its `bounds`; or a probability per act (`ss.peract`) is outside 0 to 1 """

class UnknownOutcomeSource(CodedError, KeyError):
    """ The source of an `ss.outcome` names nothing in the model """

class OutcomeCycle(CodedError):
    """ Outcomes chain back to themselves """

class DataMisaligned(CodedError):
    """ The data lies entirely outside the simulated period. (Data partly outside is used where it overlaps, with a note) """

class UnknownDataColumn(CodedError, KeyError):
    """
    A data column matches no result, when a fit compares it with the results (at `init()` it is a note). Lists the closest result names and suggests `ss.simdata(..., aliases=dict(result='Column'))`
    to rename it or `ignore_extra=True` to drop it. Also data by stratum for a level the dimension does not have, or for a
    result that is not kept by group, or kept by group only every few steps
    """

class OverlappingSources(CodedError):
    """ A treatment moves people in one state along two transitions of the same disease (`ss.Treatment(moves=)`); each person treated can move along only one transition of a disease """

class TriggerLoop(CodedError):
    """ Moves with no wait (branches nobody waits in) lead back to a state they started from, so people passed through at once would never settle; names the loop """

class ClockOutsideDomain(CodedError):
    """ A clock (`X.time`, `(A | B).time`, `X.progress`) is read where it has no value: by a transition out of a state outside its states, or by a modifier whose `where=` is not inside them (at build; a `where=` that is a function is checked when it runs, where a NaN amount is an error) """

class NoScheduledDwell(CodedError):
    """ `X.progress` is read, but `X` has only rates out of it, so no dwell time is scheduled when people enter it """

class PeopleAndCounts(CodedError):
    """ A transition moves between states of people and states that are counts (not people); names them, and says to leave out `people=False` if they are people """

class InitWhereNobodyIs(CodedError):
    """ `init=` gives people a state only at levels of the stratifications where the split puts nobody, so nobody would start there (given at some such levels only, it is a note) """

class CapabilityError(CodedError):
    """
    The requested solver cannot run part of the model. Three cases: there is no sensible approximation (contact tracing,
    transmission trees, agent-only custom code); a function or condition asked for something that exists only per agent in a
    compartmental run; or an approximation would be needed and `approx=False`. Names the part, what it asked for, and what
    would fix it
    """

class TimeNotKept(CodedError):
    """
    A declared disease run as agents keeps the time each agent entered a state (`ti_X`) only for the states whose times
    the engine reads (for scheduled exits, clocks, and `after=`/`before=`), unless `ss.options.keep_times = True`. Raised
    when code reads one of the others, e.g. an analyzer or a script after the run; names the array and says to set the option
    """

class TimeOutsideSim(CodedError):
    """
    A time given to a delivery (`start=`, a round in `at=`), `ss.modify` (`start=`), or `ss.change` (`at=`) is
    before the start of the sim or after its end. A `stop=` after the end is allowed, and means it never stops. A number is a
    time in the sim's unit (in a sim with dates, that long after the start, or a whole year within the sim), and the message
    says so
    """

class MissingDimension(CodedError, KeyError):
    """ An `ss.transfer` names a stratification the model does not have (a boolean coordinate that does not exist yet is created instead); or a transition uses a value by level of a dimension (`ss.by()`) that the states it moves have no level of; or a value by position along a dimension (`ss.by(age=[...])`) that the sim is not stratified by """

class AlreadyRunError(CodedError, RuntimeError):
    """ A sim that has already run (or been finalized) is stepped, finalized, or added to again; `sim.run()` runs it again from the start """
    def __init__(self, *args):
        default = 'To run it again from the start, call sim.run()'
        args = (f'{args[0]}\n{default}',) + args[1:] if args else (default,)
        super().__init__(*args)

class DistNotInitializedError(CodedError, RuntimeError):
    """ A distribution made with `strict=True` is sampled before it is initialized (by the sim, or with `dist.init()` or `dist.mock()`) """
    def __init__(self, dist=None, msg=None):
        if msg is None:
            msg = f'{dist} has not been initialized; please set strict=False when creating the distribution, or call dist.mock() or dist.init().'
        super().__init__(msg)

class DistNotReadyError(CodedError, RuntimeError):
    """ A distribution made with `strict=True` is sampled twice in one step, which would reuse its random numbers; `dist.jump()` resets it """
    def __init__(self, dist=None, msg=None):
        if msg is None:
            msg = f'{dist} is not ready. This is likely caused by calling a distribution multiple times in a single step. Call dist.jump() to reset.'
        super().__init__(msg)

class DistSeedRepeatError(CodedError, RuntimeError):
    """ Two distributions of a sim have the same seed, or a distribution is jumped back to a state it has passed, so that it would draw the same random numbers twice """
    def __init__(self, dist1=None, dist2=None, msg=None):
        if msg is None:
            msg = f'A common seed was found between {dist1} and {dist2}. This is likely caused by incorrect initialization of the parent Dists object.'
        super().__init__(msg)

class BooleanOperationError(CodedError, NotImplementedError):
    """ A logical operation (and, or, xor, not) is applied to an array of people's values that is not Boolean, e.g. `ppl.age & ppl.female` """
    def __init__(self, arr):
        super().__init__(f'Logical operations are only valid on Boolean arrays, not {arr.dtype}')

class NothingToFit(CodedError):
    """ `fit()` or `sensitivity()` is called and no parameter has `bounds` """

class NoData(CodedError):
    """ `fit()`, `compare_models()`, or `log_density()` is called and no result has data """

class NoPosterior(CodedError):
    """
    `method='nuts'` or `method='pmcmc'` is asked for, and not every data series has a likelihood, or the sim cannot be sampled
    that way (no gradients, for NUTS; not a stochastic compartmental sim that the kernels run whole, for particle MCMC)
    """


#%% Warnings

class TargetStatsNotRecovered(StarsimWarning):
    """ `net.check()` finds that a network does not reproduce a target statistic it was given """

class AtBound(StarsimWarning):
    """ A best-fit value is at or near one of its bounds """

class StageStopped(StarsimWarning):
    """ A stage of a staged fit did not converge; the sequence stops and the last completed fit is returned """

class NotConverged(StarsimWarning):
    """ The chains of a posterior disagree (R-hat above 1.01), or some draws diverged """

class UndrivenUnused(StarsimWarning):
    """ A transition with no rate, duration, or `p=` (undriven) is named by no product, delivery, or `on=` in the sim, so only custom code can move people along it """

class LossyExport(StarsimWarning):
    """ An export to another format, or to code, left something out or changed it. Lists each item """

class ImportIncomplete(StarsimWarning):
    """
    An import could not translate something in the source. Lists each item by name, says what to ask an AI assistant with the
    `import-model` skill to finish it, and warns again when the sim is first run
    """


#%% Notes

class UnitInferred(Note):
    """
    The sim's unit was not given with `unit=` and was taken from `dt`, from `start` and `stop`, or from the typed values. Says
    which unit and where it came from (`unit='day' (from start and stop dates)`). Not said when nothing has a unit: time is
    then in timesteps and nothing was inferred
    """

class LargeStep(Note):
    """ `rate × dt > 0.1` for a converted rate, or a mean dwell time shorter than the step; says how far `rate × dt` is from `1 − exp(−rate × dt)` and points to `sim.check_dt()` """

class MatrixAggregated(Note):
    """
    In a compartmental run, a contact matrix whose age groups are not the model's was aggregated (or split) to the model's groups,
    weighted by population. Names the matrices (once for all of them, e.g. the layers of a country's) and both sets of groups. Agents use the matrix's own groups and get no note
    """

class OverlappingSet(Note):
    """ Two `set=` modifiers are active for the same people at once; the one later in the loop is used. Names both """

class ClippedToRange(Note):
    """
    The modifiers of a rate, duration, or multiplier made it negative, so it was set to 0 where it was; or made the
    probabilities of the branches out of a state come to more than 1, so they were scaled to sum to 1. Said once per quantity
    """

class NotCompiled(Note):
    """ A function or method given to `ss.jit` can't be compiled (or traced by JAX), so it runs as written; says why (e.g. what Numba couldn't compile, or what a method reads that compiled code can't take). Said once per function, or once per sim for a method (in `sim.notes`) """

class SmallCounts(Note):
    """ `ode` with a disease that starts with only a few infected, so that whether it takes off is a matter of chance """

class ExtinctionPossible(Note):
    """ `ode` on a model where the number infected falls so low that the epidemic could plausibly die out by chance """

class StagesUnderAuto(Note):
    """
    A compartmental run represented a dwell time that is not exponential or Erlang by fitted Erlang stages. Names the
    transition, the number of stages, and the mean and standard deviation kept. `stages=` on the transition silences it, and
    `approx=True` keeps it without printing it; with `approx=False` it is a `CapabilityError` instead, and `solver='auto'` runs the
    model as agents
    """

class Unidentifiable(Note):
    """
    Two fitted parameters appear in the model only as a product (or a ratio), or are almost perfectly correlated in the fit:
    over the draws of a posterior, or in how they change the fitted results at the best fit
    """


#%% The catalog

def classes():
    """ The named errors, warnings, and notes, in the order they are defined above """
    return [obj for obj in globals().values() if isinstance(obj, type) and issubclass(obj, (CodedError, StarsimWarning)) and obj not in [CodedError, StarsimWarning, Note]]


def catalog():
    """
    The catalog of every named error, warning, and note, as Markdown tables

    This is the section "The catalog" of the specification's `15-errors.md`, which a
    test checks against it.
    """
    def when(cls):
        return ' '.join(cls.__doc__.split())

    def kind(cls):
        if issubclass(cls, Note):
            return 'note'
        elif issubclass(cls, StarsimWarning):
            return 'warning'
        builtins = [base.__name__ for base in cls.__bases__ if base is not CodedError]
        return 'error' + (f' (also a `{builtins[0]}`)' if builtins else '')

    found = classes()
    counts = {k:sum(kind(cls).startswith(k) for cls in found) for k in ['error', 'warning', 'note']}
    lines = [f'{len(found)} names: {counts["error"]} errors, {counts["warning"]} warnings, and {counts["note"]} notes. Each is `ss.errors.<Name>` (and `ss.CapabilityError` and `ss.AlreadyRunError` are also at the root); the errors are subclasses of `ss.CodedError` (itself a `ValueError`), and the warnings and notes of `ss.StarsimWarning` (itself a `RuntimeWarning`).', '']
    lines += ['| Name | Kind | When |', '|---|---|---|']
    lines += [f'| `{cls.__name__}` | {kind(cls)} | {when(cls)} |' for cls in found]
    return '\n'.join(lines) + '\n'


def check_catalog(filename, write=False):
    """
    Check that the catalog in a Markdown file (between the lines `<!-- catalog start -->`
    and `<!-- catalog end -->`) matches `catalog()`, or write it there

    Returns whether it matched.
    """
    text = sc.loadtext(filename)
    pattern = r'(<!-- catalog start -->\n)(.*?)(<!-- catalog end -->)'
    match = re.search(pattern, text, flags=re.S)
    if match is None:
        errormsg = f'No catalog markers found in {filename}'
        raise ValueError(errormsg)
    new = catalog()
    same = match.group(2).strip() == new.strip()
    if write and not same:
        sc.savetext(filename, text[:match.start(2)] + '\n' + new + '\n' + text[match.end(2):])
    return same

