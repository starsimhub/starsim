"""
Errors, warnings, and notes.

Most errors are Python's own (`ValueError`, `TypeError`, a `KeyError` for a name
that is not found); the few kept as classes here are those that code may want to
catch by kind, e.g. `ss.CapabilityError`. A note is a line about something that
was assumed or approximated: it is kept in `sim.notes` and printed when `verbose`
is 1 or more (see `ss.note()`).
"""
import sciris as sc
import starsim as ss

__all__ = ['note', 'CapabilityError', 'AlreadyRunError'] # The error for what a solver cannot run and the error for a sim that has already run are also ss.<Name>; the rest are ss.errors.<Name>


def note(msg, sim=None, verbose=None):
    """
    Say something that was assumed or approximated, in one line

    The note is kept in `sim.notes` (if a sim is supplied), whatever the verbosity,
    and is printed if `verbose` (by default the sim's) is 1 or more.

    Args:
        msg (str): what was assumed or approximated
        sim (Sim): the sim to keep the note in
        verbose (int): whether to print it

    **Example**:

        ss.note("unit='day' (from start and stop dates)", sim=sim)
    """
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

class CapabilityError(ValueError):
    """
    The requested solver cannot run part of the model. Three cases: there is no sensible approximation (contact tracing,
    transmission trees, agent-only custom code); a function or condition asked for something that exists only per agent in a
    compartmental run; or an approximation would be needed and `approx=False`. Names the part, what it asked for, and what
    would fix it
    """

class AlreadyRunError(RuntimeError):
    """ A sim that has already run (or been finalized) is stepped, finalized, or added to again; `sim.run()` runs it again from the start """
    def __init__(self, *args):
        default = 'To run it again from the start, call sim.run()'
        args = (f'{args[0]}\n{default}',) + args[1:] if args else (default,)
        super().__init__(*args)

class DistNotInitializedError(RuntimeError):
    """ A distribution made with `strict=True` is sampled before it is initialized (by the sim, or with `dist.init()` or `dist.mock()`) """
    def __init__(self, dist=None, msg=None):
        if msg is None:
            msg = f'{dist} has not been initialized; please set strict=False when creating the distribution, or call dist.mock() or dist.init().'
        super().__init__(msg)

class DistSeedRepeatError(RuntimeError):
    """ Two distributions of a sim have the same seed, or a distribution is jumped back to a state it has passed, so that it would draw the same random numbers twice """
    def __init__(self, dist1=None, dist2=None, msg=None):
        if msg is None:
            msg = f'A common seed was found between {dist1} and {dist2}. This is likely caused by incorrect initialization of the parent Dists object.'
        super().__init__(msg)

class BooleanOperationError(NotImplementedError):
    """ A logical operation (and, or, xor, not) is applied to an array of people's values that is not Boolean, e.g. `ppl.age & ppl.female` """
    def __init__(self, arr):
        super().__init__(f'Logical operations are only valid on Boolean arrays, not {arr.dtype}')


#%% Warnings

class AtBound(RuntimeWarning):
    """ A best-fit value is at or near one of its bounds """

class NotConverged(RuntimeWarning):
    """ The chains of a posterior disagree (R-hat above 1.01), or some draws diverged """

class LossyExport(RuntimeWarning):
    """ An export to another format, or to code, left something out or changed it. Lists each item """

class ImportIncomplete(RuntimeWarning):
    """
    An import could not translate something in the source. Lists each item by name, says what to ask an AI assistant with the
    `import-model` skill to finish it, and warns again when the sim is first run
    """
