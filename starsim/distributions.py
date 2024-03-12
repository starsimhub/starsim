""" 
Distribution support extending scipy with two key functionalities:
1. Callable parameters
2. Ability to use MultiRNG for common random number support
"""

from copy import deepcopy
import numpy as np
import starsim as ss
# from starsim.random import SingleRNG, MultiRNG
from starsim import options
from scipy.stats import (bernoulli, expon, lognorm, norm, poisson, randint, rv_discrete, 
                         uniform, rv_histogram, weibull_min)
from scipy.stats._discrete_distns import bernoulli_gen # TODO: can we remove this?


__all__ = ['ScipyDistribution', 'ScipyHistogram']
__all__ += ['bernoulli', 'expon', 'lognorm', 'norm', 'poisson', 'randint', 'rv_discrete', 
            'uniform', 'weibull_min'] # Add common distributions so they can be imported directly


class ScipyDistribution():
    def __init__(self, gen, rng=None):
        self._gen = gen
        class starsim_gen(type(gen.dist)):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.sim = None
                return

            def initialize(self, sim, context):
                self.sim = sim
                self.context = context
                return

            def rvs(self, *args, **kwargs):
                """
                Return a specified number of samples from the distribution
                """

                size = kwargs['size']
                slots = None
                repeat_slot_flag = False
                self.repeat_slot_handling = {}
                # Work out how many samples to draw. If sampling by UID, this depends on the slots assigned to agents.
                if np.isscalar(size):
                    if type(size) not in [int, np.int64, np.int32]: # CK: TODO: need to refactor
                        raise Exception('Input "size" must be an integer')
                    if size < 0:
                        raise Exception('Input "size" cannot be negative')
                    elif size == 0:
                        return np.array([], dtype=int) # int dtype allows use as index, e.g. when filtering
                    else:
                        n_samples = size
                elif len(size) == 0:
                    return np.array([], dtype=int)
                elif size.dtype == bool:
                    n_samples = len(size) if options.multirng else size.sum()
                elif size.dtype in [int, np.int64, np.int32]: # CK: TODO: need to refactor
                    if not options.multirng:
                        n_samples = len(size)
                    else:
                        try:
                            slots = self.random_state.slots[size]
                            max_slot = slots.max()
                        except AttributeError as e:
                            if not isinstance(self.random_state, ss.RNG):
                                raise Exception('With options.multirng and passing agent UIDs to a distribution, the random_state of the distribution must be a MultiRNG.')
                            else:
                                if not self.random_state.initialized:
                                    raise Exception('The MultiRNG instance must be initialized before use.')
                            raise e

                        if max_slot == ss.INT_NAN:
                            raise Exception('Attempted to sample from an INT_NAN slot')
                        n_samples = max_slot + 1
                else:
                    raise Exception("Unrecognized input type")

                # Now handle distribution arguments
                for pname in [p.name for p in self._param_info()]:
                    if pname in kwargs and callable(kwargs[pname]):
                        kwargs[pname] = kwargs[pname](self.context, self.sim, size)

                    # Now do slotting if MultiRNG
                    if options.multirng and (pname in kwargs) and (not np.isscalar(kwargs[pname])) and (len(kwargs[pname]) != n_samples):
                        # Fill in the blank. The number of UIDs provided is
                        # hopefully consistent with the length of pars
                        # provided, but we need to expand out the pars to be
                        # n_samples in length.
                        if len(kwargs[pname]) not in [len(size), sum(size)]: # Could handle uid and bool separately? len(size) for uid and sum(size) for bool
                            raise Exception('When providing an array of parameters, the length of the parameters must match the number of agents for the selected size (uids).')
                        pars_slots = np.full(n_samples, fill_value=1, dtype=kwargs[pname].dtype) # self.fill_value
                        if slots is not None:
                            if len(slots) != len(np.unique(slots)):
                                # Tricky - repeated slots!
                                if not repeat_slot_flag:
                                    repeat_slot_u, repeat_slot_ind, inv, cnt = np.unique(slots, return_index=True, return_inverse=True, return_counts=True)
                                self.repeat_slot_handling[pname] = kwargs[pname].__array__()  # Save full pars for handling later. Use .__array__() here to provide seamless interoperability with States, UIDArrays, and np.ndarrays
                                pars_slots[repeat_slot_u] = self.repeat_slot_handling[pname][repeat_slot_ind] # Take first instance of each
                                repeat_slot_flag = True
                            else:
                                pars_slots[slots] = kwargs[pname]
                        else:
                            pars_slots[size] = kwargs[pname]
                        kwargs[pname] = pars_slots

                kwargs['size'] = n_samples
                
                # If multirng, make sure the generator is ready to avoid multiple calls without jumping in between
                if options.multirng and not self.random_state.ready:
                    raise ss.NotReadyException(self.random_state.name)

                # Actually sample the random values
                vals = super().rvs(*args, **kwargs)

                # Again if multirng, mark the generator as not ready (needs to be jumped)
                if options.multirng:
                    self.random_state.ready = False

                if repeat_slot_flag:
                    # Handle repeated slots
                    repeat_slot_vals = np.full(len(slots), np.nan)
                    repeat_slot_vals[repeat_slot_ind] = vals[repeat_slot_u] # Store results
                    todo_inds = np.where(np.isnan(repeat_slot_vals))[0]

                    if options.verbose > 1 and cnt.max() > 2:
                        print(f'MultiRNG slots are repeated up to {cnt.max()} times.')

                    #repeat_degree = repeat_slot_cnt.max()
                    while len(todo_inds):
                        repeat_slot_u, repeat_slot_ind, inv, cnt = np.unique(slots.values[todo_inds], return_index=True, return_inverse=True, return_counts=True)
                        cur_inds = todo_inds[repeat_slot_ind] # Absolute positions being filled this pass

                        # Reset RNG, note that ti=0 on initialization and ti+1
                        # there after, including ti=0. Assuming that repeat
                        # slots are not encountered during sim initialization.
                        if self.sim is not None:
                            self.random_state.step(self.sim.ti+1) 
                        else:
                            # Likely from a test? Just reset the random state.
                            self.random_state.reset()

                        for pname in [p.name for p in self._param_info()]:
                            if pname in self.repeat_slot_handling:
                                kwargs_pname = self.repeat_slot_handling[pname][cur_inds]
                                pars_slots = np.full(n_samples, fill_value=1, dtype=kwargs_pname.dtype) # self.fill_value
                                pars_slots[repeat_slot_u] = kwargs_pname # Take first instance of each
                                kwargs[pname] = pars_slots

                        vals = super().rvs(*args, **kwargs) # Draw again for slot repeat
                        #assert np.allclose(slots[cur_inds], repeat_slot_u) # TEMP: Check alignment
                        repeat_slot_vals[cur_inds] = vals[repeat_slot_u]
                        todo_inds = np.where(np.isnan(repeat_slot_vals))[0]

                    vals = repeat_slot_vals
                
                if isinstance(self, bernoulli_gen):
                    vals = vals.astype(bool)

                # _select:
                if not options.multirng or repeat_slot_flag:
                    return vals

                if np.isscalar(size):
                    return vals
                elif size.dtype == bool:
                    return vals[size]
                else:
                    return vals[slots] # slots defined above

        rng = self.set_rng(rng, gen)
        self.gen = starsim_gen(name=gen.dist.name, seed=rng)(**gen.kwds)
        return

    @staticmethod
    def set_rng(rng, gen):
        # Handle random generators
        ret = gen.random_state # Default
        if options.multirng and rng and (gen.random_state == np.random.mtrand._rand):
            # MultiRNG, rng not none, and the current "random_state" is the
            # numpy global singleton... so let's override
            if isinstance(rng, str):
                ret = ss.RNG(rng) # Crate a new generator with the user-provided string
            elif isinstance(rng, np.random.Generator):
                ret = rng
            else:
                raise Exception(f'The rng must be a string or a np.random.Generator instead of {type(rng)}')
        return ret

    def initialize(self, sim, context):
        # Passing sim and context here allow callables to receive "self" and sim pointers
        self.gen.dist.initialize(sim, context)
        if isinstance(self.rng, ss.RNG):
            self.rng.initialize(sim.dists, sim.people.slot)
        return

    @property
    def rng(self):
        return self.dist.random_state

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))

        if self.gen.random_state == np.random.mtrand._rand:
            # The gen is using the centralized numpy random number generator
            # If we copy over the state, we'll get _separate_ random number generators
            # for each distribution which do not draw from the _centralized_ generator
            # in the new instance.
            # Not clear what to do, I suppose keep as centralized?
            result.gen.random_state = np.random.mtrand._rand

        return result

    def __getstate__(self):
        dct = self.__dict__.copy()
        dct.pop('gen')
        return dct

    def __setstate__(self, state):
        self.__init__(state['_gen'])
        return
    
    def __getattr__(self, attr):
        # Returns wrapped generator.(attr) if not a property
        if attr in ['__await__']:
            return None
        try:
            return self.__getattribute__(attr)
        except Exception:
            try:
                return getattr(self.gen, attr) # .dist?
            except Exception:
                errormsg = f'"{attr}" is not a member of this class or the underlying scipy stats class'
                raise Exception(errormsg)

    def filter(self, size, **kwargs):
        return size[self.gen.rvs(size, **kwargs)]


class ScipyHistogram(rv_histogram):
    def __init__(self, *args, rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = self.set_rng(rng)
        return

    def initialize(self, sim, context):
        # Note: context not used here, but maintained for consistency with ScipyDistribution
        # Passing sim and context here allow callables to receive "self" and sim pointers
        if isinstance(self.random_state, ss.RNG):
            self.random_state.initialize(sim.rngs, sim.people.slot)
        return

    def set_rng(self, rng):
        # Handle random generators
        ret = self.random_state # Default
        if options.multirng and rng and (self.random_state == np.random.mtrand._rand):
            # MultiRNG, rng not none, and the current "random_state" is the
            # numpy global singleton... so let's override
            if isinstance(rng, str):
                ret = ss.RNG(rng) # Crate a new generator with the user-provided string
            elif isinstance(rng, np.random.Generator):
                ret = rng
            else:
                raise Exception(f'The rng must be a string or a np.random.Generator instead of {type(rng)}')
        return ret

    def rvs(self, *args, **kwargs):
        """
        Return a specified number of samples from the distribution
        """

        size = kwargs['size']
        slots = None
        # Work out how many samples to draw. If sampling by UID, this depends on the slots assigned to agents.
        if np.isscalar(size):
            if not isinstance(size, int):
                raise Exception('Input "size" must be an integer')
            if size < 0:
                raise Exception('Input "size" cannot be negative')
            elif size == 0:
                return np.array([], dtype=int) # int dtype allows use as index, e.g. when filtering
            else:
                n_samples = size
        elif len(size) == 0:
            return np.array([], dtype=int)
        elif size.dtype == bool:
            n_samples = len(size) if options.multirng else size.sum()
        elif size.dtype in [int, np.int64, np.int32]: # CK: TODO -- need to refactor
            if not options.multirng:
                n_samples = len(size)
            else:
                try:
                    slots = self.random_state.slots[size]
                    max_slot = slots.max()
                except AttributeError as e:
                    if not isinstance(self.random_state, ss.RNG):
                        raise Exception('With options.multirng and passing agent UIDs to a distribution, the random_state of the distribution must be a MultiRNG.')
                    else:
                        if not self.random_state.initialized:
                            raise Exception('The MultiRNG instance must be initialized before use.')
                    raise e

                if max_slot == ss.INT_NAN:
                    raise Exception('Attempted to sample from an INT_NAN slot')
                n_samples = max_slot + 1
        else:
            raise Exception("Unrecognized input type")

        kwargs['size'] = n_samples
        vals = super().rvs(*args, **kwargs)
        
        # _select:
        if not options.multirng:
            return vals

        if np.isscalar(size):
            return vals
        elif size.dtype == bool:
            return vals[size]
        else:
            return vals[slots] # slots defined above

'''
from scipy.stats import bernoulli
class rate(ScipyDistribution):
    """
    Exponentially distributed, accounts for dt.
    Assumes the rate is constant over each dt interval.
    """
    def __init__(self, p, rng=None):
        dist = bernoulli(p=p)
        super().__init__(dist, rng)
        self.rate = rate
        self.dt = None
        return

    def initialize(self, sim, rng):
        self.dt = sim.dt
        self.rng = self.set_rng(rng, self.gen)
        super().initialize(sim)
        return

    def sample(self, size=None):
        n_samples, pars = super().sample(size, rate=self.rate)
        prob = 1 - np.exp(-pars['rate'] * self.dt)
        vals = self.rng.random(size=n_samples)
        vals = self._select(vals, size)
        return vals < prob
'''
