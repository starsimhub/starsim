""" 
Distribution support extending scipy with two key functionalities:
1. Callable parameters
2. Ability to use MultiRNG for common random number support
"""

import numpy as np
from stisim.utils import INT_NAN
from stisim.random import SingleRNG, MultiRNG
from stisim import options, int_
from copy import deepcopy

__all__ = ['ScipyDistribution']

from scipy.stats._discrete_distns import bernoulli_gen


class ScipyDistribution():
    def __init__(self, gen, rng=None):
        self.gen = None
        class starsim_gen(type(gen.dist)):
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
                    if not isinstance(size, (int, np.int64, int_)):
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
                elif size.dtype in [int, np.int64, int_]:
                    if not options.multirng:
                        n_samples = len(size)
                    else:
                        v = size.__array__() # TODO - check if this works without calling __array__()?
                        try:
                            slots = self.random_state.slots[v].__array__()
                            max_slot = slots.max()
                        except AttributeError as e:
                            if not isinstance(self.random_state, MultiRNG):
                                raise Exception('With options.multirng and passing agent UIDs to a distribution, the random_state of the distribution must be a MultiRNG.')
                            else:
                                if not self.random_state.initialized:
                                    raise Exception('The MultiRNG instance must be initialized before use.')
                            raise e

                        if max_slot == INT_NAN:
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
                                self.repeat_slot_handling[pname] = kwargs[pname].__array__().copy() # Save full pars for handling later
                                pars_slots[repeat_slot_u] = self.repeat_slot_handling[pname][repeat_slot_ind] # Take first instance of each
                                repeat_slot_flag = True
                            else:
                                pars_slots[slots] = kwargs[pname]
                        else:
                            pars_slots[size] = kwargs[pname]
                        kwargs[pname] = pars_slots

                kwargs['size'] = n_samples
                vals = super().rvs(*args, **kwargs)
                if repeat_slot_flag:
                    # Handle repeated slots
                    if not self.sim.initialized:
                        raise Exception('Repeat slots before sim is fully initialized?')

                    repeat_slot_vals = np.full(len(slots), np.nan)
                    repeat_slot_vals[repeat_slot_ind] = vals[repeat_slot_u] # Store results
                    todo_inds = np.where(np.isnan(repeat_slot_vals))[0]

                    if options.verbose > 1 and cnt.max() > 2:
                        print(f'MultiRNG slots are repeated up to {cnt.max()} times.')

                    #repeat_degree = repeat_slot_cnt.max()
                    while len(todo_inds):
                        repeat_slot_u, repeat_slot_ind, inv, cnt = np.unique(slots[todo_inds], return_index=True, return_inverse=True, return_counts=True)
                        cur_inds = todo_inds[repeat_slot_ind] # Absolute positions being filled this pass
                        self.random_state.step(self.sim.ti+1) # Reset RNG
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
            


        self.rng = self.set_rng(rng, gen)
        self.gen = starsim_gen(name=gen.dist.name, seed=self.rng)(**gen.kwds)
        return

    @staticmethod
    def set_rng(rng, gen):
        # Handle random generators
        ret = gen.random_state # Default
        if options.multirng and rng and (gen.random_state == np.random.mtrand._rand):
            # MultiRNG, rng not none, and the current "random_state" is the
            # numpy global singleton... so let's override
            if isinstance(rng, str):
                ret = MultiRNG(rng) # Crate a new generator with the user-provided string
            elif isinstance(rng, np.random.Generator):
                ret = rng
            else:
                raise Exception(f'The rng must be a string or a np.random.Generator instead of {type(rng)}')
        return ret

    def initialize(self, sim, context):
        # Passing sim and context here allow callables to receive "self" and sim pointers
        self.gen.dist.initialize(sim, context)
        if isinstance(self.rng, (SingleRNG, MultiRNG)):
            self.rng.initialize(sim.rng_container, sim.people.slot)
        return

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
        return result
    
    def __getattr__(self, attr):
        # Returns wrapped generator.(attr) if not a property
        if attr == '__getstate__':
            # Must be from pickle, return a callable function that returns None
            return lambda: None
        #elif attr == '__deepcopy__':
        #    return self
        elif attr in ['__setstate__', '__await__']:
            # Must be from pickle, async programming, copy
            return None
        try:
            return self.__getattribute__(attr)
        except Exception:
            try:
                return getattr(self.gen, attr) # .dist?
            except Exception as e:
                errormsg = f'"{attr}" is not a member of this class or the underlying scipy stats class'
                raise Exception(errormsg)

    def filter(self, size, **kwargs):
        return size[self.gen.rvs(size, **kwargs)]

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