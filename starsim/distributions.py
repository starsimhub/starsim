""" 
Distribution support extending scipy with two key functionalities:
1. Callable parameters
2. Ability to use MultiRNG for common random number support
"""

import numpy as np
from starsim.utils import INT_NAN
from starsim.random import SingleRNG, MultiRNG
from starsim import options, int_
from copy import deepcopy
from scipy.stats._discrete_distns import bernoulli_gen
from scipy.stats._distn_infrastructure import rv_sample
from scipy.stats import rv_histogram

__all__ = ['ScipyDistribution', 'ScipyDiscrete', 'ScipyHistogram']


class ScipyBase:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim = None
        self.params = None # To be set in subclasses
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
        for pname in self.params: #[p.name for p in self._param_info()]: # Set in init, xk, pk for rv_sample
            if pname in kwargs and callable(kwargs[pname]):
                kwargs[pname] = kwargs[pname](self.context, self.sim, size)

            # Now do slotting if MultiRNG
            if options.multirng and (pname in kwargs) and (not np.isscalar(kwargs[pname])) and (len(kwargs[pname]) != n_samples):
                # Handle an edge case in which the user provides a custom
                # parameter value for each UID and has multirng enabled. In this
                # case, we sample the distribution more times than needed, and
                # throw away the excess samples.
                
                # The random draws agents receive is determined by their slot.
                # Slots cannot be guaranteed to be unique, so two agents may
                # re-use the same underlying draw in rare instances. The chance
                # of slot reuse can be reduced by increasing the slot_scale
                # parameter. However, the challenged addressed here is that two
                # agents with the same slot might have different parameters
                # assigned. And because the parameter values are used in
                # determining the draw value deep within SciPy, we must repeat
                # the draw for each slot repeat.

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
            repeat_slot_vals = np.full(len(slots), np.nan)
            repeat_slot_vals[repeat_slot_ind] = vals[repeat_slot_u] # Store results
            todo_inds = np.where(np.isnan(repeat_slot_vals))[0]

            if options.verbose > 1 and cnt.max() > 2:
                print(f'MultiRNG slots are repeated up to {cnt.max()} times.')

            #repeat_degree = repeat_slot_cnt.max()
            while len(todo_inds):
                repeat_slot_u, repeat_slot_ind, inv, cnt = np.unique(slots[todo_inds], return_index=True, return_inverse=True, return_counts=True)
                cur_inds = todo_inds[repeat_slot_ind] # Absolute positions being filled this pass

                # Reset RNG, note that ti=0 on initialization and ti+1
                # there after, including ti=0. Assuming that repeat
                # slots are not encountered during sim initialization.
                if self.sim is not None:
                    self.random_state.step(self.sim.ti+1) 
                else:
                    # Likely from a test? Just reset the random state.
                    self.random_state.reset()

                for pname in self.params: #[p.name for p in self._param_info()]:
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

class ScipyDistribution():
    def __init__(self, gen, rng=None):
        self.gen = None
        class starsim_gen(ScipyBase, type(gen.dist)):
            pass

        self.rng = self.set_rng(rng, gen)
        self.gen = starsim_gen(name=gen.dist.name, seed=self.rng)(**gen.kwds)
        self.gen.dist.params = [p.name for p in self.gen.dist._param_info()] # Set in init, xk, pk for rv_sample
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


class ScipyDiscrete(ScipyBase, rv_sample):

    def __new__(cls, a=0, b=np.inf, name=None, badvalue=None,
                moment_tol=1e-8, values=None, inc=1, longname=None,
                shapes=None, seed=None):

        if values is not None:
            # dispatch to a subclass
            return super().__new__(ScipyDiscrete)
        else:
            # business as usual
            return super().__new__(cls)


    def __init__(self, *args, rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = ['xk', 'pk', 'loc', 'scale']
        self.random_state = self.set_rng(rng)
        return

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.random_state = self.set_rng(rng)
        return


    def initialize(self, sim, context):
        # Note: context not used here, but maintained for consistency with ScipyDistribution
        # Passing sim and context here allow callables to receive "self" and sim pointers
        self.sim = sim
        self.context = context
        if isinstance(self.random_state, (SingleRNG, MultiRNG)):
            self.random_state.initialize(sim.rng_container, sim.people.slot)
        return
    '''

    def set_rng(self, rng):
        # Handle random generators
        ret = self.random_state # Default
        if options.multirng and rng and (self.random_state == np.random.mtrand._rand):
            # MultiRNG, rng not none, and the current "random_state" is the
            # numpy global singleton... so let's override
            if isinstance(rng, str):
                ret = MultiRNG(rng) # Crate a new generator with the user-provided string
            elif isinstance(rng, np.random.Generator):
                ret = rng
            else:
                raise Exception(f'The rng must be a string or a np.random.Generator instead of {type(rng)}')
        return ret

    def set_values(self, xk=None, pk=None):
        '''Used to set or reset "values" as setting xk or pk directly fails to update qvals, which are used in sampling.'''

        if xk is None:
            xk = self.xk
        if pk is None:
            pk = self.pk

        if np.shape(xk) != np.shape(pk):
            raise ValueError("xk and pk must have the same shape.")
        if np.less(pk, 0.0).any():
            raise ValueError("All elements of pk must be non-negative.")
        if not np.allclose(np.sum(pk), 1):
            raise ValueError("The sum of provided pk is not 1.")

        indx = np.argsort(np.ravel(xk))
        self.xk = np.take(np.ravel(xk), indx, 0)
        self.pk = np.take(np.ravel(pk), indx, 0)
        self.a = self.xk[0]
        self.b = self.xk[-1]

        self.qvals = np.cumsum(self.pk, axis=0) # Recompute qvals, as these are used when sampling values
        return

    '''
    def rvs(self, *args, **kwargs):
        """
        Return a specified number of samples from the distribution
        """

        size = kwargs['size']
        slots = None
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

class ScipyHistogram(rv_histogram):
    def __init__(self, *args, rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = self.set_rng(rng)
        return

    def initialize(self, sim, context=None):
        # Note: context not used here, but maintained for consistency with ScipyDistribution
        # Passing sim and context here allow callables to receive "self" and sim pointers
        if isinstance(self.random_state, (SingleRNG, MultiRNG)):
            self.random_state.initialize(sim.rng_container, sim.people.slot)
        return

    def set_rng(self, rng):
        # Handle random generators
        ret = self.random_state # Default
        if options.multirng and rng and (self.random_state == np.random.mtrand._rand):
            # MultiRNG, rng not none, and the current "random_state" is the
            # numpy global singleton... so let's override
            if isinstance(rng, str):
                ret = MultiRNG(rng) # Crate a new generator with the user-provided string
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
