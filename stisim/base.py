"""
Base classes for *sim models
"""

# import numpy as np
# import sciris as sc
# import functools
# from . import settings as sss

# # Specify all externally visible classes this file defines
# __all__ = []#['BasePeople']

# # Default object getter/setter
# obj_set = object.__setattr__
# base_key = 'uid'  # Define the key used by default for getting length, etc.








# class OldBase:


#     def _brief(self):
#         """
#         Return a one-line description of the people -- used internally and by repr();
#         see people.brief() for the user version.
#         """
#         try:
#             string = f'People(n={len(self):0n})'
#         except Exception as E:  # pragma: no cover
#             string = sc.objectid(self)
#             string += f'Warning, sim appears to be malformed:\n{str(E)}'
#         return string

#     def set(self, key, value):
#         """
#         Set values. Note that this will raise an exception the shapes don't match,
#         and will automatically cast the value to the existing type
#         """
#         self[key][:] = value[:]

#     def get(self, key):
#         """ Convenience method -- key can be string or list of strings """
#         if isinstance(key, str):
#             return self[key]
#         elif isinstance(key, list):
#             arr = np.zeros((len(self), len(key)))
#             for k, ky in enumerate(key):
#                 arr[:, k] = self[ky]
#             return arr


#     @property
#     def f_inds(self):
#         """ Indices of everyone female """
#         return self.true('female')

#     @property
#     def m_inds(self):
#         """ Indices of everyone male """
#         return self.false('female')


#     @property
#     def int_age(self):
#         """ Return ages as an integer """
#         return np.array(self.age, dtype=sss.default_int)

#     @property
#     def round_age(self):
#         """ Rounds age up to the next highest integer"""
#         return np.array(np.ceil(self.age))

#     @property
#     def alive_inds(self):
#         """ Indices of everyone alive """
#         return self.true('alive')

#     @property
#     def n_alive(self):
#         """ Number of people alive """
#         return len(self.alive_inds)

#     def true(self, key):
#         """ Return indices matching the condition """
#         return self[key].nonzero()[-1]

#     def false(self, key):
#         """ Return indices not matching the condition """
#         return (~self[key]).nonzero()[-1]

#     def defined(self, key):
#         """ Return indices of people who are not-nan """
#         return (~np.isnan(self[key])).nonzero()[0]

#     def undefined(self, key):
#         """ Return indices of people who are nan """
#         return np.isnan(self[key]).nonzero()[0]

#     def count(self, key, weighted=True):
#         """ Count the number of people for a given key """
#         inds = self[key].nonzero()[0]
#         if weighted:
#             out = self.scale[inds].sum()
#         else:
#             out = len(inds)
#         return out

#     def count_any(self, key, weighted=True):
#         """ Count the number of people for a given key for a 2D array if any value matches """
#         inds = self[key].sum(axis=0).nonzero()[0]
#         if weighted:
#             out = self.scale[inds].sum()
#         else:
#             out = len(inds)
#         return out

#     def keys(self):
#         """ Returns keys for all non-derived properties of the people object """
#         return [state.name for state in self.states]

#     def indices(self):
#         """ The indices of each people array """
#         return np.arange(len(self))

#     def to_arr(self):
#         """ Return as numpy array """
#         arr = np.empty((len(self), len(self.keys())), dtype=sss.default_float)
#         for k, key in enumerate(self.keys()):
#             if key == 'uid':
#                 arr[:, k] = np.arange(len(self))
#             else:
#                 arr[:, k] = self[key]
#         return arr

#     def to_list(self):
#         """ Return all people as a list """
#         return list(self)