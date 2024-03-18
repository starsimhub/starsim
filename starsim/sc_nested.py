"""
Temporary copy of sc_nested  until Sciris 3.1.5 is released
"""

import collections as co
import numpy as np
import pandas as pd
import sciris as sc

# Define objects for which it doesn't make sense to descend further -- used here and sc.equal()
atomic_classes = [np.ndarray, pd.Series, pd.DataFrame, pd.core.indexes.base.Index]


class IterObj(object):
    """
    Object iteration manager
    """    
    def __init__(self, obj, func=None, inplace=False, copy=False, leaf=False, recursion=0, 
                 atomic='default', rootkey='root', verbose=False, _memo=None, _trace=None, _output=None, 
                 custom_type=None, custom_iter=None, custom_get=None, custom_set=None, *args, **kwargs):
        
        # Default argument
        self.obj       = obj
        self.func      = func
        self.inplace   = inplace
        self.copy      = copy
        self.leaf      = leaf
        self.recursion = recursion
        self.atomic    = atomic
        self.rootkey   = rootkey
        self.verbose   = verbose
        self._memo     = _memo
        self._trace    = _trace
        self._output   = _output
        self.func_args = args
        self.func_kw   = kwargs
        
        # Custom arguments
        self.custom_type = custom_type
        self.custom_iter = custom_iter
        self.custom_get  = custom_get
        self.custom_set  = custom_set
        
        # Handle inputs
        if self.func is None: # Define the default function
            self.func = lambda obj: obj 
            
        atomiclist = sc.tolist(self.atomic)
        if 'default' in atomiclist: # Handle objects to not descend into
            atomiclist.remove('default')
            atomiclist = atomic_classes + atomiclist
        self.atomic = tuple(atomiclist)
            
        if self._memo is None:
            self._memo = co.defaultdict(int)
            self._memo[id(obj)] = 1 # Initialize the memo with a count of this object
            
        if self._trace is None:
            self._trace = [] # Handle where we are in the object
            if inplace and copy: # Only need to copy once
                self.obj = sc.dcp(obj)
        
        if self._output is None: # Handle the output at the root level
            self._output = sc.objdict()
            if not inplace:
                self._output[self.rootkey] = self.func(self.obj, *args, **kwargs)
                
        # Check what type of object we have
        self.itertype = self.check_iter_type(self.obj)
        
        return
    
    def indent(self, string='', space='  '):
        """ Print, with output indented successively """
        if self.verbose:
            print(space*len(self._trace) + string)
        return
        
    def iteritems(self):
        """ Return an iterator over items in this object """
        self.indent(f'Iterating with type "{self.itertype}"')
        out = None
        if self.custom_iter:
            out = self.custom_iter(self.obj)
        if out is None:
            if self.itertype == 'dict':
                out = self.obj.items()
            elif self.itertype == 'list':
                out = enumerate(self.obj)
            elif self.itertype == 'object':
                out = self.obj.__dict__.items()
            else:
                out = {}.items() # Return nothing if not recognized
        return out
    
    def getitem(self, key):
        """ Get the value for the item """
        self.indent(f'Getting key "{key}"')
        if self.itertype in ['dict', 'list']:
            return self.obj[key]
        elif self.itertype == 'object':
            return self.obj.__dict__[key]
        elif self.custom_get:
            return self.custom_get(self.obj, key)
        else:
            return None
    
    def setitem(self, key, value):
        """ Set the value for the item """
        self.indent(f'Setting key "{key}"')
        if self.itertype in ['dict', 'list']:
            self.obj[key] = value
        elif self.itertype == 'object':
            self.obj.__dict__[key] = value
        elif self.custom_set:
            self.custom_set(self.obj, key, value)
        return
    
    def check_iter_type(self, obj):
        """ Shortcut to check_iter_type() """
        return sc.sc_nested.check_iter_type(obj, known=self.atomic, custom=self.custom_type)
    
    def iterate(self):
        """ Actually perform the iteration over the object """
        
        # Iterate over the object
        for key,subobj in self.iteritems():
            newid = id(subobj)
            if (newid in self._memo) and (self._memo[newid] > self.recursion): # If we've already parsed this object, don't parse it again
                continue
            else:
                self._memo[newid] += 1
                trace = self._trace + [key]
                newobj = subobj
                subitertype = self.check_iter_type(subobj)
                self.indent(f'Working on {trace}, leaf={self.leaf}, type={str(subitertype)}')
                if not (self.leaf and subitertype):
                    newobj = self.func(subobj, *self.func_args, **self.func_kw)
                    if self.inplace:
                        self.setitem(key, newobj)
                    else:
                        self._output[tuple(trace)] = newobj
                io = IterObj(self.getitem(key), self.func, inplace=self.inplace, leaf=self.leaf, recursion=self.recursion,  # Create a new instance
                        atomic=self.atomic, verbose=self.verbose, _memo=self._memo, _trace=trace, _output=self._output, 
                        custom_type=self.custom_type, custom_iter=self.custom_iter, custom_get=self.custom_get, custom_set=self.custom_set,
                        *self.func_args, **self.func_kw)
                io.iterate() # Run recursively
            
        if self.inplace:
            newobj = self.func(self.obj, *self.func_args, **self.func_kw) # Set at the root level
            return newobj
        else:
            if (not self._trace) and (len(self._output)>1) and self.leaf: # We're at the top level, we have multiple entries, and only leaves are requested
                self._output.pop('root') # Remove "root" with leaf=True if it's not the only node
            return self._output