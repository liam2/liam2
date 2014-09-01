import numpy as np
from expr import always, FunctionExpr, firstarg_dtype
from exprbases import NumpyRandom
from utils import argspec


class Uniform(NumpyRandom):
    np_func = np.random.uniform
    # The docstring was wrong in np1.7: the default size is None instead of 1.
    # Issue reported as: https://github.com/numpy/numpy/pull/4611
    argspec = argspec(('low', 0.0), ('high', 1.0), ('size', None),
                      **NumpyRandom.kwonlyargs)


class Normal(NumpyRandom):
    np_func = np.random.normal
    argspec = argspec(('loc', 0.0), ('scale', 1.0), ('size', None),
                      **NumpyRandom.kwonlyargs)


class RandInt(NumpyRandom):
    np_func = np.random.randint
    argspec = argspec('low', ('high', None), ('size', None),
                      **NumpyRandom.kwonlyargs)
    dtype = always(int)


class Gumbel(NumpyRandom):
    np_func = np.random.gumbel
    argspec = argspec(('loc', 0.0), ('scale', 1.0), ('size', None),
                      **NumpyRandom.kwonlyargs)


# not inheriting from NumpyRandom as it would get the argspec from an
# nonexistent np_func
class Choice(FunctionExpr):
    funcname = 'choice'

    def compute(self, context, choices, p=None, size=None, replace=True):
        #TODO: __init__ should detect when all args are constants and run
        # a "check_arg_values" method if present
        #TODO: document the change in behavior for the case where the sum of
        # probabilities is != 1
        # random.choice only checks that the error is < 1e-8 but always
        # divides probabilities by sum(p). It is probably a better choice
        # because it distributes the error to all bins instead of only
        # adjusting the probability of the last choice.
        if size is None:
            size = len(context)
        return np.random.choice(choices, size=size, replace=replace, p=p)

    dtype = firstarg_dtype

#------------------------------------

functions = {
    # random
    'uniform': Uniform,
    'normal': Normal,
    'gumbel': Gumbel,
    'choice': Choice,
    'randint': RandInt,
}
