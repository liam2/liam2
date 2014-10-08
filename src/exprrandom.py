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


class Choice(NumpyRandom):
    np_func = np.random.choice
    argspec = argspec('choices', ('p', None), ('size', None), ('replace', True),
                      **NumpyRandom.kwonlyargs)

    #TODO: document the change in behavior for the case where the sum of
    # probabilities is != 1
    # random.choice only checks that the error is < 1e-8 but always
    # divides probabilities by sum(p). It is probably a better choice
    # because it distributes the error to all bins instead of only
    # adjusting the probability of the last choice.

    # We override _eval_args only to change the order of arguments because we
    # do not use the same order than numpy
    def _eval_args(self, context):
        (a, p, size, replace), kwargs = NumpyRandom._eval_args(self, context)
        return (a, size, replace, p), kwargs

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
