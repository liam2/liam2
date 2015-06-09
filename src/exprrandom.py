# encoding: utf-8
from __future__ import division, print_function

import numpy as np

from expr import firstarg_dtype
from exprbases import NumpyRandom, make_np_class, make_np_classes
from utils import argspec


def make_random(docstring, dtypefunc):
    return make_np_class(NumpyRandom, docstring, dtypefunc)


class Choice(NumpyRandom):
    np_func = np.random.choice
    # choice(a, size=None, replace=True, p=None)
    argspec = argspec('choices, p=None, size=None, replace=True',
                      **NumpyRandom.kwonlyargs)

    # TODO: document the change in behavior for the case where the sum of
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

# need to be explicitly defined because used internally
Normal = make_random("normal(loc=0.0, scale=1.0, size=None)", float)
# WARNING: the docstring was wrong (size is None instead of 1) in numpy < 1.9 !
# Issue: https://github.com/numpy/numpy/pull/4611
Uniform = make_random("uniform(low=0.0, high=1.0, size=None)", float)

functions = {
    'choice': Choice,
    'normal': Normal,
    'uniform': Uniform,
}

functions.update(make_np_classes(NumpyRandom, """
beta(a, b, size=None)
chisquare(df, size=None)
dirichlet(alpha, size=None)
exponential(scale=1.0, size=None)
f(dfnum, dfden, size=None)
gamma(shape, scale=1.0, size=None)
gumbel(loc=0.0, scale=1.0, size=None)
laplace(loc=0.0, scale=1.0, size=None)
lognormal(mean=0.0, sigma=1.0, size=None)
multivariate_normal(mean, cov, size=None)
noncentral_chisquare(df, nonc, size=None)
noncentral_f(dfnum, dfden, nonc, size=None)
pareto(a, size=None)
power(a, size=None)
rayleigh(scale=1.0, size=None)
standard_cauchy(size=None)
standard_exponential(size=None)
standard_gamma(shape, size=None)
standard_normal(size=None)
standard_t(df, size=None)
triangular(left, mode, right, size=None)
vonmises(mu, kappa, size=None)
wald(mean, scale, size=None)
weibull(a, size=None)""", float))

# return integers
functions.update(make_np_classes(NumpyRandom, """
binomial(n, p, size=None)
geometric(p, size=None)
hypergeometric(ngood, nbad, nsample, size=None)
logseries(p, size=None)
negative_binomial(n, p, size=None)
poisson(lam=1.0, size=None)
randint(low, high=None, size=None)
zipf(a, size=None)""", int))

# returns an array of integers (not sure it will work)
functions.update(make_np_classes(NumpyRandom, """
multinomial(n, pvals, size=None)""", int))

if __name__ == '__main__':
    import types

    # set_state(state)
    # get_state()
    # seed(seed=None)
    # rand(d0, d1, ..., dn)
    # randn(d0, d1, ..., dn)
    # bytes(length)
    # shuffle(x)
    # permutation(x)
    # logistic(loc=0.0, scale=1.0, size=None)
    to_ignore = {
        # internal stuff I am not sure I should expose
        "set_state", "get_state",
        # functions with a special signature, which we should probably support
        # but for which make_np_classes would not work
        "seed", "rand", "randn", "bytes", "shuffle", "permutation",
        # aliases we do not want to support (just use randint/uniform)
        "random_integers", "ranf", "random_sample", "random", "sample",
        # we use specific classes to handle them (for various reasons)
        "choice",  'normal', 'uniform', 'randint',
        # we provide a function with the same name
        "logistic"
    }
    d = np.random.__dict__
    for name in sorted(d.keys()):
        if name not in to_ignore:
            func = d[name]
            if isinstance(func, types.BuiltinMethodType):
                print(func.__doc__.splitlines()[1].strip())