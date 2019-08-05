# encoding: utf-8
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import larray as la

from liam2 import config
from liam2.compat import basestring
from liam2.expr import firstarg_dtype, ComparisonOp, Variable, expr_eval, index_array_by_variables
from liam2.exprbases import NumpyRandom, make_np_class, make_np_classes
from liam2.exprmisc import Where
from liam2.importer import load_ndarray
from liam2.utils import argspec


def make_random(docstring, dtypefunc):
    return make_np_class(NumpyRandom, docstring, dtypefunc)


class Choice(NumpyRandom):
    np_func = np.random.choice
    # choice(a, size=None, replace=True, p=None)
    argspec = argspec('choices, p=None, size=None, replace=True',
                      **NumpyRandom.kwonlyargs)

    def __init__(self, *args, **kwargs):
        NumpyRandom.__init__(self, *args, **kwargs)

        probabilities = self.args[0]
        if isinstance(probabilities, basestring):
            fpath = os.path.join(config.input_directory, probabilities)
            probabilities = load_ndarray(fpath)
            # XXX: store args in a list so that we can modify it?
            # self.args[1] = load_ndarray(fpath, float)
            # XXX: but we should be able to do better than a list, eg.
            # self.args.need = load_ndarray(fpath, float)
            self.args = (probabilities,) + self.args[1:]

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

    def compute(self, context, a, size=None, replace=True, p=None):
        if isinstance(a, la.Array):
            assert p is None
            # FIXME13: rename to outcome or __outcome__ and make the dimension name configurable
            # TODO: add support for generating several variables at once
            # e.g.
            # agegroup, gender
            #         , False, True
            #        0,  0.17, 0.16
            #       30,  0.16, 0.15
            #       60,  0.13, 0.14
            #       90,  0.04, 0.05
            outcomes_axis = a.axes['outcomes']
            outcomes = outcomes_axis.labels
            other_axes = a.axes - outcomes_axis

            if other_axes:
                a = index_array_by_variables(a, context, other_axes)
                p = np.asarray(a.transpose('outcomes'))
            else:
                p = np.asarray(a)
            a = outcomes

        if isinstance(p, (list, np.ndarray)) and len(p) and not np.isscalar(p[0]):
            assert len(p) == len(a)
            assert all(len(px) == size for px in p)
            assert len(a) >= 2

            if isinstance(p, list) and any(isinstance(px, la.Array) for px in p):
                p = [np.asarray(px) for px in p]
            ap = np.asarray(p)
            cdf = ap.cumsum(axis=0)

            # copied & adapted from numpy/random/mtrand/mtrand.pyx
            atol = np.sqrt(np.finfo(np.float64).eps)
            if np.issubdtype(ap.dtype, np.floating):
                atol = max(atol, np.sqrt(np.finfo(ap.dtype).eps))

            if np.any(np.abs(cdf[-1] - 1.) > atol):
                raise ValueError("probabilities do not sum to 1")

            cdf /= cdf[-1]

            # I have not found a way to do this without an explicit loop as
            # np.digitize only supports a 1d array for bins. What we do is
            # worse than a linear "search" since we always evaluate all
            # possibilities (there is no shortcut when the value is found).
            # It might be faster to rewrite this using numba + np.digitize
            # for each individual (assuming it has a low setup overhead).

            # the goal is to build something like:
            # if(u < proba1, outcome1,
            #    if(u < proba2, outcome2,
            #       outcome3))

            data = {'u': np.random.uniform(size=size)}
            expr = a[-1]
            # iterate in reverse and skip last
            pairs = zip(cdf[-2::-1], a[-2::-1])
            for i, (proba_x, outcome_x) in enumerate(pairs):
                data['p%d' % i] = proba_x
                expr = Where(ComparisonOp('<', Variable(None, 'u'),
                                          Variable(None, 'p%d' % i)),
                             outcome_x, expr)
            local_ctx = context.clone(fresh_data=True, entity_data=data)
            return expr.evaluate(local_ctx)
        else:
            return NumpyRandom.compute(self, context, a, size, replace, p)

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
