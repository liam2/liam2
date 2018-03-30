# encoding: utf-8
from __future__ import print_function, division

import numpy as np

from expr import (Variable, BinaryOp, getdtype, expr_eval,
                  ispresent, FunctionExpr, always, firstarg_dtype, ComparisonOp, missing_values)
from exprbases import NumpyAggregate, FilteredExpression, WeightedFilteredAggregateFunction
import exprmisc
from context import context_length
from utils import removed, argspec

try:
    import bottleneck as bn
    nanmin = bn.nanmin
    nanmax = bn.nanmax
    nansum = bn.nansum
    # nanmedian, nanstd, nanmean, nanvar
except ImportError:
    nanmin = np.nanmin
    nanmax = np.nanmax
    nansum = np.nansum


class All(NumpyAggregate):
    np_func = np.all
    dtype = always(bool)


class Any(NumpyAggregate):
    np_func = np.any
    dtype = always(bool)


# XXX: inherit from FilteredExpression instead?
class Count(FunctionExpr):
    def compute(self, context, filter=None):
        if filter is None:
            return context_length(context)
        else:
            # TODO: check this at "compile" time (in __init__), though for
            # that we need to know the type of all temporary variables
            # first
            if not np.issubdtype(filter.dtype, bool):
                raise ValueError("count filter must be a boolean expression")
            return np.sum(filter)

    dtype = always(int)


class Min(NumpyAggregate):
    np_func = np.amin
    nan_func = (nanmin,)
    dtype = firstarg_dtype
    # manually defined argspec so that is works with bottleneck (which is a
    # builtin function)
    argspec = argspec('a, axis=None', **NumpyAggregate.kwonlyargs)


class Max(NumpyAggregate):
    np_func = np.amax
    nan_func = (nanmax,)
    dtype = firstarg_dtype
    # manually defined argspec so that is works with bottleneck (which is a
    # builtin function)
    argspec = argspec('a, axis=None', **NumpyAggregate.kwonlyargs)


def na_sum(a, overwrite=False):
    if np.issubdtype(a.dtype, np.inexact):
        func = nansum
    else:
        func = np.sum
        if overwrite:
            a *= ispresent(a)
        else:
            a = a * ispresent(a)
    return func(a)


# class Sum(NumpyAggregate):
#    np_func = np.sum
#    nan_func = (nansum,)
#
#    def dtype(self, context):
#        # TODO: merge this typemap with tsum's
#        typemap = {bool: int, int: int, float: float}
#        return typemap[dtype(self.args[0], context)]


# TODO: inherit from NumpyAggregate, to get support for the axis argument
class Sum(FilteredExpression):
    no_eval = ('expr', 'filter', 'weights')

    def compute(self, context, expr, filter=None, weights=None, skip_na=True):
        filter_expr = self._getfilter(context, filter)
        if filter_expr is not None:
            expr = BinaryOp('*', expr, filter_expr)
        if weights is not None:
            expr_dtype = getdtype(expr, context)
            # missing (-1) * weight should be missing (-1)
            if skip_na and np.issubdtype(expr_dtype, int):
                # expr = where(expr != -1, expr * weights, -1)
                missing_int = missing_values[int]
                expr = exprmisc.Where(ComparisonOp('!=', expr, missing_int),
                                      BinaryOp('*', expr, weights),
                                      missing_int)
            else:
                # expr = expr * weights
                expr = BinaryOp('*', expr, weights)

        values = expr_eval(expr, context)
        values = np.asarray(values)
        return na_sum(values) if skip_na else np.sum(values)

    def dtype(self, context):
        # TODO: merge this typemap with tsum's
        typemap = {bool: int, int: int, float: float}
        return typemap[getdtype(self.args[0], context)]


# class Average(NumpyAggregate):
#    funcname = 'avg'
#    np_func = np.mean
#    nan_func = (nanmean,)
#    dtype = always(float)


# TODO: inherit from NumpyAggregate, to get support for the axis argument
# TODO: use nanmean (np & bn)
class Average(FilteredExpression):
    funcname = 'avg'
    no_eval = ('expr',)

    def compute(self, context, expr, filter=None, skip_na=True):
        # FIXME: either take "contextual filter" into account here (by using
        # self._getfilter), or don't do it in sum & gini
        if filter is not None:
            tmpvar = self.add_tmp_var(context, filter)
            if getdtype(expr, context) is bool:
                # convert expr to int because mul_bbb is not implemented in
                # numexpr
                # expr *= 1
                expr = BinaryOp('*', expr, 1)
            # expr *= filter_values
            expr = BinaryOp('*', expr, tmpvar)
        else:
            filter = True

        values = expr_eval(expr, context)
        values = np.asarray(values)

        if skip_na:
            # we should *not* use an inplace operation because filter can be a
            # simple variable
            filter = filter & ispresent(values)

        if filter is True:
            numrows = len(values)
        else:
            numrows = np.sum(filter)

        if numrows:
            if skip_na:
                return na_sum(values) / float(numrows)
            else:
                return np.sum(values) / float(numrows)
        else:
            return float('nan')

    dtype = always(float)


# TODO: use nanstd (np & bn)
class Std(WeightedFilteredAggregateFunction):
    dtype = always(float)

    def compute(self, context, expr, filter=None, skip_na=True, weights=None):
        values, weights = self.get_filtered_values_weights(expr, filter_values=filter, weights=weights, skip_na=skip_na)
        if weights is None:
            return np.std(values)
        else:
            average = np.average(values, weights=weights)
            variance = np.average((values - average) ** 2, weights=weights)
            return np.sqrt(variance)


def wpercentile(a, weights=None, q=50, weights_type='freq'):
    """
    Calculates percentiles associated with a (possibly weighted) array. Ignores data points with 0 weight.

    Parameters
    ----------
    a : array-like
        The input array from which to calculate percentiles
    weights : array-like, optional
        The weights to assign to values of a. See weights_type for how they are interpreted. Defaults to None (weights
        equal 1).
    q : scalar or array-like, optional
        The percentile(s) to calculate (0 to 100). Defaults to 50.
    weights_type : 'freq'|'sampling', optional
        'freq': frequency weights. They are assumed to be positive integers.
        'sampling': sampling weights. Assumed to be between 0 and 1. In this case, weights are normalized so that they
                    sum to the number of non-missing data points.

    Returns
    -------
    scalar or np.ndarray
        The value(s) associated with the specified percentile(s).

    Notes
    -----
    For both kinds of weights, this implementation gives identical results to np.percentile when all weights are equal,
    but it is only equivalent to calling np.percentile on an expanded array (using weights as the number of times
    each value must be repeated) for weights_type='freq'.

    Examples
    --------

    >>> a = [4, 9, 0, 2, 0, 3, 2, 7, 3, 9]
    >>> round(np.percentile(a, q=40), 2)
    2.6
    >>> w = np.ones_like(a)
    >>> round(wpercentile(a, w, 40), 2)
    2.6

    >>> round(np.percentile([0, 2, 3, 5, 6, 7, 9], q=40), 2)
    3.8
    >>> a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> w = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    >>> round(wpercentile(a, w, 40), 2)
    3.8

    >>> round(np.percentile([0, 1, 2, 3, 4], q=40), 2)
    1.6
    >>> a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> w = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    >>> round(wpercentile(a, w, 40), 2)
    1.6

    >>> a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> w = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    >>> round(np.percentile([5, 6, 7, 8, 9], q=40), 2)
    6.6
    >>> round(wpercentile(a, w, 40), 2)
    6.6

    >>> a = [1, 2, 3]
    >>> w = [3, 5, 4]
    >>> np.repeat(a, w)
    array([1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3])

    >>> np.percentile(np.repeat(a, w), q=[0, 25, 50, 100])
    array([ 1.  ,  1.75,  2.  ,  3.  ])
    >>> wpercentile(a, w, q=[0, 25, 50, 100], weights_type='freq')
    array([ 1.  ,  1.75,  2.  ,  3.  ])

    >>> q = [0, 10, 40, 50, 60, 90, 100]
    >>> np.percentile([1, 2, 3], q=q)
    array([ 1. ,  1.2,  1.8,  2. ,  2.2,  2.8,  3. ])

    >>> wpercentile([1, 2, 3], [0.1, 0.1, 0.1], q=q, weights_type='other')
    array([ 1. ,  1.2,  1.8,  2. ,  2.2,  2.8,  3. ])

    >>> np.percentile([1, 2, 3, 4], q=q)
    array([ 1. ,  1.3,  2.2,  2.5,  2.8,  3.7,  4. ])
    >>> wpercentile([1, 2, 3, 4], [0.1, 0.1, 0.1, 0.1], q=q, weights_type='other')
    array([ 1. ,  1.3,  2.2,  2.5,  2.8,  3.7,  4. ])

    >>> wpercentile([1, 2, 3, 4], [.2, .9, .3, .1], q=q, weights_type='other')
    array([ 2. ,  2. ,  2.2,  2.5,  2.8,  3.7,  4. ])
    >>> wpercentile([1, 2, 3, 4], [.9, .2, .3, .1], q=q, weights_type='other')
    array([ 1. ,  1. ,  1.4,  2. ,  2.6,  3.7,  4. ])

    Check using duplicated values

    >>> wpercentile([1, 2, 2, 3], [.2, .1, .4, .1], q=q, weights_type='other')
    array([ 1. ,  1.3,  2. ,  2. ,  2. ,  2.7,  3. ])
    """
    if not np.isscalar(q):
        q = np.asarray(q)
    if np.any(q < 0) or np.any(q > 100):
        raise ValueError("percentile must be between 0 and 100")
    q = q / 100
    a = np.asarray(a)
    if weights is None:
        weights = np.ones(a.size)
    else:
        weights = np.asarray(weights)
    to_ignore = np.isnan(a) | np.isnan(weights) | (weights <= 0)
    if np.any(to_ignore):
        a = a[~to_ignore]
        weights = weights[~to_ignore]
    if weights_type == 'freq':
        n = np.sum(weights)
    else:
        n = len(a)
        # normalize weights so that they sum to n (do NOT use an inplace op to not modify input)
        # rounding is done to avoid a few pathological cases where wpercentile(a, q=0) > np.min(a)
        # or wpercentile(a, constant_w, q=q) != np.percentile(a, constant_w)
        weights = np.round(weights * n / np.sum(weights), 14)

    assert len(a) == len(weights)
    assert len(a) > 0

    ind_sorted = np.argsort(a)
    sorted_values = a[ind_sorted]
    sorted_weights = weights[ind_sorted]
    cum_weight = np.cumsum(sorted_weights)
    assert np.isclose(cum_weight[-1], n), "cum_weight (%f) != n (%f)" % (cum_weight[-1], n)
    cum_weight_m1 = cum_weight - 1

    # compute target cumulative weight for requested percentile(s)
    target_weight = q * (n - 1)
    # compute the two integral weights bounding each target_weight
    low_target = np.floor(target_weight)
    # high_target is potentially > n - 1, but it does not matter
    high_target = low_target + 1
    # where are we between the two bounds? (== target % 1)
    frac = target_weight - low_target
    # find first value where cum_weight - 1 >= each bound
    low_idx = np.searchsorted(cum_weight_m1, low_target)
    # equivalent to sorted_values[low_idx] but with clipping
    low_value = np.take(sorted_values, low_idx, mode='clip')
    # this is an optimization to avoid searching again the whole array but it only works when q is a scalar
    # high_idx = low_idx + np.searchsorted(cum_weight_m1[low_idx:], high_target)
    high_idx = np.searchsorted(cum_weight_m1, high_target)
    high_value = np.take(sorted_values, high_idx, mode='clip')
    # interpolate the two values
    return low_value + (high_value - low_value) * frac


# TODO: use nanpercentile (np only)
class Percentile(WeightedFilteredAggregateFunction):
    dtype = always(float)

    def compute(self, context, expr, q, filter=None, skip_na=True, weights=None, weights_type='sampling'):
        values, weights = self.get_filtered_values_weights(expr, filter_values=filter, weights=weights, skip_na=skip_na)
        if weights is None:
            return np.percentile(values, q)
        else:
            return wpercentile(values, weights, q, weights_type=weights_type)


# TODO: use nanmedian (np & bn)
class Median(WeightedFilteredAggregateFunction):
    dtype = always(float)

    def compute(self, context, expr, filter=None, skip_na=True, weights=None, weights_type='sampling'):
        values, weights = self.get_filtered_values_weights(expr, filter_values=filter, weights=weights, skip_na=skip_na)
        if weights is None:
            return np.median(values)
        else:
            return wpercentile(values, weights, 50, weights_type=weights_type)


# TODO: filter and skip_na should be provided by an "Aggregate" mixin that is
# used both here and in NumpyAggregate
class Gini(WeightedFilteredAggregateFunction):
    def compute(self, context, expr, filter=None, skip_na=True, weights=None):
        values, weights = self.get_filtered_values_weights(expr, filter_values=filter, weights=weights, skip_na=skip_na)
        if weights is not None:
            # ported from a GPL algorithm written in R found at:
            # https://rdrr.io/cran/acid/src/R/weighted.gini.R
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]
            sorted_weights = weights[sorted_indices]
            # force float to avoid overflows with integer inputs
            cumw = np.cumsum(sorted_weights, dtype=float)
            cumvalw = np.cumsum(sorted_values * sorted_weights, dtype=float)
            sumw = cumw[-1]
            sumvalw = cumvalw[-1]
            if sumvalw == 0:
                print("WARNING: gini(%s, filter=%s): value * weight is all zeros (or nan) for filter"
                      % (self.args[0], self.args[1]))
            # FWIW, this formula with all weights equal to 1 simplifies to the "usual" gini formula without weights,
            # as seen below. Using c = cumxw for concision:
            # cumw = np.arange(1, n + 1)
            # gini = sum(c[1] * 1 - c[0] * 2 + c[2] * 2 - c[1] * 3 + ... + c[-1] * n-1 - c[-2] * n) / (c[-1] * n)
            # gini = sum(- 2 * c[0] - 2 * c[1] - 2 * c[2] - ... - 2 * c[-2] + c[-1] * n-1) / (c[-1] * n)
            # gini = (- 2 * sum(c) + (n + 1) * c[-1]) / (c[-1] * n)
            # gini = (n + 1 - 2 * sum(c) / c[-1]) / n
            return np.sum(cumvalw[1:] * cumw[:-1] - cumvalw[:-1] * cumw[1:]) / (sumvalw * sumw)
        else:
            sorted_values = np.sort(values)
            n = len(values)
            # force float to avoid overflows with integer input expressions
            cumval = np.cumsum(sorted_values, dtype=float)
            sumval = cumval[-1]
            if sumval == 0:
                print("WARNING: gini(%s, filter=%s): expression is all zeros (or nan) for filter"
                      % (self.args[0], self.args[1]))
            # From Wikipedia (https://en.wikipedia.org/wiki/Gini_coefficient)
            # G = 1/n * (n + 1 - 2 * (sum((n + 1 - i) * a[i]) / sum(a[i])))
            #                        i=1..n                    i=1..n
            # but since in Python we are indexing from 0, a[i] should be written a[i - 1].
            # The central part is thus:
            #  = sum((n + 1 - i) * a[i - 1])
            #   i=1..n
            #  = sum((n - i) * a[i])
            #   i=0..n-1
            #  = n * a[0] + (n - 1) * a[1] + (n - 2) * a[2] + ... + 1 * a[n - 1]
            #  = sum(cumsum(a))
            return (n + 1 - 2 * np.sum(cumval) / sumval) / n

    dtype = always(float)


def make_dispatcher(agg_func, elem_func):
    def dispatcher(*args, **kwargs):
        func = agg_func if len(args) == 1 else elem_func
        return func(*args, **kwargs)

    return dispatcher


functions = {
    'all': All, 'any': Any, 'count': Count,
    'min': make_dispatcher(Min, exprmisc.Min),
    'max': make_dispatcher(Max, exprmisc.Max),
    'sum': Sum, 'avg': Average, 'std': Std,
    'median': Median, 'percentile': Percentile,
    'gini': Gini
}

for k, v in functions.items():
    functions['grp' + k] = removed(v, 'grp' + k, k)
