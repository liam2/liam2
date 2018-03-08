# encoding: utf-8
from __future__ import print_function, division

import numpy as np

from expr import (Variable, BinaryOp, getdtype, expr_eval,
                  ispresent, FunctionExpr, always, firstarg_dtype, ComparisonOp, missing_values)
from exprbases import NumpyAggregate, FilteredExpression
import exprmisc
from context import context_length
from utils import removed, argspec

try:
    import bottleneck as bn
    nanmin = bn.nanmin
    nanmax = bn.nanmax
    nansum = bn.nansum
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

    def compute(self, context, expr, filter=None, skip_na=True, weights=None):
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
class Std(NumpyAggregate):
    np_func = np.std
    dtype = always(float)


# TODO: use nanmedian (np & bn)
class Median(NumpyAggregate):
    np_func = np.median
    dtype = always(float)


# TODO: use nanpercentile (np only)
class Percentile(NumpyAggregate):
    np_func = np.percentile
    dtype = always(float)


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

    >>> np.percentile([1, 2, 3], q=[0, 10, 40, 50, 60, 100])
    array([ 1. ,  1.2,  1.8,  2. ,  2.2,  3. ])
    >>> wpercentile([1, 2, 3], [0.1, 0.1, 0.1], q=[0, 10, 40, 50, 60, 100], weights_type='other')
    array([ 1. ,  1.2,  1.8,  2. ,  2.2,  3. ])

    >>> np.percentile([1, 2, 3, 4], q=[40, 50])
    array([ 2.2,  2.5])
    >>> wpercentile([1, 2, 3, 4], [0.1, 0.1, 0.1, 0.1], q=[40, 50], weights_type='other')
    array([ 2.2,  2.5])
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
        weights = weights * n / np.sum(weights)

    assert len(a) == len(weights)
    assert len(a) > 0

    ind_sorted = np.argsort(a)
    sorted_values = a[ind_sorted]
    sorted_weights = weights[ind_sorted]
    cum_sorted_weight = np.cumsum(sorted_weights)
    assert np.isclose(cum_sorted_weight[-1], n), "cum_sorted_weight (%f) != n (%f)" % (cum_sorted_weight[-1], n)

    if weights_type == 'freq':
        # compute target cumulative weight for requested percentile(s)
        target = q * (n - 1)
        # find indices which bound this cumweight
        idx_left = np.searchsorted(cum_sorted_weight, target, side='right')
        idx_right = np.searchsorted(cum_sorted_weight, target + 1, side='right')
        idx_right = np.minimum(idx_right, len(sorted_values) - 1)
        # where are we between the two bounds?
        frac = target - np.floor(target)
        # get two values to interpolate between
        v_left = sorted_values[idx_left]
        v_right = sorted_values[idx_right]
        return v_left + (v_right - v_left) * frac
    else:
        p = (cum_sorted_weight - sorted_weights) / (n - 1)
        return np.interp(q, p, sorted_values)


# TODO: filter and skip_na should be provided by an "Aggregate" mixin that is
# used both here and in NumpyAggregate
class Gini(FilteredExpression):
    no_eval = ('filter',)

    def compute(self, context, expr, filter=None, skip_na=True):
        values = np.asarray(expr)

        filter_expr = self._getfilter(context, filter)
        if filter_expr is not None:
            filter_values = expr_eval(filter_expr, context)
        else:
            filter_values = True
        if skip_na:
            # we should *not* use an inplace operation because filter_values
            # can be a simple variable
            filter_values = filter_values & ispresent(values)
        if filter_values is not True:
            values = values[filter_values]

        # from Wikipedia:
        # G = 1/n * (n + 1 - 2 * (sum((n + 1 - i) * a[i]) / sum(a[i])))
        #                        i=1..n                    i=1..n
        # but sum((n + 1 - i) * a[i])
        #    i=1..n
        #   = sum((n - i) * a[i] for i in range(n))
        #   = sum(cumsum(a))
        sorted_values = np.sort(values)
        n = len(values)

        # force float to avoid overflows with integer input expressions
        cumsum = np.cumsum(sorted_values, dtype=float)
        values_sum = cumsum[-1]
        if values_sum == 0:
            print("gini(%s, filter=%s): expression is all zeros (or nan) "
                  "for filter" % (self.args[0], filter))
        return (n + 1 - 2 * np.sum(cumsum) / values_sum) / n

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
