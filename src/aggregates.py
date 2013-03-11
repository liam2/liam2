import numpy as np

from expr import (Variable, dtype, expr_eval,
                  collect_variables, traverse_expr, get_tmp_varname,
                  ispresent)
from exprbases import EvaluableExpression, NumpyAggregate, FilteredExpression
from context import context_length
from utils import nansum


class GroupMin(NumpyAggregate):
    func_name = 'grpmin'
    np_func = (np.amin,)
    nan_func = (np.nanmin,)
    arg_names = ('a', 'axis', 'out')

    def dtype(self, context):
        return dtype(self.args[0], context)


class GroupMax(NumpyAggregate):
    func_name = 'grpmax'
    np_func = (np.amax,)
    nan_func = (np.nanmax,)
    arg_names = ('a', 'axis', 'out')

    def dtype(self, context):
        return dtype(self.args[0], context)


#FIXME: inherit from NumpyAggregate to have access to all arguments of np.sum
class GroupSum(FilteredExpression):
    func_name = 'grpsum'

    def evaluate(self, context):
        expr = self.expr
        filter_expr = self._getfilter(context)
        if filter_expr is not None:
            expr *= filter_expr
        return nansum(expr_eval(expr, context))

    def dtype(self, context):
        #TODO: merge this typemap with tsum's
        typemap = {bool: int, int: int, float: float}
        return typemap[dtype(self.args[0], context)]


class GroupStd(NumpyAggregate):
    func_name = 'grpstd'
    np_func = (np.std,)
    arg_names = ('a', 'axis', 'dtype', 'out', 'ddof')

    def dtype(self, context):
        return float


class GroupMedian(NumpyAggregate):
    func_name = 'grpmedian'
    np_func = (np.median,)
    arg_names = ('a', 'axis', 'out', 'overwrite_input')

    def dtype(self, context):
        return float


class GroupPercentile(NumpyAggregate):
    func_name = 'grppercentile'
    np_func = (np.percentile,)
    arg_names = ('a', 'q', 'axis', 'out', 'overwrite_input')

    def dtype(self, context):
        return float


class GroupGini(FilteredExpression):
    func_name = 'grpgini'

    def evaluate(self, context):
        # from Wikipedia:
        # G = 1/n * (n + 1 - 2 * (sum((n + 1 - i) * a[i]) / sum(a[i])))
        #                        i=1..n                    i=1..n
        # but sum((n + 1 - i) * a[i])
        #    i=1..n
        #   = sum((n - i) * a[i] for i in range(n))
        #   = sum(cumsum(a))
        values = expr_eval(self.expr, context)
        if isinstance(values, (list, tuple)):
            values = np.array(values)

        filter_expr = self._getfilter(context)
        if filter_expr is not None:
            filter_values = expr_eval(filter_expr, context)
        else:
            filter_values = True
        filter_values &= ispresent(values)
        values = values[filter_values]
        sorted_values = np.sort(values)
        n = len(values)

        # force float to avoid overflows with integer input expressions
        cumsum = np.cumsum(sorted_values, dtype=float)
        values_sum = cumsum[-1]
        if values_sum == 0:
            print "grpgini(%s, filter=%s): expression is all zeros (or nan) " \
                  "for filter" % (self.expr, filter_expr)
        return (n + 1 - 2 * np.sum(cumsum) / values_sum) / n

    def dtype(self, context):
        return float


class GroupCount(EvaluableExpression):
    def __init__(self, filter=None):
        self.filter = filter

    def evaluate(self, context):
        if self.filter is None:
            return context_length(context)
        else:
            #TODO: check this at "compile" time (in __init__), though for
            # that we need to know the type of all temporary variables
            # first
            if dtype(self.filter, context) is not bool:
                raise Exception("grpcount filter must be a boolean expression")
            return np.sum(expr_eval(self.filter, context))

    def dtype(self, context):
        return int

    def traverse(self, context):
        for node in traverse_expr(self.filter, context):
            yield node
        yield self

    def collect_variables(self, context):
        return collect_variables(self.filter, context)

    def __str__(self):
        filter_str = str(self.filter) if self.filter is not None else ''
        return "grpcount(%s)" % filter_str


# we could transform this into a CompoundExpression:
# grpsum(expr, filter=filter) / grpcount(filter) but that would be inefficient.
class GroupAverage(FilteredExpression):
    func_name = 'grpavg'

    def evaluate(self, context):
        expr = self.expr
        #FIXME: either take "contextual filter" into account here (by using
        # self._getfilter), or don't do it in grpsum (& grpgini?)
        if self.filter is not None:
            filter_values = expr_eval(self.filter, context)
            tmp_varname = get_tmp_varname()
            context = context.copy()
            context[tmp_varname] = filter_values
            if dtype(expr, context) is bool:
                # convert expr to int because mul_bbb is not implemented in
                # numexpr
                expr *= 1
            expr *= Variable(tmp_varname)
        else:
            filter_values = True
        values = expr_eval(expr, context)
        filter_values &= np.isfinite(values)
        numrows = np.sum(filter_values)
        if numrows:
            return nansum(values) / float(numrows)
        else:
            return float('nan')

    def dtype(self, context):
        return float


functions = {
    'grpcount': GroupCount,
    'grpmin': GroupMin,
    'grpmax': GroupMax,
    'grpsum': GroupSum,
    'grpavg': GroupAverage,
    'grpstd': GroupStd,
    'grpmedian': GroupMedian,
    'grppercentile': GroupPercentile,
    'grpgini': GroupGini,
}
