# encoding: utf-8
from __future__ import print_function

import numpy as np

from expr import expr_eval, collect_variables, not_hashable
from exprbases import TableExpression
from utils import prod, LabeledArray
from aggregates import Count
from partition import partition_nd


class GroupBy(TableExpression):
    funcname = 'groupby'
    no_eval = ('expressions', 'expr')
    kwonlyargs = {'expr': None, 'filter': None, 'percent': False,
                  'pvalues': None}

    # noinspection PyNoneFunctionAssignment
    def compute(self, context, *expressions, **kwargs):
        if not expressions:
            raise TypeError("groupby() takes at least 1 argument")

        # TODO: allow lists/tuples of arguments to group by the combinations
        # of keys
        for expr in expressions:
            if isinstance(expr, (bool, int, float)):
                raise TypeError("groupby() does not work with constant "
                                "arguments")
            if isinstance(expr, (tuple, list)):
                raise TypeError("groupby() takes expressions as arguments, "
                                "not a list of expressions")

        # On python 3, we could clean up this code (keyword only arguments).
        expr = kwargs.pop('expr', None)
        if expr is None:
            expr = Count()

#        by = kwargs.pop('by', None)
        filter_value = kwargs.pop('filter', None)
        percent = kwargs.pop('percent', False)
        possible_values = kwargs.pop('pvalues', None)

        expr_vars = [v.name for v in collect_variables(expr)]
        labels = [str(e) for e in expressions]
        columns = [expr_eval(e, context) for e in expressions]

        if filter_value is not None:
            # TODO: make a function out of this, I think we have this pattern
            # in several places
            filtered_columns = [col[filter_value]
                                if isinstance(col, np.ndarray) and col.shape
                                else [col]
                                for col in columns]
            # FIXME: use the actual filter_expr instead of not_hashable
            filtered_context = context.subset(filter_value, expr_vars,
                                              not_hashable)
        else:
            filtered_columns = columns
            filtered_context = context

        if possible_values is None:
            possible_values = [np.unique(col) for col in filtered_columns]

        # We pre-filtered columns instead of passing the filter to partition_nd
        # because it is a bit faster this way. The indices are still correct,
        # because we use them on a filtered_context.
        groups = partition_nd(filtered_columns, True, possible_values)
        if not groups:
            return LabeledArray([], labels, possible_values)

        # evaluate the expression on each group
        # we use not_hashable to avoid storing the subset in the cache
        contexts = [filtered_context.subset(indices, expr_vars, not_hashable)
                    for indices in groups]
        data = [expr_eval(expr, c) for c in contexts]

        # TODO: use group_indices_nd directly to avoid using np.unique
        # this is twice as fast (unique is very slow) but breaks because
        # the rest of the code assumes all combinations are present
#        if self.filter is not None:
#            filter_value = expr_eval(self.filter, context)
#        else:
#            filter_value = True
#
#        d = group_indices_nd(columns, filter_value)
#        pvalues = sorted(d.keys())
#        ndim = len(columns)
#        possible_values = [[pv[i] for pv in pvalues]
#                           for i in range(ndim)]
#        groups = [d[k] for k in pvalues]

        # groups is a (flat) list of list.
        # the first variable is the outer-most "loop",
        # the last one the inner most.

        # add total for each row
        len_pvalues = [len(vals) for vals in possible_values]
        width = len_pvalues[-1]
        height = prod(len_pvalues[:-1])

        rows_indices = [np.concatenate([groups[y * width + x]
                                        for x in range(width)])
                        for y in range(height)]
        cols_indices = [np.concatenate([groups[y * width + x]
                                        for y in range(height)])
                        for x in range(width)]
        cols_indices.append(np.concatenate(cols_indices))

        # evaluate the expression on each "combined" group (ie compute totals)
        row_ctxs = [filtered_context.subset(indices, expr_vars, not_hashable)
                    for indices in rows_indices]
        row_totals = [expr_eval(expr, ctx) for ctx in row_ctxs]
        col_ctxs = [filtered_context.subset(indices, expr_vars, not_hashable)
                    for indices in cols_indices]
        col_totals = [expr_eval(expr, ctx) for ctx in col_ctxs]

        if percent:
            # convert to np.float64 to get +-inf if total_value is int(0)
            # instead of Python's built-in behaviour of raising an exception.
            # This can happen at least when using the default expr (count())
            # and the filter yields empty groups
            total_value = np.float64(col_totals[-1])
            data = [100.0 * value / total_value for value in data]
            row_totals = [100.0 * value / total_value for value in row_totals]
            col_totals = [100.0 * value / total_value for value in col_totals]

#        if self.by or self.percent:
#            if self.percent:
#                total_value = data[-1]
#                divisors = [total_value for _ in data]
#            else:
#                num_by = len(self.by)
#                inc = prod(len_pvalues[-num_by:])
#                num_groups = len(groups)
#                num_categories = prod(len_pvalues[:-num_by])
#
#                categories_groups_idx = [range(cat_idx, num_groups, inc)
#                                         for cat_idx in range(num_categories)]
#
#                divisors = ...
#
#            data = [100.0 * value / divisor
#                    for value, divisor in izip(data, divisors)]

        # convert to a 1d array. We don't simply use data = np.array(data),
        # because if data is a list of ndarray (for example if we use
        # groupby(a, expr=id), *and* all the ndarrays have the same length,
        # the result is a 2d array instead of an array of ndarrays like we
        # need (at this point).
        arr = np.empty(len(data), dtype=type(data[0]))
        arr[:] = data
        data = arr

        # and reshape it
        data = data.reshape(len_pvalues)
        return LabeledArray(data, labels, possible_values,
                            row_totals, col_totals)


functions = {
    'groupby': GroupBy
}
