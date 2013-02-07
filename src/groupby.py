from __future__ import print_function

import numpy as np

from expr import expr_eval, collect_variables, traverse_expr
from context import context_subset
from utils import prod, LabeledArray
from properties import TableExpression, GroupCount
from partition import partition_nd


class GroupBy(TableExpression):
#    func_name = 'groupby'

    def __init__(self, *args, **kwargs):
        assert len(args), "groupby needs at least one expression"
        #TODO: allow lists/tuples of arguments to group by the combinations
        # of keys
        for arg in args:
            if isinstance(arg, (bool, int, float)):
                raise TypeError("groupby takes expressions as arguments, "
                                "not scalar values")
            if isinstance(arg, (tuple, list)):
                raise TypeError("groupby takes expressions as arguments, "
                                "not a list of expressions")
        self.expressions = args

        # On python 3, we could clean up this code (keyword only arguments).
        expr = kwargs.pop('expr', None)
        if expr is None:
            expr = GroupCount()
        self.expr = expr

#        by = kwargs.pop('by', None)
#        if isinstance(by, Expr):
#            by = (by,)
#        self.by = by

        self.filter = kwargs.pop('filter', None)
        self.percent = kwargs.pop('percent', False)
        self.pvalues = kwargs.pop('pvalues', None)

        if kwargs:
            kwarg, _ = kwargs.popitem()
            raise TypeError("'%s' is an invalid keyword argument for groupby()"
                            % kwarg)

    def evaluate(self, context):
        expressions = self.expressions
        columns = [expr_eval(e, context) for e in expressions]
        if self.filter is not None:
            filter_value = expr_eval(self.filter, context)
            #TODO: make a function out of this, I think we have this pattern
            # in several places
            filtered_columns = [col[filter_value]
                                   if isinstance(col, np.ndarray) and col.shape
                                   else [col]
                                for col in columns]
        else:
            filtered_columns = columns

        possible_values = self.pvalues
        if possible_values is None:
            possible_values = [np.unique(col) for col in filtered_columns]
        groups = partition_nd(filtered_columns, True, possible_values)
        if not groups:
            return

        # evaluate the expression on each group
        expr = self.expr
        used_vars = expr.collect_variables(context)
        data = [expr_eval(expr, context_subset(context, indices, used_vars))
                for indices in groups]

        #TODO: use group_indices_nd directly to avoid using np.unique
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
        row_totals = [expr_eval(expr, context_subset(context, inds, used_vars))
                      for inds in rows_indices]
        col_totals = [expr_eval(expr, context_subset(context, inds, used_vars))
                      for inds in cols_indices]

        if self.percent:
            # convert to np.float64 to get +-inf if total_value is int(0)
            # instead of Python's built-in behaviour of raising an exception.
            # This can happen at least when using the default expr (grpcount())
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

        # add headers
        labels = [str(e) for e in expressions]
        data = np.array(data)
        data = data.reshape(len_pvalues)
        return LabeledArray(data, labels, possible_values,
                            row_totals, col_totals)

    def traverse(self, context):
        for expr in self.expressions:
            for node in traverse_expr(expr, context):
                yield node
        for node in traverse_expr(self.expr, context):
            yield node
        for node in traverse_expr(self.filter, context):
            yield node
        yield self

    def collect_variables(self, context):
        variables = set.union(*[collect_variables(expr, context)
                                for expr in self.expressions])
        variables |= collect_variables(self.filter, context)
        variables |= collect_variables(self.expr, context)
        return variables


functions = {
    'groupby': GroupBy
}
