from __future__ import print_function

from itertools import product

import numpy as np

from expr import expr_eval, collect_variables, traverse_expr
from context import context_subset
from utils import PrettyTable, prod
from properties import TableExpression, GroupCount
from partition import partition_nd


# this is a quick hack, I should use "standard" GroupBy instead but I'm
# running out of time, so quick hack it is...
#TODO: somehow move "headers"/totals out of GroupBy
def groupby(filtered_columns, possible_values=None):
    if possible_values is None:
        possible_values = [np.unique(col) for col in filtered_columns]
    #TODO: use _group_labels directly because we do not need the
    # indices themselves, only the number of individuals.
    # We could even create a custom function because we don't need the label
    # vector nor the reverse dict, though I am unsure it would gain us much
    # (I guess the big time spender is the hash map lookup).

    # Note that when len(filtered_columns) == 1 we could use np.bincount
    # instead but bincount does not support multiple columns nor record arrays
    groups = partition_nd(filtered_columns, True, possible_values)
    data = [len(member_indices) for member_indices in groups]
    data = np.array(data)
    shape = tuple(len(pv) for pv in possible_values)
    return data.reshape(shape)


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

        possible_values = [np.unique(col) for col in filtered_columns]
        groups = partition_nd(filtered_columns, True, possible_values)

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
        folded_exprs = len(expressions) - 1
        len_pvalues = [len(vals) for vals in possible_values]
        width = len_pvalues[-1]
        height = prod(len_pvalues[:-1])

        def xy_to_idx(x, y):
            # divide by the prod of possible values of expressions to its
            # right, mod by its own number of possible values
            offsets = [(y / prod(len_pvalues[v + 1:folded_exprs]))
                       % len_pvalues[v]
                       for v in range(folded_exprs)]
            return sum(v * prod(len_pvalues[i + 1:])
                       for i, v in enumerate(offsets)) + x

        groups_wh_totals = []
        for y in range(height):
            line_indices = []
            for x in range(width):
                member_indices = groups[xy_to_idx(x, y)]
                groups_wh_totals.append(member_indices)
                line_indices.extend(member_indices)
            groups_wh_totals.append(line_indices)

        # width just increased because of totals
        width += 1

        # add total for each column (including the "total per row" one)
        for x in range(width):
            column_indices = []
            for y in range(height):
                column_indices.extend(groups_wh_totals[y * width + x])
            groups_wh_totals.append(column_indices)

        # evaluate the expression on each group
        expr = self.expr
        used_variables = expr.collect_variables(context)
        #TODO: only add it when really needed, as it makes context_subset much
        # faster in the usual GroupCount() case.
        # Ironically, I think I added this
        # for GroupCount() because otherwise the context was empty, but it
        # is not needed in the end.
        # I'll need to test whether simply removing this works for MIDAS.
        # it does work on the test model
#        used_variables.add('id')

        data = []
        for member_indices in groups_wh_totals:
            local_context = context_subset(context, member_indices,
                                           used_variables)
            data.append(expr_eval(expr, local_context))

        if self.percent:
            # convert to np.float64 to get +-inf if total_value is int(0)
            # instead of Python's built-in behaviour of raising an exception.
            # This can happen at least when using the default expr (grpcount())
            # and the filter yields empty groups
            total_value = np.float64(data[-1])
            data = [100.0 * value / total_value for value in data]

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

        # gender |      |
        #  False | True | total
        #     20 |   16 |    35

        #   dead | gender |      |
        #        |  False | True | total
        #  False |     20 |   15 |    35
        #   True |      0 |    1 |     1
        #  total |     20 |   16 |    36

        # agegroup | gender |  dead |      |
        #          |        | False | True | total
        #        5 |  False |    20 |   15 |    xx
        #        5 |   True |     0 |    1 |    xx
        #       10 |  False |    25 |   10 |    xx
        #       10 |   True |     1 |    1 |    xx
        #          |  total |    xx |   xx |    xx

        # add headers
        result = [[str(e) for e in expressions] +
                  [''] * (width - 1),
                  # 2nd line
                  [''] * folded_exprs +
                  list(possible_values[-1]) +
                  ['total']]
        if folded_exprs:
            categ_values = list(product(*possible_values[:-1]))
            last_line = [''] * (folded_exprs - 1) + ['total']
            categ_values.append(last_line)
            height += 1
            for y in range(height):
                result.append(list(categ_values[y]) +
                              data[y * width:(y + 1) * width])
        else:
            for y in range(height):
                result.append(data[y * width:(y + 1) * width])

        return PrettyTable(result)

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
