from __future__ import print_function

from itertools import izip, product, count
import random
import os

import numpy as np

import config
from expr import Expr, Variable, expr_eval, collect_variables, traverse_expr
from context import context_length, context_subset
from utils import PrettyTable, prod
from properties import (FilteredExpression, TableExpression, GroupCount,
                        add_individuals)
from importer import load_ndarray

try:
    from groupby import filter_to_indices
except ImportError:
    def filter_to_indices(filter_value):
        return filter_value.nonzero()[0]

try:
    from groupby import group_indices_nd

    def partition_nd(columns, filter_value, possible_values):
        # note that since we don't iterate through the columns many times,
        # it's not worth it to copy non contiguous columns in this version
        d = group_indices_nd(columns, filter_value)

        if len(columns) > 1:
            pvalues = product(*possible_values)
        else:
            pvalues = possible_values[0]

        empty_list = np.empty(0, dtype=int)
        return [d.get(pv, empty_list) for pv in pvalues]
except ImportError:
    #TODO: make possible_values a list of combinations of value. In some cases,
    # (eg GroupBy), we are not interested in all possible combinations.
    def partition_nd(columns, filter_value, possible_values):
        """
        * columns is a list of columns containing the data to be partitioned
        * filter_value is a vector of booleans which selects individuals
        * possible_values is an matrix with N vectors containing the possible
          values for each column
        * returns a 1d array of lists of indices
        """
        # make a copy of non contiguous columns. It is only worth it when the
        # number of possible values for that column is large enough to
        # compensate for the cost of the copy, and it is usually the case.
        #XXX: we might want to be more precise about this.
        # 1e5 arrays
        # * not aligned (nor contiguous): always worth it
        # * aligned but not contiguous: never worth it
        # 1e6 arrays
        # * not aligned (nor contiguous): worth it from 6 values
        # * aligned but not contiguous: worth it from 12 values
        contiguous_columns = []
        for col in columns:
            if isinstance(col, np.ndarray) and col.shape:
                if not col.flags.contiguous:
                    col = col.copy()
            else:
                col = [col]
            contiguous_columns.append(col)
        columns = contiguous_columns

        size = tuple([len(colvalues) for colvalues in possible_values])

        #TODO: build result as a flattened array directly instead of calling
        # ravel afterwards

        # initialise result with empty lists
        result = np.empty(size, dtype=list)

        # for each combination of i, j, k:
        for idx in np.ndindex(*size):
            # local_filter = filter & (data0 == values0[i])
            #                       & (data1 == values1[j])
            # it is a bit faster to do: v = expr; v &= b
            # than
            # v = copy(b); v &= expr
            parts = zip(idx, possible_values, columns)
            if parts:
                first_i, first_colvalues, first_coldata = parts[0]
                local_filter = first_coldata == first_colvalues[first_i]
                for i, colvalues, coldata in parts[1:]:
                    local_filter &= coldata == colvalues[i]
                local_filter &= filter_value
            else:
                # filter_value can be a simple boolean, in that case, we
                # get a 0-d array.
                local_filter = np.copy(filter_value)
            if local_filter.shape:
                result[idx] = filter_to_indices(local_filter)
            else:
                # local_filter = True
                assert local_filter
                result[idx] = np.arange(len(columns[0]))

        # pure-python version. It is 10x slower than the NumPy version above
        # but it might be a better starting point to translate to C,
        # especially given that the possible_values are usually sorted (we
        # could sort them too), so we could use some bisect algorithm to find
        # which category it belongs to. python built-in bisect is faster
        # (average time on all indexes) than list.index() starting from ~20
        # elements. We usually have even less elements than that though :(.
        # Strangely bisect on a list is 2x faster than np.searchsorted on an
        # array, even with large array sizes (10^6).
#        fill_with_empty_list = np.frompyfunc(lambda _: [], 1, 1)
#        fill_with_empty_list(result, result)
#
#        for idx, row in enumerate(izip(*columns)):
#            # returns a tuple with the position of the group this row belongs
#            # to. eg. (0, 1, 5)
#            #XXX: this uses strict equality, partitioning using
#            # inequalities might be useful in some cases
#            if filter[idx]:
#                try:
##                    pos = tuple([values_i.index(vi) for vi, values_i
#                    pos = tuple([np.searchsorted(values_i, vi)
#                                 for vi, values_i
#                                 in izip(row, possible_values)])
#                    result[pos].append(idx)
#                except ValueError:
#                    #XXX: issue a warning?
#                    pass
#        for idx in np.ndindex(*size):
#            result[idx] = np.array(result[idx])
        return np.ravel(result)


def kill_axis(axis_name, value, expressions, possible_values, proportions):
    '''possible_values is a list of ndarrays'''

    str_expressions = [str(e) for e in expressions]
    axis_num = str_expressions.index(axis_name)
    expressions = expressions[:axis_num] + expressions[axis_num + 1:]
    possible_values = possible_values[:]
    axis_values = possible_values.pop(axis_num)

    #TODO: make sure possible_values are sorted and use searchsorted instead
    is_wanted_value = axis_values == value
    value_idx = is_wanted_value.nonzero()[0]
    num_idx = len(value_idx)
    if num_idx < 1:
        raise Exception('missing alignment data for %s %s'
                        % (axis_name, value))
    if num_idx > 1:
        raise Exception('invalid alignment data for %s %s: there are %d cells'
                        'for that value (instead of one)'
                        % (num_idx, axis_name, value))
    value_idx = value_idx[0]
    complete_idx = [slice(None) for _ in range(proportions.ndim)]
    complete_idx[axis_num] = value_idx
    proportions = proportions[complete_idx]
    return expressions, possible_values, proportions


def align_get_indices_nd(context, filter_value, score,
                         expressions, possible_values, proportions,
                         take_filter=None, leave_filter=None, weights=None,
                         past_error=None):
    assert len(expressions) == len(possible_values)
    if filter_value is not None:
        num_to_align = np.sum(filter_value)
    else:
        num_to_align = context_length(context)

    if 'period' in [str(e) for e in expressions]:
        period = context['period']
        expressions, possible_values, proportions = \
            kill_axis('period', period, expressions, possible_values,
                      proportions)

    if expressions:
        #TODO: we should also accept a flat version as long as the number of
        # elements is the same (that's how we use it anyway)
        shape1 = proportions.shape
        shape2 = tuple(len(pv) for pv in possible_values)
        assert shape1 == shape2, "%s != %s" % (shape1, shape2)

        # retrieve the columns we need to work with
        columns = [expr_eval(expr, context) for expr in expressions]

        # kill any axis where the value is constant for all individuals
        # satisfying the filter
#        tokill = [(expr, column[0])
#                  for expr, column in zip(expressions, columns)
#                  if isconstant(column, filter_value)]
#        for expr, value in tokill:
#            expressions, possible_values, proportions = \
#                kill_axis(str(expr), value, expressions, possible_values,
#                          proportions)

        if filter_value is not None:
            groups = partition_nd(columns, filter_value, possible_values)
        else:
            groups = partition_nd(columns, True, possible_values)
    else:
        if filter_value is not None:
            groups = [filter_value.nonzero()[0]]
        else:
            groups = [np.arange(num_to_align)]
        assert len(proportions) == 1

    # the sum is not necessarily equal to len(a), because some individuals
    # might not fit in any group (eg if some alignment data is missing)
    num_aligned = sum(len(g) for g in groups)
    if num_aligned < num_to_align:
        if filter_value is not None:
            to_align = set(filter_value.nonzero()[0])
        else:
            to_align = set(xrange(num_to_align))
        aligned = set()
        for member_indices in groups:
            aligned |= set(member_indices)
        unaligned = to_align - aligned
        print("Warning: %d individual(s) do not fit in any alignment category"
              % len(unaligned))
        print(PrettyTable([['id'] + expressions] +
                          [[col[row] for col in [context['id']] + columns]
                           for row in unaligned]))

    if filter_value is not None:
        bool_filter_value = filter_value.copy()
    else:
        bool_filter_value = True

    maybe_filter = bool_filter_value
    if take_filter is not None:
        #XXX: I wonder if users would prefer if filter_value was taken into
        # account or not. This only impacts what it displayed on the console,
        # but still...
        take = np.sum(take_filter)
        #XXX: it would probably be faster to leave the filters as boolean
        # vector and do
        #     take_members = take_filter[member_indices]
        #     group_always = member_indices[take_members]
        # instead of
        #     group_always = np.intersect1d(members_indices, take_indices,
        #                                   assume_unique=True)

        take_indices = (take_filter & bool_filter_value).nonzero()[0]
        maybe_filter &= ~take_filter
    else:
        take = 0
        take_indices = None

    if leave_filter is not None:
        leave = np.sum(leave_filter)
        maybe_filter &= ~leave_filter
    else:
        leave = 0

    if take_filter is not None or leave_filter is not None:
        maybe_indices = maybe_filter.nonzero()[0]
    else:
        maybe_indices = None

    total_underflow = 0
    total_overflow = 0
    total_affected = 0
    total_indices = []
    to_split_indices = []
    to_split_overflow = []
    for group_idx, members_indices, probability in izip(count(), groups,
                                                        proportions.flat):
        if len(members_indices):
            if weights is None:
                expected = len(members_indices) * probability
            else:
                expected = np.sum(weights[members_indices]) * probability
            affected = int(expected)
            if past_error is not None:
                group_overflow = past_error[group_idx]
                if group_overflow != 0:
                    affected -= group_overflow
                past_error[group_idx] = 0

            if random.random() < expected - affected:
                affected += 1
            total_affected += affected

            if take_indices is not None:
                group_always = np.intersect1d(members_indices, take_indices,
                                              assume_unique=True)
                if weights is None:
                    num_always = len(group_always)
                else:
                    num_always = np.sum(weights[group_always])
                total_indices.extend(group_always)
            else:
                num_always = 0

            if affected > num_always:
                if maybe_indices is not None:
                    group_maybe_indices = np.intersect1d(members_indices,
                                                         maybe_indices,
                                                         assume_unique=True)
                else:
                    group_maybe_indices = members_indices
                if isinstance(score, np.ndarray):
                    maybe_members_rank_value = score[group_maybe_indices]
                    sorted_local_indices = np.argsort(maybe_members_rank_value)
                    sorted_global_indices = \
                        group_maybe_indices[sorted_local_indices]
                else:
                    assert isinstance(score, (bool, int, float))
                    # if the score expression is a constant, we don't need to
                    # sort indices. In that case, the alignment will take
                    # the last individuals created first (highest id).
                    sorted_global_indices = group_maybe_indices

                # maybe_to_take is always > 0
                maybe_to_take = affected - num_always
                if weights is None:
                    # take the last X individuals (ie those with the highest
                    # score)
                    indices_to_take = sorted_global_indices[-maybe_to_take:]
                else:
                    maybe_weights = weights[sorted_global_indices]

                    # we need to invert the order because members are sorted
                    # on score ascendingly and we need to take those with
                    # highest score.
                    weight_sums = np.cumsum(maybe_weights[::-1])

                    threshold_idx = np.searchsorted(weight_sums, maybe_to_take)
                    if threshold_idx < len(weight_sums):
                        num_to_take = threshold_idx + 1
                        # if there is enough weight to reach "maybe_to_take"
                        overflow = weight_sums[threshold_idx] - maybe_to_take
                        if overflow > 0:
                            # the next individual has too much weight, so we
                            # need to split it.
                            id_to_split = sorted_global_indices[threshold_idx]
                            past_error[group_idx] = overflow
                            to_split_indices.append(id_to_split)
                            to_split_overflow.append(overflow)
                        else:
                            # we got exactly the number we wanted
                            assert overflow == 0
                    else:
                        # we can't reach our target number of individuals
                        # (probably because of a "leave" filter), so we
                        # take all the ones we have
                        #XXX: should we add *this* underflow to the past_error
                        # too? It would probably accumulate!
                        num_to_take = len(weight_sums)
                    indices_to_take = sorted_global_indices[-num_to_take:]

                underflow = maybe_to_take - len(indices_to_take)
                if underflow > 0:
                    total_underflow += underflow
                total_indices.extend(indices_to_take)
            elif affected < num_always:
                total_overflow += num_always - affected
# this assertion is only valid in the non weighted case
#    assert len(total_indices) == \
#           total_affected + total_overflow - total_underflow
    print(" %d/%d" % (len(total_indices), num_aligned), end=" ")
    if (take_filter is not None) or (leave_filter is not None):
        print("[take %d, leave %d]" % (take, leave), end=" ")
    if total_underflow:
        print("UNDERFLOW: %d" % total_underflow, end=" ")
    if total_overflow:
        print("OVERFLOW: %d" % total_overflow, end=" ")

    if to_split_indices:
        return total_indices, (to_split_indices, to_split_overflow)
    else:
        return total_indices, None


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

        # gender | False | True | total
        #        |    20 |   16 |    35

        # gender | False | True |
        #   dead |       |      | total
        #  False |    20 |   15 |    35
        #   True |     0 |    1 |     1
        #  total |    20 |   16 |    36

        #          |   dead | False | True |
        # agegroup | gender |       |      | total
        #        5 |  False |    20 |   15 |    xx
        #        5 |   True |     0 |    1 |    xx
        #       10 |  False |    25 |   10 |    xx
        #       10 |   True |     1 |    1 |    xx
        #          |  total |    xx |   xx |    xx

        # add headers
        labels = [str(e) for e in expressions]
        if folded_exprs:
            result = [[''] * (folded_exprs - 1) +
                      [labels[-1]] +
                      list(possible_values[-1]) +
                      [''],
                      # 2nd line
                      labels[:-1] +
                      [''] * len(possible_values[-1]) +
                      ['total']]
            categ_values = list(product(*possible_values[:-1]))
            last_line = [''] * (folded_exprs - 1) + ['total']
            categ_values.append(last_line)
            height += 1
        else:
            # if there is only one expression, the headers are different
            result = [[labels[-1]] + list(possible_values[-1]) + ['total']]
            categ_values = [['']]

        for y in range(height):
            result.append(list(categ_values[y]) +
                          data[y * width:(y + 1) * width])

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


class Alignment(FilteredExpression):
    func_name = 'align'

    def __init__(self, score_expr, filter=None, take=None, leave=None,
                 fname=None,
                 expressions=None, possible_values=None, proportions=None,
                 on_overflow='default'):
        super(Alignment, self).__init__(score_expr, filter)

        #TODO: make it possible to override expressions/pvalues given in
        # the file
        #TODO: make score_expr optional and rename it to "score"

        # Q: make the "need" argument (=proportions) the first one and
        #    accept a file name in that argument
        #      align(xyz, fname='al_p_dead_m.csv')
        #    ->
        #      align('al_p_dead_m.csv', xyz)
        # A: I personally would prefer that, but this is backward incompatible,
        #    so I guess users will not like the change ;-).
        #    >>> Ask for more opinions

        # Q: switch to absolute values like align_other?
        #      align(0.0, fname='al_p_dead_m.csv')
        #    ->
        #      align(0.0, AL_P_DEAD_M * groupby(age))
        # A: no, as in that case we have to "duplicate" the information about
        #    columns/dimension (age in groupby() while it is already defined
        #    in the alignment file.
        #    so the solution is to introduce a new "select" method for that
        #    purpose:
        #      align(0.0, AL_P_DEAD_M)
        #      align(proportions=AL_P_DEAD_M)
        #      select(AL_P_DEAD_M * groupby(age), 0.0)

        if possible_values is not None:
            if expressions is None or len(possible_values) != len(expressions):
                raise Exception("align() expressions and possible_values "
                                "arguments should have the same length")

        if proportions is None and fname is None:
            raise Exception("align() needs either a filename or proportions")

        if fname is not None:
            self.load(fname)
        else:
            if expressions is None:
                expressions = []
            self.expressions = [Variable(e) if isinstance(e, basestring) else e
                                for e in expressions]
            if possible_values is None:
                possible_values = []
            self.possible_values = [np.array(pv) for pv in possible_values]

            # e -> e
            # v -> array([v])
            # [v1, v2] -> array([v1, v2])
            # [e1, e2] -> [e1, e2]
            if not isinstance(proportions, (tuple, list, Expr)):
                proportions = [proportions]

            if not any(isinstance(p, Expr) for p in proportions):
                self.proportions = np.array(proportions)
            else:
                self.proportions = proportions

        self.take_filter = take
        self.leave_filter = leave
        self.on_overflow = on_overflow
        self.overflows = None

    def traverse(self, context):
        for node in FilteredExpression.traverse(self, context):
            yield node
        for expr in self.expressions:
            for node in traverse_expr(expr, context):
                yield node
        for node in traverse_expr(self.take_filter, context):
            yield node
        for node in traverse_expr(self.leave_filter, context):
            yield node
        yield self

    def collect_variables(self, context):
        variables = FilteredExpression.collect_variables(self, context)
        if self.expressions:
            variables |= set.union(*[collect_variables(expr, context)
                                     for expr in self.expressions])
        variables |= collect_variables(self.take_filter, context)
        variables |= collect_variables(self.leave_filter, context)
        return variables

    def load(self, fname):
        from exprparser import parse
        fpath = os.path.join(config.input_directory, fname)
        header, possible_values, array = load_ndarray(fpath, float)
        self.expressions = [parse(expr, autovariables=True) for expr in header]
        self.possible_values = possible_values
        self.proportions = array

    def evaluate(self, context):
        scores = expr_eval(self.expr, context)

        on_overflow = self.on_overflow
        if on_overflow == 'default':
            on_overflow = context.get('__on_align_overflow__', 'carry')

        #XXX: I should try to pre-parse weight_col in the entity, rather than
        # here, possibly allowing expressions. Not sure it has any use, but it
        # should not cost us much
        weight_col = context.get('__weight_col__')
        if weight_col is not None:
            weights = expr_eval(Variable(weight_col), context)
            if on_overflow == 'carry' and self.overflows is None:
                self.overflows = np.zeros(len(self.proportions))
        else:
            weights = None

        ctx_filter = context.get('__filter__')
        if self.filter is not None:
            if ctx_filter is not None:
                filter_expr = ctx_filter & self.filter
            else:
                filter_expr = self.filter
        else:
            if ctx_filter is not None:
                filter_expr = ctx_filter
            else:
                filter_expr = None

        if filter_expr is not None:
            filter_value = expr_eval(filter_expr, context)
        else:
            filter_value = None

        take_filter = expr_eval(self.take_filter, context) \
                      if self.take_filter is not None else None
        leave_filter = expr_eval(self.leave_filter, context) \
                       if self.leave_filter is not None \
                       else None

        if isinstance(self.proportions, list):
            proportions = np.array([expr_eval(p, context)
                                      for p in self.proportions])
        elif isinstance(self.proportions, Expr):
            proportions = expr_eval(self.proportions, context)
            if not (isinstance(proportions, np.ndarray) and
                    proportions.shape):
                proportions = np.array([proportions])
        else:
            assert isinstance(self.proportions, np.ndarray)
            proportions = self.proportions

        indices, overflows = \
            align_get_indices_nd(context, filter_value, scores,
                                 self.expressions, self.possible_values,
                                 proportions,
                                 take_filter, leave_filter, weights,
                                 self.overflows)

        if overflows is not None:
            to_split_indices, to_split_overflow = overflows
            if on_overflow == 'split':
                num_birth = len(to_split_indices)
                source_entity = context['__entity__']
                target_entity = source_entity
                array = target_entity.array
                clones = array[to_split_indices]
                id_to_rownum = target_entity.id_to_rownum
                num_individuals = len(id_to_rownum)
                clones['id'] = np.arange(num_individuals,
                                         num_individuals + num_birth)
                #FIXME: self.weight_col is not defined
                clones[self.weight_col] = to_split_overflow
                array[self.weight_col][to_split_indices] -= to_split_overflow
                add_individuals(context, clones)

        return {'values': True, 'indices': indices}

    def dtype(self, context):
        return bool


functions = {
    'align': Alignment,
    'groupby': GroupBy
}
