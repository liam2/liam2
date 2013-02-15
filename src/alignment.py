from __future__ import print_function

from itertools import izip
import os

import numpy as np

import config
from expr import Expr, Variable, expr_eval, collect_variables, traverse_expr
from context import context_length
from utils import PrettyTable, LabeledArray
from properties import FilteredExpression
from importer import load_ndarray
from partition import partition_nd, filter_to_indices


def kill_axis(axis_name, value, expressions, possible_values, need):
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
    complete_idx = [slice(None) for _ in range(need.ndim)]
    complete_idx[axis_num] = value_idx
    need = need[complete_idx]
    return expressions, possible_values, need


def align_get_indices_nd(groups, need, filter_value, score,
                         take_filter=None, leave_filter=None,
                         past_error=None):
    assert score is None or isinstance(score, (bool, int, float, np.ndarray))

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
        take_indices = filter_to_indices(take_filter & bool_filter_value)
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
        maybe_indices = filter_to_indices(maybe_filter)
    else:
        maybe_indices = None

    total_underflow = 0
    total_overflow = 0
    total_affected = 0
    total_indices = []

    #TODO: add other options to handle fractional persons
    int_need = need.astype(int)
    u = np.random.uniform(size=need.shape)
    actual_need = int_need + (u < need - int_need)
    if past_error is not None:
        print("adding %d individuals from last period error"
              % np.sum(past_error))
        need += past_error

    for members_indices, group_need in izip(groups, actual_need.flat):
        if len(members_indices):
            affected = group_need
            total_affected += affected

            if take_indices is not None:
                group_always = np.intersect1d(members_indices, take_indices,
                                              assume_unique=True)
                num_always = len(group_always)
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
                    # if the score expression is a constant, we don't need to
                    # sort indices. In that case, the alignment will take
                    # the last individuals created first (highest id).
                    sorted_global_indices = group_maybe_indices

                # maybe_to_take is always > 0
                maybe_to_take = affected - num_always
                # take the last X individuals (ie those with the highest score)
                indices_to_take = sorted_global_indices[-maybe_to_take:]

                underflow = maybe_to_take - len(indices_to_take)
                if underflow > 0:
                    total_underflow += underflow
                total_indices.extend(indices_to_take)
            elif affected < num_always:
                total_overflow += num_always - affected
    # this assertion is only valid in the non weighted case
    assert len(total_indices) == \
           total_affected + total_overflow - total_underflow
    num_aligned = sum(len(g) for g in groups)
    print(" %d/%d" % (len(total_indices), num_aligned), end=" ")
    if (take_filter is not None) or (leave_filter is not None):
        print("[take %d, leave %d]" % (take, leave), end=" ")
    if total_underflow:
        print("UNDERFLOW: %d" % total_underflow, end=" ")
    if total_overflow:
        print("OVERFLOW: %d" % total_overflow, end=" ")

    return total_indices


class AlignmentAbsoluteValues(FilteredExpression):
    func_name = 'align_abs'

#TODO: make it possible to override expressions/pvalues given in
# the file
    def __init__(self, score, need,
                 filter=None, take=None, leave=None,
                 expressions=None, possible_values=None,
                 errors='default'):
        super(AlignmentAbsoluteValues, self).__init__(score, filter)

        if possible_values is not None:
            if expressions is None or len(possible_values) != len(expressions):
                raise Exception("align() expressions and possible_values "
                                "arguments should have the same length")

        if isinstance(need, basestring):
            self.load(need)
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
            if not isinstance(need, (tuple, list, Expr)):
                need = [need]

            if not any(isinstance(p, Expr) for p in need):
                self.need = np.array(need)
            else:
                self.need = need

        self.take_filter = take
        self.leave_filter = leave
        self.errors = errors
        self.past_error = None

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
        array = load_ndarray(fpath, float)
        self.expressions = [parse(expr, autovariables=True)
                            for expr in array.dim_names]
        self.possible_values = array.pvalues
        self.need = array

    def evaluate(self, context):
        scores = expr_eval(self.expr, context)

        errors = self.errors
        if errors == 'default':
            errors = context.get('__on_align_error__')

        if errors == 'carry' and self.past_error is None:
            self.past_error = np.zeros(len(self.need))

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

        expressions = self.expressions
        possible_values = self.possible_values

        if isinstance(self.need, list):
            need = np.array([expr_eval(e, context) for e in self.need])
        elif isinstance(self.need, Expr):
            need = expr_eval(self.need, context)
            if not (isinstance(need, np.ndarray) and need.shape):
                need = np.array([need])
            if isinstance(need, LabeledArray):
                if not expressions:
                    expressions = [Variable(name)
                                   for name in need.dim_names]
                if not possible_values:
                    possible_values = need.pvalues
        else:
            assert isinstance(self.need, np.ndarray)
            need = self.need
        assert len(expressions) == len(possible_values)

        if filter_value is not None:
            num_to_align = np.sum(filter_value)
        else:
            num_to_align = context_length(context)

        if 'period' in [str(e) for e in expressions]:
            period = context['period']
            expressions, possible_values, need = \
                kill_axis('period', period, expressions, possible_values,
                          need)

        # kill any axis where the value is constant for all individuals
        # satisfying the filter
#        tokill = [(expr, column[0])
#                  for expr, column in zip(expressions, columns)
#                  if isconstant(column, filter_value)]
#        for expr, value in tokill:
#            expressions, possible_values, need = \
#                kill_axis(str(expr), value, expressions, possible_values,
#                          need)

        if expressions:
            # retrieve the columns we need to work with
            columns = [expr_eval(expr, context) for expr in expressions]
            if filter_value is not None:
                groups = partition_nd(columns, filter_value, possible_values)
            else:
                groups = partition_nd(columns, True, possible_values)
        else:
            if filter_value is not None:
                groups = [filter_to_indices(filter_value)]
            else:
                groups = [np.arange(num_to_align)]

        # the sum is not necessarily equal to len(a), because some individuals
        # might not fit in any group (eg if some alignment data is missing)
        num_aligned = sum(len(g) for g in groups)
        if num_aligned < num_to_align:
            if filter_value is not None:
                to_align = set(filter_to_indices(filter_value))
            else:
                to_align = set(xrange(num_to_align))
            aligned = set()
            for member_indices in groups:
                aligned |= set(member_indices)
            unaligned = to_align - aligned
            print("Warning: %d individual(s) do not fit in any alignment "
                  "category" % len(unaligned))
            print(PrettyTable([['id'] + expressions] +
                              [[col[row] for col in [context['id']] + columns]
                               for row in unaligned]))

        need = need * self._get_need_correction(groups, possible_values)
        aligned = \
            align_get_indices_nd(groups, need, filter_value, scores,
                                 take_filter, leave_filter,
                                 self.past_error)

        return {'values': True, 'indices': aligned}

    def _get_need_correction(self, groups, possible_values):
        return 1

    def dtype(self, context):
        return bool


class Alignment(AlignmentAbsoluteValues):
    func_name = 'align'

    def __init__(self, score=None, proportions=None,
                 filter=None, take=None, leave=None,
                 expressions=None, possible_values=None, 
                 errors='default', fname=None):

        if possible_values is not None:
            if expressions is None or len(possible_values) != len(expressions):
                raise Exception("align() expressions and possible_values "
                                "arguments should have the same length")

        if proportions is None and fname is None:
            raise Exception("align() needs either an fname or proportions "
                            "arguments")
        if proportions is not None and fname is not None:
            raise Exception("align() cannot have both fname and proportions "
                            "arguments")
        if fname is not None:
            proportions = fname

        super(Alignment, self).__init__(score, proportions,
                                        filter, take, leave,
                                        expressions, possible_values,
                                        errors)

    def _get_need_correction(self, groups, possible_values):
        data = np.array([len(group) for group in groups])
        len_pvalues = [len(vals) for vals in possible_values]
        return data.reshape(len_pvalues)


functions = {
    'align_abs': AlignmentAbsoluteValues,
    'align': Alignment
}
