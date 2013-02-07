from __future__ import print_function

from itertools import izip, count
import random
import os

import numpy as np

import config
from expr import Expr, Variable, expr_eval, collect_variables, traverse_expr
from context import context_length
from utils import PrettyTable
from properties import FilteredExpression, add_individuals
from importer import load_ndarray
from partition import partition_nd, filter_to_indices


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
            groups = [filter_to_indices(filter_value)]
        else:
            groups = [np.arange(num_to_align)]
        assert len(proportions) == 1

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
                    assert score is None or isinstance(score,
                                                       (bool, int, float))
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


class Alignment(FilteredExpression):
    func_name = 'align'

    def __init__(self, score=None, filter=None, take=None, leave=None,
                 fname=None,
                 expressions=None, possible_values=None, proportions=None,
                 on_overflow='default'):
        super(Alignment, self).__init__(score, filter)

        #TODO: make it possible to override expressions/pvalues given in
        # the file

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
    'align': Alignment
}
