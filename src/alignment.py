from __future__ import print_function

from itertools import izip
import os

import numpy as np

import config
from align_link import align_link_nd
from context import context_length, EntityContext
from expr import (Expr, Variable, expr_eval, collect_variables, traverse_expr,
                  missing_values)
from groupby import GroupBy
from links import Link, LinkValue
from partition import partition_nd, filter_to_indices
from properties import FilteredExpression
from importer import load_ndarray
from registry import entity_registry
from utils import PrettyTable, LabeledArray


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


def align_get_indices_nd(ctx_length, groups, need, filter_value, score,
                         take_filter=None, leave_filter=None):
    assert isinstance(need, np.ndarray) and \
           issubclass(need.dtype.type, np.integer)
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

    aligned = np.zeros(ctx_length, dtype=bool)
    for members_indices, group_need in izip(groups, need.flat):
        if len(members_indices):
            affected = group_need
            total_affected += affected

            if take_indices is not None:
                group_always = np.intersect1d(members_indices, take_indices,
                                              assume_unique=True)
                num_always = len(group_always)
                aligned[group_always] = True
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
                    # sort indices. In that case, the alignment will first take
                    # the individuals created last (highest id).
                    sorted_global_indices = group_maybe_indices

                # maybe_to_take is always > 0
                maybe_to_take = affected - num_always
                # take the last X individuals (ie those with the highest score)
                indices_to_take = sorted_global_indices[-maybe_to_take:]

                underflow = maybe_to_take - len(indices_to_take)
                if underflow > 0:
                    total_underflow += underflow
                aligned[indices_to_take] = True
            elif affected < num_always:
                total_overflow += num_always - affected

    num_aligned = np.sum(aligned)
    # this assertion is only valid in the non weighted case
    assert num_aligned == total_affected + total_overflow - total_underflow
    num_partitioned = sum(len(g) for g in groups)
    print(" %d/%d" % (num_aligned, num_partitioned), end=" ")
    if (take_filter is not None) or (leave_filter is not None):
        print("[take %d, leave %d]" % (take, leave), end=" ")
    if total_underflow:
        print("UNDERFLOW: %d" % total_underflow, end=" ")
    if total_overflow:
        print("OVERFLOW: %d" % total_overflow, end=" ")

    return aligned


class AlignmentAbsoluteValues(FilteredExpression):
    func_name = 'align_abs'

    def __init__(self, score, need,
                 filter=None, take=None, leave=None,
                 expressions=None, possible_values=None,
                 errors='default', frac_need='uniform',
                 link=None, secondary_axis=None):
        super(AlignmentAbsoluteValues, self).__init__(score, filter)

        if isinstance(need, basestring):
            fpath = os.path.join(config.input_directory, need)
            need = load_ndarray(fpath, float)

        # need is a single scalar
        if not isinstance(need, (tuple, list, Expr, np.ndarray)):
            need = [need]

        # need is a simple list (no expr inside)
        if isinstance(need, (tuple, list)) and \
           not any(isinstance(p, Expr) for p in need):
            need = np.array(need)

        self.need = need

        if expressions is None:
            expressions = []
        self.expressions = expressions

        if possible_values is None:
            possible_values = []
        else:
            possible_values = [np.array(pv) for pv in possible_values]
        self.possible_values = possible_values

        self.take_filter = take
        self.leave_filter = leave

        self.errors = errors
        self.past_error = None

        self.frac_need = frac_need
        if frac_need not in ('uniform', 'cutoff', 'round'):
            raise Exception("frac_need should be one of: 'uniform', 'cutoff' "
                            "or 'round'")

        self.link = link
        if secondary_axis is not None and link is None:
            raise Exception("the 'secondary_axis' argument is only valid in "
                            "combination with the 'link' argument")
        if not isinstance(secondary_axis, (type(None), int, Variable)):
            raise Exception("'secondary_axis' should be either an integer or "
                            "an axis name")
        self.secondary_axis = secondary_axis

    def traverse(self, context):
        for node in FilteredExpression.traverse(self, context):
            yield node
        for expr in self.expressions:
            for node in traverse_expr(expr, context):
                yield node
        for node in traverse_expr(self.need, context):
            yield node
        for node in traverse_expr(self.take_filter, context):
            yield node
        for node in traverse_expr(self.leave_filter, context):
            yield node
        yield self

    def collect_variables(self, context):
        variables = FilteredExpression.collect_variables(self, context)
        if self.expressions and self.link is None:
            variables |= set.union(*[collect_variables(expr, context)
                                     for expr in self.expressions])
        variables |= collect_variables(self.need, context)
        variables |= collect_variables(self.take_filter, context)
        variables |= collect_variables(self.leave_filter, context)
        return variables

    def _eval_need(self, context):
        expressions = self.expressions
        possible_values = self.possible_values
        if isinstance(self.need, (tuple, list)):
            need = np.array([expr_eval(e, context) for e in self.need])
        elif isinstance(self.need, Expr):
            need = expr_eval(self.need, context)
            # need was a *scalar* expr
            if not (isinstance(need, np.ndarray) and need.shape):
                need = np.array([need])
        else:
            need = self.need

        if isinstance(need, LabeledArray):
            if not expressions:
                expressions = [Variable(name)
                               for name in need.dim_names]
            if not possible_values:
                possible_values = need.pvalues

        assert isinstance(need, np.ndarray)

        if len(expressions) != len(possible_values):
            raise Exception("align() expressions and possible_values "
                            "have different length: %d vs %d"
                            % (len(expressions), len(possible_values)))

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

        return need, expressions, possible_values

    def _handle_frac_need(self, need):
        # handle the "fractional people problem"
        if not issubclass(need.dtype.type, np.integer):
            if self.frac_need == 'uniform':
                int_need = need.astype(int)
                u = np.random.uniform(size=need.shape)
                need = int_need + (u < need - int_need)
            elif self.frac_need == 'cutoff':
                int_need = need.astype(int)
                frac_need = need - int_need
                need = int_need

                # the sum of fractional objects number of extra objects we want
                # aligned
                extra_wanted = int(round(np.sum(frac_need)))
                if extra_wanted:
                    # search cutoff that yield
                    # sum(frac_need >= cutoff) == extra_wanted
                    sorted_frac_need = frac_need.flatten()
                    sorted_frac_need.sort()
                    cutoff = sorted_frac_need[-extra_wanted]
                    extra = frac_need >= cutoff
                    if np.sum(extra) > extra_wanted:
                        # This case can only happen when several bins have the
                        # same frac_need. In this case we could try to be even
                        # closer to our target by randomly selecting X out of
                        # the Y bins which have a frac_need equal to the
                        # cutoff.
                        assert np.sum(frac_need == cutoff) > 1
                    need += extra
            elif self.frac_need == 'round':
                # always use 0.5 as a cutoff point
                need = (need + 0.5).astype(int)

        assert issubclass(need.dtype.type, np.integer)
        return need

    def _add_past_error(self, need, context):
        errors = self.errors
        if errors == 'default':
            errors = context.get('__on_align_error__')

        if errors == 'carry':
            if self.past_error is None:
                self.past_error = np.zeros(need.shape, dtype=int)

            print("adding %d individuals from last period error"
                  % np.sum(self.past_error))
            need += self.past_error

        return need

    def _display_unaligned(self, expressions, ids, columns, unaligned):
        print("Warning: %d individual(s) do not fit in any alignment "
              "category" % np.sum(unaligned))
        header = ['id'] + [str(e) for e in expressions]
        columns = [ids] + columns
        num_rows = len(ids)
        print(PrettyTable([header] +
                          [[col[row] for col in columns]
                           for row in range(num_rows) if unaligned[row]]))

    def evaluate(self, context):
        if self.link is None:
            return self.align_no_link(context)
        else:
            return self.align_link(context)

    def align_no_link(self, context):
        ctx_length = context_length(context)

        scores = expr_eval(self.expr, context)

        need, expressions, possible_values = self._eval_need(context)

        filter_value = expr_eval(self._getfilter(context), context)
        take_filter = expr_eval(self.take_filter, context)
        leave_filter = expr_eval(self.leave_filter, context)

        if filter_value is not None:
            num_to_align = np.sum(filter_value)
        else:
            num_to_align = ctx_length

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
        if sum(len(g) for g in groups) < num_to_align:
            unaligned = np.ones(ctx_length, dtype=bool)
            if filter_value is not None:
                unaligned[~filter_value] = False
            for member_indices in groups:
                unaligned[member_indices] = False
            self._display_unaligned(expressions, context['id'], columns,
                                    unaligned)

        need = need * self._get_need_correction(groups, possible_values)
        need = self._handle_frac_need(need)
        need = self._add_past_error(need, context)

        return align_get_indices_nd(ctx_length, groups, need, filter_value,
                                    scores, take_filter, leave_filter)

    #TODO: somehow merge these two functions with LinkExpression or move them
    # to the Link class
    def target_entity(self, context):
        return entity_registry[self.link._target_entity]

    def target_context(self, context):
        target_entity = self.target_entity(context)
        return EntityContext(target_entity,
                             {'period': context['period'],
                             '__globals__': context['__globals__']})

    def align_link(self, context):
        scores = expr_eval(self.expr, context)

        need, expressions, possible_values = self._eval_need(context)
        need = self._handle_frac_need(need)
        need = self._add_past_error(need, context)

        # handle secondary axis
        secondary_axis = self.secondary_axis
        if isinstance(secondary_axis, Expr):
            axis_name = str(secondary_axis)
            try:
                secondary_axis = need.dim_names.index(axis_name)
            except ValueError:
                raise ValueError("invalid value for secondary_axis: there is "
                                 "no axis named '%s' in the need array"
                                 % axis_name)
        else:
            if secondary_axis >= need.ndim:
                raise Exception("%d is an invalid value for secondary_axis: "
                                "it should be smaller than the number of "
                                "dimension of the need array (%d)"
                                % (secondary_axis, need.ndim))

        # evaluate columns
        target_context = self.target_context(context)
        target_columns = [expr_eval(e, target_context) for e in expressions]
        # this is a one2many, so the link column is on the target side
        link_column = expr_eval(Variable(self.link._link_field),
                                target_context)

        filter_expr = self._getfilter(context)
        if filter_expr is not None:
            reverse_link = Link("reverse", "many2one", self.link._link_field,
                                context['__entity__'].name)
            target_filter = LinkValue(reverse_link, filter_expr, False)
            target_filter_value = expr_eval(target_filter, target_context)

            # It is often not a good idea to pre-filter columns like this
            # because we loose information about "indices", but in this case,
            # it is fine, because we do not need that information afterwards.
            filtered_columns = [col[target_filter_value]
                                  if isinstance(col, np.ndarray) and col.shape
                                  else [col]
                                for col in target_columns]

            link_column = link_column[target_filter_value]
        else:
            filtered_columns = target_columns
            target_filter_value = None

        # compute labels for filtered columns
        # -----------------------------------
        # We can't use _group_labels_light because group_labels assigns labels
        # on a first come, first served basis, not using the order they are
        # in pvalues
        fcols_labels = []
        filtered_length = len(filtered_columns[0])
        unaligned = np.zeros(filtered_length, dtype=bool)
        for fcol, pvalues in zip(filtered_columns, need.pvalues):
            pvalues_index = dict((v, i) for i, v in enumerate(pvalues))
            fcol_labels = np.empty(filtered_length, dtype=np.int32)
            for i in range(filtered_length):
                value_idx = pvalues_index.get(fcol[i], -1)
                if value_idx == -1:
                    unaligned[i] = True
                fcol_labels[i] = value_idx
            fcols_labels.append(fcol_labels)

        num_unaligned = np.sum(unaligned)
        if num_unaligned:
            # further filter label columns and link_column
            validlabels = ~unaligned
            fcols_labels = [labels[validlabels] for labels in fcols_labels]
            link_column = link_column[validlabels]

            # display who are the evil ones
            ids = target_context['id']
            if target_filter_value is not None:
                filtered_ids = ids[target_filter_value]
            else:
                filtered_ids = ids
            self._display_unaligned(expressions, filtered_ids,
                                    filtered_columns, unaligned)
        else:
            del unaligned

        id_to_rownum = context.id_to_rownum
        missing_int = missing_values[int]
        source_ids = link_column

        if len(id_to_rownum):
            source_rows = id_to_rownum[source_ids]
            # filter out missing values: those where the value of the link
            # points to nowhere (-1)
            source_rows[source_ids == missing_int] = missing_int
        else:
            assert np.all(source_ids == missing_int)
            source_rows = []

        # filtered_columns are not filtered further on invalid labels
        # (num_unaligned) but this is not a problem since those will be
        # ignored by GroupBy anyway.
        groupby_expr = GroupBy(*filtered_columns, pvalues=possible_values)

        # target_context is not technically correct, as it is not "filtered"
        # while filtered_columns are, but since we don't use the context
        # "columns", it does not matter.
        num_candidates = expr_eval(groupby_expr, target_context)

        # fetch the list of linked individuals for each local individual.
        # e.g. the list of person ids for each household
        hh = np.empty(context_length(context), dtype=object)
        # we can't use .fill([]) because it reuses the same list for all
        # objects
        for i in range(len(hh)):
            hh[i] = []

        # Even though this is highly sub-optimal, the time taken to create
        # those lists of ids is very small compared to the total time taken
        # for align_other (0.2s vs 4.26), so I shouldn't care too much about
        # it for now.

        # target_row is an index valid for *filtered/label* columns !
        for target_row, source_row in enumerate(source_rows):
            if source_row == -1:
                continue
            hh[source_row].append(target_row)

        aligned, error = \
            align_link_nd(scores, need, num_candidates, hh, fcols_labels,
                          secondary_axis)
        self.past_error = error
        return aligned

    def _get_need_correction(self, groups, possible_values):
        return 1

    def dtype(self, context):
        return bool


class Alignment(AlignmentAbsoluteValues):
    func_name = 'align'

    def __init__(self, score=None, proportions=None,
                 filter=None, take=None, leave=None,
                 expressions=None, possible_values=None,
                 errors='default', frac_need='uniform',
                 fname=None):

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
                                        errors, frac_need)

    def _get_need_correction(self, groups, possible_values):
        data = np.array([len(group) for group in groups])
        len_pvalues = [len(vals) for vals in possible_values]
        return data.reshape(len_pvalues)


functions = {
    'align_abs': AlignmentAbsoluteValues,
    'align': Alignment
}
