# encoding: utf-8
from __future__ import print_function

from itertools import izip
import os

import numpy as np

import config
from align_link import align_link_nd
from context import context_length
from expr import Expr, Variable, expr_eval, missing_values, always
from exprbases import FilteredExpression
from groupby import GroupBy
from links import LinkGet, Many2One
from partition import partition_nd, filter_to_indices
from importer import load_ndarray
from utils import PrettyTable, LabeledArray


def kill_axis(axis_name, value, expressions, possible_values, need):
    """possible_values is a list of ndarrays"""

    # When we transition to LArray, this whole function could be replaced by:
    # need = need.filter(axis[value])
    str_expressions = [str(e) for e in expressions]
    axis_num = str_expressions.index(axis_name)
    expressions = expressions[:axis_num] + expressions[axis_num + 1:]
    possible_values = possible_values[:]
    axis_values = possible_values.pop(axis_num)

    # TODO: make sure possible_values are sorted and use searchsorted instead
    is_wanted_value = axis_values == value
    value_idx = is_wanted_value.nonzero()[0]
    num_idx = len(value_idx)
    if not num_idx:
        raise Exception('missing alignment data for %s %s' % (axis_name, value))
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
                         take_filter=None, leave_filter=None,
                         method="bysorting"):
    assert isinstance(need, np.ndarray) and \
        np.issubdtype(need.dtype, np.integer)
    assert score is None or isinstance(score, (bool, int, float, np.ndarray))

    if filter_value is not None:
        bool_filter_value = filter_value.copy()
    else:
        bool_filter_value = True

    maybe_filter = bool_filter_value
    if take_filter is not None:
        take_intersect = take_filter & bool_filter_value
        take = np.sum(take_intersect)

        # XXX: it would probably be faster to leave the filters as boolean
        # vector and do
        #     take_members = take_filter[member_indices]
        #     group_always = member_indices[take_members]
        # instead of
        #     group_always = np.intersect1d(members_indices, take_indices,
        #                                   assume_unique=True)
        take_indices = filter_to_indices(take_intersect)
        maybe_filter &= ~take_filter
    else:
        take = 0
        take_indices = None

    if leave_filter is not None:
        leave = np.sum(leave_filter & bool_filter_value)
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

    if method == 'sidewalk':
        score_max = max(score)
        score_min = min(score)
        if score_max > 1 or score_min < 0:
            raise Exception("""Score values are in the interval {} - {}.
Sidewalk alignment can only be used with a score between 0 and 1.
You may want to use a logistic function.
""".format(score_min, score_max))

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
                    if method == 'bysorting':
                        maybe_members_rank_value = score[group_maybe_indices]
                        # TODO: use np.partition (np1.8+)
                        sorted_local_indices = np.argsort(maybe_members_rank_value)
                        sorted_global_indices = \
                            group_maybe_indices[sorted_local_indices]
                    elif method == 'sidewalk':
                        sorted_global_indices = \
                            np.random.permutation(group_maybe_indices)
                else:
                    # if the score expression is a constant, we don't need to
                    # sort indices. In that case, the alignment will first take
                    # the individuals created last (highest id).
                    sorted_global_indices = group_maybe_indices

                # maybe_to_take is always > 0
                maybe_to_take = affected - num_always
                if method == 'bysorting':
                    # take the last X individuals (ie those with the highest
                    # score)
                    indices_to_take = sorted_global_indices[-maybe_to_take:]
                elif method == 'sidewalk':
                    proba_sum = sum(score[sorted_global_indices])
                    if maybe_to_take > round(proba_sum):
                        raise ValueError(
                            "Cannot use 'sidewalk' with need = {} > sum of probabilities = {}".format(
                                maybe_to_take, proba_sum
                                )
                            )
                    u = np.random.uniform() + np.arange(maybe_to_take)
                    # on the random sample, score are cumulated and then, we
                    # extract indices of each value before each value of u
                    cum_score = np.cumsum(score[sorted_global_indices])
                    indices_to_take = \
                        sorted_global_indices[np.searchsorted(cum_score, u)]

                underflow = maybe_to_take - len(indices_to_take)
                if underflow > 0:
                    total_underflow += underflow
                aligned[indices_to_take] = True
            elif affected < num_always:
                total_overflow += num_always - affected

    num_aligned = int(np.sum(aligned))
    # this assertion is only valid in the non weighted case
    assert num_aligned == total_affected + total_overflow - total_underflow
    num_partitioned = sum(len(g) for g in groups)
    if config.log_level == "processes":
        print(" %d/%d" % (num_aligned, num_partitioned), end=" ")
        if (take_filter is not None) or (leave_filter is not None):
            print("[take %d, leave %d]" % (take, leave), end=" ")
        if total_underflow:
            print("UNDERFLOW: %d" % total_underflow, end=" ")
        if total_overflow:
            print("OVERFLOW: %d" % total_overflow, end=" ")

    return aligned


# noinspection PyProtectedMember
class AlignmentAbsoluteValues(FilteredExpression):
    funcname = 'align_abs'
    no_eval = ('filter', 'secondary_axis', 'expressions',
               'method')

    def __init__(self, *args, **kwargs):
        super(AlignmentAbsoluteValues, self).__init__(*args, **kwargs)

        need = self.args[1]
        if isinstance(need, basestring):
            fpath = os.path.join(config.input_directory, need)
            need = load_ndarray(fpath, float)
            # XXX: store args in a list so that we can modify it?
            self.args = (self.args[0], need) + self.args[2:]
        self.past_error = None

    def collect_variables(self):
        # args[9] is the "link" argument
        # if self.args.link is None:
        if self.args[9] is None:
            return FilteredExpression.collect_variables(self)
        else:
            # in this case, it's tricky
            return set()

    def _eval_need(self, context, need, expressions, possible_values,
                   expressions_context=None):
        assert isinstance(need, np.ndarray)
        if expressions_context is None:
            expressions_context = context
        # When given a 0d array, we convert it to 1d. This can happen e.g. for
        # >>> b = True; x = ne.evaluate('where(b, 0.1, 0.2)')
        # >>> isinstance(x, np.ndarray)
        # True
        # >>> x.shape
        # ()
        if not need.shape:
            need = np.array([need])

        if isinstance(need, LabeledArray):
            if not expressions:
                expressions = [Variable(expressions_context.entity, name)
                               for name in need.dim_names]
            if not possible_values:
                possible_values = need.pvalues

        assert isinstance(need, np.ndarray)

        if len(expressions) != len(possible_values):
            raise Exception("align() expressions and possible_values "
                            "have different length: %d vs %d"
                            % (len(expressions), len(possible_values)))

        if 'period' in [str(e) for e in expressions]:
            period = context.period
            expressions, possible_values, need = \
                kill_axis('period', period, expressions, possible_values, need)

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

    def _handle_frac_need(self, need, method):
        # handle the "fractional people problem"
        if not np.issubdtype(need.dtype, np.integer):
            if method == 'uniform':
                int_need = need.astype(int)
                if config.debug and config.log_level == "processes":
                    print()
                    print("random sequence position before:",
                          np.random.get_state()[2])
                u = np.random.uniform(size=need.shape)
                if config.debug and config.log_level == "processes":
                    print("random sequence position after:",
                          np.random.get_state()[2])
                need = int_need + (u < need - int_need)
            elif method == 'cutoff':
                int_need = need.astype(int)
                frac_need = need - int_need
                need = int_need

                # the sum of fractional objects is the number of extra objects
                # we want aligned
                extra_wanted = int(round(np.sum(frac_need)))
                if extra_wanted:
                    # search the cutoff point yielding:
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
            elif method == 'round':
                # always use 0.5 as a cutoff point
                need = (need + 0.5).astype(int)

        assert np.issubdtype(need.dtype, np.integer)
        return need

    def _add_past_error(self, context, need, method='default'):
        if method == 'default':
            method = context.get('__on_align_error__')

        if method == 'carry':
            if self.past_error is None:
                # TODO: we should store this somewhere in the context instead
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

    def compute(self, context, score, need=None,
                filter=None, take=None, leave=None,
                expressions=None, possible_values=None, errors='default',
                frac_need='uniform', link=None, secondary_axis=None,
                method='bysorting'):

        if method not in ("bysorting", "sidewalk"):
            raise Exception("Method for alignment should be either 'bysorting' "
                            "or 'sidewalk'")
        if method == 'bysorting' and need is None:
            raise Exception("need argument is required when using the "
                            "'bysorting' method (which is the default)")

        if method == "sidewalk":
            # need is calculated over score and we could think of
            # calculate without leave_filter and without take_filter
            if need is None:
                need = sum(score)
            need = np.floor(need).astype(int)

        # need is a single scalar
        if np.isscalar(need):
            need = [need]

        # need is a non-ndarray sequence
        if isinstance(need, (tuple, list)):
            need = np.array(need)
        assert isinstance(need, np.ndarray)

        if expressions is None:
            expressions = []

        if possible_values is None:
            possible_values = []
        else:
            possible_values = [np.array(pv) for pv in possible_values]

        if frac_need not in ('uniform', 'cutoff', 'round'):
            cls = ValueError if isinstance(frac_need, basestring) else TypeError
            raise cls("frac_need should be one of: 'uniform', 'cutoff' or "
                      "'round'")

        if secondary_axis is not None and link is None:
            raise Exception("the 'secondary_axis' argument is only valid in "
                            "combination with the 'link' argument")
        if not isinstance(secondary_axis, (type(None), int, Variable)):
            raise Exception("'secondary_axis' should be either an integer or "
                            "an axis name (but got '%s' which is of type '%s')"
                            % (secondary_axis, type(secondary_axis)))

        func = self.align_no_link if link is None else self.align_link
        return func(context, score, need, filter, take, leave, expressions,
                    possible_values, errors, frac_need, link, secondary_axis,
                    method)

    def align_no_link(self, context, score, need, filter, take, leave,
                      expressions, possible_values, errors, frac_need, link,
                      secondary_axis, method):

        ctx_length = context_length(context)

        need, expressions, possible_values = \
            self._eval_need(context, need, expressions, possible_values)

        filter_value = expr_eval(self._getfilter(context, filter), context)

        if filter_value is not None:
            num_to_align = np.sum(filter_value)
        else:
            num_to_align = ctx_length

        # retrieve the columns we need to work with
        if expressions:
            columns = [expr_eval(expr, context) for expr in expressions]
            if filter_value is not None:
                groups = partition_nd(columns, filter_value, possible_values)
            else:
                groups = partition_nd(columns, True, possible_values)
        else:
            columns = []
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

        # noinspection PyAugmentAssignment
        need = need * self._get_need_correction(groups, possible_values)
        need = self._handle_frac_need(need, frac_need)
        need = self._add_past_error(context, need, errors)
        return align_get_indices_nd(ctx_length, groups, need, filter_value,
                                    score, take, leave, method)

    def align_link(self, context, score, need, filter, take, leave,
                   expressions, possible_values, errors, frac_need, link,
                   secondary_axis, method):
        target_context = link._target_context(context)
        need, expressions, possible_values = \
            self._eval_need(context, need, expressions, possible_values,
                            target_context)
        need = self._handle_frac_need(need, frac_need)
        need = self._add_past_error(context, need, errors)

        # handle secondary axis
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
        target_columns = [expr_eval(e, target_context) for e in expressions]
        # this is a one2many, so the link column is on the target side
        link_column = target_context[link._link_field]

        filter_expr = self._getfilter(context, filter)
        if filter_expr is not None:
            reverse_link = Many2One("reverse", link._link_field,
                                    context.entity.name)
            target_filter = LinkGet(reverse_link, filter_expr, False)
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

        # FIXME: target_context is not correct, as it is not filtered while
        # filtered_columns are. Since we do not use the context "columns" it
        # mostly works but I had to disable an assertion in utils.expand
        # because the length of the context is not correct.
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

        # target_row (row of person) is an index valid for *filtered/label*
        # columns !
        for target_row, source_row in enumerate(source_rows):
            if source_row == -1:
                continue
            hh[source_row].append(target_row)

        aligned, error = \
            align_link_nd(score, need, num_candidates, hh, fcols_labels,
                          secondary_axis)
        self.past_error = error
        return aligned

    def _get_need_correction(self, groups, possible_values):
        return 1

    dtype = always(bool)


class Alignment(AlignmentAbsoluteValues):
    funcname = 'align'

    def __init__(self, score=None, proportions=None,
                 filter=None, take=None, leave=None,
                 expressions=None, possible_values=None,
                 errors='default', frac_need='uniform',
                 fname=None, method='bysorting'):

        if possible_values is not None:
            if expressions is None or len(possible_values) != len(expressions):
                raise Exception("align() expressions and possible_values "
                                "arguments should have the same length")

        if method == 'sidewalk':
            raise Exception("sidewalk method is not supported for align(), "
                            "please use align_abs() instead")

        if proportions is None and fname is None:
            raise Exception("align() needs either an fname or proportions "
                            "arguments")
        elif proportions is not None and fname is not None:
            raise Exception("align() cannot have both fname and proportions "
                            "arguments")
        if fname is not None:
            proportions = fname

        super(Alignment, self).__init__(score, proportions,
                                        filter, take, leave,
                                        expressions, possible_values,
                                        errors, frac_need,
                                        method=method)

    def _get_need_correction(self, groups, possible_values):
        data = np.array([len(group) for group in groups])
        len_pvalues = [len(vals) for vals in possible_values]
        return data.reshape(len_pvalues)


functions = {
    'align_abs': AlignmentAbsoluteValues,
    'align': Alignment
}
