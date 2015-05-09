from __future__ import print_function

from itertools import izip
from fractions import gcd
import os

import numpy as np

import config
from align_link import align_link_nd
from context import context_length
from expr import Expr, Variable, expr_eval, traverse_expr, missing_values, \
    always
from exprbases import FilteredExpression
from groupby import GroupBy
from links import LinkGet, Many2One
from partition import partition_nd, filter_to_indices
from importer import load_ndarray
from utils import PrettyTable, LabeledArray
from random import random
from utils import time_period 


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    return [l[i:i+n] for i in range(0, len(l), n)]

<<<<<<< HEAD
def kill_axis(axis_name, value, expressions, possible_values, need, periodicity):
    '''possible_values is a list of ndarrays'''
=======
    #When we transition to LArray, this whole function could be replaced by:
    # need = need.filter(axis[value])
>>>>>>> liam2/master
    str_expressions = [str(e) for e in expressions]
    axis_num = str_expressions.index(axis_name)
    expressions = expressions[:axis_num] + expressions[axis_num + 1:]
    possible_values = possible_values[:]
    axis_values = possible_values.pop(axis_num)

    #TODO: make sure possible_values are sorted and use searchsorted instead
    
    str_expressions.pop(axis_num)
    if 'age' in str_expressions:
        axis_age_num = str_expressions.index('age')
#     if axis_name in ['age']:
#     import pdb 
#     pdb.set_trace()
    
    if axis_name in ['period']:
        axis_values[axis_values<3000] = axis_values[axis_values<3000]*100 + 1
        if value < 9999:
            value = value*100 + 1
        is_wanted_value_period = axis_values/100 == value/100
        periodicity_axis = 12/len(is_wanted_value_period.nonzero()[0])
        value_idx_period = is_wanted_value_period.nonzero()[0]

        if periodicity_axis > periodicity:
            if not isinstance(periodicity_axis/periodicity,int):
                raise Exception("can't do anything if time period is"
                                " not a multiple of time given in alignment data")
            pdb.set_trace()
            chunk = chunks(value_idx_period, periodicity_axis/periodicity)[value % 10 - 1]
            axis_values[value_idx_period[0]] = value
            axis_values[value_idx_period[1:]] = int(value/100)*100            
            need.base[:,value_idx_period[0]] = need.base[:,chunk].sum(axis=1)
            
        if periodicity_axis < periodicity:
            if not isinstance(periodicity/periodicity_axis,int):
                raise Exception("can't do anything if time period is"
                                " not a multiple of time given in alignment data")                
            # which season ? 
            time_value = value % 100
            if time_value > 12:
                time_value = value % 10
            season = int(time_value / periodicity * periodicity_axis-0.01)
            axis_values[value_idx_period] = int(value/100)*100
            axis_values[value_idx_period[season]] = value
            need.base[:,value_idx_period[season]] = \
                    need.base[:,value_idx_period[season]] * periodicity_axis/periodicity
        else:
            axis_values[value_idx_period] = value
            
                    
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
                         take_filter=None, leave_filter=None, method="default" ):
    assert isinstance(need, np.ndarray) and \
        np.issubdtype(need.dtype, np.integer)
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
                    if method=='default':
                        maybe_members_rank_value = score[group_maybe_indices]
                        sorted_local_indices = np.argsort(maybe_members_rank_value)
                        sorted_global_indices = \
                            group_maybe_indices[sorted_local_indices]
                    if method=='sidewalk':
                        if max(score[group_maybe_indices]) > 1 or min(score[group_maybe_indices]) < 0:
                            raise Exception("Sidewalk method can be used only with a"
                                            " score between 0 and 1. You may want to use"
                                            " a logistic function ")
                        sorted_global_indices = \
                          np.random.permutation(group_maybe_indices)

                    maybe_members_rank_value = score[group_maybe_indices]
                    #TODO: use np.partition (np1.8+)
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
                if method=='default':
                    # take the last X individuals (ie those with the highest score)
                    indices_to_take = sorted_global_indices[-maybe_to_take:]
                elif method=='sidewalk':
                    if maybe_to_take > sum(score[sorted_global_indices]):
                        raise Exception("Can't use Sidewalk with need > sum of probabilities")
                    U=random()+np.arange(maybe_to_take)             
                    #on the random sample, score are cumulated and then, we extract indices
                    #of each value before each value of U
                    indices_to_take = np.searchsorted(np.cumsum(score[sorted_global_indices]), U)
                    indices_to_take = sorted_global_indices[indices_to_take] 
                    
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
    if config.log_level == "processes":
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
    no_eval = ('filter', 'secondary_axis', 'expressions')

    def __init__(self, score, need = None,
                 filter=None, take=None, leave=None,
                 expressions=None, possible_values=None,
                 errors='default', frac_need='uniform',
                 method='default', periodicity_given='year',
                 link=None, secondary_axis=None):
        super(AlignmentAbsoluteValues, self).__init__(score, filter)

    def post_init(self):
        need = self.args[1]
        if isinstance(need, basestring):
            fpath = os.path.join(config.input_directory, need)
            need = load_ndarray(fpath, float)
#<<<<<<< HEAD

        # need is a single scalar
        if not isinstance(need, (tuple, list, Expr, np.ndarray)):
            need = [need]

        # need is a simple list (no expr inside)
        if isinstance(need, (tuple, list)) and \
           not any(isinstance(p, Expr) for p in need):
            need = np.array(need)
            
        if need is None and method != 'sidewalk':
            raise Exception("No default value for need if method is not sidewalk")
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
#=======
#            #XXX: store args in a #list so that we can modify it?
#            self.args = (self.args#[0], need) + self.args[2:]
#>>>>>>> liam2/master
        self.past_error = None
        
        assert(periodicity_given in time_period)
        self.periodicity_given = time_period[periodicity_given]


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

        if method in ("default","sidewalk") :
            self.method = method
        else: 
            raise Exception("Method for alignment should be either 'default' "
                            "either 'sidewalk'")

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

    def _eval_need(self, context, scores, filter_value):
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
        
        if self.need[0] is None and self.method == "sidewalk":
        #Note: need is calculated over score and we could think of 
        # calculate without leave_filter and without take_filter
            if filter_value is not None:
                scores = scores[filter_value]
            need = int(sum(scores)) 
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
                kill_axis('period', period, expressions, possible_values,
                          need, abs(self.periodicity_given))

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
                #TODO: we should store this somewhere in the context instead
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

    def compute(self, context, score, need, filter=None, take=None, leave=None,
                expressions=None, possible_values=None, errors='default',
                frac_need='uniform', link=None, secondary_axis=None):
        # need is a single scalar
        # if not isinstance(need, (tuple, list, np.ndarray)):
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

        scores = expr_eval(self.expr, context)
        filter_value = expr_eval(self._getfilter(context), context)
        
        need, expressions, possible_values = self._eval_need(context, scores, filter_value)

        take_filter = expr_eval(self.take_filter, context)
        leave_filter = expr_eval(self.leave_filter, context)

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

        periodicity = context['periodicity']
        if context['format_date'] == 'year0':
            periodicity = periodicity*12 #give right periodicity/self.periodicity_given whereas self.periodicity_given/12 doesn't
            
        #sign(self.periodicity_given) = sign(periodicity)
        self.periodicity_given = \
            self.periodicity_given * (self.periodicity_given*periodicity)/abs(self.periodicity_given*periodicity)
        if gcd(periodicity,self.periodicity_given) not in [periodicity,self.periodicity_given] : 
            raise( "mix of quarter and triannual impossible")
        
        need = need*periodicity/self.periodicity_given
        if scores is not None:            
            scores = scores*periodicity/self.periodicity_given 
            
        #noinspection PyAugmentAssignment
        need = need * self._get_need_correction(groups, possible_values)
        need = self._handle_frac_need(need, method=frac_need)
        need = self._add_past_error(context, need, method=errors)

        return align_get_indices_nd(ctx_length, groups, need, filter_value,
                                    scores, take_filter, leave_filter, method=self.method)

    def align_link(self, context):
        scores = expr_eval(self.expr, context)
        # TODO: filter_values ? Check for sidewalk
        need, expressions, possible_values = self._eval_need(context, scores, [])
        need = self._handle_frac_need(need)
        need = self._add_past_error(need, context)

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
                 fname=None,
                 method='default', periodicity_given='year'):

        if possible_values is not None:
            if expressions is None or len(possible_values) != len(expressions):
                raise Exception("align() expressions and possible_values "
                                "arguments should have the same length")

        if proportions is None and fname is None:
            if method=='default':
                raise Exception("align() needs either an fname or proportions "
                                "arguments")
            if method=='sidewalk':
                raise Exception("If alignment on sum of score is wanted "
                                "then use align_abs")            
        if proportions is not None and fname is not None:
            raise Exception("align() cannot have both fname and proportions "
                            "arguments")
        if fname is not None:
            proportions = fname

        super(Alignment, self).__init__(score, proportions,
                                        filter, take, leave,
                                        expressions, possible_values,
                                        errors, frac_need, method, periodicity_given)
        
    def _get_need_correction(self, groups, possible_values):
        data = np.array([len(group) for group in groups])
        len_pvalues = [len(vals) for vals in possible_values]
        return data.reshape(len_pvalues)


functions = {
    'align_abs': AlignmentAbsoluteValues,
    'align': Alignment
}
