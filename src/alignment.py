from __future__ import print_function

from itertools import izip, chain, product, count
import random
import operator
import csv
import os

import numpy as np

from expr import functions, Expr, Variable, expr_eval, parse
from entities import context_length
from utils import skip_comment_cells, strip_rows, PrettyTable
import simulation
from properties import FilteredExpression, GroupCount, add_individuals

#XXX: it might be faster to partition using a formula giving the index of the
# group instead of checking all possible value combination.
# eg. idx = where(agegroup <= 50, agegroup / 5, 5 + agegroup / 10) + male * 15
# however, in numpy I would still have to do (idx == 0).nonzero(),
# (idx == 1).nonzero(), ... 
def partition_nd(columns, filter, possible_values):
    """
    * columns is a list of columns containing the data to be partitioned
    * filter is a vector of booleans which selects individuals to be counted  
    * values is an matrix with N vectors containing the possible values for
      each column
    * returns a 1d array of lists of indices 
    """
    size = tuple([len(colvalues) for colvalues in possible_values])
    
    # initialise result with empty lists
    result = np.empty(size, dtype=list)    

    # for each combination of i, j, k:
    for idx in np.ndindex(*size):
        #TODO: use numexpr instead
        # for the usual case (2 dimensions), it is faster as soon as we have 
        # more than ~20000 rows, which will probably be the usual case.

        # local_filter = filter & (data0 == values0[i]) & (data1 == values1[j])
        local_filter = np.copy(filter)
        for i, colvalues, coldata in izip(idx, possible_values, columns):
            local_filter &= coldata == colvalues[i]
        result[idx] = local_filter.nonzero()[0]

    # pure-python version. It is 10x slower than the NumPy version above
    # but it might be a better starting point to translate to C, 
    # especially given that the possible_values are usually sorted (we could
    # sort them too), so we could use some bisect algorithm to find which
    # category it belongs to. 
#    fill_with_empty_list = np.frompyfunc(lambda _: [], 1, 1)
#    fill_with_empty_list(result, result)
    
#    for idx, row in enumerate(izip(*data_columns)):
#        # returns a tuple with the position of the group this row belongs to
#        # eg. (0, 1, 5)
#        #XXX: this uses strict equality, partitioning using 
#        # inequalities might be useful in some cases
#        try:
#            pos = tuple([values_i.index(vi) for vi, values_i 
#                                            in izip(row, possible_values)])
#            result[pos].append(idx)
#        except ValueError:
#            #XXX: issue a warning?
#            pass
    return np.ravel(result)

def extract_period(period, expressions, possible_values, probabilities):
    #TODO: allow any temporal variable
    str_expressions = [str(e) for e in expressions]
    periodcol = str_expressions.index('period')
    expressions = expressions[:periodcol] + expressions[periodcol+1:]
    possible_values = possible_values[:]
    period_values = possible_values.pop(periodcol)
    num_periods = len(period_values)
    try:
        period_idx = period_values.index(period)
        #TODO: allow period in any dimension
        assert periodcol == 1
        probabilities = probabilities[period_idx::num_periods]
    except ValueError:
        raise Exception('missing alignment data for period %d' % period)

    return expressions, possible_values, probabilities
    
def align_get_indices_nd(context, filter, score,
                         expressions, possible_values, probabilities,
                         take_filter=None, leave_filter=None, weights=None,
                         past_error=None):
    assert len(expressions) == len(possible_values)

    num_to_align = np.sum(filter)
    if expressions:
        #TODO: allow any temporal variable
        if 'period' in [str(e) for e in expressions]:
            period = context['period']
            expressions, possible_values, probabilities = \
                extract_period(period, expressions, possible_values,
                               probabilities)
        assert len(probabilities) == \
               reduce(operator.mul, [len(vals) for vals in possible_values], 1)
        # retrieve the columns we need to work with
        columns = [expr_eval(expr, context) for expr in expressions]
        groups = partition_nd(columns, filter, possible_values)
    else:
        groups = np.array([filter.nonzero()[0]])
        assert len(probabilities) == 1

    # the sum is not necessarily equal to len(a), because some individuals
    # might not fit in any group (eg if some alignment data is missing)
    num_aligned = sum(len(g) for g in groups) 
    if num_aligned < num_to_align:
        to_align = set(filter.nonzero()[0])
        aligned = set()
        for member_indices in groups:
            aligned |= set(member_indices)
        unaligned = to_align - aligned
        print("Warning: %d individual(s) do not fit in any alignment category"
              % len(unaligned))
        print(" | ".join(str(expr) for expr in ['id'] + expressions))
        for row in unaligned:
            print(" | ".join(str(col[row])
                             for col in [context['id']] + columns))

    take = 0
    leave = 0
    if take_filter is not None:
        take = np.sum(filter & take_filter) 
        take_indices = take_filter.nonzero()[0]
        if leave_filter is not None: 
            maybe_indices = ((~take_filter) & (~leave_filter)).nonzero()[0]
        else:
            maybe_indices = (~take_filter).nonzero()[0]
    elif leave_filter is not None:
        take_indices = None
        maybe_indices = (~leave_filter).nonzero()[0]
    else:
        take_indices = None
        maybe_indices = None

    if leave_filter is not None:
        leave = np.sum(filter & leave_filter)

    total_underflow = 0
    total_overflow = 0
    total_affected = 0
    total_indices = []
    to_split_indices = []
    to_split_overflow = []
    for group_idx, members_indices, probability in izip(count(), groups,
                                                        probabilities):
        
        if len(members_indices):
            if weights is None:
                member_weights = None
                expected = len(members_indices) * probability
            else:
                member_weights = weights[members_indices]
                expected = np.sum(member_weights) * probability
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


def prod(values):
    return reduce(operator.mul, values, 1)

class GroupBy(FilteredExpression):
    func_name = 'groupby'

    def __init__(self, *args, **kwargs):
        # On python 3, we could clean up this code (keyword only arguments).
        self.percent = kwargs.get('percent', False)
        expr = kwargs.get('expr')
        if expr is None:
            expr = GroupCount()
        
        filter = kwargs.get('filter')
        super(GroupBy, self).__init__(expr, filter)
        assert len(args), "groupby needs at least one expression"
        assert isinstance(args[0], Expr), "groupby takes expressions as " \
                                          "arguments, not a list of expressions"
        self.expressions = args

    def eval(self, context):
        if self.filter is not None:
            filter_value = expr_eval(self.filter, context)
        else:
            filter_value = np.ones(context_length(context), dtype=bool)

        expressions = self.expressions
        columns = [expr_eval(e, context) for e in expressions]
        possible_values = [np.unique(column[filter_value])
                           for column in columns]
        groups = partition_nd(columns, filter_value, possible_values)
        
        # groups is a (flat) list of list.
        # the first variable is the outer-most "loop", 
        # the last one the inner most.
        
        # add total for each row        
        folded_vars = len(expressions) - 1
        len_pvalues = [len(vals) for vals in possible_values]
        width = len_pvalues[-1]
        height = prod(len_pvalues[:-1])

        def xy_to_idx(x, y):
            # divide by the prod of possible values of expressions to its right, 
            # mod by its own number of possible values
            vars = [(y / prod(len_pvalues[v + 1:folded_vars])) % len_pvalues[v]
                    for v in range(folded_vars)]
            return sum(v * prod(len_pvalues[i + 1:])
                       for i, v in enumerate(vars)) + x
        
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

        data = []
        for member_indices in groups_wh_totals:
            local_context = dict((v, context[v][member_indices])
                                 for v in used_variables)
            local_context['__len__'] = len(member_indices)
            local_context['period'] = context['period']
            local_context['__entity__'] = context['__entity__']
            #FIXME: this should come from somewhere else
            local_context['nan'] = float('nan')
            data.append(expr_eval(expr, local_context))

        if self.percent:
            total_count = np.sum(filter_value)
            data = [100.0 * count / total_count for count in data]
            
        # gender | False | True | total
        #        |    20 |   15 |    xx

        #   dead | False | True | 
        # gender |       |      | total
        #  False |    20 |   15 |    xx
        #   True |     0 |    1 |    xx
        #  total |    xx |   xx |    xx

        #          |   dead | False | True | 
        # agegroup | gender |       |      | total
        #        5 |  False |    20 |   15 |    xx
        #        5 |   True |     0 |    1 |    xx
        #       10 |  False |    25 |   10 |    xx
        #       10 |   True |     1 |    1 |    xx
        #          |  total |    xx |   xx |    xx

        # add headers
        labels = [str(e) for e in expressions]
        if folded_vars:
            result = [[''] * (folded_vars - 1) +
                      [labels[-1]] + 
                      list(possible_values[-1]) +
                      [''],
                      # 2nd line
                      labels[:-1] +
                      [''] * len(possible_values[-1]) +
                      ['total']]
            categ_values = list(product(*possible_values[:-1]))
            last_line = [''] * (folded_vars - 1) + ['total']
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


class Alignment(FilteredExpression):
    func_name = 'align'

    def __init__(self, score_expr, filter=None, take=None, leave=None,
                 fname=None,
                 expressions=None, possible_values=None, probabilities=None,
                 on_overflow='default'):
        super(Alignment, self).__init__(score_expr, filter)
        
        assert ((expressions is not None and 
                 possible_values is not None and 
                 probabilities is not None) or
                (fname is not None))
        
        if fname is not None:
            self.load(fname)
        else:
            self.expressions = [Variable(e) if isinstance(e, basestring) else e
                                for e in expressions]
            self.possible_values = possible_values
            self.probabilities = probabilities

        self.take_filter = take
        self.leave_filter = leave
        self.on_overflow = on_overflow
        self.overflows = None

    def load(self, fpath):
        with open(os.path.join(simulation.input_directory, fpath), "rb") as f:
            reader = csv.reader(f)
            lines = skip_comment_cells(strip_rows(reader))
            header = lines.next()
            self.expressions = [parse(s, autovariables=True) for s in header]
            table = []
            for line in lines:
                assert all(value != '' for value in line), \
                       "empty cell found in %s" % fpath 
                table.append([eval(value) for value in line]) 
        #TODO: ndimensional
        headers1 = table.pop(0)
        headers2 = [line.pop(0) for line in table]
        self.possible_values = [headers2, headers1]
        self.probabilities = list(chain.from_iterable(table))
        assert len(self.probabilities) == len(headers1) * len(headers2), \
               'incoherent alignment data: %d actual data\n' \
               'but %d * %d = %d in headers' % (len(self.probabilities),
                                                len(headers1), len(headers2),
                                                len(headers1) * len(headers2))
        
    def eval(self, context):
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
                self.overflows = np.zeros(len(self.probabilities))
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
            filter_value = np.ones(context_length(context), dtype=bool)

        take_filter = expr_eval(self.take_filter, context) \
                      if self.take_filter is not None else None
        leave_filter = expr_eval(self.leave_filter, context) \
                       if self.leave_filter is not None \
                       else None

        indices, overflows = \
            align_get_indices_nd(context, filter_value, scores,
                                 self.expressions, self.possible_values,
                                 self.probabilities,
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
                clones[self.weight_col] = to_split_overflow
                array[self.weight_col][to_split_indices] -= to_split_overflow
                add_individuals(context, clones)

        return {'values': True, 'indices': indices}

    def dtype(self, context):
        return bool


functions.update({
    'align': Alignment,
    'groupby': GroupBy
})
