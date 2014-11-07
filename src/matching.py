from __future__ import print_function

import numpy as np
import pandas as pd
import random

from expr import expr_eval, always, expr_cache
from exprbases import FilteredExpression
from context import context_length, context_delete
from utils import loop_wh_progress


def df_by_cell(used_variables, setfilter, context):
    """return a DataFrame, with id list and group size"""

    subset = context_subset(context, setfilter, used_variables)
    used_set = dict((k, subset[k]) for k in used_variables)
    used_set = pd.DataFrame(used_set)
    used_variables.remove('id')
    grouped = used_set.groupby(used_variables)
    idx = grouped.apply(lambda x: list(x['id'].values)).reset_index()
    idx.rename(columns={0: 'idx'}, inplace=True)
    return idx


class ScoreMatching(FilteredExpression):
    """General framework for a Matching based on score

    In general that kind of matching doesn't provide the best matching meaning
    it doesn't optimize an overall penalty function. For example, if we have a
    distance function, the function doesn't always return the match with the
    lowest sum of distanced between all matched pairs. A Score matching need two
    things:
      - an order for the two sets
      - a way of selecting a match
    """
    def __init__(self, set1filter, set2filter, orderby1, orderby2):
        self.set1filter = set1filter
        self.set2filter = set2filter
        self.orderby1_expr = orderby1
        self.orderby2_expr = orderby2

    def traverse(self, context):
        #FIXME: we should not override the parent traverse method, so that all
        # "child" expressions are traversed too.
        # This is not done currently, because it would traverse score_expr.
        # This is a problem because traverse is used by collect_variables and
        # the presence of variables is checked in expr.expr_eval() before
        # the evaluate method is called and the context is completed during
        # evaluation (__other_xxx is added during evaluation).
        yield self

    def _get_filters(self, context):
        set1filterexpr = self._getfilter(context, self.set1filter)
        set2filterexpr = self._getfilter(context, self.set2filter)
        return expr_eval((set1filterexpr, set1filterexpr), context)

    dtype = always(int)



class RankingMatching(ScoreMatching):
    """
    Matching based on score
        set 1 is ranked by decreasing orderby1
        set 2 is ranked by decreasing orderby2
        Then individuals in the nth position in each list are matched together.
    """
    def _match(self, set1tomatch, set2tomatch, set1, set2, context):
        result = np.empty(context_length(context), dtype=int)
        result.fill(-1)

        id_to_rownum = context.id_to_rownum
        id1 = set1['id'][set1tomatch]
        id2 = set2['id'][set2tomatch]
        result[id_to_rownum[id1]] = id2
        result[id_to_rownum[id2]] = id1
        return result

    # def compute(self, context, set1filter, set2filter, score, orderby):

    def evaluate(self, context):
        set1filter, set2filter = self._get_filters(context)
        set1len = set1filter.sum()
        set2len = set2filter.sum()
        tomatch = min(set1len, set2len)

        orderby1 = expr_eval(self.orderby1_expr, context)
        orderby2 = expr_eval(self.orderby2_expr, context)

        sorted_set1_indices = orderby1[set1filter].argsort()[::-1]
        sorted_set2_indices = orderby2[set2filter].argsort()[::-1]

        set1tomatch = sorted_set1_indices[:tomatch]
        set2tomatch = sorted_set2_indices[:tomatch]

        used_variables1 = self.orderby1_expr.collect_variables(context)
        used_variables1.add('id')
        used_variables2 = self.orderby2_expr.collect_variables(context)
        used_variables2.add('id')

        #TODO: we should detect whether or not we are using non-simple
        # expressions (EvaluableExpression children) and pre-evaluate them,
        # because otherwise they are re-evaluated on all of set2 for each
        # individual in set1. See https://github.com/liam2/liam2/issues/128
        set1 = context_subset(context, set1filter, used_variables1)
        set2 = context_subset(context, set2filter, used_variables2)

        # set1 = context.subset(set1filtervalue, used_variables1, set1filterexpr)
        # set2 = context.subset(set2filtervalue, used_variables2, set2filterexpr)
        # 
        # # subset creates a dict for the current entity, so .entity_data is a
        # # dict
        # set1 = set1.entity_data
        # set2 = set2.entity_data
        # 
        # set1len = set1filtervalue.sum()
        # set2len = set2filtervalue.sum()
        # tomatch = min(set1len, set2len)
        # sorted_set1_indices = orderby[set1filtervalue].argsort()[::-1]
        # set1tomatch = sorted_set1_indices[:tomatch]
        print("matching with %d/%d individuals" % (set1len, set2len))
        return self._match(set1tomatch, set2tomatch, set1, set2, context)


class SequentialMatching(ScoreMatching):
    """
    Matching base on researching the best match one by one.
        - orderby gives the way individuals of 1 are sorted before matching
        The first on the list will be matched with its absolute best match
        The last on the list will be matched with the best match among the
        remaining pool
        - orederby can be :
            - an expression (usually a variable name)
            - a string : the name of a method to sort individuals to be match,
            it is supposed to be
             a "difficulty" because it's better (according to a general
             objective score)
             to match hard-to-match people first. The possible difficulty order
             are:
                - 'EDtM' : 'Euclidian Distance to the Mean', note it is the
                reduce euclidan distance that is
                           used which is a common convention when there is more
                           than one variable
                - 'SDtOM' : 'Score Distance to the Other Mean'
            The SDtOM is the most relevant distance.
    """
    funcname = 'matching'
    no_eval = ('set1filter', 'set2filter', 'score')
    
    def __init__(self, set1filter, set2filter, score, orderby, pool_size=None):
        ScoreMatching.__init__(self, set1filter, set2filter, orderby, None)
        self.score_expr = score
        if pool_size is not None:
            assert isinstance(pool_size, int)
            assert pool_size > 0
        self.pool_size = pool_size

#    def compute(self, context, set1filter, set2filter, score, orderby):


    def _get_used_variables_match(self, context):
        used_variables = self.score_expr.collect_variables(context)
        used_variables1 = [v for v in used_variables
                           if not v.startswith('__other_')]
        used_variables2 = [v[8:] for v in used_variables
                           if v.startswith('__other_')]
        used_variables1 += ['id']
        used_variables2 += ['id']
        return used_variables1, used_variables2

    def _match(self, set1tomatch, set1, set2,
               used_variables1, used_variables2, context):
        global matching_ctx

        score_expr = self.score_expr
        result = np.empty(context_length(context), dtype=int)
        result.fill(-1)
        id_to_rownum = context.id_to_rownum

        matching_ctx = dict(('__other_' + k if k in used_variables2 else k, v)
                            for k, v in set2.iteritems())

        #noinspection PyUnusedLocal
        def match_one_set1_individual(idx, sorted_idx, pool_size):
            global matching_ctx

            set2_size = context_length(matching_ctx)
            if not set2_size:
                raise StopIteration

            if pool_size is not None and set2_size > pool_size:
                pool = random.sample(xrange(set2_size), pool_size)
                local_ctx = context_subset(matching_ctx, pool, None)
            else:
                local_ctx = matching_ctx.copy()

            local_ctx.update((k, set1[k][sorted_idx]) for k in used_variables1)

            eval_ctx = context.clone(entity_data=local_ctx)
            set2_scores = expr_eval(score, eval_ctx)

            individual2_idx = np.argmax(set2_scores)

            id1 = local_ctx['id']
            id2 = local_ctx['__other_id'][individual2_idx]
            if pool_size is not None and set2_size > pool_size:
                individual2_idx = pool[individual2_idx]
            matching_ctx = context_delete(matching_ctx, individual2_idx)

            #FIXME: the expr gets cached for the full matching_ctx at the
            # beginning and then when another women with the same values is
            # found, it thinks it can reuse the expr but it breaks because it
            # has not the correct length.

            # the current workaround is to invalidate the whole cache for the
            # current entity but this is not the right way to go.
            # * disable the cache for matching?
            # * use a local cache so that methods after matching() can use
            # what was in the cache before matching(). Shouldn't the cache be
            # stored inside the context anyway?
            expr_cache.invalidate(context.period, context.entity_name)

            result[id_to_rownum[id1]] = id2
            result[id_to_rownum[id2]] = id1

        loop_wh_progress(match_one_set1_individual, set1tomatch,
                         title="Matching...", pool_size=self.pool_size)
        return result

    def evaluate(self, context):
        set1filter, set2filter = self._get_filters(context)
        set1len = set1filter.sum()
        set2len = set2filter.sum()

        used_variables1, used_variables2 = \
            self._get_used_variables_match(context)
        set1 = context_subset(context, set1filter, used_variables1)
        set2 = context_subset(context, set2filter, used_variables2)

        orderby1_expr = self.orderby1_expr
        if isinstance(orderby1_expr, str):
            assert orderby1_expr == 'EDtM'
            order = np.zeros(context_length(context), dtype=int)
            for var in used_variables1:
                col = set1[var]
                order[set1filter] += (col - col.mean()) ** 2 / col.var()
        else:
            order = expr_eval(orderby1_expr, context)

        sorted_set1_indices = order[set1filter].argsort()[::-1]

        tomatch = min(set1len, set2len)
        set1tomatch = sorted_set1_indices[:tomatch]

        print("matching with %d/%d individuals" % (set1len, set2len))
        return self._match(set1tomatch, set1, set2,
                           used_variables1, used_variables2, context)


class OptimizedSequentialMatching(SequentialMatching):
    """ Here, the matching is optimized since we work on
        sets grouped with values. Doing so, we work with
        smaller sets and we can improve running time.
    """
    funcname = 'optimized_matching'

    def __init__(self, set1filter, set2filter, score, orderby):
        SequentialMatching.__init__(self, set1filter, set2filter, score,
                                    orderby, pool_size=None)

    def evaluate(self, context):
        global matching_ctx

        set1filter, set2filter = self._get_filters(context)
        set1len = set1filter.sum()
        set2len = set2filter.sum()
        print("matching with %d/%d individuals" % (set1len, set2len))
        
        used_variables1, used_variables2 = \
            self._get_used_variables_match(context)
        order = self.orderby1_expr
        if not isinstance(order, str):
            var_match = order.collect_variables(context)
            used_variables1 += list(var_match)

        df1 = df_by_cell(used_variables1, set1filter, context)
        df2 = df_by_cell(used_variables2, set2filter, context)

        # Sort df1: 
        if isinstance(order, str):
            assert order == 'EDtM'
            orderby = pd.Series(len(df1), dtype=int)
            for var in used_variables1:
                orderby += (df1[var] - df1[var].mean())**2 / df1[var].var()
        else:
            orderby = df1.eval(order)

        df1 = df1.loc[orderby.order().index]
        
        score_expr = self.score_expr
        result = np.empty(context_length(context), dtype=int)
        result.fill(-1)
        id_to_rownum = context.id_to_rownum
        
        matching_ctx = dict(('__other_' + k, v.values)
                            for k, v in df2.iteritems())
        matching_ctx['__len__'] = len(df2)
        for varname, col in df1.iteritems():
            matching_ctx[varname] = np.empty(1, dtype=col.dtype)

        def match_cell(idx, row):
            global matching_ctx
            if matching_ctx['__len__'] == 0:
                raise StopIteration()
            
            size1 = len(row['idx'])
            for var in df1.columns:
                matching_ctx[var] = row[var]
    
            cell_idx = expr_eval(score_expr, matching_ctx).argmax()
            size2 = len(matching_ctx['__other_idx'][cell_idx])
            nb_match = min(size1, size2)

            # we could introduce a random choice her but it's not
            # much necessary. In that case, it should be done in df_by_cell
            idx1 = row['idx'][:nb_match]
            idx2 = matching_ctx['__other_idx'][cell_idx][:nb_match]
            
            result[id_to_rownum[idx1]] = idx2
            result[id_to_rownum[idx2]] = idx1
            
            if nb_match == size2:
                matching_ctx = context_delete(matching_ctx, cell_idx)
            else:
                matching_ctx['__other_idx'][cell_idx] = \
                    matching_ctx['__other_idx'][cell_idx][nb_match:]

            if nb_match < size1:
                row['idx'] = row['idx'][nb_match:]
                match_cell(idx, row)
                    
        loop_wh_progress(match_cell, df1.to_records())
        return result


functions = {
    'matching': SequentialMatching,
    'rank_matching': RankingMatching,
    'optimized_matching': OptimizedSequentialMatching
}
