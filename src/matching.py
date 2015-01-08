from __future__ import print_function

import numpy as np
import pandas as pd
import random

from expr import expr_eval, always, expr_cache
from exprbases import FilteredExpression
from context import context_length, context_delete, context_subset
from utils import loop_wh_progress


def df_by_cell(used_variables, setfilter, context):
    """return a DataFrame, with id list and group size"""

    subset = context.subset(setfilter, used_variables)
    used_set = dict((k, subset[k]) for k in used_variables)
    used_set = pd.DataFrame(used_set)
    used_variables.remove('id')
    grouped = used_set.groupby(list(used_variables), sort=False)
    idx = grouped.apply(lambda x: list(x['id'].values)).reset_index()
    idx.rename(columns={0: 'idx'}, inplace=True)
    return idx



class Matching(FilteredExpression):
    """
    Base class for matching functions
    """
    dtype = always(int)


class RankMatching(Matching):
    """
    Matching based on rank/order
    * set 1 is ranked by decreasing orderby1
    * set 2 is ranked by decreasing orderby2
    * individuals in the nth position in each list are matched together.
    """
    funcname = 'rank_matching'
    no_eval = ('set1filter', 'set2filter')

    def compute(self, context, set1filter, set2filter, orderby1, orderby2):
        set1filterexpr = self._getfilter(context, set1filter)
        set1filtervalue = expr_eval(set1filterexpr, context)
        set2filterexpr = self._getfilter(context, set2filter)
        set2filtervalue = expr_eval(set2filterexpr, context)
        set1len = set1filtervalue.sum()
        set2len = set2filtervalue.sum()
        numtomatch = min(set1len, set2len)
        print("matching with %d/%d individuals" % (set1len, set2len))
        result = np.empty(context_length(context), dtype=int)
        result.fill(-1)
        if not numtomatch:
            return result

        sorted_set1_indices = orderby1[set1filtervalue].argsort()[-numtomatch:]
        sorted_set2_indices = orderby2[set2filtervalue].argsort()[-numtomatch:]

        set1ids = context['id'][set1filtervalue]
        set2ids = context['id'][set2filtervalue]

        id_to_rownum = context.id_to_rownum
        id1 = set1ids[sorted_set1_indices]
        id2 = set2ids[sorted_set2_indices]
        # cannot use sorted_setX_indices because those are "local" indices
        result[id_to_rownum[id1]] = id2
        result[id_to_rownum[id2]] = id1
        return result


class SequentialMatching(Matching):
    """
    Matching based on searching for the best match one by one.

    In general that kind of matching does not provide the best matching,
    because it does not optimize the *overall* distance function (it does not
    necessarily return the match with the lowest sum of distances between all
    matched pairs).

    - orderby gives the way individuals of set 1 are sorted before matching.
      The first individual will be matched with the highest scoring individual
      from set 2. The next individual in set 1 will be matched with the highest
      scoring individual among the remaining individuals in set 2.

    - orderby can be :
        - an expression (usually a variable name). It is supposed to be
          a "difficulty" because it is better (according to a general
          objective score) to match hard-to-match people first.
        - the string 'EDtM', in which case, the (reduced) "Euclidean Distance to
          the Mean" is used to order individuals.
    """
    funcname = 'matching'
    no_eval = ('set1filter', 'set2filter', 'score')

    def traverse(self, context):
        #FIXME: we should not override the parent traverse method, so that all
        # "child" expressions are traversed too.
        # This is not done currently, because it would traverse score_expr.
        # This is a problem because traverse is used by collect_variables and
        # the presence of variables is checked in expr.expr_eval() before
        # the evaluate method is called and the context is completed during
        # evaluation (__other_xxx is added during evaluation).
        yield self

    def _get_score_variables(self, score_expr, context):
        names = {v.name for v in score_expr.collect_variables(context)}
        used_variables1 = {n for n in names if not n.startswith('__other_')}
        used_variables2 = {n[8:] for n in names if n.startswith('__other_')}
        return used_variables1, used_variables2

    def compute(self, context, set1filter, set2filter, score, orderby,
                pool_size=None):
        global matching_ctx

        if pool_size is not None:
            assert isinstance(pool_size, int)
            assert pool_size > 0

        set1filterexpr = self._getfilter(context, set1filter)
        set1filtervalue = expr_eval(set1filterexpr, context)
        set2filterexpr = self._getfilter(context, set2filter)
        set2filtervalue = expr_eval(set2filterexpr, context)
        set1len = set1filtervalue.sum()
        set2len = set2filtervalue.sum()
        tomatch = min(set1len, set2len)
        print("matching with %d/%d individuals" % (set1len, set2len))

        used_variables1, used_variables2 = \
            self._get_score_variables(score, context)

        #TODO: we should detect whether or not we are using non-simple
        # expressions (EvaluableExpression children) and pre-evaluate them,
        # because otherwise they are re-evaluated on all of set2 for each
        # individual in set1. See https://github.com/liam2/liam2/issues/128
        set1 = context.subset(set1filtervalue, {'id'} | used_variables1,
                              set1filterexpr)
        set2 = context.subset(set2filtervalue, {'id'} | used_variables2,
                              set2filterexpr)

        # subset creates a dict for the current entity, so .entity_data is a
        # dict
        set1 = set1.entity_data
        set2 = set2.entity_data

        if isinstance(orderby, str):
            assert orderby == 'EDtM'
            order = np.zeros(context_length(context))
            #FIXME: remove "id"
            for var in used_variables1:
                col = set1[var]
                order[set1filter] += (col - col.mean()) ** 2 / col.var()
        else:
            #XXX: shouldn't orderby be computed only on the filtered set? (
            # but used_variables might be different than in the set,
            # so it might not be worth it.
            order = orderby

        sorted_set1_indices = order[set1filtervalue].argsort()[::-1]

        set1tomatch = sorted_set1_indices[:tomatch]

        result = np.empty(context_length(context), dtype=int)
        result.fill(-1)
        id_to_rownum = context.id_to_rownum

        # prefix all keys except __len__
        matching_ctx = {'__other_' + k if k != '__len__' else k: v
                        for k, v in set2.iteritems()}

        #noinspection PyUnusedLocal
        def match_one_set1_individual(idx, sorted_idx, pool_size):
            global matching_ctx

            set2_size = context_length(matching_ctx)
            if not set2_size:
                raise StopIteration

            if pool_size is not None and set2_size > pool_size:
                pool = random.sample(xrange(set2_size), pool_size)
                local_ctx = context_subset(matching_ctx, pool)
            else:
                local_ctx = matching_ctx.copy()

            local_ctx.update((k, set1[k][sorted_idx])
                             for k in {'id'} | used_variables1)

            eval_ctx = context.clone(entity_data=local_ctx)
            set2_scores = expr_eval(score, eval_ctx)

            individual2_idx = set2_scores.argmax()

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
                         pool_size=pool_size)
        return result


class OptimizedSequentialMatching(SequentialMatching):
    """
    Here, the matching is optimized since we work on sets grouped by values.
    Doing so, we work with smaller sets and we can improve running time.
    """
    funcname = 'optimized_matching'
    no_eval = ('set1filter', 'set2filter', 'score', 'orderby')

    def compute(self, context, set1filter, set2filter, score, orderby):
        global matching_ctx

        set1filterexpr = self._getfilter(context, set1filter)
        set1filtervalue = expr_eval(set1filterexpr, context)
        set2filterexpr = self._getfilter(context, set2filter)
        set2filtervalue = expr_eval(set2filterexpr, context)
        set1len = set1filtervalue.sum()
        set2len = set2filtervalue.sum()
        print("matching with %d/%d individuals" % (set1len, set2len), end='')

        used_variables1, used_variables2 = \
            self._get_score_variables(score, context)

        if not isinstance(orderby, str):
            var_match = orderby.collect_variables(context)
            #FIXME: all used_variables are used in groupby. We need to keep the
            # orderby variables in the context subset but they should NOT be
            # used in the groupby.
            used_variables1 |= {v.name for v in var_match}

        df1 = df_by_cell(used_variables1, set1filtervalue, context)
        df2 = df_by_cell(used_variables2, set2filtervalue, context)
        print(" (%d/%d groups)" % (len(df1), len(df2)))

        # Sort df1: 
        if isinstance(orderby, str):
            assert orderby == 'EDtM'
            orderbyvalue = pd.Series(np.zeros(len(df1)))
            #FIXME: id should not be in here
            for var in used_variables1:
                orderbyvalue += (df1[var] - df1[var].mean())**2 / df1[var].var()
        else:
            orderbyvalue = df1.eval(orderby)

        df1 = df1.loc[orderbyvalue.order()[::-1].index]
        
        result = np.empty(context_length(context), dtype=int)
        result.fill(-1)
        id_to_rownum = context.id_to_rownum
        
        matching_ctx = dict(('__other_' + k, v.values)
                            for k, v in df2.iteritems())
        matching_ctx['__len__'] = len(df2)
        def match_cell(idx, row):
            global matching_ctx

            if matching_ctx['__len__'] == 0:
                raise StopIteration

            for var in df1.columns:
                matching_ctx[var] = row[var]

            eval_ctx = context.clone(entity_data=matching_ctx)
            set2_scores = expr_eval(score, eval_ctx)
            cell2_idx = set2_scores.argmax()

            # reverse to mimic non-optimized argsort()[::-1]
            cell1ids = row['idx'][::-1]
            cell2ids = matching_ctx['__other_idx'][cell2_idx]

            cell1size = len(cell1ids)
            cell2size = len(cell2ids)
            nb_match = min(cell1size, cell2size)

            # we could introduce a random choice here but it is not
            # much necessary. In that case, it should be done in df_by_cell
            ids1 = cell1ids[:nb_match]
            ids2 = cell2ids[:nb_match]
            
            result[id_to_rownum[ids1]] = ids2
            result[id_to_rownum[ids2]] = ids1
            
            if nb_match == cell2size:
                matching_ctx = context_delete(matching_ctx, cell2_idx)
            else:
                # other variables do not need to be modified since the cell
                # only got smaller and was not deleted
                matching_ctx['__other_idx'][cell2_idx] = cell2ids[nb_match:]

            if nb_match < cell1size:
                row['idx'] = cell1ids[nb_match:]
                match_cell(idx, row)
                    
        loop_wh_progress(match_cell, df1.to_records())
        return result


functions = {
    'matching': SequentialMatching,
    'optimized_matching': OptimizedSequentialMatching
    'rank_matching': RankMatching,
}
