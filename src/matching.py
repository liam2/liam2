from __future__ import print_function

import numpy as np
import random

from expr import expr_eval, collect_variables, traverse_expr
from exprbases import EvaluableExpression
from context import context_length, context_subset, context_delete
from utils import loop_wh_progress

# TODO: a quoi sert le collect variable ?
implemented_difficulty_methods = ['EDtM', 'SDtOM']

class ScoreMatching(EvaluableExpression):
    ''' General framework for a Matching based on score
        That kind of matching doesn't
        A Score matching need two things :
          - an order for the two sets
          - a way of selecting a match
    '''
    def __init__(self, set1filter, set2filter, orderby1, orderby2):
        self.set1filter = set1filter
        self.set2filter = set2filter
        self.orderby1_expr = orderby1
        self.orderby2_expr = orderby2
#         TDOD: To remove because of case of orderby is in implemented_difficulty_methods
#         if isinstance(orderby1, basestring) | isinstance(orderby2, basestring):
#             raise Exception("Using a string for the orderby expression is not "
#                             "supported anymore. You should use a normal "
#                             "expression (ie simply remove the quotes).")

    def traverse(self, context):
        for node in traverse_expr(self.set1filter, context):
            yield node
        for node in traverse_expr(self.set2filter, context):
            yield node
        for node in traverse_expr(self.orderby1_expr, context):
            yield node
        for node in traverse_expr(self.orderby2_expr, context):
            yield node
        yield self

    def collect_variables(self, context):
        expr_vars = collect_variables(self.set1filter, context)
        expr_vars |= collect_variables(self.set2filter, context)
        #FIXME: add variables from orderby_expr. This is not done currently,
        # because the presence of variables is done in expr.expr_eval before
        # the evaluate method is called and the context is completed during
        # evaluation (__other_xxx is added during evaluation).
#        expr_vars |= collect_variables(self.score_expr, context)
        return expr_vars

    def _get_filters(self, context):
        ctx_filter = context.get('__filter__')
        # at some point ctx_filter will be cached automatically, so we don't
        # need to take care of it manually here
        if ctx_filter is not None:
            set1filter = expr_eval(ctx_filter & self.set1filter, context)
            set2filter = expr_eval(ctx_filter & self.set2filter, context)
        else:
            set1filter = expr_eval(self.set1filter, context)
            set2filter = expr_eval(self.set2filter, context)
        return set1filter, set2filter

    #noinspection PyUnusedLocal
    def dtype(self, context):
        return int


class RankingMatching(ScoreMatching):
    '''
    Matching based on score
        set 1 is ranked by decreasing orderby1
        set 2 is ranked by decreasing orderby2
        Then individuals in the nth position in each list are matched together.
        The reverse options allow, if True, to sort by increasing orderby
    '''
    def __init__(self, set1filter, set2filter, orderby1, orderby2,
                 reverse1=False, reverse2=False):
        ScoreMatching.__init__(self, set1filter, set2filter, orderby1, orderby2)
        self.reverse1 = reverse1
        self.reverse2 = reverse2

    def _get_used_variables_order(self, context):
        orderby1_expr = self.orderby1_expr
        orderby2_expr = self.orderby2_expr
        used_variables1 = orderby1_expr.collect_variables(context)
        used_variables2 = orderby2_expr.collect_variables(context)
        used_variables1.add('id')
        used_variables2.add('id')
        return used_variables1, used_variables2

    def _get_sorted_indices(self, set1filter, set2filter, context):
        orderby1_expr = self.orderby1_expr
        orderby2_expr = self.orderby2_expr
        orderby1 = expr_eval(orderby1_expr, context)
        orderby2 = expr_eval(orderby2_expr, context)
        if self.reverse1:
            orderby1 = - orderby1  # reverse sorting
        if self.reverse2:
            orderby2 = - orderby2  # reverse sorting
        sorted_set1_indices = orderby1[set1filter].argsort()[::-1]
        sorted_set2_indices = orderby2[set2filter].argsort()[::-1]
        return sorted_set1_indices, sorted_set2_indices

    def _match(self, set1tomatch, set2tomatch, set1, set2, context):
        result = np.empty(context_length(context), dtype=int)
        result.fill(-1)

        id_to_rownum = context.id_to_rownum
        id1 = set1['id'][set1tomatch]
        id2 = set2['id'][set2tomatch]
        result[id_to_rownum[id1]] = id2
        result[id_to_rownum[id2]] = id1
        return result

    def evaluate(self, context):
        set1filter, set2filter = self._get_filters(context)
        set1len = set1filter.sum()
        set2len = set2filter.sum()
        tomatch = min(set1len, set2len)
        
        sorted_set1_indices, sorted_set2_indices = \
                self._get_sorted_indices(set1filter, set2filter, context)
        set1tomatch = sorted_set1_indices[:tomatch]
        set2tomatch = sorted_set2_indices[:tomatch]

        used_variables1, used_variables2 = self._get_used_variables_order(context)
        set1 = context_subset(context, set1filter, used_variables1)
        set2 = context_subset(context, set2filter, used_variables2)

        print("matching with %d/%d individuals" % (set1len, set2len))
        return self._match(set1tomatch, set2tomatch, set1, set2, context)

class SequentialMatching(ScoreMatching):
    '''
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
    '''
    def __init__(self, set1filter, set2filter, score, orderby, pool_size=None):
        ScoreMatching.__init__(self, set1filter, set2filter, orderby, None)
        self.score_expr = score
        if pool_size is not None:
            assert isinstance(pool_size, int)
            assert pool_size > 0
        self.pool_size = pool_size

    def _get_used_variable_order(self, context):
        orderby1_expr = self.orderby1_expr
        used_variables1 = orderby1_expr.collect_variables(context)
        used_variables1 += ['id']
        if self.orderby2_expr is None:
            return used_variables1, None
        orderby2_expr = self.orderby2_expr
        used_variables2 = orderby2_expr.collect_variables(context)
        used_variables2 += ['id']
        return used_variables1, used_variables2

    def _get_used_variables_match(self, context):
        score_expr = self.score_expr
        used_variables = score_expr.collect_variables(context)
        used_variables1 = [v for v in used_variables
                                    if not v.startswith('__other_')]
        used_variables2 = [v[8:] for v in used_variables
                                    if v.startswith('__other_')]
        used_variables1 += ['id']
        used_variables2 += ['id']
        return used_variables1, used_variables2

    def _get_sorted_indices(self, set1filter, set2filter, context):
        orderby1_expr = self.orderby1_expr
        if not isinstance(orderby1_expr, str):
            order = expr_eval(orderby1_expr, context)
        else:
            order = np.zeros(context_length(context), dtype=int)
            used_variables1, used_variables2 = \
                self._get_used_variables_match(context)
            set1 = context_subset(context, set1filter, used_variables1)
            set2 = context_subset(context, set2filter, used_variables2)

            if orderby1_expr == 'EDtM':
                for var in used_variables1:
                    order[set1filter] += (set1[var] - set1[var].mean())**2 \
                                          / set1[var].var()
            if orderby1_expr == 'SDtOM':
                orderby_ctx = dict((k if k in used_variables1 else k, v)
                                 for k, v in set1.iteritems())
                orderby_ctx.update(('__other_' + k, set2[k].mean())
                                 for k in used_variables2)
                order[set1filter] = expr_eval(orderby1_expr, orderby_ctx)

        sorted_set1_indices = order[set1filter].argsort()[::-1]
        return sorted_set1_indices, None

    def _match(self, set1tomatch, set1, set2,
              used_variables1, used_variables2, context):
        global matching_ctx

        score_expr = self.score_expr
        result = np.empty(context_length(context), dtype=int)
        id_to_rownum = context.id_to_rownum
        used_variables1
        result.fill(-1)

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

            local_ctx.update((k, set1[k][sorted_idx])
                                  for k in used_variables1)
            # pk = tuple(individual1[fname] for fname in pk_names)
            # optimized_expr = optimized_exprs.get(pk)
            # if optimized_expr is None:
            # for name in pk_names:
            # fake_set1['__f_%s' % name].value = individual1[name]
            # optimized_expr = str(symbolic_expr.simplify())
            # optimized_exprs[pk] = optimized_expr
            # set2_scores = evaluate(optimized_expr, mm_dict, set2)
            set2_scores = expr_eval(score_expr, local_ctx)

            individual2_idx = np.argmax(set2_scores)

            id1 = local_ctx['id']
            id2 = local_ctx['__other_id'][individual2_idx]

            matching_ctx = context_delete(matching_ctx, individual2_idx)

            result[id_to_rownum[id1]] = id2
            result[id_to_rownum[id2]] = id1


        loop_wh_progress(match_one_set1_individual, set1tomatch,
                         pool_size=self.pool_size)
        return result

    def evaluate(self, context):
        set1filter, set2filter = self._get_filters(context)
        set1len = set1filter.sum()
        set2len = set2filter.sum()
        tomatch = min(set1len, set2len)

        sorted_set1_indices, _ = \
                self._get_sorted_indices(set1filter, set2filter, context)
        set1tomatch = sorted_set1_indices[:tomatch]
        print("matching with %d/%d individuals" % (set1len, set2len))

        #TODO: compute pk_names automatically: variables which are either
        # boolean, or have very few possible values and which are used more
        # than once in the expression and/or which are used in boolean
        # expressions
#        pk_names = ('eduach', 'work')
#        optimized_exprs = {}

        used_variables1, used_variables2 = \
            self._get_used_variables_match(context)
        set1 = context_subset(context, set1filter, used_variables1)
        set2 = context_subset(context, set2filter, used_variables2)
        return self._match(set1tomatch, set1, set2,
                          used_variables1, used_variables2, context)

functions = {'matching': SequentialMatching, 'rank_matching': RankingMatching}
