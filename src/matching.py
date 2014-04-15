from __future__ import print_function

import numpy as np
import itertools
import random

from expr import expr_eval, collect_variables, traverse_expr
from exprbases import EvaluableExpression
from context import context_length, context_subset, context_delete
from utils import loop_wh_progress

#TODO:  
class RankingMatching(EvaluableExpression):
    ''' 
    Matching based on rank 
        set 1 is ranked by ascending rank1 
        set 2 is ranked by ascending rank2
        Then individuals in the nth position in each list are matched together.
        The ascending options allow, if False, to sort by decreasing rank
    '''
    def __init__(self, set1filter, set2filter, rank1, rank2, ascending1=True, ascending2=True):
        self.set1filter = set1filter
        self.set2filter = set2filter
        self.rank1_expr = rank1
        self.rank2_expr = rank2
        self.ascending1 = ascending1
        self.ascending2 = ascending2
        
    def traverse(self, context):
        for node in traverse_expr(self.set1filter, context):
            yield node
        for node in traverse_expr(self.set2filter, context):
            yield node
        for node in traverse_expr(self.rank1_expr, context):
            yield node
        for node in traverse_expr(self.rank2_expr, context):
            yield node
        yield self

    def collect_variables(self, context):
        expr_vars = collect_variables(self.set1filter, context)
        expr_vars |= collect_variables(self.set2filter, context)
        #FIXME: add variables from score_expr. This is not done currently,
        # because the presence of variables is done in expr.expr_eval before
        # the evaluate method is called and the context is completed during
        # evaluation (__other_xxx is added during evaluation).
#        expr_vars |= collect_variables(self.rank1_expr, context)
#        expr_vars |= collect_variables(self.rank2_expr, context)
        return expr_vars

    def evaluate(self, context):
        ctx_filter = context.get('__filter__')
        id_to_rownum = context.id_to_rownum

        # at some point ctx_filter will be cached automatically, so we don't
        # need to take care of it manually here
        if ctx_filter is not None:
            set1filter = expr_eval(ctx_filter & self.set1filter, context)
            set2filter = expr_eval(ctx_filter & self.set2filter, context)
        else:
            set1filter = expr_eval(self.set1filter, context)
            set2filter = expr_eval(self.set2filter, context)

        rank1_expr = self.rank1_expr
        rank2_expr = self.rank2_expr
        used_variables1 = rank1_expr.collect_variables(context)
        used_variables2 = rank2_expr.collect_variables(context)
        used_variables1.add('id')
        used_variables2.add('id')
        set1 = context_subset(context, set1filter, used_variables1)
        set2 = context_subset(context, set2filter, used_variables2)
        set1len = set1filter.sum()
        set2len = set2filter.sum()
        tomatch = min(set1len, set2len)
        order1 = expr_eval(rank1_expr, context)
        order2 = expr_eval(rank2_expr, context)
        if not self.ascending1: 
            order1 = - order1       # reverse sorting
        if not self.ascending2:
            order2 = - order2       # reverse sorting

        sorted_set1_indices = order1[set1filter].argsort()
        sorted_set2_indices = order2[set2filter].argsort()
        idx1 = sorted_set1_indices[:tomatch]
        idx2 = sorted_set2_indices[:tomatch]
        print("matching with %d/%d individuals" % (set1len, set2len))
        
        result = np.empty(context_length(context), dtype=int)
        result.fill(-1)
        
        id1 = set1['id'][idx1]
        id2 = set2['id'][idx2]
        result[id_to_rownum[id1]] = id2
        result[id_to_rownum[id2]] = id1

        return result
        
    #noinspection PyUnusedLocal
    def dtype(self, context):
        return int
                      
class ScoreMatching(EvaluableExpression):
    ''' General framework for a Matching based on score '''
    def __init__(self, set1filter, set2filter, score):
        self.set1filter = set1filter
        self.set2filter = set2filter
        self.score_expr = score
        if isinstance(score, basestring):
            raise Exception("Using a string for the score expression is not "
                            "supported anymore. You should use a normal "
                            "expression (ie simply remove the quotes).")
        
    def traverse(self, context):
        for node in traverse_expr(self.set1filter, context):
            yield node
        for node in traverse_expr(self.set2filter, context):
            yield node
        for node in traverse_expr(self.score_expr, context):
            yield node
        yield self

    def collect_variables(self, context):
        expr_vars = collect_variables(self.set1filter, context)
        expr_vars |= collect_variables(self.set2filter, context)
        #FIXME: add variables from score_expr. This is not done currently,
        # because the presence of variables is done in expr.expr_eval before
        # the evaluate method is called and the context is completed during
        # evaluation (__other_xxx is added during evaluation).
#        expr_vars |= collect_variables(self.score_expr, context)
        return expr_vars

    def evaluate(self, context):
        raise NotImplementedError

    #noinspection PyUnusedLocal
    def dtype(self, context):
        return int

difficulty_methods = ['EDtM', 'SDtOM']  
class SequentialMatching(ScoreMatching):
    ''' 
    Matching base on researching the best match one by one.
        - orderby gives the way individuals of 1 are sorted before matching
        The first on the list will be matched with its absolute best match
        The last on the list will be matched with the best match among the remaining pool 
        - orederby can be :
            - an expression (usually a variable name) 
            - a string : the name of a method to sort individuals to be match, it is supposed to be
             a "difficulty" because it's better (according to a general objective score) 
             to match hard-to-match people first. The possible difficulty order are : 
                - 'EDtM' : 'Euclidian Distance to the Mean', note it is the reduce euclidan distance that is
                           used which is a common convention when there is more than one variable 
                - 'SDtOM' : 'Score Distance to the Other Mean'
            The SDtOM is the most relevant distance.
    '''
    def __init__(self, set1filter, set2filter, score, orderby, pool_size=None):
        ScoreMatching.__init__(self, set1filter, set2filter, score)
        
        if pool_size is not None:
            assert isinstance(pool_size, int) and pool_size > 0 
        if isinstance(orderby, str):
            if orderby not in difficulty_methods:
                raise Exception("The given method is not implemented, you can try with "
                                "%s "  % (' or '.join(difficulty_methods)))
        self.orderby = orderby
        self.pool_size = pool_size
        
    def traverse(self, context):
        return itertools.chain(traverse_expr(self.orderby, context), ScoreMatching.traverse(self,context))
                
    def collect_variables(self, context):
        expr_vars = ScoreMatching.collect_variables(self, context)
        expr_vars |= collect_variables(self.orderby, context)
        return expr_vars 

    def evaluate(self, context):
        global local_ctx

        ctx_filter = context.get('__filter__')

        id_to_rownum = context.id_to_rownum

        # at some point ctx_filter will be cached automatically, so we don't
        # need to take care of it manually here
        if ctx_filter is not None:
            set1filter = expr_eval(ctx_filter & self.set1filter, context)
            set2filter = expr_eval(ctx_filter & self.set2filter, context)
        else:
            set1filter = expr_eval(self.set1filter, context)
            set2filter = expr_eval(self.set2filter, context)

        score_expr = self.score_expr

        used_variables = score_expr.collect_variables(context)
        used_variables1 = [v for v in used_variables
                                    if not v.startswith('__other_')]
        used_variables2 = [v[8:] for v in used_variables
                                    if v.startswith('__other_')]

        set1 = context_subset(context, set1filter, ['id'] + used_variables1)
        set2 = context_subset(context, set2filter, ['id'] + used_variables2)
        set1len = set1filter.sum()
        set2len = set2filter.sum()
        tomatch = min(set1len, set2len)
        
        orderby = self.orderby
        if not isinstance(orderby, str):
            order = expr_eval(orderby, context)
        else: 
            order = np.zeros(context_length(context), dtype=int)
            if orderby == 'EDtM':
                for var in used_variables1:
                    order[set1filter] += (set1[var] -  set1[var].mean())**2/set1[var].var()
            if orderby == 'SDtOM':
                order_ctx = dict((k if k in used_variables1 else k, v)
                             for k, v in set1.iteritems())
                order_ctx.update(('__other_' + k, set2[k].mean()) for k in used_variables2)
                order[set1filter] = expr_eval(score_expr, order_ctx)               
        
        sorted_set1_indices = order[set1filter].argsort()[::-1]
        set1tomatch = sorted_set1_indices[:tomatch]
        print("matching with %d/%d individuals" % (set1len, set2len))

        #TODO: compute pk_names automatically: variables which are either
        # boolean, or have very few possible values and which are used more
        # than once in the expression and/or which are used in boolean
        # expressions
#        pk_names = ('eduach', 'work')
#        optimized_exprs = {}

        result = np.empty(context_length(context), dtype=int)
        result.fill(-1)

        local_ctx = dict(('__other_' + k if k in ['id'] + used_variables2 else k, v)
                         for k, v in set2.iteritems())

        if self.pool_size is None:
            #noinspection PyUnusedLocal
            def match_one_set1_individual(idx, sorted_idx):
                global local_ctx
    
                if not context_length(local_ctx):
                    raise StopIteration
    
                local_ctx.update((k, set1[k][sorted_idx]) for k in ['id'] + used_variables1)
    
    #            pk = tuple(individual1[fname] for fname in pk_names)
    #            optimized_expr = optimized_exprs.get(pk)
    #            if optimized_expr is None:
    #                for name in pk_names:
    #                    fake_set1['__f_%s' % name].value = individual1[name]
    #                optimized_expr = str(symbolic_expr.simplify())
    #                optimized_exprs[pk] = optimized_expr
    #            set2_scores = evaluate(optimized_expr, mm_dict, set2)
    
                set2_scores = expr_eval(score_expr, local_ctx)
    
                individual2_idx = np.argmax(set2_scores)
    
                id1 = local_ctx['id']
                id2 = local_ctx['__other_id'][individual2_idx]
    
                local_ctx = context_delete(local_ctx, individual2_idx)
    
                result[id_to_rownum[id1]] = id2
                result[id_to_rownum[id2]] = id1            
            
            loop_wh_progress(match_one_set1_individual, set1tomatch)
        else:
            pool_size = self.pool_size
            #noinspection PyUnusedLocal
            def match_one_set1_individual_pool(idx, sorted_idx, pool_size):
                global local_ctx
                
                set2_size = context_length(local_ctx)
                if not set2_size:
                    raise StopIteration
                
                if set2_size > pool_size:
                    pool = random.sample(xrange(context_length(local_ctx)), pool_size)
                else:
                    pool = range(set2_size)

                sub_local_ctx = context_subset(local_ctx, pool, None)
                sub_local_ctx.update((k, set1[k][sorted_idx]) for k in ['id'] + used_variables1)
                
                set2_scores = expr_eval(score_expr, sub_local_ctx)
    
                individual2_pool_idx = np.argmax(set2_scores)
                individual2_idx = pool[individual2_pool_idx]
                
                id1 = sub_local_ctx['id']
                id2 = local_ctx['__other_id'][individual2_idx]
    
                local_ctx = context_delete(local_ctx, individual2_idx)
    
                result[id_to_rownum[id1]] = id2
                result[id_to_rownum[id2]] = id1
                
            loop_wh_progress(match_one_set1_individual_pool, set1tomatch, pool_size=10)
            
        return result
    
functions = {'matching': SequentialMatching, 'rank_matching': RankingMatching}
