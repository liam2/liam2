from __future__ import print_function

import numpy as np

from expr import expr_eval, collect_variables, traverse_expr
from exprbases import EvaluableExpression
from context import context_length, context_subset, context_delete
from utils import loop_wh_progress
from timeit import itertools

#TODO:  
# class Ranking_Matching(EvaluableExpression):
           
class Score_Matching(EvaluableExpression):
    ''' General frame for a Matching based on score '''
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
        NotImplementedError

    #noinspection PyUnusedLocal
    def dtype(self, context):
        return int

class Closest_Neighbourg(Score_Matching):
    ''' add a variable orderby to class Score_Matching) 
        set the evaluate method
    '''
    def __init__(self, set1filter, set2filter, score, orderby):
        Score_Matching.__init__(self, set1filter, set2filter, score)
        self.orderby = orderby
        
    def traverse(self, context):
        return itertools.chain(traverse_expr(self.orderby, context), Score_Matching.traverse(self,context))
                
    def collect_variables(self, context):
        expr_vars = Score_Matching.collect_variables(self, context)
        expr_vars |= collect_variables(self.orderby, context)
        return expr_vars 

    def orderby_method(self, context):
        NotImplementedError

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
        used_variables1 = ['id'] + [v for v in used_variables
                                    if not v.startswith('__other_')]
        used_variables2 = ['id'] + [v[8:] for v in used_variables
                                    if v.startswith('__other_')]

        set1 = context_subset(context, set1filter, used_variables1)
        set2 = context_subset(context, set2filter, used_variables2)
        set1len = set1filter.sum()
        set2len = set2filter.sum()
        tomatch = min(set1len, set2len)
        orderby = self.orderby_method(context)
        sorted_set1_indices = orderby[set1filter].argsort()[::-1]
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

        local_ctx = dict(('__other_' + k if k in used_variables2 else k, v)
                         for k, v in set2.iteritems())

        #noinspection PyUnusedLocal
        def match_one_set1_individual(idx, sorted_idx):
            global local_ctx

            if not context_length(local_ctx):
                raise StopIteration

            local_ctx.update((k, set1[k][sorted_idx]) for k in used_variables1)

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
        return result
    
class Classic_CN_Matching(Closest_Neighbourg):
    
    def orderby_method(self, context):
        return expr_eval(self.orderby, context)
    
difficulty_methods = ['sum of squares', 'square distance to the other mean']
class ODD_CN_Matching(Closest_Neighbourg):
    def __init__(self, set1filter, set2filter, score, difficulty='sum of squares'):
        Closest_Neighbourg.__init__(self, set1filter, set2filter, score, None)
        if difficulty not in difficulty_methods:
            raise Exception("The given method is not implemented, you can try with "
                            "%s "  % (' or '.join(difficulty_methods))) 
        self.difficulty = difficulty
        
    def orderby_method(self, context):
        order = np.zeros(context_length(context), dtype=int)
        score_expr = self.score_expr
        ctx_filter = context.get('__filter__')    
        # at some point ctx_filter will be cached automatically, so we don't
        # need to take care of it manually here
        if ctx_filter is not None:
            set1filter = expr_eval(ctx_filter & self.set1filter, context)
        else:
            set1filter = expr_eval(self.set1filter, context)
        
        used_variables = score_expr.collect_variables(context)
        used_variables1 = [v for v in used_variables
                                    if not v.startswith('__other_')]
        set1 = context_subset(context, set1filter, used_variables1)
        
        if self.difficulty == 'sum of squares':
            for var in used_variables1:
                order[set1filter] += (set1[var] -  set1[var].mean())**2
            return order
            
        if self.difficulty == 'square distance to the other mean':
            used_variables2 = [v[8:] for v in used_variables
                                        if v.startswith('__other_')]
            if ctx_filter is not None:
                set2filter = expr_eval(ctx_filter & self.set2filter, context)
            else:
                set2filter = expr_eval(self.set2filter, context)
            set2 = context_subset(context, set2filter, used_variables2)
            
            local_ctx = dict((k if k in used_variables1 else k, v)
                         for k, v in set1.iteritems())
            local_ctx.update(('__other_' + k, set2[k].mean()) for k in used_variables2) 

            order[set1filter] = expr_eval(score_expr, local_ctx)
            return order
            
    
functions = {'matching': Classic_CN_Matching,
             'matching_odd' : ODD_CN_Matching}
