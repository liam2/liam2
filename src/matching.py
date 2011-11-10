import numpy as np

from expr import evaluate, Variable, functions, expr_eval, collect_variables
from entities import context_length, context_subset, context_delete
from properties import EvaluableExpression
from utils import loop_wh_progress


class Matching(EvaluableExpression):
    def __init__(self, set1filter, set2filter, score, orderby):
        self.set1filter = set1filter
        self.set2filter = set2filter
        self.score_expr = score
        self.orderby = orderby
    
    def collect_variables(self, context):
        #FIXME: add score_expr 
        expr_vars = collect_variables(self.set1filter, context)
        expr_vars |= collect_variables(self.set2filter, context)
        expr_vars |= collect_variables(self.orderby, context)
        return expr_vars
    
    def eval(self, context):
        global set2

        ctx_filter = context.get('__filter__')
        
        id_to_rownum = context.id_to_rownum

        score_expr = self.score_expr.strip()
        score_expr = score_expr.replace('other.', '__o_') 
        
        field_names = [name for name in context.keys()
                       if not name.startswith('__') and name != 'nan']
        variables = dict((name, Variable('__self_%s' % name))
                         for name in field_names)
        variables.update(('__o_%s' % name, Variable(name))
                         for name in field_names)
        # parse string
        symbolic_expr = eval(score_expr, variables)
        score_expr = str(symbolic_expr)

        # at some point ctx_filter will be cached automatically, so we don't
        # need to take care of it manually here 
        if ctx_filter is not None:
            set1filter = expr_eval(ctx_filter & self.set1filter, context)
            set2filter = expr_eval(ctx_filter & self.set2filter, context)
        else:
            set1filter = expr_eval(self.set1filter, context)
            set2filter = expr_eval(self.set2filter, context)

        used_variables = symbolic_expr.collect_variables(context)
        used_variables1 = ['id'] + [v[7:] for v in used_variables
                                    if v.startswith('__self_')]
        used_variables2 = ['id'] + [v for v in used_variables
                                    if not v.startswith('__self_')]

        set1 = context_subset(context, set1filter, used_variables1)
        set2 = context_subset(context, set2filter, used_variables2)
        
        orderby = expr_eval(self.orderby, context)
        sorted_set1_indices = orderby[set1filter].argsort()[::-1]
        
        print "matching with %d/%d individuals" % (set1filter.sum(),
                                                   set2filter.sum())

#        #TODO: compute pk_names automatically: variables which are either
#        # boolean, or have very few possible values and which are used more
#        # than once in the expression and/or which are used in boolean 
#        # expressions
#        pk_names = ('eduach', 'work')        
#        optimized_exprs = {}

        result = np.empty(context_length(context), dtype=int)
        result.fill(-1)

        mm_dict = {}
        def match_one_set1_individual(idx, sorted_idx):
            global set2
            
            if not context_length(set2):
                raise StopIteration                

            mm_dict.update(('__self_%s' % name, set1[name][sorted_idx])
                           for name in used_variables1)

#            pk = tuple(individual1[fname] for fname in pk_names)
#            optimized_expr = optimized_exprs.get(pk)
#            if optimized_expr is None:
#                for name in pk_names:
#                    fake_set1['__f_%s' % name].value = individual1[name] 
#                optimized_expr = str(symbolic_expr.simplify())
#                optimized_exprs[pk] = optimized_expr
#            set2_scores = evaluate(optimized_expr, mm_dict, set2)

            #TODO: use eval_expr instead
            set2_scores = evaluate(score_expr, mm_dict, set2)
            
            individual2_idx = np.argmax(set2_scores)       
            
            id1 = set1['id'][sorted_idx]
            id2 = set2['id'][individual2_idx]
            
            set2 = context_delete(set2, individual2_idx)
            
            result[id_to_rownum[id1]] = id2
            result[id_to_rownum[id2]] = id1
            
        loop_wh_progress(match_one_set1_individual, sorted_set1_indices)
        return result
    
    def dtype(self, context):
        return int
    
functions['matching'] = Matching