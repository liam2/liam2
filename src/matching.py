from __future__ import print_function

import numpy as np

from expr import expr_eval, traverse_expr, LogicalOp, always, FunctionExpr
from context import context_length, context_delete
from utils import loop_wh_progress


class Matching(FunctionExpr):
    func_name = 'matching'
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

    def compute(self, context, set1filter, set2filter, score, orderby):
        global local_ctx

        if isinstance(score, basestring):
            raise Exception("Using a string for the score expression is not "
                            "supported anymore. You should use a normal "
                            "expression (ie simply remove the quotes).")

        ctx_filter = context.filter_expr
        id_to_rownum = context.id_to_rownum

        # at some point ctx_filter will be cached automatically, so we don't
        # need to take care of it manually here
        if ctx_filter is not None:
            set1filter = expr_eval(LogicalOp('&', ctx_filter, set1filter),
                                   context)
            set2filter = expr_eval(LogicalOp('&', ctx_filter, set2filter),
                                   context)
        else:
            set1filter = expr_eval(set1filter, context)
            set2filter = expr_eval(set2filter, context)

        used_variables = score.collect_variables(context)
        used_variables1 = ['id'] + [v for v in used_variables
                                    if not v.startswith('__other_')]
        used_variables2 = ['id'] + [v[8:] for v in used_variables
                                    if v.startswith('__other_')]

        set1 = context.subset(set1filter, used_variables1)
        set2 = context.subset(set2filter, used_variables2)
        set1 = set1.entity_data
        set2 = set2.entity_data

        set1len = set1filter.sum()
        set2len = set2filter.sum()
        tomatch = min(set1len, set2len)
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

            set2_scores = expr_eval(score, local_ctx)

            individual2_idx = np.argmax(set2_scores)

            id1 = local_ctx['id']
            id2 = local_ctx['__other_id'][individual2_idx]

            local_ctx = context_delete(local_ctx, individual2_idx)

            result[id_to_rownum[id1]] = id2
            result[id_to_rownum[id2]] = id1

        loop_wh_progress(match_one_set1_individual, set1tomatch)
        return result

    dtype = always(int)


functions = {'matching': Matching}
