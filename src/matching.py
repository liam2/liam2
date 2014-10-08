from __future__ import print_function

import numpy as np

from expr import expr_eval, always
from exprbases import FilteredExpression
from context import context_length, context_delete, EvaluationContext
from utils import loop_wh_progress


class Matching(FilteredExpression):
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

    def compute(self, context, set1filter, set2filter, score, orderby):
        global matching_ctx

        if isinstance(score, basestring):
            raise Exception("Using a string for the score expression is not "
                            "supported anymore. You should use a normal "
                            "expression (ie simply remove the quotes).")

        id_to_rownum = context.id_to_rownum

        set1filterexpr = self._getfilter(context, set1filter)
        set1filtervalue = expr_eval(set1filterexpr, context)
        set2filterexpr = self._getfilter(context, set2filter)
        set2filtervalue = expr_eval(set2filterexpr, context)

        used_variables = score.collect_variables(context)
        used_variables1 = ['id'] + [v for v in used_variables
                                    if not v.startswith('__other_')]
        used_variables2 = ['id'] + [v[8:] for v in used_variables
                                    if v.startswith('__other_')]

        #TODO: we should detect whether or not we are using non-simple
        # expressions (EvaluableExpression children) and pre-evaluate them,
        # because otherwise they are re-evaluated on all of set2 for each
        # individual in set1. See https://github.com/liam2/liam2/issues/128
        set1 = context.subset(set1filtervalue, used_variables1, set1filterexpr)
        set2 = context.subset(set2filtervalue, used_variables2, set2filterexpr)

        # subset creates a dict for the current entity, so .entity_data is a
        # dict
        set1 = set1.entity_data
        set2 = set2.entity_data

        set1len = set1filtervalue.sum()
        set2len = set2filtervalue.sum()
        tomatch = min(set1len, set2len)
        sorted_set1_indices = orderby[set1filtervalue].argsort()[::-1]
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

        matching_ctx = dict(('__other_' + k if k in used_variables2 else k, v)
                            for k, v in set2.iteritems())

        #noinspection PyUnusedLocal
        def match_one_set1_individual(idx, sorted_idx):
            global matching_ctx

            if not context_length(matching_ctx):
                raise StopIteration

            local_ctx = matching_ctx.copy()
            local_ctx.update((k, set1[k][sorted_idx]) for k in used_variables1)
            eval_ctx = context.clone(entity_data=local_ctx)

#            pk = tuple(individual1[fname] for fname in pk_names)
#            optimized_expr = optimized_exprs.get(pk)
#            if optimized_expr is None:
#                for name in pk_names:
#                    fake_set1['__f_%s' % name] = individual1[name]
#                optimized_expr = str(symbolic_expr.simplify())
#                optimized_exprs[pk] = optimized_expr
#            set2_scores = evaluate(optimized_expr, mm_dict, set2)
            set2_scores = expr_eval(score, eval_ctx)

            individual2_idx = np.argmax(set2_scores)

            id1 = local_ctx['id']
            id2 = matching_ctx['__other_id'][individual2_idx]
            matching_ctx = context_delete(matching_ctx, individual2_idx)

            result[id_to_rownum[id1]] = id2
            result[id_to_rownum[id2]] = id1

        loop_wh_progress(match_one_set1_individual, set1tomatch,
                         title="Matching...")
        return result

    dtype = always(int)


functions = {'matching': Matching}
