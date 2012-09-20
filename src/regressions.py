import numpy as np

from properties import Log, Exp, Normal, Max, CompoundExpression
from alignment import Alignment
from expr import Expr, Variable, ShortLivedVariable, get_tmp_varname
from context import context_length


#TODO: make those available
def logit(expr):
    return Log(expr / (1.0 - expr))


def logistic(expr):
    return 1.0 / (1.0 + Exp(-expr))


class Regression(CompoundExpression):
    '''abstract base class for all regressions'''

    def __init__(self, expr, filter=None):
        CompoundExpression.__init__(self)
        self.expr = expr
        self.filter = filter

    def build_context(self, context):
        return context

#TODO: this fixes the filter problem for non-aligned regression, but breaks
#      aligned ones (logit_regr):
#      * the score expr is Alignment(LogitScore(xxx), filter=filter)
#      * Alignment.eval() returns {'values': True, 'indices': indices}
#      * Regression.eval() returns {'filter': filter,
#                                   'values': {'values': Trues,
#                                              'indices': indices}}
#      * this is not supported by Assignment.store_result

#    def eval(self, context):
#        context = self.build_context(context)
#        result = self.complete_expr.eval(context)
#        if self.filter is not None:
#            filter = expr_eval(self.filter, context)
#            return {'filter': filter, 'values': result}
#        else:
#            return result

    def dtype(self, context):
        return float


class LogitScore(CompoundExpression):
    func_name = 'logit_score'

    def __init__(self, expr):
        CompoundExpression.__init__(self)
        self.expr = expr
        self.u_varname = get_tmp_varname()

    def build_context(self, context):
        context[self.u_varname] = \
            np.random.uniform(size=context_length(context))
        return context

    def build_expr(self):
        expr = self.expr
        u = ShortLivedVariable(self.u_varname, float)
        # expr in (0, 0.0, False, '')
        if not isinstance(expr, Expr) and not expr:
            expr = u
        else:
            # In some case, the expression could crash LIAM's interpreter:
            # logit(-1000) => logistic(-1000.0 + epsilon) => exp(1000)
            # => overflow, so LIAM uses "exp(min(expr, 99))" instead.
            # However, this is not needed here since numpy/numexpr handles
            # overflows nicely with "inf".
            # The maximum value before exp overflows is 709.
            epsilon = logit(u)
            expr = logistic(expr - epsilon)
        return expr

    def __str__(self):
        return '%s(%s)' % (self.func_name, self.expr)

    def dtype(self, context):
        return float


class LogitRegr(Regression):
    func_name = 'logit_regr'

    def __init__(self, expr, filter=None, align=False):
        Regression.__init__(self, expr, filter)
        if isinstance(align, float):
            align_kwargs = {'expressions': [],
                            'possible_values': [],
                            'probabilities': [align]}
        elif isinstance(align, basestring):
            align_kwargs = {'fname': align}
        else:
            assert not align, "invalid value for align argument"
            align_kwargs = None
        self.align_kwargs = align_kwargs

    def build_context(self, context):
        return context

    def build_expr(self):
        score_expr = LogitScore(self.expr)
        if self.align_kwargs is not None:
            return Alignment(score_expr, self.filter,
                             **self.align_kwargs)
        else:
            return score_expr > 0.5

    def dtype(self, context):
        return bool


class ContRegr(Regression):
    func_name = 'cont_regr'

    def __init__(self, expr, filter=None, mult=0.0, error_var=None):
        Regression.__init__(self, expr, filter)
        self.mult = mult
        self.error_var = error_var

    def build_expr(self):
        expr = self.expr
        if self.error_var is not None:
            expr += Variable(self.error_var)
        if self.mult:
            expr += Normal(0, 1) * self.mult
        return expr


class ClipRegr(ContRegr):
    func_name = 'clip_regr'

    def build_expr(self):
        return Max(ContRegr.build_expr(self), 0)


class LogRegr(ContRegr):
    func_name = 'log_regr'

    def build_expr(self):
        # exp(x) overflows for x > 709 but that is handled gracefully by numpy
        # and numexpr
        return Exp(ContRegr.build_expr(self))


functions = {
    'logit_score': LogitScore,
    'logit_regr': LogitRegr,
    'cont_regr': ContRegr,
    'clip_regr': ClipRegr,
    'log_regr': LogRegr,
}
