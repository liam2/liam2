import numpy as np

from expr import (Expr, Variable, ShortLivedVariable, get_tmp_varname,
                  missing_values, dtype, expr_eval)
from exprbases import CompoundExpression
from exprmisc import Log, Exp, Normal, Max, Where
from alignment import Alignment

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

    def build_expr(self):
        raise NotImplementedError

    def add_filter(self, expr, context):
        if self.filter is not None:
            missing_value = missing_values[dtype(expr, context)]
            return Where(self.filter, expr, missing_value)
        else:
            return expr

    def evaluate(self, context):
        context = self.build_context(context)
        expr = self.add_filter(self.complete_expr, context)
        return expr_eval(expr, context)

    def as_simple_expr(self, context):
        context = self.build_context(context)
        expr = self.add_filter(self.complete_expr, context)
        return expr.as_simple_expr(context)

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
            epsilon = logit(u)
            expr = logistic(expr - epsilon)
        return expr

    def __str__(self):
        return '%s(%s)' % (self.func_name, self.expr)

    def dtype(self, context):
        return float


class LogitRegr(Regression):
    func_name = 'logit_regr'

    def __init__(self, expr, filter=None, align=None):
        Regression.__init__(self, expr, filter)
        self.align = align

    def build_context(self, context):
        return context

    def build_expr(self):
        score_expr = LogitScore(self.expr)
        if self.align is not None:
            return Alignment(score_expr, self.align, filter=self.filter)
        else:
            return score_expr > 0.5

    # this is an optimisation: Alignment already handles the filter
    def add_filter(self, expr, context):
        if self.align is not None:
            return expr
        else:
            return super(LogitRegr, self).add_filter(expr, context)

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
        return Exp(ContRegr.build_expr(self))


functions = {
    'logit_score': LogitScore,
    'logit_regr': LogitRegr,
    'cont_regr': ContRegr,
    'clip_regr': ClipRegr,
    'log_regr': LogRegr,
}
