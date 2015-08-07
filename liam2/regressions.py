from __future__ import print_function

from alignment import Alignment
from expr import (Expr, Variable, BinaryOp, ComparisonOp, missing_values,
                  getdtype, always)
from exprbases import CompoundExpression
from exprmisc import Exp, Max, Where, Logit, Logistic
from exprrandom import Normal, Uniform


class Regression(CompoundExpression):
    """abstract base class for all regressions"""

    @staticmethod
    def add_filter(expr, filter):
        if filter is not None:
            missing_value = missing_values[getdtype(expr, None)]
            return Where(filter, expr, missing_value)
        else:
            return expr

    dtype = always(float)


class LogitScore(CompoundExpression):
    funcname = 'logit_score'

    def build_expr(self, expr):
        u = Uniform()
        # expr in (0, 0.0, False, '')
        if not isinstance(expr, Expr) and not expr:
            expr = u
        else:
            epsilon = Logit(u)
            # expr = logistic(expr - epsilon)
            expr = Logistic(BinaryOp('-', expr, epsilon))
        return expr

    dtype = always(float)


class LogitRegr(Regression):
    funcname = 'logit_regr'

    def build_expr(self, expr, filter=None, align=None):
        score_expr = LogitScore(expr)
        if align is not None:
            # we do not need add_filter because Alignment already handles it
            return Alignment(score_expr, align, filter=filter)
        else:
            return self.add_filter(ComparisonOp('>', score_expr, 0.5), filter)

    dtype = always(bool)


class ContRegr(Regression):
    funcname = 'cont_regr'

    # TODO: deprecate error_var in favor of an "error" argument (which would
    # be an Expr instead of a string). This would allow any expression instead
    # of only simple variables and would not require quotes in the latter case
    def build_expr(self, expr, filter=None, mult=0.0, error_var=None):
        regr_expr = self.build_regression_expr(expr, mult, error_var)
        return self.add_filter(regr_expr, filter)

    def build_regression_expr(self, expr, mult=0.0, error_var=None):
        if error_var is not None:
            # expr += error_var
            expr = BinaryOp('+', expr, Variable(None, error_var))
        if mult:
            # expr += normal(0, 1) * mult
            expr = BinaryOp('+', expr, BinaryOp('*', Normal(0, 1), mult))
        return expr


class ClipRegr(ContRegr):
    funcname = 'clip_regr'

    def build_regression_expr(self, *args, **kwargs):
        return Max(ContRegr.build_regression_expr(self, *args, **kwargs), 0)


class LogRegr(ContRegr):
    funcname = 'log_regr'

    def build_regression_expr(self, *args, **kwargs):
        return Exp(ContRegr.build_regression_expr(self, *args, **kwargs))


functions = {
    'logit_score': LogitScore,
    'logit_regr': LogitRegr,
    'cont_regr': ContRegr,
    'clip_regr': ClipRegr,
    'log_regr': LogRegr,
}
