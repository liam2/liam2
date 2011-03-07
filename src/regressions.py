import numpy as np

from properties import Log, Exp, Normal, Max, \
                       CompoundExpression
import properties
from alignment import Alignment
from expr import Expr, Variable, functions, collect_variables
from entities import context_length

#TODO: make those available
def logit(expr):
    return Log(expr / (1.0 - expr))

def logistic(expr):
    return 1.0 / (1.0 + Exp(-expr))


class Regression(CompoundExpression):
    def __init__(self, expr, filter=None, align=False):
        CompoundExpression.__init__(self)
        self.expr = expr
        self.filter = filter
        if isinstance(align, float):
            align_kwargs = {'variables': [],
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
        if self.align_kwargs is not None:
            return Alignment(self.build_score_expr(), self.filter,
                             **self.align_kwargs)
        else:
            return self.build_score_expr() > 0.5

    def collect_variables(self, context):
        return collect_variables(self.expr, context)


class LogitScore(CompoundExpression):
    func_name = 'logit_score'

    def __init__(self, expr):
        self.expr = expr
        self.u_varname = "temp_%d" % properties.num_tmp
        properties.num_tmp += 1

    def build_context(self, context):
        context[self.u_varname] = \
            np.random.uniform(size=context_length(context)) 
        return context

    def build_expr(self):
        expr = self.expr
        u = Variable(self.u_varname, float)
        if not isinstance(expr, Expr) and not expr: # expr in (0, 0.0, False, '')
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
    

class LogitRegr(Regression):
    func_name = 'logit_regr'

    def build_score_expr(self):
        return LogitScore(self.expr)


class ContRegr(Regression):
    func_name = 'cont_regr'

    def __init__(self, expr, filter=None, align=False, mult=0.0,
                 error_var=None):
        Regression.__init__(self, expr, filter, align)
        self.mult = mult
        self.error_var = error_var
        
    def build_score_expr(self):
        expr = self.expr
        if self.error_var:
            expr += Variable(self.error_var)
        if self.mult:
            expr += Normal(0, 1) * self.mult
        return expr

       
class ClipRegr(ContRegr):
    func_name = 'clip_regr'
    
    def build_score_expr(self):
        return Max(ContRegr.build_score_expr(self), 0)


class LogRegr(ContRegr):
    func_name = 'log_regr'
    
    def build_score_expr(self):
        # exp(x) overflows for x > 709 but that is handled gracefully by numpy
        # and numexpr
        return Exp(ContRegr.build_score_expr(self))


functions.update({
    'logit_score': LogitScore,
    'logit_regr': LogitRegr,
    'cont_regr': ContRegr,
    'clip_regr': ClipRegr,
    'log_regr': LogRegr,
})
