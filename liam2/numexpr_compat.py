from typing import Optional, Dict

import numpy as np

from numexpr.necompiler import getExprNames, getArguments, getType, evaluate_lock
from numexpr import NumExpr


class CompiledExpression:
    def __init__(self, expr: str,
                 local_dict: Optional[Dict] = None,
                 global_dict: Optional[Dict] = None,
                 optimization='aggressive',
                 truediv=True,
                 sanitize=True, _frame_depth=2):
        context = {'optimization': optimization, 'truediv': truediv}
        argnames, uses_vml = getExprNames(expr, context, sanitize=sanitize)
        self.argnames = argnames
        self.uses_vml = uses_vml
        argvalues = getArguments(argnames, local_dict, global_dict,
                                 _frame_depth=_frame_depth)
        # Create a signature
        signature = [(name, getType(arg)) for (name, arg) in
                     zip(argnames, argvalues)]
        self.compiled_ex = NumExpr(expr, signature, sanitize=sanitize, **context)

    def evaluate(self,
                 local_dict: Optional[Dict] = None,
                 global_dict: Optional[Dict] = None,
                 out: np.ndarray = None,
                 order: str = 'K',
                 casting: str = 'safe',
                 _frame_depth: int = 2):
        args = getArguments(self.argnames, local_dict, global_dict, _frame_depth)
        kwargs = {'out': out, 'order': order, 'casting': casting,
                  'ex_uses_vml': self.uses_vml}
        with evaluate_lock:
            return self.compiled_ex(*args, **kwargs)


class JITExpression:
    def __init__(self, expr: str, optimization='aggressive',
                 truediv=True, sanitize=True):
        context = {'optimization': optimization, 'truediv': truediv}
        argnames, uses_vml = getExprNames(expr, context, sanitize=sanitize)
        self.expr = expr
        self.context = context
        self.argnames = argnames
        self.uses_vml = uses_vml
        self.sanitize = sanitize
        self._compiled_expr_per_sig = {}

    def evaluate(self,
                 local_dict: Optional[Dict] = None,
                 global_dict: Optional[Dict] = None,
                 out: np.ndarray = None,
                 order: str = 'K',
                 casting: str = 'safe',
                 _frame_depth: int = 2):
        args = getArguments(self.argnames, local_dict, global_dict, _frame_depth)
        signature = tuple([(name, getType(arg)) for (name, arg) in
                           zip(self.argnames, args)])
        compiled_ex = self._compiled_expr_per_sig.get(signature)
        if compiled_ex is None:
            compiled_ex = NumExpr(self.expr, signature, sanitize=self.sanitize,
                                  **self.context)
            self._compiled_expr_per_sig[signature] = compiled_ex
        kwargs = {'out': out, 'order': order, 'casting': casting,
                  'ex_uses_vml': self.uses_vml}
        with evaluate_lock:
            return compiled_ex(*args, **kwargs)
