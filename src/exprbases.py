from __future__ import print_function

import numpy as np

import config
from context import context_length
from expr import (Expr, AbstractExprCall, EvaluableExpression, expr_eval, traverse_expr,
                  getdtype, as_simple_expr, as_string, get_missing_value,
                  ispresent, LogicalOp)


class CompoundExpression(Expr):
    """expression written in terms of other expressions"""

    def __init__(self):
        self._complete_expr = None

    def evaluate(self, context):
        context = self.build_context(context)
        return expr_eval(self.complete_expr, context)

    def as_simple_expr(self, context):
        context = self.build_context(context)
        return self.complete_expr.as_simple_expr(context)

    def build_context(self, context):
        return context

    def build_expr(self):
        raise NotImplementedError()

    def traverse(self, context):
        for node in traverse_expr(self.complete_expr, context):
            yield node
        yield self

    @property
    def complete_expr(self):
        if self._complete_expr is None:
            self._complete_expr = self.build_expr()
        return self._complete_expr


#TODO: generalise to a function with several arguments
# - factorize with NumpyFunction, AbstractExprCall, FilteredExpression
class FunctionExpression(EvaluableExpression):
    func_name = None

    def __init__(self, expr):
        self.expr = expr

    def traverse(self, context):
        for node in traverse_expr(self.expr, context):
            yield node
        yield self

    def __str__(self):
        return '%s(%s)' % (self.func_name, self.expr)


class FilteredExpression(FunctionExpression):
    def __init__(self, expr, filter=None):
        super(FilteredExpression, self).__init__(expr)
        self.filter = filter

    def traverse(self, context):
        for node in traverse_expr(self.filter, context):
            yield node
        for node in FunctionExpression.traverse(self, context):
            yield node

    def _getfilter(self, context):
        ctx_filter = context.filter_expr
        if self.filter is not None and ctx_filter is not None:
            filter_expr = LogicalOp('&', ctx_filter, self.filter)
        elif self.filter is not None:
            filter_expr = self.filter
        elif ctx_filter is not None:
            filter_expr = ctx_filter
        else:
            filter_expr = None
        if filter_expr is not None and getdtype(filter_expr, context) is not bool:
            raise Exception("filter must be a boolean expression")
        return filter_expr

    def __str__(self):
        filter_str = ', %s' % self.filter if self.filter is not None else ''
        return '%s(%s%s)' % (self.func_name, self.expr, filter_str)


class NumpyFunction(AbstractExprCall):
    func_name = None  # optional (for display)
    np_func = (None,)
    # arg_names can be set automatically by using inspect.getargspec, but it
    # only works for pure Python functions, so I decided to avoid it because
    # when you add a function, it is hard to know whether it is implemented
    # in C or not.
    arg_names = None
    # all subclasses support a filter keyword-only argument
    kwargs_names = ('filter',)

    def __init__(self, *args, **kwargs):
        if len(args) > len(self.arg_names):
            # + 1 to be consistent with Python (to account for self)
            raise TypeError("takes at most %d arguments (%d given)" %
                            (len(self.arg_names) + 1, len(args) + 1))
        allowed_kwargs = set(self.arg_names) | set(self.kwargs_names)
        extra_kwargs = set(kwargs.keys()) - allowed_kwargs
        if extra_kwargs:
            extra_kwargs = [repr(arg) for arg in extra_kwargs]
            raise TypeError("got an unexpected keyword argument %s" %
                            extra_kwargs[0])
        AbstractExprCall.__init__(self, *args, **kwargs)

    @property
    def func_name(self):
        return self.np_func[0].__name__


class NumpyChangeArray(NumpyFunction):
    def __init__(self, *args, **kwargs):
        # the first argument should be the array to work on ('a')
        assert self.arg_names[0] == 'a'
        NumpyFunction.__init__(self, *args, **kwargs)

    def _compute(self, *args, **kwargs):
        func = self.np_func[0]
        new_values = func(*args, **kwargs)

        filter_value = kwargs.pop('filter', None)
        if filter_value is None:
            return new_values
        else:
            # we cannot do this yet because dtype() currently requires
            # context (and I don't want to change the signature of compute
            # just for that) assert dtype(old_values) == dtype(new_values)
            old_values = args[0]
            return np.where(filter_value, new_values, old_values)


class NumpyCreateArray(NumpyFunction):
    def _compute(self, *args, **kwargs):
        func = self.np_func[0]
        values = func(*args, **kwargs)

        filter_value = kwargs.pop('filter', None)
        if filter_value is None:
            return values
        else:
            missing_value = get_missing_value(values)
            return np.where(filter_value, values, missing_value)


class NumpyRandom(NumpyCreateArray):
    def _eval_args(self, context):
        args, kwargs = NumpyCreateArray._eval_args(self, context)
        if 'size' in self.arg_names and 'size' not in kwargs:
            kwargs['size'] = context_length(context)
        return args, kwargs

    def _compute(self, *args, **kwargs):
        if config.debug:
            print()
            print("random sequence position before:", np.random.get_state()[2])
        res = super(NumpyRandom, self)._compute(*args, **kwargs)
        if config.debug:
            print("random sequence position after:", np.random.get_state()[2])
        return res


class NumpyAggregate(NumpyFunction):
    nan_func = (None,)
    kwargs_names = ('filter', 'skip_na')

    def __init__(self, *args, **kwargs):
        # the first argument should be the array to work on ('a')
        assert self.arg_names[0] == 'a'
        NumpyFunction.__init__(self, *args, **kwargs)

    def _compute(self, *args, **kwargs):
        filter_value = kwargs.pop('filter', None)
        skip_na = kwargs.pop('skip_na', True)

        values, args = args[0], args[1:]
        values = np.asanyarray(values)

        if (skip_na and issubclass(values.dtype.type, np.inexact) and
                self.nan_func[0] is not None):
            usenanfunc = True
            func = self.nan_func[0]
        else:
            usenanfunc = False
            func = self.np_func[0]

        if values.shape:
            if values.ndim == 1:
                if skip_na and not usenanfunc:
                    if filter_value is not None:
                        # we should *not* use an inplace operation because
                        # filter_value can be a simple variable
                        filter_value = filter_value & ispresent(values)
                    else:
                        filter_value = ispresent(values)
                if filter_value is not None and filter_value is not True:
                    values = values[filter_value]
            elif values.ndim > 1 and filter_value is not None:
                raise Exception("filter argument is not supported on arrays "
                                "with more than 1 dimension")
        args = (values,) + args
        return func(*args, **kwargs)


class NumexprFunction(Expr):
    """For functions which are present as-is in numexpr"""
    func_name = None

    def __init__(self, expr):
        self.expr = expr

    def as_simple_expr(self, context):
        return self.__class__(as_simple_expr(self.expr, context))

    def as_string(self):
        return '%s(%s)' % (self.func_name, as_string(self.expr))

    def __str__(self):
        return '%s(%s)' % (self.func_name, self.expr)

    def traverse(self, context):
        for node in traverse_expr(self.expr, context):
            yield node


class TableExpression(EvaluableExpression):
    pass
