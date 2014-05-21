from __future__ import print_function

import types

import numpy as np

import config
from context import context_length
from expr import (Expr, FunctionExpr, EvaluableExpression, expr_eval,
                  traverse_expr, getdtype, as_simple_expr, as_string,
                  get_missing_value, ispresent, LogicalOp)
from utils import classproperty


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


class FilteredExpression(FunctionExpr):
    @staticmethod
    def _getfilter(context, filter):
        ctx_filter = context.filter_expr
        if filter is not None and ctx_filter is not None:
            filter_expr = LogicalOp('&', ctx_filter, filter)
        elif filter is not None:
            filter_expr = filter
        elif ctx_filter is not None:
            filter_expr = ctx_filter
        else:
            filter_expr = None
        if filter_expr is not None and \
                getdtype(filter_expr, context) is not bool:
            raise Exception("filter must be a boolean expression")
        return filter_expr


class NumpyFunction(FunctionExpr):
    np_func = None
    # all subclasses support a filter keyword-only argument
    kwonlyargs = {'filter': None}

    @classmethod
    def get_compute_func(cls):
        func = cls.np_func
        if func is None:
            return None
        elif isinstance(func, types.BuiltinFunctionType):
            # Note that types.BuiltinFunctionType and types.BuiltinMethodType
            # are the same object (equal to <type 'builtin_function_or_method'>)
            return func
        else:
            # This is necessary because class attributes set to functions are
            # automatically converted to methods !
            # >>> def f():
            # ...     pass
            # >>> class A(object):
            # ...     m = f
            # >>> f
            # <function f at 0x02844470>
            # >>> A.m
            # <unbound method A.f>
            assert isinstance(func, types.MethodType)
            return func.im_func

    # subclasses can override this by a class-constant
    @classproperty
    @classmethod
    def funcname(cls):
        return cls.get_compute_func().__name__


class NumpyChangeArray(NumpyFunction):
    def __init__(self, *args, **kwargs):
        # the first argument should be the array to work on ('a')
        assert self.argspec.args[0] == 'a'
        NumpyFunction.__init__(self, *args, **kwargs)

    def compute(self, context, *args, **kwargs):
        filter_value = kwargs.pop('filter', None)

        func = self.get_compute_func()
        new_values = func(*args, **kwargs)

        if filter_value is None:
            return new_values
        else:
            # we cannot do this yet because dtype() currently requires
            # context (and I don't want to change the signature of compute
            # just for that) assert dtype(old_values) == dtype(new_values)
            old_values = args[0]
            return np.where(filter_value, new_values, old_values)


class NumpyCreateArray(NumpyFunction):
    def compute(self, context, *args, **kwargs):
        filter_value = kwargs.pop('filter', None)

        func = self.get_compute_func()
        values = func(*args, **kwargs)

        if filter_value is None:
            return values
        else:
            missing_value = get_missing_value(values)
            return np.where(filter_value, values, missing_value)


class NumpyRandom(NumpyCreateArray):
    def _eval_args(self, context):
        args, kwargs = NumpyCreateArray._eval_args(self, context)
        if 'size' in self.argspec.args:
            pos = self.argspec.args.index('size')
            size = args[pos]

            # The original functions return a scalar when size is None, and an
            # array of length one when size is 1.
            #TODO: users should have a way to have the "size=None" behavior. We
            # could differentiate whether None was explicitly passed or comes
            # from the default value (as we did previously: 'size' not in
            # kwargs), but I do not think it is a good idea. Adding a new
            # "sentinel" value (e.g. -1 or "scalar") is probably better.
            if size is None:
                args = args[:pos] + (context_length(context),) + args[pos + 1:]
        return args, kwargs

    def compute(self, context, *args, **kwargs):
        if config.debug:
            print()
            print("random sequence position before:", np.random.get_state()[2])
        res = super(NumpyRandom, self).compute(context, *args, **kwargs)
        if config.debug:
            print("random sequence position after:", np.random.get_state()[2])
        return res


class NumpyAggregate(NumpyFunction):
    nan_func = (None,)
    kwonlyargs = {'filter': None, 'skip_na': True}

    def __init__(self, *args, **kwargs):
        # the first argument should be the array to work on ('a')
        assert self.argspec.args[0] == 'a'
        NumpyFunction.__init__(self, *args, **kwargs)

    def compute(self, context, *args, **kwargs):
        filter_value = kwargs.pop('filter', None)
        skip_na = kwargs.pop('skip_na', True)

        values, args = args[0], args[1:]
        values = np.asanyarray(values)

        if (skip_na and np.issubdtype(values.dtype, np.inexact) and
                self.nan_func[0] is not None):
            usenanfunc = True
            func = self.nan_func[0]
        else:
            usenanfunc = False
            func = self.get_compute_func()

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
    funcname = None

    def __init__(self, expr):
        self.expr = expr

    def as_simple_expr(self, context):
        return self.__class__(as_simple_expr(self.expr, context))

    def as_string(self):
        return '%s(%s)' % (self.funcname, as_string(self.expr))

    def __str__(self):
        return '%s(%s)' % (self.funcname, self.expr)

    def traverse(self, context):
        for node in traverse_expr(self.expr, context):
            yield node


class TableExpression(FunctionExpr):
    pass
