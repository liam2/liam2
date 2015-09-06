# encoding: utf-8
from __future__ import print_function

import types

import numpy as np

import config
from context import context_length
from expr import (FunctionExpr, not_hashable,
                  getdtype, as_simple_expr, as_string,
                  get_missing_value, ispresent, LogicalOp, AbstractFunction,
                  always, FillArgSpecMeta)
from utils import classproperty, argspec, split_signature


# class CompoundExpression(Expr):
#     """function expression written in terms of other expressions"""
#     def __init__(self):
#         self._complete_expr = None
#
#     def evaluate(self, context):
#         context = self.build_context(context)
#         return expr_eval(self.complete_expr, context)
#
#     def as_simple_expr(self, context):
#         context = self.build_context(context)
#         return self.complete_expr.as_simple_expr(context)
#
#     def build_context(self, context):
#         return context
#
#     def build_expr(self):
#         raise NotImplementedError()
#
#     def traverse(self, context):
#         for node in traverse_expr(self.complete_expr, context):
#             yield node
#         yield self
#
#     @property
#     def complete_expr(self):
#         if self._complete_expr is None:
#             self._complete_expr = self.build_expr()
#         return self._complete_expr


class CompoundExpression(AbstractFunction):
    """
    function expression written in terms of other expressions
    """

    __metaclass__ = FillArgSpecMeta

    kwonlyargs = {}

    @classmethod
    def get_compute_func(cls):
        return cls.build_expr

    def as_simple_expr(self, context):
        # This will effectively trigger evaluation of expressions arguments
        # which are not handled by numexpr functions such has all expressions
        # inheriting from EvaluableExpression (e.g, uniform()) and their result
        # will be stored as a temporary variables in the context. The subtlety
        # to remember is that if a CompoundExpression "duplicates" arguments
        # (such as Logit), those must be either duplicate-safe or
        # EvaluableExpression. For example, if numexpr someday supports random
        # generators, we will be in trouble if we use it as-is. This means we
        # cannot keep the "compiled" expression, because the "temporary
        # variables" would only have a value in the first period, when the
        # expr is "compiled". This would tick the balance in favor of keeping a
        # build_context method.
        args = [as_simple_expr(arg, context) for arg in self.args]
        kwargs = {name: as_simple_expr(arg, context)
                  for name, arg in self.kwargs}
        expr = self.build_expr(*args, **kwargs)
        # We need this because self.build_expr returns an Expr which can
        # contain CompoundExpressions
        return expr.as_simple_expr(context)

    def build_expr(self, *args, **kwargs):
        raise NotImplementedError()


class FilteredExpression(FunctionExpr):
    @staticmethod
    def _getfilter(context, filter):
        ctx_filter = context.filter_expr
        # FIXME: this is a hack and shows that the not_hashable filter_expr in
        #  context is not really a good solution. We should rather add a flag
        # in the context "ishardsubset" or something like that.
        if filter is not_hashable:
            filter_expr = ctx_filter
        elif ctx_filter is not_hashable:
            filter_expr = filter
        elif filter is not None and ctx_filter is not None:
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
            # TODO: users should have a way to have the "size=None" behavior. We
            # could differentiate whether None was explicitly passed or comes
            # from the default value (as we did previously: 'size' not in
            # kwargs), but I do not think it is a good idea. Adding a new
            # "sentinel" value (e.g. -1 or "scalar") is probably better.
            if size is None:
                args = args[:pos] + (context_length(context),) + args[pos + 1:]
        return args, kwargs

    def compute(self, context, *args, **kwargs):
        if config.debug and config.log_level == "processes":
            print()
            print("random sequence position before:", np.random.get_state()[2])
        res = super(NumpyRandom, self).compute(context, *args, **kwargs)
        if config.debug and config.log_level == "processes":
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


class NumexprFunction(AbstractFunction):
    """For functions which are present as-is in numexpr"""
    # argspec need to be given manually for each function
    argspec = None

    def as_simple_expr(self, context):
        args, kwargs = as_simple_expr((self.args, self.kwargs), context)
        return self.__class__(*args, **dict(kwargs))

    def as_string(self):
        args, kwargs = as_string((self.args, self.kwargs))
        return '%s(%s)' % (self.funcname, self.format_args_str(args, kwargs))


class TableExpression(FunctionExpr):
    pass


def make_np_class(baseclass, docstring, dtypefunc):
    name, args = split_signature(docstring)
    if isinstance(dtypefunc, type):
        dtypefunc = always(dtypefunc)

    # we need to explicitly set funcname, because the usual mechanism of
    # getting it from the class name during class creation (in the metaclass)
    # does not work because the class name is not set yet.
    class FuncClass(baseclass):
        np_func = getattr(np.random, name)
        funcname = name
        argspec = argspec(args, **baseclass.kwonlyargs)
        if dtypefunc is not None:
            dtype = dtypefunc
    FuncClass.__name__ = name.capitalize()
    return FuncClass


def make_np_classes(baseclass, s, dtypefunc):
    for line in s.splitlines():
        if line:
            c = make_np_class(baseclass, line, dtypefunc)
            yield c.__name__.lower(), c

