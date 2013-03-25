import numpy as np

import config
from context import context_length
from expr import (Expr, EvaluableExpression, expr_eval,
                  traverse_expr, collect_variables, dtype,
                  as_simple_expr, as_string,
                  get_missing_value, ispresent)


class CompoundExpression(Expr):
    '''expression written in terms of other expressions'''

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

    def collect_variables(self, context):
        return collect_variables(self.complete_expr, context)

    @property
    def complete_expr(self):
        if self._complete_expr is None:
            self._complete_expr = self.build_expr()
        return self._complete_expr


#TODO: generalise to a function with several arguments
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

    def collect_variables(self, context):
        return collect_variables(self.expr, context)


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
        ctx_filter = context.get('__filter__')
        if self.filter is not None and ctx_filter is not None:
            filter_expr = ctx_filter & self.filter
        elif self.filter is not None:
            filter_expr = self.filter
        elif ctx_filter is not None:
            filter_expr = ctx_filter
        else:
            filter_expr = None
        if filter_expr is not None and dtype(filter_expr, context) is not bool:
            raise Exception("filter must be a boolean expression")
        return filter_expr

    def __str__(self):
        filter_str = ', %s' % self.filter if self.filter is not None else ''
        return '%s(%s%s)' % (self.func_name, self.expr, filter_str)

    def collect_variables(self, context):
        expr_vars = collect_variables(self.expr, context)
        expr_vars |= collect_variables(self.filter, context)
        return expr_vars


class NumpyFunction(EvaluableExpression):
    func_name = None  # optional (for display)
    np_func = (None,)
    # arg_names can be set automatically by using inspect.getargspec,
    # but it only works for pure Python functions, so I decided to avoid it
    # because when you add a function, it is hard to know whether it is
    # implemented in C or not.
    arg_names = None
    allow_filter = True

    def __init__(self, *args, **kwargs):
        EvaluableExpression.__init__(self)
        if len(args) > len(self.arg_names):
            # + 1 to be consistent with Python (to account for self)
            raise TypeError("takes at most %d arguments (%d given)" %
                            (len(self.arg_names) + 1, len(args) + 1))
        if self.allow_filter:
            self.filter_expr = kwargs.pop("filter", None)
        else:
            self.filter_expr = None
        extra_kwargs = set(kwargs.keys()) - set(self.arg_names)
        if extra_kwargs:
            extra_kwargs = [repr(arg) for arg in extra_kwargs]
            raise TypeError("got an unexpected keyword argument %s" %
                            extra_kwargs[0])
        self.args = args
        self.kwargs = kwargs

    def evaluate(self, context):
        args = [expr_eval(arg, context) for arg in self.args]
        kwargs = dict((k, expr_eval(v, context))
                      for k, v in self.kwargs.iteritems())
        if 'size' in self.arg_names and 'size' not in kwargs:
            kwargs['size'] = context_length(context)
        #XXX: use _getfilter?
        if self.filter_expr is None:
            filter_value = None
        else:
            filter_value = expr_eval(self.filter_expr, context)
        func = self.np_func[0]
        return self.compute(func, args, kwargs, filter_value)

    def compute(self, func, args, kwargs, filter_value=None):
        raise NotImplementedError()

    def __str__(self):
        func_name = self.func_name
        if func_name is None:
            func_name = self.np_func[0].__name__
        kwargs = self.kwargs
        values = zip(self.arg_names, self.args)
        for name in self.arg_names[len(self.args):]:
            if name in kwargs:
                values.append((name, kwargs[name]))
        str_args = ', '.join('%s=%s' % (name, value) for name, value in values)
        return '%s(%s)' % (func_name, str_args)

    def traverse(self, context):
        for arg in self.args:
            for node in traverse_expr(arg, context):
                yield node
        for kwarg in self.kwargs.itervalues():
            for node in traverse_expr(kwarg, context):
                yield node
        yield self

    def collect_variables(self, context):
        args_vars = [collect_variables(arg, context) for arg in self.args]
        args_vars.extend(collect_variables(v, context)
                         for v in self.kwargs.itervalues())
        return set.union(*args_vars) if args_vars else set()


class NumpyChangeArray(NumpyFunction):
    def compute(self, func, args, kwargs, filter_value=None):
        # the first argument should be the array to work on ('a')
        assert self.arg_names[0] == 'a'
        old_values = args[0]
        new_values = func(*args, **kwargs)

        # we cannot do this yet because dtype() currently requires context
        # (and I don't want to change the signature of compute just for that)
#        assert dtype(old_values) == dtype(new_values)
        if filter_value is None:
            return new_values
        else:
            return np.where(filter_value, new_values, old_values)


class NumpyCreateArray(NumpyFunction):
    def compute(self, func, args, kwargs, filter_value=None):
        values = func(*args, **kwargs)
        if filter_value is None:
            return values
        else:
            missing_value = get_missing_value(values)
            return np.where(filter_value, values, missing_value)


class NumpyRandom(NumpyCreateArray):
    def compute(self, *args, **kwargs):
        if config.debug:
            print
            print "random sequence position before:", np.random.get_state()[2]
        res = super(NumpyRandom, self).compute(*args, **kwargs)
        if config.debug:
            print "random sequence position after:", np.random.get_state()[2]
        return res


class NumpyAggregate(NumpyFunction):
    nan_func = (None,)

    def __init__(self, *args, **kwargs):
        self.skip_na = kwargs.pop("skip_na", True)
        NumpyFunction.__init__(self, *args, **kwargs)

    def compute(self, func, args, kwargs, filter_value=None):
        # the first argument should be the array to work on ('a')
        assert self.arg_names[0] == 'a'

        values, args = args[0], args[1:]
        values = np.asanyarray(values)

        usenanfunc = False
        if (self.skip_na and issubclass(values.dtype.type, np.inexact) and
            self.nan_func[0] is not None):
            usenanfunc = True
            func = self.nan_func[0]
        if values.shape:
            if values.ndim == 1:
                if self.skip_na and not usenanfunc:
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
        return func(values, *args, **kwargs)


class NumexprFunction(Expr):
    '''For functions which are present as-is in numexpr'''
    func_name = None

    def __init__(self, expr):
        self.expr = expr

    def collect_variables(self, context):
        return collect_variables(self.expr, context)

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
