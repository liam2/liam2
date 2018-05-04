# encoding: utf-8
from __future__ import absolute_import, division, print_function

from itertools import chain
import os
import random

import numpy as np
try:
    import scipy
    import scipy.special as special
except ImportError:
    scipy = None
import larray as la

from liam2 import config
from liam2.compat import zip, basestring, long
from liam2.expr import (Variable, UnaryOp, BinaryOp, ComparisonOp, DivisionOp, LogicalOp, getdtype, coerce_types,
                        expr_eval, as_simple_expr, as_string, collect_variables, get_default_array, get_default_vector,
                        FunctionExpr, always, firstarg_dtype, expr_cache, index_array_by_variables)
from liam2.exprbases import FilteredExpression, CompoundExpression, NumexprFunction, TableExpression, NumpyChangeArray
from liam2.context import context_length
from liam2.importer import load_ndarray, load_table
from liam2.utils import PrettyTable, argspec


# TODO: implement functions in expr to generate "Expr" nodes at the python level
# less painful
class Min(CompoundExpression):
    def build_expr(self, context, *args):
        assert len(args) >= 2

        expr1, expr2 = args[:2]
        expr = Where(ComparisonOp('<', expr1, expr2), expr1, expr2)
        for arg in args[2:]:
            expr = Where(ComparisonOp('<', expr, arg), expr, arg)

            # args = [Symbol('x%d' % i) for i in range(len(self.args))]
            # ctx = {'__entity__': 'x',
            #        'x': {'x%d' % i: a for i, a in enumerate(self.args)}}
            # where = Symbol('where')
            # expr = where(a < b, a, b)
            # for arg in self.args[2:]:
            #     expr = where(expr < arg, expr, arg)
            # expr = expr.to_ast(ctx)

            # expr1, expr2 = self.args[:2]
            # expr = parse('if(a < b, a, b)',
            #              {'__entity__': 'x', 'x': {'a': expr1, 'b': expr2}})
            # for arg in self.args[2:]:
            #     expr = parse('if(a < b, a, b)',
            #                  {'__entity__': 'x', 'x': {'a': expr, 'b': arg}})

            # expr = Where(expr1 < expr2, expr1, expr2)
            # for arg in self.args[2:]:
            #     expr = Where(expr < arg, expr, arg)

        #        Where(Where(expr1 < expr2, expr1, expr2) < expr3,
        #              Where(expr1 < expr2, expr1, expr2),
        #              expr3)
        #        3 where, 3 comparisons = 6 op (or 4 if optimised)
        #
        #        Where(Where(Where(expr1 < expr2, expr1, expr2) < expr3,
        #                    Where(expr1 < expr2, expr1, expr2),
        #                    expr3) < expr4,
        #              Where(Where(expr1 < expr2, expr1, expr2) < expr3,
        #                    Where(expr1 < expr2, expr1, expr2),
        #                    expr3),
        #              expr4)
        #        7 where, 7 comp = 14 op (or 6 if optimised)

        # this version scales better in theory (but in practice, it will depend
        # if numexpr factorize the common subexpression in the above version
        # or not)
        #        Where(expr1 < expr2 & expr1 < expr3,
        #              expr1,
        #              Where(expr2 < expr3, expr2, expr3))
        #        2 where, 3 comparisons, 1 and = 6 op
        #
        #        Where(expr1 < expr2 & expr1 < expr3 & expr1 < expr4,
        #              expr1,
        #              Where(expr2 < expr3 & expr2 < expr4,
        #                    expr2
        #                    Where(expr3 < expr4,
        #                          expr3,
        #                          expr4)))
        #        3 where, 6 comp, 3 and = 12 op
        return expr


class Max(CompoundExpression):
    def build_expr(self, context, *args):
        assert len(args) >= 2

        expr1, expr2 = args[:2]
        # if(x > y, x, y)
        expr = Where(ComparisonOp('>', expr1, expr2), expr1, expr2)
        for arg in args[2:]:
            # if(e > z, e, z)
            expr = Where(ComparisonOp('>', expr, arg), expr, arg)
        return expr


class Logit(CompoundExpression):
    def build_expr(self, context, expr):
        # log(x / (1 - x))
        return Log(DivisionOp('/', expr, BinaryOp('-', 1.0, expr)))


class Logistic(CompoundExpression):
    def build_expr(self, context, expr):
        # 1 / (1 + exp(-x))
        return DivisionOp('/', 1.0,
                          BinaryOp('+', 1.0, Exp(UnaryOp('-', expr))))


class ZeroClip(CompoundExpression):
    def build_expr(self, context, expr, expr_min, expr_max):
        # if(minv <= x <= maxv, x, 0)
        return Where(LogicalOp('&', ComparisonOp('>=', expr, expr_min),
                               ComparisonOp('<=', expr, expr_max)), expr,
                     0)

    # We do not have to coerce with self.expr_min & expr_max because they
    # are only used in the comparisons, not in the result.
    dtype = firstarg_dtype


# >>> mi = 1
# >>> ma = 10
# >>> a = np.arange(1e7)
#
# >>> timeit np.clip(a, mi, ma)
# 10 loops, best of 3: 127 ms per loop
# >>> timeit np.clip(a, mi, ma, a)
# 10 loops, best of 3: 26.2 ms per loop
# >>> timeit ne.evaluate('where(a < mi, mi, where(a > ma, ma, a))')
# 10 loops, best of 3: 94.1 ms per loop
class Clip(NumpyChangeArray):
    np_func = np.clip


class Sort(NumpyChangeArray):
    np_func = np.sort


# ------------------------------------


class Round(NumpyChangeArray):
    np_func = np.round
    dtype = firstarg_dtype


class Trunc(FunctionExpr):
    # TODO: check that the dtype is correct at compilation time (__init__ is too
    # early since we do not have the context yet)
    # assert getdtype(self.args[0], context) == float
    def compute(self, context, expr):
        if isinstance(expr, np.ndarray):
            return expr.astype(int)
        else:
            return int(expr)

    dtype = always(int)


class Erf(FunctionExpr):
    def compute(self, context, expr):
        if scipy is None:
            raise ImportError(
                "Failed to import scipy, which is required for erf(). Please make sure scipy is installed and working.",
                )
        if isinstance(expr, np.ndarray):
            return special.erf(expr)
        else:
            return scipy.math.erf(expr)

    dtype = always(float)

# ------------------------------------


class Abs(NumexprFunction):
    argspec = argspec('expr')
    dtype = always(float)


class Log(NumexprFunction):
    argspec = argspec('expr')
    dtype = always(float)


class Exp(NumexprFunction):
    argspec = argspec('expr')
    dtype = always(float)


def add_individuals(target_context, children):
    target_entity = target_context.entity
    id_to_rownum = target_entity.id_to_rownum
    array = target_entity.array
    num_rows = len(array)
    num_birth = len(children)
    if config.log_level == "processes":
        print("%d new %s(s) (%d -> %d)" % (num_birth, target_entity.name,
                                           num_rows, num_rows + num_birth),
              end=' ')

    target_entity.array.append(children)

    temp_variables = target_entity.temp_variables
    for name, temp_value in temp_variables.items():
        # FIXME: OUCH, this is getting ugly, I'll need a better way to
        # differentiate nd-arrays from "entity" variables
        # I guess having the context contain all entities and a separate
        # globals namespace should fix this problem. Well, no it would not
        # fix the problem by itself, as this would only move the problem
        # to the "store" part of Assignment processes which would need to be
        # able to differentiate between an "entity temp" and a global temp.
        # I think this can be done by inspecting the expressions that generate
        # them: no non-aggregated entity var => global temp. It would be nice
        # to further distinguish between aggregated entity var and other global
        # temporaries to store them in the entity somewhere, but I am unsure
        # whether it is possible.
        if (isinstance(temp_value, np.ndarray) and
                temp_value.shape == (num_rows,)):
            extra = get_default_vector(num_birth, temp_value.dtype)
            temp_variables[name] = np.concatenate((temp_value, extra))

    extra_variables = target_context.entity_data.extra
    for name, temp_value in extra_variables.items():
        if name == '__globals__':
            continue
        if isinstance(temp_value, np.ndarray) and temp_value.shape:
            extra = get_default_vector(num_birth, temp_value.dtype)
            extra_variables[name] = np.concatenate((temp_value, extra))

    id_to_rownum_tail = np.arange(num_rows, num_rows + num_birth)
    target_entity.id_to_rownum = np.concatenate(
        (id_to_rownum, id_to_rownum_tail))


class New(FilteredExpression):
    no_eval = ('filter', 'kwargs')

    def _initial_values(self, array, to_give_birth, num_birth, default_values):
        return get_default_array(num_birth, array.dtype, default_values)

    @classmethod
    def _collect_kwargs_variables(cls, kwargs):
        used_variables = set()
        # kwargs are stored as a list of (k, v) pairs
        for k, v in kwargs.items():
            used_variables.update(collect_variables(v))
        return used_variables

    def compute(self, context, entity_name=None, filter=None, number=None,
                **kwargs):
        if filter is not None and number is not None:
            # Having neither is allowed, though, as there can be a contextual
            # filter. Also, there is no reason to prevent the whole
            # population giving birth, even though the usefulness of such
            # usage seem dubious.
            raise ValueError("new() 'filter' and 'number' arguments are "
                             "mutually exclusive")
        source_entity = context.entity
        if entity_name is None:
            target_entity = source_entity
        else:
            target_entity = context.entities[entity_name]

        # target context is the context where the new individuals will be
        # created
        if target_entity is source_entity:
            target_context = context
        else:
            # we do need to copy the data (.extra) because we will insert into
            # the entity.array anyway => fresh_data=True
            target_context = context.clone(fresh_data=True,
                                           entity_name=target_entity.name)

        filter_expr = self._getfilter(context, filter)
        if filter_expr is not None:
            to_give_birth = expr_eval(filter_expr, context)
            num_birth = to_give_birth.sum()
        elif number is not None:
            to_give_birth = None
            num_birth = number
        else:
            to_give_birth = np.ones(len(context), dtype=bool)
            num_birth = len(context)

        array = target_entity.array
        default_values = target_entity.fields.default_values

        id_to_rownum = target_entity.id_to_rownum
        num_individuals = len(id_to_rownum)

        children = self._initial_values(array, to_give_birth, num_birth,
                                        default_values)
        if num_birth:
            children['id'] = np.arange(num_individuals,
                                       num_individuals + num_birth)
            children['period'] = context.period

            used_variables = [v.name for v in
                              self._collect_kwargs_variables(kwargs)]
            if to_give_birth is None:
                assert not used_variables
                child_context = context.empty(num_birth)
            else:
                child_context = context.subset(to_give_birth, used_variables,
                                               filter_expr)
            for k, v in kwargs.items():
                if k not in array.dtype.names:
                    print("WARNING: {} is unknown, ignoring it!".format(k))
                    continue
                children[k] = expr_eval(v, child_context)

        add_individuals(target_context, children)

        expr_cache.invalidate(context.period, context.entity_name)

        # result is the ids of the new individuals corresponding to the source
        # entity
        if to_give_birth is not None:
            result = np.full(context_length(context), -1, dtype=int)
            if source_entity is target_entity:
                extra_bools = np.zeros(num_birth, dtype=bool)
                to_give_birth = np.concatenate((to_give_birth, extra_bools))
            # Note that np.place is a bit faster, but is currently buggy when
            # working with columns of structured arrays.
            # See https://github.com/numpy/numpy/issues/2462
            result[to_give_birth] = children['id']
            return result
        else:
            return None

    dtype = always(int)


class Clone(New):
    def __init__(self, filter=None, **kwargs):
        New.__init__(self, None, filter, None, **kwargs)

    def _initial_values(self, array, to_give_birth, num_birth, default_values):
        return array[to_give_birth]


class Dump(TableExpression):
    no_eval = ('args',)
    kwonlyargs = {'filter': None, 'missing': None, 'header': True,
                  'limit': None}

    def compute(self, context, *args, **kwargs):
        filter_value = kwargs.pop('filter', None)
        missing = kwargs.pop('missing', None)
        # periods = kwargs.pop('periods', None)
        header = kwargs.pop('header', True)
        limit = kwargs.pop('limit', None)
        entity = context.entity

        if args:
            expressions = list(args)
        else:
            # extra=False because we don't want globals nor "system" variables
            # (nan, period, __xxx__)
            # FIXME: we should also somehow "traverse" expressions in this case
            # too (args is ()) => all keys in the current context
            expressions = [Variable(entity, name)
                           for name in context.keys(extra=False)]

        str_expressions = [str(e) for e in expressions]
        if 'id' not in str_expressions:
            str_expressions.insert(0, 'id')
            expressions.insert(0, Variable(entity, 'id'))
            id_pos = 0
        else:
            id_pos = str_expressions.index('id')

        #        if (self.periods is not None and len(self.periods) and
        #            'period' not in str_expressions):
        #            str_expressions.insert(0, 'period')
        #            expressions.insert(0, Variable('period'))
        #            id_pos += 1

        columns = []
        for expr in expressions:
            if filter_value is False:
                # dtype does not matter much
                expr_value = np.empty(0)
            else:
                # TODO: set filter before evaluating expressions
                expr_value = expr_eval(expr, context)
                if (filter_value is not None and
                        isinstance(expr_value, np.ndarray) and
                        expr_value.shape):
                    expr_value = expr_value[filter_value]
            columns.append(expr_value)

        ids = columns[id_pos]
        if isinstance(ids, np.ndarray) and ids.shape:
            numrows = len(ids)
        else:
            # FIXME: we need a test for this case (no idea how this can happen)
            numrows = 1

        # expand scalar columns to full columns in memory
        # TODO: handle or explicitly reject columns wh ndim > 1
        for idx, col in enumerate(columns):
            dtype = None
            if not isinstance(col, np.ndarray):
                dtype = type(col)
            elif not col.shape:
                dtype = col.dtype.type

            if dtype is not None:
                # TODO: try using itertools.repeat instead as it seems to be a
                # bit faster and would consume less memory (however, it might
                # not play very well with Pandas.to_csv)
                newcol = np.full(numrows, col, dtype=dtype)
                columns[idx] = newcol
            elif col.ndim > 1:
                # move last axis (should be id axis) first
                # np.moveaxis requires numpy >= 1.11
                # columns[idx] = np.moveaxis(col, -1, 0)
                columns[idx] = col.transpose((-1,) + tuple(range(col.ndim - 1)))

        assert all(isinstance(col, np.ndarray) for col in columns)
        bad_lengths = {str_expr: col.shape for col, str_expr in zip(columns, str_expressions)
                       if col.shape[0] != numrows}
        if bad_lengths:
            raise ValueError("first dimension of some columns are not the same length as the id column (%d): %s"
                             % (numrows, str(bad_lengths)))

        if limit is not None:
            assert isinstance(limit, (int, long))
            columns = [col[:limit] for col in columns]

        data = zip(*columns)
        table = chain([str_expressions], data) if header else data
        return PrettyTable(table, missing)

    dtype = always(None)


class Where(NumexprFunction):
    funcname = 'if'
    argspec = argspec('cond, iftrue, iffalse')

    @property
    def cond(self):
        return self.args[0]

    @property
    def iftrue(self):
        return self.args[1]

    @property
    def iffalse(self):
        return self.args[2]

    def as_simple_expr(self, context):
        cond = as_simple_expr(self.cond, context)

        # filter is stored as an unevaluated expression
        context_filter = context.filter_expr
        local_ctx = context.clone()
        if context_filter is None:
            local_ctx.filter_expr = self.cond
        else:
            # filter = filter and cond
            local_ctx.filter_expr = LogicalOp('&', context_filter, self.cond)
        iftrue = as_simple_expr(self.iftrue, local_ctx)

        if context_filter is None:
            local_ctx.filter_expr = UnaryOp('~', self.cond)
        else:
            # filter = filter and not cond
            local_ctx.filter_expr = LogicalOp('&', context_filter,
                                              UnaryOp('~', self.cond))
        iffalse = as_simple_expr(self.iffalse, local_ctx)
        return Where(cond, iftrue, iffalse)

    def as_string(self):
        args = as_string((self.cond, self.iftrue, self.iffalse))
        return 'where(%s)' % self.format_args_str(args, [])

    def dtype(self, context):
        assert getdtype(self.cond, context) == bool
        return coerce_types(context, self.iftrue, self.iffalse)


def _plus(a, b):
    return BinaryOp('+', a, b)


def _mul(a, b):
    return BinaryOp('*', a, b)


# TODO: add class to handle loading a single coefficient (array) from a file
# TODO: rename to something else.
# * LinearExpression (linear_expr?) but that would
#   make it awkward if we want to support expression with non linear variables
# * CoefficientsExpression (coef_expr)
class ExtExpr(FunctionExpr):
    def __init__(self, *args, **kwargs):
        # to initialize .args, .kwargs, .original_args, ...
        FunctionExpr.__init__(self, *args, **kwargs)

        coefficients = self.args[0]
        if isinstance(coefficients, basestring):
            fpath = os.path.join(config.input_directory, coefficients)
            coefficients = load_ndarray(fpath)
            # XXX: store args in a list so that we can modify it?
            # self.args[1] = load_ndarray(fpath, float)
            # XXX: but we should be able to do better than a list, eg.
            # self.args.need = load_ndarray(fpath, float)
            self.args = (coefficients,) + self.args[1:]

    def compute(self, context, coefficients):
        assert isinstance(coefficients, la.LArray)

        # XXX: change to "variable"? because we can use temporary variables too!
        #      or even to "expressions" if we want to support expressions.
        # FIXME013: in any case, it should be singular
        field_axis = coefficients.axes['fields']
        other_axes = coefficients.axes - field_axis

        expr = None
        # XXX: instead of retrieving labels along a dimension & splitting manually,
        #      we should have a "split" operation in LArray (opposite of stack)
        for name in field_axis.labels:
            coef_value = coefficients[name]

            # automatically index other (remaining) dimensions
            if other_axes:
                coef_value = index_array_by_variables(coef_value, context, other_axes)

            coef_var = self.add_tmp_var(context, coef_value)
            if name != 'constant':
                # XXX: should I reuse variables instances defined in the entity at
                # context.entity.variables[name]
                # XXX: parse expressions instead of only simple Variable?
                term = _mul(Variable(context.entity, name), coef_var)
            else:
                term = coef_var
            if expr is None:
                expr = term
            else:
                expr = _plus(expr, term)
        return expr_eval(expr, context)

    # def __repr__(self):
    #     return "yada"


class Seed(FunctionExpr):
    def compute(self, context, seed=None):
        if seed is not None:
            seed = long(seed)
            print("using fixed random seed: %d" % seed)
        else:
            print("resetting random seed")
        random.seed(seed)
        np.random.seed(seed)


class Array(FunctionExpr):
    def compute(self, context, expr):
        return np.array(expr)

    # XXX: is this correct?
    dtype = firstarg_dtype


class Load(FunctionExpr):
    def compute(self, context, fname, type=None, fields=None):
        # TODO: move those checks to __init__
        if type is None and fields is None:
            raise ValueError("type or fields must be specified")
        if type is not None and fields is not None:
            raise ValueError("cannot specify both type and fields")
        # using abspath does not change anything except it makes relative paths using ".." easier to read
        file_path = os.path.abspath(os.path.join(config.input_directory, fname))
        if type is not None:
            return load_ndarray(file_path, type)
        elif fields is not None:
            return load_table(file_path, fields)


class View(FunctionExpr):
    def compute(self, context, expr):
        # XXX: is expr_eval necessary here?
        la.view(expr_eval(expr, context))

    dtype = None


functions = {
    # element-wise functions
    # Min and Max are in aggregates.py.functions (because of the dispatcher)
    'abs': Abs,
    'clip': Clip,
    'zeroclip': ZeroClip,
    'round': Round,
    'trunc': Trunc,
    'exp': Exp,
    'erf': Erf,
    'log': Log,
    'logit': Logit,
    'logistic': Logistic,
    'where': Where,
    # misc
    'sort': Sort,
    'new': New,
    'clone': Clone,
    'dump': Dump,
    'extexpr': ExtExpr,
    'seed': Seed,
    'array': Array,
    'load': Load,
    'view': View,
}
