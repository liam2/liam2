from __future__ import print_function

from itertools import izip, chain

import numpy as np

from expr import (Variable, UnaryOp, BinaryOp, ComparisonOp, DivisionOp,
                  LogicalOp, getdtype, coerce_types, expr_eval, as_simple_expr,
                  as_string, collect_variables,
                  get_missing_record, get_missing_vector, FunctionExpr,
                  always, firstarg_dtype)
from exprbases import (FilteredExpression, CompoundExpression, NumexprFunction,
                       TableExpression, NumpyRandom, NumpyChangeArray)
from context import context_length
from utils import PrettyTable, argspec


#TODO: implement functions in expr to generate "Expr" nodes at the python level
# less painful
class Min(CompoundExpression):
    def __init__(self, *args):
        assert len(args) >= 2
        CompoundExpression.__init__(self, *args)

    def build_expr(self, *args):
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

    def __str__(self):
        return 'min(%s)' % ', '.join(str(arg) for arg in self.args)


class Max(CompoundExpression):
    def __init__(self, *args):
        assert len(args) >= 2
        CompoundExpression.__init__(self, *args)

    def build_expr(self, *args):
        expr1, expr2 = args[:2]
        # if(x > y, x, y)
        expr = Where(ComparisonOp('>', expr1, expr2), expr1, expr2)
        for arg in args[2:]:
            # if(e > z, e, z)
            expr = Where(ComparisonOp('>', expr, arg), expr, arg)
        return expr

    def __str__(self):
        return 'max(%s)' % ', '.join(str(arg) for arg in self.args)


class Logit(CompoundExpression):
    def build_expr(self, expr):
        # log(x / (1 - x))
        return Log(DivisionOp('/', expr, BinaryOp('-', 1.0, expr)))

    def __str__(self):
        return 'logit(%s)' % self.args[0]


class Logistic(CompoundExpression):
    def build_expr(self, expr):
        # 1 / (1 + exp(-x))
        return DivisionOp('/', 1.0,
                          BinaryOp('+', 1.0, Exp(UnaryOp('-', expr))))

    def __str__(self):
        return 'logistic(%s)' % self.args[0]


class ZeroClip(CompoundExpression):
    def build_expr(self, expr, expr_min, expr_max):
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


#------------------------------------


class Uniform(NumpyRandom):
    np_func = np.random.uniform
    # The docstring was wrong in np1.7: the default size is None instead of 1.
    # Issue reported as: https://github.com/numpy/numpy/pull/4611
    argspec = argspec(('low', 0.0), ('high', 1.0), ('size', None),
                      **NumpyRandom.kwonlyargs)


class Normal(NumpyRandom):
    np_func = np.random.normal
    argspec = argspec(('loc', 0.0), ('scale', 1.0), ('size', None),
                      **NumpyRandom.kwonlyargs)


class Gumbel(NumpyRandom):
    np_func = np.random.gumbel
    argspec = argspec(('loc', 0.0), ('scale', 1.0), ('size', None),
                      **NumpyRandom.kwonlyargs)


class RandInt(NumpyRandom):
    np_func = np.random.randint
    argspec = argspec('low', ('high', None), ('size', None),
                      **NumpyRandom.kwonlyargs)
    dtype = always(int)


# not inheriting from NumpyRandom as it would get the argspec from an
# nonexistent np_func
class Choice(FunctionExpr):
    funcname = 'choice'

    def compute(self, context, choices, p=None, size=None, replace=True):
        #TODO: __init__ should detect when all args are constants and run
        # a "check_arg_values" method if present
        #TODO: document the change in behavior for the case where the sum of
        # probabilities is != 1
        # random.choice only checks that the error is < 1e-8 but always
        # divides probabilities by sum(p). It is probably a better choice
        # because it distributes the error to all bins instead of only
        # adjusting the probability of the last choice.
        if size is None:
            size = len(context)
        return np.random.choice(choices, size=size, replace=replace, p=p)

    dtype = firstarg_dtype

#------------------------------------


class Round(NumpyChangeArray):
    funcname = 'round'  # np.round redirects to np.round_
    np_func = np.round
    dtype = firstarg_dtype


class Trunc(FunctionExpr):
    funcname = 'trunc'

    def compute(self, context, expr):
        return expr.astype(int)

    #TODO: do the check in __init__ and use dtype = always(int)
    def dtype(self, context):
        assert getdtype(self.expr, context) == float
        return int


#------------------------------------


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
    print("%d new %s(s) (%d -> %d)" % (num_birth, target_entity.name, num_rows,
                                       num_rows + num_birth), end=' ')

    target_entity.array.append(children)

    temp_variables = target_entity.temp_variables
    for name, temp_value in temp_variables.iteritems():
        #FIXME: OUCH, this is getting ugly, I'll need a better way to
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
        if (isinstance(temp_value, np.ndarray) and temp_value.shape == (
        num_rows,)):
            extra = get_missing_vector(num_birth, temp_value.dtype)
            temp_variables[name] = np.concatenate((temp_value, extra))

    extra_variables = target_context.entity_data.extra
    for name, temp_value in extra_variables.iteritems():
        if name == '__globals__':
            continue
        if isinstance(temp_value, np.ndarray) and temp_value.shape:
            extra = get_missing_vector(num_birth, temp_value.dtype)
            extra_variables[name] = np.concatenate((temp_value, extra))

    id_to_rownum_tail = np.arange(num_rows, num_rows + num_birth)
    target_entity.id_to_rownum = np.concatenate(
        (id_to_rownum, id_to_rownum_tail))


class New(FilteredExpression):
    no_eval = ('filter', 'kwargs')

    def _initial_values(self, array, to_give_birth, num_birth):
        #TODO: use default values for fields which have one
        children = np.empty(num_birth, dtype=array.dtype)
        children[:] = get_missing_record(array)
        return children

    @classmethod
    def _collect_kwargs_variables(cls, kwargs, context):
        used_variables = set()
        # kwargs are stored as a list of (k, v) pairs
        for k, v in kwargs.iteritems():
            used_variables.update(collect_variables(v, context))
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

        id_to_rownum = target_entity.id_to_rownum
        num_individuals = len(id_to_rownum)

        children = self._initial_values(array, to_give_birth, num_birth)
        if num_birth:
            children['id'] = np.arange(num_individuals,
                                       num_individuals + num_birth)
            children['period'] = context.period

            used_variables = self._collect_kwargs_variables(kwargs, context)
            if to_give_birth is None:
                assert not used_variables
                child_context = context.empty(num_birth)
            else:
                child_context = context.subset(to_give_birth, used_variables)
            for k, v in kwargs.iteritems():
                children[k] = expr_eval(v, child_context)

        add_individuals(target_context, children)

        # result is the ids of the new individuals corresponding to the source
        # entity
        if to_give_birth is not None:
            result = np.empty(context_length(context), dtype=int)
            result.fill(-1)
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

    def _initial_values(self, array, to_give_birth, num_birth):
        return array[to_give_birth]


class Dump(TableExpression):
    no_eval = ('args',)
    kwonlyargs = {'filter': None, 'missing': None, 'header': True}

    def compute(self, context, *args, **kwargs):
        filter_value = kwargs.pop('filter', None)
        missing = kwargs.pop('missing', None)
        # periods = kwargs.pop('periods', None)
        header = kwargs.pop('header', True)

        if args:
            expressions = list(args)
        else:
            # extra=False because we don't want globals nor "system" variables
            # (nan, period, __xxx__)
            #FIXME: we should also somehow "traverse" expressions in this case
            # too (args is ()) => all keys in the current context
            expressions = [Variable(name) for name in context.keys(extra=False)]

        str_expressions = [str(e) for e in expressions]
        if 'id' not in str_expressions:
            str_expressions.insert(0, 'id')
            expressions.insert(0, Variable('id'))
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
            expr_value = expr_eval(expr, context)
            if (filter_value is not None and isinstance(expr_value,
                                                        np.ndarray) and
                    expr_value.shape):
                expr_value = expr_value[filter_value]
            columns.append(expr_value)

        ids = columns[id_pos]
        if isinstance(ids, np.ndarray) and ids.shape:
            numrows = len(ids)
        else:
            numrows = 1

        # expand scalar columns to full columns in memory
        for idx, col in enumerate(columns):
            dtype = None
            if not isinstance(col, np.ndarray):
                dtype = type(col)
            elif not col.shape:
                dtype = col.dtype.type
            if dtype is not None:
                newcol = np.empty(numrows, dtype=dtype)
                newcol.fill(col)
                columns[idx] = newcol

        data = izip(*columns)
        table = chain([str_expressions], data) if header else data
        return PrettyTable(table, missing)

    dtype = always(None)


class Where(NumexprFunction):
    funcname = 'if'
    argspec = argspec('cond', 'iftrue', 'iffalse')

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
        return 'where(%s)' % self.format_args_str(*as_string(self.children))

    def dtype(self, context):
        assert getdtype(self.cond, context) == bool
        return coerce_types(context, self.iftrue, self.iffalse)


functions = {
    # random
    'uniform': Uniform,
    'normal': Normal,
    'gumbel': Gumbel,
    'choice': Choice,
    'randint': RandInt,
    # element-wise functions
    # Min and Max are in aggregates.py.functions (because of the dispatcher)
    'abs': Abs,
    'clip': Clip,
    'zeroclip': ZeroClip,
    'round': Round,
    'trunc': Trunc,
    'exp': Exp,
    'log': Log,
    'logit': Logit,
    'logistic': Logistic,
    'where': Where,
    # misc
    'sort': Sort,
    'new': New,
    'clone': Clone,
    'dump': Dump
}
