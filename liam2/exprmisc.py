# encoding: utf-8
from __future__ import print_function

from itertools import izip, chain

import numpy as np


import config
from expr import (EvaluableExpression, Expr, Variable, UnaryOp, BinaryOp, ComparisonOp, DivisionOp,
                  LogicalOp, getdtype, coerce_types, expr_eval, as_simple_expr,
                  as_string, collect_variables, traverse_expr,
                  get_missing_record, get_missing_vector, FunctionExpr,
                  always, firstarg_dtype, expr_cache)
from exprbases import (CompoundExpression,
                       NumexprFunction, FilteredExpression,
                       FunctionExpr, TableExpression,
                       NumpyRandom, NumpyChangeArray)
from context import (EntityContext, context_length, context_subset,
                     new_context_like)
from utils import PrettyTable, argspec

from til.pgm.run_pension import get_pension

# TODO: implement functions in expr to generate "Expr" nodes at the python level
# less painful


class Min(CompoundExpression):
    def build_expr(self, *args):
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
    def build_expr(self, *args):
        assert len(args) >= 2

        expr1, expr2 = args[:2]
        # if(x > y, x, y)
        expr = Where(ComparisonOp('>', expr1, expr2), expr1, expr2)
        for arg in args[2:]:
            # if(e > z, e, z)
            expr = Where(ComparisonOp('>', expr, arg), expr, arg)
        return expr


class Logit(CompoundExpression):
    def build_expr(self, expr):
        # log(x / (1 - x))
        return Log(DivisionOp('/', expr, BinaryOp('-', 1.0, expr)))


class Logistic(CompoundExpression):
    def build_expr(self, expr):
        # 1 / (1 + exp(-x))
        return DivisionOp('/', 1.0,
                          BinaryOp('+', 1.0, Exp(UnaryOp('-', expr))))


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

# ------------------------------------

class TimeScale(FunctionExpr):
    func_name = 'period'

    def compute(self, context, expr):
        return expr_eval(expr, context) + context.periodicity

    dtype = always(int)


class Year(FunctionExpr):
    func_name = 'year'

    def compute(self, context, expr):
        return int(expr_eval(expr, context)/100)

    dtype = always(int)


class Month(FunctionExpr):
    func_name = 'month'

    def compute(self, context, expr):
        return (expr_eval(expr, context) % 100)

    dtype = always(int)


class AddTime(FunctionExpr):
    func_name = 'add_time'

    def compute(self, context, expr):
        periodicity = context.periodicity
        init_value = expr_eval(expr, context)
        #TODO: be more general with periodicity > 12
        if periodicity > 0:
            change_year = (init_value % 100) + periodicity >= 12
            value = init_value + periodicity*(1 - change_year) + (100 - 12 + periodicity)*(change_year)
        if periodicity < 0:
            change_year = (init_value % 100) + periodicity < 1
            value = init_value + periodicity*(1 - change_year) + (-100 + 12 + periodicity)*(change_year)
        return value

    dtype = always(int)

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
    if config.log_level == "processes":
        print("%d new %s(s) (%d -> %d)" % (num_birth, target_entity.name,
                                           num_rows, num_rows + num_birth),
              end=' ')

    target_entity.array.append(children)

    temp_variables = target_entity.temp_variables
    for name, temp_value in temp_variables.iteritems():
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
        # TODO: use default values for fields which have one
        children = np.empty(num_birth, dtype=array.dtype)
        children[:] = get_missing_record(array)
        return children

    @classmethod
    def _collect_kwargs_variables(cls, kwargs):
        used_variables = set()
        # kwargs are stored as a list of (k, v) pairs
        for k, v in kwargs.iteritems():
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

        id_to_rownum = target_entity.id_to_rownum
        num_individuals = len(id_to_rownum)

        children = self._initial_values(array, to_give_birth, num_birth)
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
            for k, v in kwargs.iteritems():
                if k not in array.dtype.names:
                    print("WARNING: {} is unknown, ignoring it!".format(k))
                    continue
                children[k] = expr_eval(v, child_context)

        add_individuals(target_context, children)

        expr_cache.invalidate(context.period, context.entity_name)

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
                newcol = np.empty(numrows, dtype=dtype)
                newcol.fill(col)
                columns[idx] = newcol

        data = izip(*columns)
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


class Pension(FilteredExpression):

    no_eval = ('filter', 'varname', 'regime')
    already_simulated = None

    @classmethod
    def no_need_to_reload(cls, context, yearleg):
        if Pension.already_simulated is None:
            return False

        try:
            # Note that period is in context
            return (
                (Pension.already_simulated['yearleg'] == yearleg) &  # legislation changes
                (set(Pension.already_simulated['context']['id']) == set(context['id'])) &
                (Pension.already_simulated['context']['period'] == context['period'])  # period changes
                )
        except:
            import pdb
            pdb.set_trace()


    def compute(self, context, varname, regime, expr=None, filter=None, yearleg=None):

        selected = expr_eval(filter, context)
        context = context.subset(selected)
        # determine yearleg
        if yearleg is None:
            yearleg = context['period'] // 100
            # if yearleg > 2009:  # TODO: remove when yearleg > 2009 possible
            #     yearleg = 2009

        if Pension.no_need_to_reload(context, yearleg):
            simul = Pension.already_simulated['simul']
        else:
            print('yearleg: {}'.format(yearleg))
            # try:
            simul = get_pension(context, yearleg)
            # except:
            #     import pdb
            #     pdb.set_trace()

        result = simul.calculate(varname, regime)
        Pension.already_simulated = {'context': context,
                                     'yearleg': yearleg,
                                     'simul': simul,
                                     }

        output = -1 * np.ones(len(selected))
        # TODO: understant why result is not float
        output[selected] = result.astype(float)
        return output


functions = {
    # element-wise functions
    # Min and Max are in aggregates.py.functions (because of the dispatcher)
    'add_time_scale': TimeScale,
    'add_time': AddTime,
    'year': Year,
    'month': Month,
    # per element
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
    'dump': Dump,
    'pension': Pension
}
