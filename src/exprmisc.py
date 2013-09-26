from __future__ import print_function

from collections import Sequence
from itertools import izip, chain

import numpy as np

import config
from expr import (Expr, Variable,
                  getdtype, coerce_types, expr_eval,
                  as_simple_expr, as_string,
                  collect_variables, traverse_expr,
                  get_missing_record, get_missing_vector)
from exprbases import (EvaluableExpression, CompoundExpression,
                       NumexprFunction,
                       FunctionExpression, TableExpression,
                       NumpyRandom, NumpyChangeArray)
from context import (EntityContext, context_length, context_subset,
                     new_context_like)
from registry import entity_registry
from utils import PrettyTable


class Min(CompoundExpression):
    def __init__(self, *args):
        CompoundExpression.__init__(self)
        assert len(args) >= 2
        self.args = args

    def build_expr(self):
        expr1, expr2 = self.args[:2]
        expr = Where(expr1 < expr2, expr1, expr2)
        for arg in self.args[2:]:
            expr = Where(expr < arg, expr, arg)

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

    def dtype(self, context):
        return coerce_types(context, *self.args)

    def __str__(self):
        return 'min(%s)' % ', '.join(str(arg) for arg in self.args)


class Max(CompoundExpression):
    def __init__(self, *args):
        CompoundExpression.__init__(self)
        assert len(args) >= 2
        self.args = args

    def build_expr(self):
        expr1, expr2 = self.args[:2]
        expr = Where(expr1 > expr2, expr1, expr2)
        for arg in self.args[2:]:
            expr = Where(expr > arg, expr, arg)
        return expr

    def dtype(self, context):
        return coerce_types(context, *self.args)

    def __str__(self):
        return 'max(%s)' % ', '.join(str(arg) for arg in self.args)


class Logit(CompoundExpression):
    def __init__(self, expr):
        CompoundExpression.__init__(self)
        self.expr = expr

    def build_expr(self):
        return Log(self.expr / (1.0 - self.expr))

    #noinspection PyUnusedLocal
    def dtype(self, context):
        return float

    def __str__(self):
        return 'logit(%s)' % self.expr


class Logistic(CompoundExpression):
    def __init__(self, expr):
        CompoundExpression.__init__(self)
        self.expr = expr

    def build_expr(self):
        return 1.0 / (1.0 + Exp(-self.expr))

    #noinspection PyUnusedLocal
    def dtype(self, context):
        return float

    def __str__(self):
        return 'logistic(%s)' % self.expr


class ZeroClip(CompoundExpression):
    def __init__(self, expr, expr_min, expr_max):
        CompoundExpression.__init__(self)
        self.expr = expr
        self.expr_min = expr_min
        self.expr_max = expr_max

    def build_expr(self):
        expr = self.expr
        return Where((expr >= self.expr_min) & (expr <= self.expr_max),
                     expr,
                     0)

    def dtype(self, context):
        return getdtype(self.expr1, context)


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
    np_func = (np.clip,)
    arg_names = ('a', 'a_min', 'a_max', 'out')


class Sort(NumpyChangeArray):
    np_func = (np.sort,)
    arg_names = ('a', 'axis', 'kind', 'order')

#------------------------------------


class Uniform(NumpyRandom):
    np_func = (np.random.uniform,)
    arg_names = ('low', 'high', 'size')


class Normal(NumpyRandom):
    np_func = (np.random.normal,)
    arg_names = ('loc', 'scale', 'size')


class RandInt(NumpyRandom):
    np_func = (np.random.randint,)
    arg_names = ('low', 'high', 'size')

    #noinspection PyUnusedLocal
    def dtype(self, context):
        return int


#XXX: use np.random.choice (new in np 1.7)
class Choice(EvaluableExpression):
    func_name = 'choice'

    def __init__(self, choices, weights=None):
        EvaluableExpression.__init__(self)
        if not isinstance(choices, Sequence):
            raise TypeError("choice() first argument should be a sequence "
                            "(tuple or list)")

        if any(isinstance(c, Expr) for c in choices):
            self.choices = choices
        else:
            self.choices = np.array(choices)

        if weights is not None:
            if not isinstance(weights, Sequence):
                raise TypeError("if provided, choice weights should be a "
                                "sequence (tuple or list)")
            if any(isinstance(w, Expr) for w in weights):
                self.bins = weights
            else:
                self.bins = self._weights_to_bins(weights)
        else:
            self.bins = None

    @staticmethod
    def _weights_to_bins(weights):
        bins = np.array([0.0] + list(np.cumsum(weights)))
        error = abs(bins[-1] - 1.0)
        if 0.0 < error <= 1e-6:
            # We correct the last bin in all cases, even though when the total
            # is > 1.0, it does not change anything (since the random numbers
            # will always be < 1, having a larger last bin -- with a total
            # > 1 -- will not increase its probability). In short, correcting
            # it just makes things explicit.
            bins[-1] = 1.0

            # only warn if the values are "visually different"
            last_weight = str(weights[-1])
            adjusted_last_weight = str(1.0 - bins[-2])
            if adjusted_last_weight != last_weight:
                print("Warning: last choice probability adjusted to %s " \
                      "instead of %s !" % (adjusted_last_weight, last_weight))
        elif error > 1e-6:
            raise Exception("the cumulative sum of choice weights must be ~1")
        return bins

    def evaluate(self, context):
        if config.debug:
            print()
            print("random sequence position before:", np.random.get_state()[2])
        num = context_length(context)
        choices = self.choices
        if num:
            bins = self.bins
            if bins is None:
                # all values have the same probability
                choices_idx = np.random.randint(len(choices), size=num)
            else:
                if any(isinstance(b, Expr) for b in bins):
                    weights = [expr_eval(expr, context) for expr in bins]
                    bins = self._weights_to_bins(weights)
                u = np.random.uniform(size=num)
                #XXX: np.choice uses searchsorted(bins, u) instead of digitize
                choices_idx = np.digitize(u, bins) - 1
        else:
            choices_idx = []

        if config.debug:
            print("random sequence position after:", np.random.get_state()[2])

        if any(isinstance(c, Expr) for c in choices):
            choices = np.array([expr_eval(expr, context) for expr in choices])
        return choices[choices_idx]

    #noinspection PyUnusedLocal
    def dtype(self, context):
        return self.choices.dtype

    def traverse(self, context):
        #FIXME: add choices & prob if they are expr
        yield self

    def collect_variables(self, context):
        #FIXME: add choices & prob if they are expr
        return set()

    def __str__(self):
        bins = self.bins
        if bins is None:
            weights_str = ""
        else:
            weights_str = ", %s" % (bins
                                    if any(isinstance(b, Expr) for b in bins)
                                    else '[%s]' %
                                         ', '.join(str(b)
                                                   for b in np.diff(bins)))
        return "%s(%s%s)" % (self.func_name, list(self.choices), weights_str)


#------------------------------------


class Round(NumpyChangeArray):
    func_name = 'round'  # np.round redirects to np.round_
    np_func = (np.round,)
    arg_names = ('a', 'decimals', 'out')

    def dtype(self, context):
        # result dtype is the same as the input dtype
        res = getdtype(self.args[0], context)
        assert res == float
        return res


class Trunc(FunctionExpression):
    func_name = 'trunc'

    def evaluate(self, context):
        return expr_eval(self.expr, context).astype(int)

    def dtype(self, context):
        assert getdtype(self.expr, context) == float
        return int

#------------------------------------


class Abs(NumexprFunction):
    func_name = 'abs'

    #noinspection PyUnusedLocal
    def dtype(self, context):
        return float


class Log(NumexprFunction):
    func_name = 'log'

    #noinspection PyUnusedLocal
    def dtype(self, context):
        return float


class Exp(NumexprFunction):
    func_name = 'exp'

    #noinspection PyUnusedLocal
    def dtype(self, context):
        return float


def add_individuals(target_context, children):
    target_entity = target_context['__entity__']
    id_to_rownum = target_entity.id_to_rownum
    array = target_entity.array
    num_rows = len(array)
    num_birth = len(children)
    print("%d new %s(s) (%d -> %d)" % (num_birth, target_entity.name,
                                       num_rows, num_rows + num_birth),
          end=' ')

    target_entity.array.append(children)

    temp_variables = target_entity.temp_variables
    for name, temp_value in temp_variables.iteritems():
        #FIXME: OUCH, this is getting ugly, I'll need a better way to
        # differentiate nd-arrays from "entity" variables
        # I guess having the context contain all entities and a separate
        # globals namespace should fix this problem
        if (isinstance(temp_value, np.ndarray) and
            temp_value.shape == (num_rows,)):
            extra = get_missing_vector(num_birth, temp_value.dtype)
            temp_variables[name] = np.concatenate((temp_value, extra))

    extra_variables = target_context.extra
    for name, temp_value in extra_variables.iteritems():
        if name == '__globals__':
            continue
        if isinstance(temp_value, np.ndarray) and temp_value.shape:
            extra = get_missing_vector(num_birth, temp_value.dtype)
            extra_variables[name] = np.concatenate((temp_value, extra))

    id_to_rownum_tail = np.arange(num_rows, num_rows + num_birth)
    target_entity.id_to_rownum = np.concatenate((id_to_rownum,
                                                 id_to_rownum_tail))


#TODO: inherit from FilteredExpression so that I can use _getfilter
#TODO: allow number to be an expression
class CreateIndividual(EvaluableExpression):
    def __init__(self, entity_name=None, filter=None, number=None, **kwargs):
        self.entity_name = entity_name
        self.filter = filter
        self.kwargs = kwargs
        self.number = number
#        assert filter is not None and number is None or \
#               number is not None and filter is None

    def _initial_values(self, array, to_give_birth, num_birth):
        #TODO: use default values for fields which have one
        children = np.empty(num_birth, dtype=array.dtype)
        children[:] = get_missing_record(array)
        return children

    def traverse(self, context):
        for node in traverse_expr(self.filter, context):
            yield node
        for kwarg in self.kwargs.itervalues():
            for node in traverse_expr(kwarg, context):
                yield node
        yield self

    def collect_variables(self, context):
        #FIXME: we need to add variables from self.filter (that's what is
        # needed for the general case -- in expr_eval)
        used_variables = self._collect_kwargs_variables(context)
        return used_variables

    def _collect_kwargs_variables(self, context):
        used_variables = set()
        for v in self.kwargs.itervalues():
            used_variables.update(collect_variables(v, context))
        return used_variables

    def evaluate(self, context):
        source_entity = context['__entity__']
        if self.entity_name is None:
            target_entity = source_entity
        else:
            target_entity = entity_registry[self.entity_name]

        if target_entity is source_entity:
            target_context = context
        else:
            target_context = \
                EntityContext(target_entity,
                              {'period': context['period'],
                               '__globals__': context['__globals__']})
        ctx_filter = context.get('__filter__')

        if self.filter is not None and ctx_filter is not None:
            filter_expr = ctx_filter & self.filter
        elif self.filter is not None:
            filter_expr = self.filter
        elif ctx_filter is not None:
            filter_expr = ctx_filter
        else:
            filter_expr = None

        if filter_expr is not None:
            to_give_birth = expr_eval(filter_expr, context)
            num_birth = to_give_birth.sum()
        elif self.number is not None:
            to_give_birth = None
            num_birth = self.number
        else:
            raise Exception('no filter nor number in "new"')

        array = target_entity.array

        id_to_rownum = target_entity.id_to_rownum
        num_individuals = len(id_to_rownum)

        children = self._initial_values(array, to_give_birth, num_birth)
        if num_birth:
            children['id'] = np.arange(num_individuals,
                                       num_individuals + num_birth)
            children['period'] = context['period']

            used_variables = self._collect_kwargs_variables(context)
            if to_give_birth is None:
                child_context = new_context_like(context, length=num_birth)
            else:
                child_context = context_subset(context, to_give_birth,
                                               used_variables)
            for k, v in self.kwargs.iteritems():
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
            # See http://projects.scipy.org/numpy/ticket/1869
            result[to_give_birth] = children['id']
            return result
        else:
            return None

    #noinspection PyUnusedLocal
    def dtype(self, context):
        return int


class Clone(CreateIndividual):
    def __init__(self, filter=None, **kwargs):
        CreateIndividual.__init__(self, None, filter, None, **kwargs)

    def _initial_values(self, array, to_give_birth, num_birth):
        return array[to_give_birth]


class Dump(TableExpression):
    #noinspection PyNoneFunctionAssignment
    def __init__(self, *args, **kwargs):
        self.expressions = args
        if len(args):
            assert all(isinstance(e, Expr) for e in args), \
                   "dump arguments must be expressions, not a list of them, " \
                   "or strings"

        self.filter = kwargs.pop('filter', None)
        self.missing = kwargs.pop('missing', None)
#        self.periods = kwargs.pop('periods', None)
        self.header = kwargs.pop('header', True)
        if kwargs:
            kwarg, _ = kwargs.popitem()
            raise TypeError("'%s' is an invalid keyword argument for dump()"
                            % kwarg)

    def evaluate(self, context):
        if self.filter is not None:
            filter_value = expr_eval(self.filter, context)
        else:
            filter_value = None

        if self.expressions:
            expressions = list(self.expressions)
        else:
            # extra=False because we don't want globals nor "system" variables
            # (nan, period, __xxx__)
            expressions = [Variable(name)
                           for name in context.keys(extra=False)]

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
            if (filter_value is not None and isinstance(expr_value, np.ndarray)
                and expr_value.shape):
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
        table = chain([str_expressions], data) if self.header else data
        return PrettyTable(table, self.missing)

    def traverse(self, context):
        for expr in self.expressions:
            for node in traverse_expr(expr, context):
                yield node
        for node in traverse_expr(self.filter, context):
            yield node
        yield self

    def collect_variables(self, context):
        if self.expressions:
            variables = set.union(*[collect_variables(expr, context)
                                    for expr in self.expressions])
        else:
            variables = set(context.keys(extra=False))
        variables |= collect_variables(self.filter, context)
        return variables

    #noinspection PyUnusedLocal
    def dtype(self, context):
        return None


#TODO: inherit from NumexprFunction
class Where(Expr):
    def __init__(self, cond, iftrue, iffalse):
        self.cond = cond
        self.iftrue = iftrue
        self.iffalse = iffalse

    def traverse(self, context):
        for node in traverse_expr(self.cond, context):
            yield node
        for node in traverse_expr(self.iftrue, context):
            yield node
        for node in traverse_expr(self.iffalse, context):
            yield node
        yield self

    def as_simple_expr(self, context):
        cond = as_simple_expr(self.cond, context)

        # filter is stored as an unevaluated expression
        filter_expr = context.get('__filter__')

        if filter_expr is None:
            context['__filter__'] = self.cond
        else:
            context['__filter__'] = filter_expr & self.cond
        iftrue = as_simple_expr(self.iftrue, context)

        if filter_expr is None:
            context['__filter__'] = ~self.cond
        else:
            context['__filter__'] = filter_expr & ~self.cond
        iffalse = as_simple_expr(self.iffalse, context)

        # This is probably useless because the only situation I can think of
        # where it could matter is inside the "iffalse" part of a nested if()
        # and in that case the contextual filter is overwritten using the value
        # of the filter at the *start* of the if() (see above), so it works
        # regardless of what we do here. It should not hurt to be correct
        # though.
        context['__filter__'] = filter_expr

        return Where(cond, iftrue, iffalse)

    def as_string(self):
        return "where(%s, %s, %s)" % (as_string(self.cond),
                                      as_string(self.iftrue),
                                      as_string(self.iffalse))

    def __str__(self):
        return "if(%s, %s, %s)" % (self.cond, self.iftrue, self.iffalse)
    __repr__ = __str__

    def dtype(self, context):
        assert getdtype(self.cond, context) == bool
        return coerce_types(context, self.iftrue, self.iffalse)

    def collect_variables(self, context):
        condvars = collect_variables(self.cond, context)
        iftruevars = collect_variables(self.iftrue, context)
        iffalsevars = collect_variables(self.iffalse, context)
        return condvars | iftruevars | iffalsevars


functions = {
    # random
    'uniform': Uniform,
    'normal': Normal,
    'choice': Choice,
    'randint': RandInt,
    # aggregates/per element combined functions
#    'min': make_dispatcher(aggregates.Min, Min), should not ...!!!!!
#    'max': make_dispatcher(aggregates.Max, Max),
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
    'new': CreateIndividual,
    'clone': Clone,
    'dump': Dump,
}
