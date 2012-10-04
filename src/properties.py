from itertools import izip, chain

import numpy as np

from expr import Expr, Variable, Where, as_string, dtype, \
                 coerce_types, type_to_idx, idx_to_type, expr_eval, \
                 collect_variables, traverse_expr, get_tmp_varname, \
                 missing_values, get_missing_record, \
                 get_missing_vector
from context import EntityContext, context_length, context_subset
from registry import entity_registry
import utils


class BreakpointException(Exception):
    pass


def ispresent(values):
    dtype = values.dtype
    if np.issubdtype(dtype, float):
        return np.isfinite(values)
    elif np.issubdtype(dtype, int):
        return values != missing_values[int]
    elif np.issubdtype(dtype, bool):
        return values != missing_values[bool]


class Process(object):
    def __init__(self):
        self.name = None
        self.entity = None

    def attach(self, name, entity):
        self.name = name
        self.entity = entity

    def run_guarded(self, simulation, globals):
        try:
            context = EntityContext(self.entity, globals.copy())
            self.run(context)
        except BreakpointException:
            simulation.stepbystep = True

    def run(self, context):
        raise NotImplementedError()

    def expressions(self):
        raise NotImplementedError()

    def __str__(self):
        return "<process '%s'>" % self.name


class Compute(Process):
    '''these processes only compute an expression and do not store their
       result (but they usually have side-effects)'''

    def __init__(self, expr):
        super(Compute, self).__init__()
        self.expr = expr

    def run(self, context):
        expr_eval(self.expr, context)

    def expressions(self):
        if isinstance(self.expr, Expr):
            yield self.expr


class Assignment(Process):
    def __init__(self, expr):
        super(Assignment, self).__init__()
        self.predictor = None
        self.kind = None  # period_individual, period, individual, global
        self.expr = expr

    def attach(self, name, entity, kind=None):
        super(Assignment, self).attach(name, entity)
        if self.predictor is None:
            self.predictor = name
        self.kind = kind

    def run(self, context):
        value = expr_eval(self.expr, context)
        self.store_result(value)

    def store_result(self, result):
        if result is None:
            return
        if self.name is None:
            raise Exception('trying to store None key')

        if isinstance(result, dict):
            indices = result.get('indices')
            filter_values = result.get('filter')
            assert filter_values is None or indices is None
            result = result['values']
        else:
            indices = None
            filter_values = None

        if isinstance(result, np.ndarray):
            res_type = result.dtype.type
        else:
            res_type = type(result)

        if self.kind == 'period_individual':
            # we cannot store/cache self.entity.array[self.name] because the
            # array object can change (eg when enlarging it due to births)
            target = self.entity.array
        else:
            target = self.entity.temp_variables

        #TODO: assert type for temporary variables too
        if self.kind is not None:
            target_type_idx = type_to_idx[target[self.predictor].dtype.type]
            res_type_idx = type_to_idx[res_type]
            if res_type_idx > target_type_idx:
                raise Exception(
                    "trying to store %s value into '%s' field which is of "
                    "type %s" % (idx_to_type[res_type_idx].__name__,
                                 self.predictor,
                                 idx_to_type[target_type_idx].__name__))

        if indices is None and filter_values is None:
            # the whole column is updated
            target[self.predictor] = result
        else:
            if isinstance(target, np.ndarray):
                hasfield = self.predictor in target.dtype.fields
            else:
                hasfield = self.predictor in target

            if not hasfield:
                assert self.kind is None, \
                       "found a missing field which is not a temporary " \
                       "variable"
                #XXX: I'm not sure we should do this at all and in any case
                # this step should be delayed further because it could
                # be unnecessary.
                target[self.predictor] = np.zeros(len(self.entity.array),
                                                  dtype=res_type)
            if indices is not None:
                np.put(target[self.predictor], indices, result)
            elif filter_values is not None:
                np.putmask(target[self.predictor], filter_values, result)

    def expressions(self):
        if isinstance(self.expr, Expr):
            yield self.expr


class ProcessGroup(Process):
    def __init__(self, name, subprocesses):
        super(ProcessGroup, self).__init__()
        self.name = name
        self.subprocesses = subprocesses

    def run_guarded(self, simulation, globals):
        print
        for k, v in self.subprocesses:
            print "    *",
            if k is not None:
                print k,
            utils.timed(v.run_guarded, simulation, globals)
#            print "done."
            simulation.start_console(v.entity, globals['period'])
        # purge all local variables
        temp_vars = self.entity.temp_variables
        all_vars = self.entity.variables
        local_vars = set(temp_vars.keys()) - set(all_vars.keys())
        for var in local_vars:
            del temp_vars[var]

    def expressions(self):
        for _, p in self.subprocesses:
            for e in p.expressions():
                yield e


class EvaluableExpression(Expr):
    def evaluate(self, context):
        raise NotImplementedError

    def as_string(self, context):
        tmp_varname = get_tmp_varname()
        result = expr_eval(self, context)
        if isinstance(result, dict):
            indices = result['indices']
            values = result['values']
        else:
            indices = None

        if indices is not None:
            if isinstance(values, np.ndarray):
                res_type = values.dtype.type
            else:
                res_type = type(values)
            result = np.zeros(context_length(context), dtype=res_type)
            np.put(result, indices, values)
        context[tmp_varname] = result
        return tmp_varname


class CompoundExpression(Expr):
    '''expression written in terms of other expressions'''

    def __init__(self):
        self._complete_expr = None

    def evaluate(self, context):
        context = self.build_context(context)
        return expr_eval(self.complete_expr, context)

    def as_string(self, context):
        context = self.build_context(context)
        return self.complete_expr.as_string(context)

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
#        3 where, 3 comparisons = 6 op (or 4 if optimized)
#
#        Where(Where(Where(expr1 < expr2, expr1, expr2) < expr3,
#                    Where(expr1 < expr2, expr1, expr2),
#                    expr3) < expr4,
#              Where(Where(expr1 < expr2, expr1, expr2) < expr3,
#                    Where(expr1 < expr2, expr1, expr2),
#                    expr3),
#              expr4)
#        7 where, 7 comp = 14 op (or 6 if optimized)

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


class ZeroClip(CompoundExpression):
    def __init__(self, expr1, expr2, expr3):
        CompoundExpression.__init__(self)
        self.expr1 = expr1
        self.expr2 = expr2
        self.expr3 = expr3

    def build_expr(self):
        return Where((self.expr1 >= self.expr2) & (self.expr1 <= self.expr3),
                     self.expr1,
                     0)

    def dtype(self, context):
        return dtype(self.expr1, context)


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

#------------------------------------


class NumpyProperty(EvaluableExpression):
    func_name = None  # optional (for display)
    np_func = (None,)
    # arg_names can be set automatically by using inspect.getargspec,
    # but it only works for pure Python functions, so I decided to avoid it
    # because when you add a function, it's hard to know whether it's
    # implemented in C or not.
    arg_names = None

    def __init__(self, *args, **kwargs):
        EvaluableExpression.__init__(self)
        if len(args) > len(self.arg_names):
            # + 1 to be consistent with Python (to account for self)
            raise TypeError("takes at most %d arguments (%d given)" %
                            (len(self.arg_names) + 1, len(args) + 1))
        extra_kwargs = set(kwargs.keys()) - set(self.arg_names)
        if extra_kwargs:
            extra_kwargs = [repr(arg) for arg in extra_kwargs]
            raise TypeError("got an unexpected keyword argument %s" %
                            extra_kwargs[0])
        self.args = args
        self.kwargs = kwargs

    def evaluate(self, context):
        eval_func = self.get_eval_func()
        args = [eval_func(arg, context) for arg in self.args]
        kwargs = dict((k, eval_func(v, context))
                      for k, v in self.kwargs.iteritems())
        if 'size' in self.arg_names and 'size' not in kwargs:
            kwargs['size'] = context_length(context)

        func = self.np_func[0]
        return func(*args, **kwargs)

    def get_eval_func(self):
        return expr_eval

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


class NumpyAggregate(NumpyProperty):
    skip_missing = False

    def get_eval_func(self):
        if self.skip_missing:
            def local_expr_eval(expr, context):
                values = expr_eval(expr, context)
                if isinstance(values, np.ndarray) and values.shape:
                    values = values[ispresent(values)]
                return values
            return local_expr_eval
        else:
            return expr_eval


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
class Clip(NumpyProperty):
    np_func = (np.clip,)
    arg_names = ('a', 'a_min', 'a_max', 'out')

#------------------------------------


class Uniform(NumpyProperty):
    np_func = (np.random.uniform,)
    arg_names = ('low', 'high', 'size')


class Normal(NumpyProperty):
    np_func = (np.random.normal,)
    arg_names = ('loc', 'scale', 'size')


class Choice(EvaluableExpression):
    func_name = 'choice'

    def __init__(self, choices, weights=None):
        #TODO: allow expressions in choices & weights
        EvaluableExpression.__init__(self)
        self.choices = np.array(choices)
        if weights is not None:
            self.bins = np.array([0.0] + list(np.cumsum(weights)))
            error = abs(self.bins[-1] - 1.0)
            if 0.0 < error <= 1e-6:
                # overshooting a bit is the lesser evil here (the last choice
                # will be picked a tad less than its probability) but we can't
                # easily "correct" that one to 1.0 because in that case, we
                # would have the last bin boundary smaller than the second last
                if str(1.0 - self.bins[-2]) != str(weights[-1]) and \
                   self.bins[-1] < 1.0:
                    print "Warning: last choice probability adjusted to %s " \
                          "instead of %s !" % (1.0 - self.bins[-2],
                                               weights[-1])
                    self.bins[-1] = 1.0
            elif error > 1e-6:
                raise Exception(
                    "the cumulative sum of choice weights must be ~1")
        else:
            self.bins = None

    def evaluate(self, context):
        num = context_length(context)

        if num:
            if self.bins is None:
                # all values have the same probability
                choices_idx = np.random.randint(len(self.choices), size=num)
            else:
                u = np.random.uniform(size=num)
                choices_idx = np.digitize(u, self.bins) - 1
        else:
            choices_idx = []
        return self.choices[choices_idx]

    def dtype(self, context):
        return self.choices.dtype

    def traverse(self, context):
        yield self

    def collect_variables(self, context):
        return set()

    def __str__(self):
        if self.bins is None:
            weights = ""
        else:
            weights = ", [%s]" % ', '.join(str(v) for v in np.diff(self.bins))
        return "%s(%s%s)" % (self.func_name, list(self.choices), weights)


class RandInt(NumpyProperty):
    np_func = (np.random.randint,)
    arg_names = ('low', 'high', 'size')

    def dtype(self, context):
        return int

#------------------------------------


class Round(NumpyProperty):
    func_name = 'round'  # np.round redirects to np.round_
    np_func = (np.round,)
    arg_names = ('a', 'decimals', 'out')

    def dtype(self, context):
        # result dtype is the same as the input dtype
        res = dtype(self.args[0], context)
        assert res == float
        return res


class Trunc(FunctionExpression):
    func_name = 'trunc'

    def evaluate(self, context):
        return expr_eval(self.expr, context).astype(int)

    def dtype(self, context):
        assert dtype(self.expr, context) == float
        return int

#------------------------------------


class GroupMin(NumpyProperty):
    func_name = 'grpmin'
    np_func = (np.amin,)
    arg_names = ('a', 'axis', 'out')

    def dtype(self, context):
        return dtype(self.args[0], context)


class GroupMax(NumpyProperty):
    func_name = 'grpmax'
    np_func = (np.amax,)
    arg_names = ('a', 'axis', 'out')

    def dtype(self, context):
        return dtype(self.args[0], context)


class GroupSum(FilteredExpression):
    func_name = 'grpsum'

    def evaluate(self, context):
        expr = self.expr
        filter_expr = self._getfilter(context)
        if filter_expr is not None:
            expr *= filter_expr

        return np.nansum(expr_eval(expr, context))

    def dtype(self, context):
        #TODO: merge this typemap with tsum's
        typemap = {bool: int, int: int, float: float}
        return typemap[dtype(self.args[0], context)]


class GroupStd(NumpyAggregate):
    func_name = 'grpstd'
    np_func = (np.std,)
    arg_names = ('a', 'axis', 'dtype', 'out', 'ddof')
    skip_missing = True

    def dtype(self, context):
        return float


class GroupMedian(NumpyAggregate):
    func_name = 'grpmedian'
    np_func = (np.median,)
    arg_names = ('a', 'axis', 'out', 'overwrite_input')
    skip_missing = True

    def dtype(self, context):
        return float


class GroupPercentile(NumpyAggregate):
    func_name = 'grppercentile'
    np_func = (np.percentile,)
    arg_names = ('a', 'q', 'axis', 'out', 'overwrite_input')
    skip_missing = True

    def dtype(self, context):
        return float


class GroupGini(FilteredExpression):
    func_name = 'grpgini'

    def evaluate(self, context):
        # from Wikipedia:
        # G = 1/n * (n + 1 - 2 * (sum((n + 1 - i) * a[i]) / sum(a[i])))
        #                        i=1..n                    i=1..n
        # but sum((n + 1 - i) * a[i])
        #    i=1..n
        #   = sum((n - i) * a[i] for i in range(n))
        #   = sum(cumsum(a))
        values = expr_eval(self.expr, context)
        if isinstance(values, (list, tuple)):
            values = np.array(values)

        filter_expr = self._getfilter(context)
        if filter_expr is not None:
            filter_values = expr_eval(filter_expr, context)
        else:
            filter_values = True
        filter_values &= ispresent(values)
        values = values[filter_values]
        sorted_values = np.sort(values)
        n = len(values)

        # force float to avoid overflows with integer input expressions
        cumsum = np.cumsum(sorted_values, dtype=float)
        values_sum = cumsum[-1]
        return (n + 1 - 2 * np.sum(cumsum) / values_sum) / n

    def dtype(self, context):
        return float


class GroupCount(EvaluableExpression):
    def __init__(self, filter=None):
        self.filter = filter

    def evaluate(self, context):
        if self.filter is None:
            return context_length(context)
        else:
            #TODO: check this at "compile" time (in __init__), though for
            # that we need to know the type of all temporary variables
            # first
            if dtype(self.filter, context) is not bool:
                raise Exception("grpcount filter must be a boolean expression")
            return np.sum(expr_eval(self.filter, context))

    def dtype(self, context):
        return int

    def traverse(self, context):
        for node in traverse_expr(self.filter, context):
            yield node
        yield self

    def collect_variables(self, context):
        return collect_variables(self.filter, context)

    def __str__(self):
        filter_str = str(self.filter) if self.filter is not None else ''
        return "grpcount(%s)" % filter_str


# we could transform this into a CompoundExpression:
# grpsum(expr, filter=filter) / grpcount(filter) but that would be inefficient.
class GroupAverage(FilteredExpression):
    func_name = 'grpavg'

    def evaluate(self, context):
        expr = self.expr
        #FIXME: either take "contextual filter" into account here (by using
        # self._getfilter), or don't do it in grpsum (& grpgini?)
        if self.filter is not None:
            filter_values = expr_eval(self.filter, context)
            tmp_varname = get_tmp_varname()
            context = context.copy()
            context[tmp_varname] = filter_values
            if dtype(expr, context) is bool:
                # convert expr to int because mul_bbb is not implemented in
                # numexpr
                expr *= 1
            expr *= Variable(tmp_varname)
        else:
            filter_values = True
        values = expr_eval(expr, context)
        filter_values &= np.isfinite(values)
        numrows = np.sum(filter_values)
        if numrows:
            return np.nansum(values) / float(numrows)
        else:
            return float('nan')

    def dtype(self, context):
        return float


class NumexprFunctionProperty(Expr):
    '''For functions which are present as-is in numexpr'''

    def __init__(self, expr):
        self.expr = expr

    def collect_variables(self, context):
        return collect_variables(self.expr, context)

    def as_string(self, context):
        return '%s(%s)' % (self.func_name, as_string(self.expr, context))

    def __str__(self):
        return '%s(%s)' % (self.func_name, self.expr)

    def traverse(self, context):
        for node in traverse_expr(self.expr, context):
            yield node


class Abs(NumexprFunctionProperty):
    func_name = 'abs'

    def dtype(self, context):
        return float


class Log(NumexprFunctionProperty):
    func_name = 'log'

    def dtype(self, context):
        return float


class Exp(NumexprFunctionProperty):
    func_name = 'exp'

    def dtype(self, context):
        return float


def add_individuals(target_context, children):
    target_entity = target_context['__entity__']
    id_to_rownum = target_entity.id_to_rownum
    array = target_entity.array
    num_rows = len(array)
    num_birth = len(children)
    print "%d new %s(s) (%d -> %d)" % (num_birth, target_entity.name,
                                       num_rows, num_rows + num_birth),

    target_entity.array = np.concatenate((array, children))
    temp_variables = target_entity.temp_variables
    for name, temp_value in temp_variables.iteritems():
        if isinstance(temp_value, np.ndarray) and temp_value.shape:
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


#TODO: inherit from FilteredExpression
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
            target_context = EntityContext(target_entity,
                                           {'period': context['period']})

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
            # Note that np.place is a tad faster, but is currently buggy when
            # working with columns of structured arrays.
            # See http://projects.scipy.org/numpy/ticket/1869
            result[to_give_birth] = children['id']
            return result
        else:
            return None

    def dtype(self, context):
        return int


class Clone(CreateIndividual):
    def __init__(self, filter=None, **kwargs):
        CreateIndividual.__init__(self, None, filter, None, **kwargs)

    def _initial_values(self, array, to_give_birth, num_birth):
        return array[to_give_birth]


class TableExpression(EvaluableExpression):
    pass


class Dump(TableExpression):
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
            expressions = [Variable(name) for name in context.keys()]

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
        return utils.PrettyTable(table, self.missing)

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
            variables = set(context.keys())
        variables |= collect_variables(self.filter, context)
        return variables

    def dtype(self, context):
        return None


functions = {
    # random
    'uniform': Uniform,
    'normal': Normal,
    'choice': Choice,
    'randint': RandInt,
    # aggregates
    'grpcount': GroupCount,
    'grpmin': GroupMin,
    'grpmax': GroupMax,
    'grpsum': GroupSum,
    'grpavg': GroupAverage,
    'grpstd': GroupStd,
    'grpmedian': GroupMedian,
    'grppercentile': GroupPercentile,
    'grpgini': GroupGini,
    # per element
    'min': Min,
    'max': Max,
    'abs': Abs,
    'clip': Clip,
    'zeroclip': ZeroClip,
    'round': Round,
    'trunc': Trunc,
    'exp': Exp,
    'log': Log,

    # misc
    'new': CreateIndividual,
    'clone': Clone,
    'dump': Dump,
}
