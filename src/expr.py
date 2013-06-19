from __future__ import division, print_function

from collections import Counter

import numpy as np

from utils import LabeledArray, ExplainTypeError, add_context
from context import EntityContext

try:
    import numexpr
#    numexpr.set_num_threads(1)
    evaluate = numexpr.evaluate
except ImportError:
    eval_context = [(name, getattr(np, name))
                    for name in ('where', 'exp', 'log', 'abs')]
    eval_context.extend([('False', False), ('True', True)])

    def evaluate(expr, globals, locals=None, **kwargs):
        complete_globals = {}
        complete_globals.update(globals)
        if locals is not None:
            if isinstance(locals, np.ndarray):
                for fname in locals.dtype.fields:
                    complete_globals[fname] = locals[fname]
            else:
                complete_globals.update(locals)
        complete_globals.update(eval_context)
        return eval(expr, complete_globals, {})

num_tmp = 0
timings = Counter()


def get_tmp_varname():
    global num_tmp

    tmp_varname = "temp_%d" % num_tmp
    num_tmp += 1
    return tmp_varname

type_to_idx = {bool: 0, np.bool_: 0,
               int: 1, np.int32: 1, np.intc: 1, np.int64: 1, np.longlong: 1,
               float: 2, np.float64: 2}
idx_to_type = [bool, int, float]

missing_values = {
#    int: -2147483648,
    # for links, we need to have abs(missing_int) < len(a) !
    #XXX: we might want to use different missing values for links and for
    #     "normal" ints
    int: -1,
    float: float('nan'),
#    bool: -1
    bool: False
}


def normalize_type(type_):
    return idx_to_type[type_to_idx[type_]]


def get_missing_value(column):
    return missing_values[normalize_type(column.dtype.type)]


def get_missing_vector(num, dtype):
    res = np.empty(num, dtype=dtype)
    res.fill(missing_values[normalize_type(dtype.type)])
    return res


def get_missing_record(array):
    row = np.empty(1, dtype=array.dtype)
    for fname in array.dtype.names:
        row[fname] = get_missing_value(row[fname])
    return row


def hasvalue(column):
    missing_value = get_missing_value(column)
    if np.isnan(missing_value):
        return ~np.isnan(column)
    else:
        return column != missing_value


def coerce_types(context, *args):
    dtype_indices = [type_to_idx[dtype(arg, context)] for arg in args]
    return idx_to_type[max(dtype_indices)]


def as_simple_expr(expr, context):
    if isinstance(expr, Expr):
        return expr.as_simple_expr(context)
    else:
        return expr


def as_string(expr):
    if isinstance(expr, Expr):
        return expr.as_string()
    else:
        return str(expr)


def traverse_expr(expr, context):
    if isinstance(expr, Expr):
        return expr.traverse(context)
    else:
        return ()


def gettype(value):
    if isinstance(value, np.ndarray):
        type_ = value.dtype.type
    elif isinstance(value, (tuple, list)):
        type_ = type(value[0])
    else:
        type_ = type(value)
    return normalize_type(type_)


def dtype(expr, context):
    if isinstance(expr, Expr):
        return expr.dtype(context)
    else:
        return gettype(expr)


def ispresent(values):
    dt = values.dtype
    if np.issubdtype(dt, float):
        return np.isfinite(values)
    elif np.issubdtype(dt, int):
        return values != missing_values[int]
    elif np.issubdtype(dt, bool):
#        return values != missing_values[bool]
        return True
    else:
        raise Exception('%s is not a supported type for ispresent' % dt)


# context is needed because in LinkValue we need to know what is the current
# entity (so that we can resolve links)
#TODO: we shouldn't resolve links during the simulation but
# rather in a "compilation" phase
def collect_variables(expr, context):
    if isinstance(expr, Expr):
        return expr.collect_variables(context)
    else:
        return set()


def expr_eval(expr, context):
    if isinstance(expr, Expr):
        globals_data = context.get('__globals__')
        if globals_data is not None:
            globals_names = set(globals_data.keys())
            if 'periodic' in globals_data:
                globals_names |= set(globals_data['periodic'].dtype.names)
        else:
            globals_names = set()

        for var_name in expr.collect_variables(context):
            if var_name not in globals_names and var_name not in context:
                raise Exception("variable '%s' is unknown (it is either not "
                                "defined or not computed yet)" % var_name)
        return expr.evaluate(context)

        # there are several flaws with this approach:
        # 1) I don't get action times (csv et al)
        # 2) these are cumulative times (they include child expr/processes)
        #    we might want to store the timings in a tree (based on call stack
        #    depth???) so that I could rebuild both cumulative and "real"
        #    timings.
        # 3) the sum of timings is wrong since children/nested expr times count
        #    both for themselves and for all their parents
#        time, res = gettime(expr.evaluate, context)
#        timings[expr.__class__.__name__] += time
#        return res
    elif isinstance(expr, list) and any(isinstance(e, Expr) for e in expr):
        return [expr_eval(e, context) for e in expr]
    elif isinstance(expr, tuple) and any(isinstance(e, Expr) for e in expr):
        return tuple([expr_eval(e, context) for e in expr])
    elif isinstance(expr, slice):
        return slice(expr_eval(expr.start, context),
                     expr_eval(expr.stop, context),
                     expr_eval(expr.step, context))
    else:
        return expr


class Expr(object):
    __metaclass__ = ExplainTypeError

    def traverse(self, context):
        raise NotImplementedError()

    def allOf(self, node_type, context=None):
        for node in self.traverse(context):
            if isinstance(node, node_type):
                yield node

    # makes sure we dont use "normal" python logical operators
    # (and, or, not)
    def __nonzero__(self):
        raise Exception("Improper use of boolean operators, you probably "
                        "forgot parenthesis around operands of an 'and' or "
                        "'or' expression. The complete expression cannot be "
                        "displayed but it contains: '%s'." % str(self))

    def __lt__(self, other):
        return ComparisonOp('<', self, other)
    def __le__(self, other):
        return ComparisonOp('<=', self, other)
    def __eq__(self, other):
        return ComparisonOp('==', self, other)
    def __ne__(self, other):
        return ComparisonOp('!=', self, other)
    def __gt__(self, other):
        return ComparisonOp('>', self, other)
    def __ge__(self, other):
        return ComparisonOp('>=', self, other)

    def __add__(self, other):
        return Addition('+', self, other)
    def __sub__(self, other):
        return Subtraction('-', self, other)
    def __mul__(self, other):
        return Multiplication('*', self, other)
    def __div__(self, other):
        return Division('/', self, other)
    def __truediv__(self, other):
        return Division('/', self, other)
    def __floordiv__(self, other):
        return Division('//', self, other)
    def __mod__(self, other):
        return BinaryOp('%', self, other)
    def __divmod__(self, other):
        #FIXME
        return BinaryOp('divmod', self, other)
    def __pow__(self, other, modulo=None):
        return BinaryOp('**', self, other)
    def __lshift__(self, other):
        return BinaryOp('<<', self, other)
    def __rshift__(self, other):
        return BinaryOp('>>', self, other)

    def __and__(self, other):
        return And('&', self, other)
    def __xor__(self, other):
        return BinaryOp('^', self, other)
    def __or__(self, other):
        return Or('|', self, other)

    def __radd__(self, other):
        return Addition('+', other, self)
    def __rsub__(self, other):
        return Subtraction('-', other, self)
    def __rmul__(self, other):
        return Multiplication('*', other, self)
    def __rdiv__(self, other):
        return Division('/', other, self)
    def __rtruediv__(self, other):
        return Division('/', other, self)
    def __rfloordiv__(self, other):
        return Division('//', other, self)
    def __rmod__(self, other):
        return BinaryOp('%', other, self)
    def __rdivmod__(self, other):
        return BinaryOp('divmod', other, self)
    def __rpow__(self, other):
        return BinaryOp('**', other, self)
    def __rlshift__(self, other):
        return BinaryOp('<<', other, self)
    def __rrshift__(self, other):
        return BinaryOp('>>', other, self)

    def __rand__(self, other):
        return And('&', other, self)
    def __rxor__(self, other):
        return BinaryOp('^', other, self)
    def __ror__(self, other):
        return Or('|', other, self)

    def __neg__(self):
        return UnaryOp('-', self)
    def __pos__(self):
        return UnaryOp('+', self)
    def __abs__(self):
        return UnaryOp('abs', self)
    def __invert__(self):
        return Not('~', self)

    def evaluate(self, context):
#        FIXME: this cannot work, because dict.__contains__(k) calls k.__eq__
#        which has a non standard meaning
#        if self in expr_cache:
#            s = expr_cache[self]
#        else:
#            s = self.as_string(context)
#            expr_cache[self] = s

        simple_expr = self.as_simple_expr(context)
        if isinstance(simple_expr, Variable) and simple_expr.name in context:
            return context[simple_expr.name]

        # check for labeled arrays, to work around the fact that numexpr
        # does not preserve ndarray subclasses.

        # avoid checking for arrays types in the past, because that is a
        # costly operation (context[var_name] fetches the column from disk
        # in that case). This probably prevents us from doing stuff like
        # align(lag(groupby() / groupby())), but it is a limitation I can
        # live with to avoid hitting the disk twice for each disk access.

        #TODO: I should rewrite this whole mess when my "dtype" method
        # supports ndarrays and LabeledArray so that I can get the dtype from
        # the expression instead of from actual values.
        labels = None
        if isinstance(context, EntityContext) and context._is_array_period:
            for var_name in simple_expr.collect_variables(context):
                # var_name should always be in the context at this point
                # because missing temporaries should have been already caught
                # in expr_eval
                value = context[var_name]
                if isinstance(value, LabeledArray):
                    if labels is None:
                        labels = (value.dim_names, value.pvalues)
                    else:
                        if labels[0] != value.dim_names:
                            raise Exception('several arrays with inconsistent '
                                            'labels (dimension names) in the '
                                            'same expression: %s vs %s'
                                            % (labels[0], value.dim_names))
                        if not np.array_equal(labels[1], value.pvalues):
                            raise Exception('several arrays with inconsistent '
                                            'axis values in the same '
                                            'expression: \n%s\n\nvs\n\n%s'
                                            % (labels[1], value.pvalues))

        s = simple_expr.as_string()
        try:
            res = evaluate(s, context, {}, truediv='auto')
            if labels is not None:
                # This is a hack which relies on the fact that currently
                # all the expression we evaluate through numexpr preserve
                # array shapes, but if we ever use numexpr reduction
                # capabilities, we will be in trouble
                res = LabeledArray(res, labels[0], labels[1])
            return res
        except KeyError, e:
            raise add_context(e, s)
        except Exception, e:
            raise

    def as_simple_expr(self, context):
        '''
        evaluate any construct that is not supported by numexpr and
        create temporary variables for them
        '''
        raise NotImplementedError()

    def as_string(self):
        raise NotImplementedError()

    def __getitem__(self, key):
        #TODO: we should be able to know at "compile" time if this is a
        # scalar or a vector and disallow getitem in case of a scalar
        return SubscriptedExpr(self, key)

    def __getattr__(self, key):
        return ExprAttribute(self, key)

    def collect_variables(self, context):
        raise NotImplementedError()


class EvaluableExpression(Expr):
    def evaluate(self, context):
        raise NotImplementedError()

    def as_simple_expr(self, context):
        tmp_varname = get_tmp_varname()
        result = self.evaluate(context)
        context[tmp_varname] = result
        return Variable(tmp_varname, gettype(result))


def non_scalar_array(a):
    return isinstance(a, np.ndarray) and a.shape


class SubscriptedExpr(EvaluableExpression):
    def __init__(self, expr, key):
        self.expr = expr
        self.key = key

    def __str__(self):
        key = self.key
        if isinstance(key, slice):
            key_str = '%s:%s' % (key.start, key.stop)
            if key.step is not None:
                key_str += ':%s' % key.step
        else:
            key_str = str(key)
        return '%s[%s]' % (self.expr, key_str)
    __repr__ = __str__

    def evaluate(self, context):
        expr_value = expr_eval(self.expr, context)
        key = expr_eval(self.key, context)

        filter_expr = context.get('__filter__')

        # When there is a contextual filter, we modify the key to avoid
        # crashes (IndexError).

        # The value returned for individuals outside the filter is
        # *undefined* ! We sometimes return missing and sometimes return the
        # value of another individual (index -1). This should not pose a
        # problem because those values should not be used anyway.
        if filter_expr is not None:
            # We need a context without __filter__ to evaluate the filter
            # (to avoid an infinite recursion)
            sub_context = context.copy()
            del sub_context['__filter__']
            filter_value = expr_eval(filter_expr, sub_context)

            def fixkey(key, filter_value):
                if non_scalar_array(key):
                    newkey = key.copy()
                else:
                    newkey = np.empty(len(filter_value), dtype=int)
                    newkey.fill(key)
                newkey[~filter_value] = -1
                return newkey

            if non_scalar_array(filter_value):
                if isinstance(key, tuple):
                    key = tuple(fixkey(k, filter_value) for k in key)
                elif isinstance(key, slice):
                    raise NotImplementedError()
                else:
                    # scalar or array key
                    key = fixkey(key, filter_value)
            else:
                if not filter_value:
                    missing_value = get_missing_value(expr_value)
                    if (non_scalar_array(key) or
                        (isinstance(key, tuple) and
                         any(non_scalar_array(k) for k in key))):
                        # scalar filter, array or tuple key
                        res = np.empty_like(expr_value)
                        res.fill(missing_value)
                        return res
                    elif isinstance(key, slice):
                        raise NotImplementedError()
                    else:
                        # scalar (or tuple of scalars) key
                        return missing_value
        return expr_value[key]

    def collect_variables(self, context):
        exprvars = collect_variables(self.expr, context)
        return exprvars | collect_variables(self.key, context)

    def traverse(self, context):
        for node in traverse_expr(self.expr, context):
            yield node
        for node in traverse_expr(self.key, context):
            yield node
        yield self


class ExprAttribute(EvaluableExpression):
    def __init__(self, expr, key):
        self.expr = expr
        self.key = key

    def __str__(self):
        return '%s.%s' % (self.expr, self.key)
    __repr__ = __str__

    def evaluate(self, context):
        # currently key can only be a string but if I ever expose getattr,
        # it could be an expr too
        return getattr(expr_eval(self.expr, context),
                       expr_eval(self.key, context))

    def __call__(self, *args, **kwargs):
        return ExprCall(self, args, kwargs)

    def collect_variables(self, context):
        exprvars = collect_variables(self.expr, context)
        return exprvars | collect_variables(self.key, context)

    def traverse(self, context):
        for node in traverse_expr(self.expr, context):
            yield node
        for node in traverse_expr(self.key, context):
            yield node
        yield self


#TODO: factorize with NumpyFunction & FunctionExpression
class ExprCall(EvaluableExpression):
    def __init__(self, expr, args, kwargs):
        self.expr = expr
        self.args = args
        self.kwargs = kwargs

    def evaluate(self, context):
        expr = expr_eval(self.expr, context)
        args = [expr_eval(arg, context) for arg in self.args]
        kwargs = dict((k, expr_eval(v, context))
                      for k, v in self.kwargs.iteritems())
        return expr(*args, **kwargs)

    def __str__(self):
        args = [repr(a) for a in self.args]
        kwargs = ['%s=%r' % (k, v) for k, v in self.kwargs.iteritems()]
        return '%s(%s)' % (self.expr, ', '.join(args + kwargs))
    __repr__ = __str__

    def collect_variables(self, context):
        args_vars = [collect_variables(arg, context) for arg in self.args]
        args_vars.extend(collect_variables(v, context)
                         for v in self.kwargs.itervalues())
        return set.union(*args_vars) if args_vars else set()

    def traverse(self, context):
        for node in traverse_expr(self.expr, context):
            yield node
        for arg in self.args:
            for node in traverse_expr(arg, context):
                yield node
        for kwarg in self.kwargs.itervalues():
            for node in traverse_expr(kwarg, context):
                yield node
        yield self


#############
# Operators #
#############

class UnaryOp(Expr):
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

    def simplify(self):
        expr = self.expr.simplify()
        if not isinstance(expr, Expr):
            return eval('%s%s' % (self.op, self.expr))
        return self

    def show(self, indent):
        print(indent, self.op)
        self.expr.show(indent + '    ')

    def as_simple_expr(self, context):
        return self.__class__(self.op, self.expr.as_simple_expr(context))

    def as_string(self):
        return "(%s%s)" % (self.op, self.expr.as_string())

    def __str__(self):
        return "(%s%s)" % (self.op, self.expr)
    __repr__ = __str__

    def collect_variables(self, context):
        return self.expr.collect_variables(context)

    def dtype(self, context):
        return dtype(self.expr, context)

    def traverse(self, context):
        for node in traverse_expr(self.expr, context):
            yield node
        yield self


class Not(UnaryOp):
    def simplify(self):
        expr = self.expr.simplify()
        if not isinstance(expr, Expr):
            return not expr
        return self


class BinaryOp(Expr):
    neutral_value = None
    overpowering_value = None

    def __init__(self, op, expr1, expr2):
        self.op = op
        self.expr1 = expr1
        self.expr2 = expr2

    def traverse(self, context):
        for c in traverse_expr(self.expr1, context):
            yield c
        for c in traverse_expr(self.expr2, context):
            yield c
        yield self

    def as_simple_expr(self, context):
        expr1 = as_simple_expr(self.expr1, context)
        expr2 = as_simple_expr(self.expr2, context)
        return self.__class__(self.op, expr1, expr2)

    # We can't simply use __str__ because of where vs if
    def as_string(self):
        return "(%s %s %s)" % (as_string(self.expr1),
                               self.op,
                               as_string(self.expr2))

    def dtype(self, context):
        return coerce_types(context, self.expr1, self.expr2)

    def __str__(self):
        return "(%s %s %s)" % (self.expr1, self.op, self.expr2)
    __repr__ = __str__

    def simplify(self):
        expr1 = self.expr1.simplify()
        if isinstance(self.expr2, Expr):
            expr2 = self.expr2.simplify()
        else:
            expr2 = self.expr2

        if self.neutral_value is not None:
            if isinstance(expr2, self.accepted_types) and \
               expr2 == self.neutral_value:
                return expr1

        if self.overpowering_value is not None:
            if isinstance(expr2, self.accepted_types) and \
               expr2 == self.overpowering_value:
                return self.overpowering_value
        if not isinstance(expr1, Expr) and not isinstance(expr2, Expr):
            return eval('%s %s %s' % (expr1, self.op, expr2))
        return BinaryOp(self.op, expr1, expr2)

    def show(self, indent=''):
        print(indent, self.op)
        if isinstance(self.expr1, Expr):
            self.expr1.show(indent=indent + '    ')
        else:
            print(indent + '    ', self.expr1)
        if isinstance(self.expr2, Expr):
            self.expr2.show(indent=indent + '    ')
        else:
            print(indent + '    ', self.expr2)

    def collect_variables(self, context):
        vars2 = collect_variables(self.expr2, context)
        return collect_variables(self.expr1, context).union(vars2)

#    def guard_missing(self):
#        dtype = self.dtype()
#        if dtype is float:
#            return self
#        else:
#            return Where(ispresent(self.expr1) & ispresent(self.expr2),
#                         self,
#                         missingvalue[dtype])


class ComparisonOp(BinaryOp):
    def dtype(self, context):
        assert coerce_types(context, self.expr1, self.expr2) is not None, \
               "operands to comparison operators need to be of compatible " \
               "types"
        return bool


class LogicalOp(BinaryOp):
    def dtype(self, context):
        def assertbool(expr):
            dt = dtype(expr, context)
            if dt is not bool:
                raise Exception("operands to logical operators need to be "
                                "boolean but %s is %s" % (expr, dt))
        assertbool(self.expr1)
        assertbool(self.expr2)
        return bool


class And(LogicalOp):
    neutral_value = True
    overpowering_value = False
    accepted_types = (bool, np.bool_)


class Or(LogicalOp):
    neutral_value = False
    overpowering_value = True
    accepted_types = (bool, np.bool_)


class Subtraction(BinaryOp):
    neutral_value = 0.0
    overpowering_value = None
    accepted_types = (float,)


class Addition(BinaryOp):
    neutral_value = 0.0
    overpowering_value = None
    accepted_types = (float,)


class Multiplication(BinaryOp):
    neutral_value = 1.0
    overpowering_value = 0.0
    accepted_types = (float,)


class Division(BinaryOp):
    neutral_value = 1.0
    overpowering_value = None
    accepted_types = (float,)

    def dtype(self, context):
        return float


#############
# Variables #
#############

class Variable(Expr):
    def __init__(self, name, dtype=None):
        self.name = name
        self._dtype = dtype

    def traverse(self, context):
        yield self

    def __str__(self):
        return self.name
    __repr__ = __str__
    as_string = __str__

    def as_simple_expr(self, context):
        return self

    def simplify(self):
        return self

    def show(self, indent):
        print(indent, self.name)

    def collect_variables(self, context):
        return set([self.name])

    def dtype(self, context):
        if self._dtype is None and self.name in context:
            type_ = context[self.name].dtype.type
            return normalize_type(type_)
        else:
            return self._dtype


class ShortLivedVariable(Variable):
    def collect_variables(self, context):
        return set()


class GlobalVariable(Variable):
    def __init__(self, tablename, name, dtype=None):
        Variable.__init__(self, name, dtype)
        self.tablename = tablename

    #XXX: inherit from EvaluableExpression?
    def as_simple_expr(self, context):
        result = self.evaluate(context)
        period = self._eval_key(context)
        if isinstance(period, int):
            tmp_varname = '__%s_%s' % (self.name, period)
            if tmp_varname in context:
                # should be consistent but nan != nan
                assert result != result or context[tmp_varname] == result
            else:
                context[tmp_varname] = result
        else:
            tmp_varname = get_tmp_varname()
            context[tmp_varname] = result
        return Variable(tmp_varname)

    def _eval_key(self, context):
        return context['period']

    def evaluate(self, context):
        key = self._eval_key(context)
        globals_data = context['__globals__']
        globals_table = globals_data[self.tablename]

        #TODO: this row computation should be encapsulated in the
        # globals_table object and the index column should be configurable
        colnames = globals_table.dtype.names
        if 'period' in colnames or 'PERIOD' in colnames:
            try:
                globals_periods = globals_table['PERIOD']
            except ValueError:
                globals_periods = globals_table['period']
            base_period = globals_periods[0]
            row = key - base_period
        else:
            row = key
        if self.name not in globals_table.dtype.fields:
            raise Exception("Unknown global: %s" % self.name)
        column = globals_table[self.name]
        numrows = len(column)
        missing_value = get_missing_value(column)
        if isinstance(row, np.ndarray) and row.shape:
            out_of_bounds = (row < 0) | (row >= numrows)
            row[out_of_bounds] = -1
            return np.where(out_of_bounds, missing_value, column[row])
        else:
            out_of_bounds = (row < 0) or (row >= numrows)
            return column[row] if not out_of_bounds else missing_value

    def __getitem__(self, key):
        return SubscriptedGlobal(self.tablename, self.name, self._dtype, key)

    def collect_variables(self, context):
        #FIXME: this is a quick hack to make "othertable" work.
        # We should return prefixed variable instead.
        if self.tablename != 'periodic':
            return set()
        else:
            return Variable.collect_variables(self, context)


class SubscriptedGlobal(GlobalVariable):
    def __init__(self, tablename, name, dtype, key):
        GlobalVariable.__init__(self, tablename, name, dtype)
        self.key = key

    def __str__(self):
        return '%s[%s]' % (self.name, self.key)
    __repr__ = __str__

    def _eval_key(self, context):
        return expr_eval(self.key, context)


class GlobalArray(Variable):
    def __init__(self, name, dtype=None):
        Variable.__init__(self, name, dtype)

    def as_simple_expr(self, context):
        globals_data = context['__globals__']
        result = globals_data[self.name]
        #XXX: maybe I should just use self.name?
        tmp_varname = '__%s' % self.name
        if tmp_varname in context:
            assert context[tmp_varname] == result
        context[tmp_varname] = result
        return Variable(tmp_varname)


class GlobalTable(object):
    def __init__(self, name, fields):
        '''fields is a list of tuples (name, type)'''

        self.name = name
        self.fields = fields
        self.fields_map = dict(fields)

    def __getattr__(self, key):
        return GlobalVariable(self.name, key, self.fields_map[key])

    def traverse(self, context):
        yield self

    def __str__(self):
        #XXX: print (a subset of) data instead?
        return 'Table(%s)' % ', '.join([name for name, _ in self.fields])
    __repr__ = __str__
