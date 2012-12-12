from __future__ import division
import re
import sys
from collections import Counter

import numpy as np

from utils import gettime
from context import context_length

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


def as_string(expr, context):
    if isinstance(expr, Expr):
        return expr.as_string(context)
    else:
        return expr


def traverse_expr(expr, context):
    if isinstance(expr, Expr):
        return expr.traverse(context)
    else:
        return ()


def dtype(expr, context):
    if isinstance(expr, Expr):
        return expr.dtype(context)
    else:
        return normalize_type(type(expr))


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
        globals_data = context['__globals__']
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
    else:
        return expr


def add_context(exception, s):
    msg = exception.args[0] if exception.args else ''
    encoding = sys.getdefaultencoding()
    expr_str = s.encode(encoding, 'replace')
    msg = "%s\n%s" % (expr_str, msg)
    cls = exception.__class__
    return cls(msg)


class ExplainTypeError(type):
    def __call__(cls, *args, **kwargs):
        try:
            return type.__call__(cls, *args, **kwargs)
        except TypeError, e:
            if hasattr(cls, 'func_name'):
                funcname = cls.func_name
            else:
                funcname = cls.__name__.lower()
            if funcname is not None:
                msg = e.args[0].replace('__init__()', funcname)
            else:
                msg = e.args[0]

            def repl(matchobj):
                needed, given = int(matchobj.group(1)), int(matchobj.group(2))
                return "%d arguments (%d given)" % (needed - 1, given - 1)
            msg = re.sub('(\d+) arguments \((\d+) given\)', repl, msg)
            raise TypeError(msg)


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
        return Substraction('-', self, other)
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
        return Substraction('-', other, self)
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
        s = self.as_string(context)
        r = context.get(s)
        if r is not None:
            return r

        try:
            return evaluate(s, context, {}, truediv='auto')
        except KeyError, e:
            raise add_context(e, s)
        except Exception, e:
            raise

    def as_string(self):
        raise NotImplementedError()

    def __getitem__(self, key):
        #TODO: we should be able to know at "compile" time if this is a
        # scalar or a vector and disallow getitem in case of a scalar
        return SubscriptedExpr(self, key)

    def __getattr__(self, key):
        return ExprAttribute(self, key)


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


class SubscriptedExpr(EvaluableExpression):
    def __init__(self, expr, key):
        self.expr = expr
        self.key = key

    def __str__(self):
        return '%s[%s]' % (self.expr, self.key)
    __repr__ = __str__

    def evaluate(self, context):
        expr = expr_eval(self.expr, context)
        key = expr_eval(self.key, context)
        #XXX: return -1 for out_of_bounds like for globals?
        return expr[key]

    #TODO: implement traverse
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


#XXX: factorize with NumpyProperty?
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


#class IsPresent(Expr):
#    def __init__(self, expr):
#        self.expr = expr
#
#    def simplify(self):
#        dtype = self.expr.dtype()
#        if np.issubdtype(dtype, float):
#            return np.isfinite(values)
#        elif np.issubdtype(dtype, int):
#            return expr != missing_values[int]
#        elif np.issubdtype(dtype, bool):
#            return expr != missing_values[bool]


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
        print indent, self.op
        self.expr.show(indent + '    ')

    def as_string(self, context):
        return "(%s%s)" % (self.op, self.expr.as_string(context))

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

    def as_string(self, context):
        expr1 = as_string(self.expr1, context)
        expr2 = as_string(self.expr2, context)
        return "(%s %s %s)" % (expr1, self.op, expr2)

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
        print indent, self.op
        if isinstance(self.expr1, Expr):
            self.expr1.show(indent=indent + '    ')
        else:
            print indent + '    ', self.expr1
        if isinstance(self.expr2, Expr):
            self.expr2.show(indent=indent + '    ')
        else:
            print indent + '    ', self.expr2

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


class Substraction(BinaryOp):
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


class Variable(Expr):
    def __init__(self, name, dtype=None):
        self.name = name
        self._dtype = dtype

    def traverse(self, context):
        yield self

    def __str__(self):
        return self.name
    __repr__ = __str__

    def as_string(self, context):
        return self.__str__()

    def simplify(self):
        return self

    def show(self, indent):
        print indent, self.name

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
    def as_string(self, context):
        result = self.evaluate(context)
        period = self._eval_period(context)
        if isinstance(period, int):
            tmp_varname = '__%s_%s' % (self.name, period)
            if tmp_varname in context:
                assert context[tmp_varname] == result
            else:
                context[tmp_varname] = result
        else:
            tmp_varname = get_tmp_varname()
            context[tmp_varname] = result
        return tmp_varname

    def _eval_period(self, context):
        return context['period']

    def evaluate(self, context):
        period = self._eval_period(context)
        globals_data = context['__globals__']
        globals_table = globals_data[self.tablename]

        #TODO: this period_idx computation should be encapsulated in the
        # globals_table object
        try:
            globals_periods = globals_table['PERIOD']
        except ValueError:
            globals_periods = globals_table['period']
        period_idx = period - globals_periods[0]

        if self.name not in globals_table.dtype.fields:
            raise Exception("Unknown global: %s" % self.name)
        column = globals_table[self.name]
        num_periods = len(globals_periods)
        missing_value = get_missing_value(column)
        if isinstance(period_idx, np.ndarray):
            out_of_bounds = (period_idx < 0) | (period_idx >= num_periods)
            period_idx[out_of_bounds] = -1
            return np.where(out_of_bounds, missing_value, column[period_idx])
        else:
            out_of_bounds = (period_idx < 0) or (period_idx >= num_periods)
            return column[period_idx] if not out_of_bounds else missing_value

    def __getitem__(self, key):
        return SubscriptedGlobal(self.tablename, self.name, self._dtype, key)


class SubscriptedGlobal(GlobalVariable):
    def __init__(self, tablename, name, dtype, key):
        GlobalVariable.__init__(self, tablename, name, dtype)
        self.key = key

    def __str__(self):
        return '%s[%s]' % (self.name, self.key)
    __repr__ = __str__

    def _eval_period(self, context):
        return expr_eval(self.key, context)


class GlobalArray(Variable):
    def __init__(self, name, dtype=None):
        Variable.__init__(self, name, dtype)

    def as_string(self, context):
        globals_data = context['__globals__']
        result = globals_data[self.name]
        #XXX: maybe I should just use self.name?
        tmp_varname = '__%s' % self.name
        if tmp_varname in context:
            assert context[tmp_varname] == result
        context[tmp_varname] = result
        return tmp_varname


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
        #TODO: print data instead
        return 'Table(%s)' % ', '.join([name for name, _ in self.fields])
    __repr__ = __str__


class Where(Expr):
    func_name = 'if'

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

    def as_string(self, context):
        cond = as_string(self.cond, context)

        # filter is stored as an unevaluated expression
        filter_expr = context.get('__filter__')

        if filter_expr is None:
            context['__filter__'] = self.cond
        else:
            context['__filter__'] = filter_expr & self.cond
        iftrue = as_string(self.iftrue, context)

        if filter_expr is None:
            context['__filter__'] = ~self.cond
        else:
            context['__filter__'] = filter_expr & ~self.cond
        iffalse = as_string(self.iffalse, context)

        context['__filter__'] = None
        return "where(%s, %s, %s)" % (cond, iftrue, iffalse)

    def __str__(self):
        return "if(%s, %s, %s)" % (self.cond, self.iftrue, self.iffalse)
    __repr__ = __str__

    def dtype(self, context):
        assert dtype(self.cond, context) == bool
        return coerce_types(context, self.iftrue, self.iffalse)

    def collect_variables(self, context):
        condvars = collect_variables(self.cond, context)
        iftruevars = collect_variables(self.iftrue, context)
        iffalsevars = collect_variables(self.iffalse, context)
        return condvars | iftruevars | iffalsevars

functions = {'where': Where}
