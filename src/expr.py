import re

import numpy as np

try:
    import numexpr
#    numexpr.set_num_threads(1)
    evaluate = numexpr.evaluate
except ImportError:
    eval_context = [(name, getattr(np, name))
                    for name in ('where', 'exp', 'log', 'abs')]
    eval_context.extend([('False', False), ('True', True)])

    def evaluate(expr, globals, locals=None):
        complete_globals = {}
        complete_globals.update(globals)
        if hasattr(globals, 'extra'):
            complete_globals.update(globals.extra)
        if locals is not None:
            if isinstance(locals, np.ndarray):
                for fname in locals.dtype.fields:
                    complete_globals[fname] = locals[fname]
            else:
                complete_globals.update(locals)
        complete_globals.update(eval_context)
        return eval(expr, complete_globals, {})

type_to_idx = {bool: 0, np.bool_: 0, 
               int: 1, np.int32: 1, np.intc: 1, np.int64: 1,
               float:2, np.float64: 2}
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

def coerce_types(context, *args):
    dtype_indices = [type_to_idx[dtype(arg, context)] for arg in args]
    return idx_to_type[max(dtype_indices)]

def as_string(expr, context):
    if isinstance(expr, Expr):
        return expr.as_string(context)
    else:
        return expr

def dtype(expr, context):
    if isinstance(expr, Expr):
        return expr.dtype(context)
    else:
        # standardise type
        return idx_to_type[type_to_idx[type(expr)]]

def collect_variables(expr, context):
    if isinstance(expr, Expr):
        return expr.collect_variables(context)
    else:
        return set()

def expr_eval(expr, context):
    if isinstance(expr, Expr):
        return expr.eval(context)
    else:
        return expr


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

    # makes sure we dont use "normal" python logical operators
    # (and, or, not) 
    def __nonzero__(self):
        raise NotImplementedError()

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

    def eval(self, context):
        s = self.as_string(context)
        r = context.get(s)
        if r is not None:
            return r
#        usual_len = None
#        for k in context.keys():
#            value = context[k]
#            if isinstance(value, np.ndarray):
#                if usual_len is not None and len(value) != usual_len: 
#                    raise Exception('incoherent array lengths: %s''s is %d '
#                                    'while the len of others is %d' %
#                                    (k, len(value), usual_len))
#
#                usual_len = len(value)
        try:
#            dt = self.dtype(context) 
            return evaluate(s, context, {})
        except Exception, e:
            msg = e.args[0] if e.args else ''
            cls = e.__class__
            raise cls("%s\n%s" % (s, msg))


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
        self.expr.show(indent+'    ')
        
    def as_string(self, context):
        return "(%s%s)" % (self.op, self.expr.as_string(context))

    def __str__(self):
        return "(%s%s)" % (self.op, self.expr)
    __repr__ = __str__

    def collect_variables(self, context):
        return self.expr.collect_variables(context)

    def dtype(self, context):
        return dtype(self.expr, context)


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
            self.expr1.show(indent=indent+'    ')
        else:
            print indent+'    ', self.expr1
        if isinstance(self.expr2, Expr):
            self.expr2.show(indent=indent+'    ')
        else:
            print indent+'    ', self.expr2

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
            assert dt is bool, \
                   "operands to logical operators need to be boolean but " \
                   "%s is %s" % (expr, dt)
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


class Variable(Expr):
    def __init__(self, name, dtype=None, value=None):
        self.name = name
        self._dtype = dtype
        self.value = value

    def __str__(self):
        if self.value is None:
            return self.name
        else:
            return str(self.value)
    __repr__ = __str__
        
    def as_string(self, context):
        return self.__str__()

    def simplify(self):
        if self.value is None:
            return self
        else:
            return self.value

    def show(self, indent):
        value = "[%s]" % self.value if self.value is not None else ''
        print indent, self.name, value
        
    def collect_variables(self, context):
        return set([self.name])
    
    def dtype(self, context):
        if self._dtype is None and self.name in context:
            dt = context[self.name].dtype.type
            # standardise type
            return idx_to_type[type_to_idx[dt]]
        else:
            return self._dtype

    
class SubscriptableVariable(Variable):
    def __getitem__(self, key):
        return SubscriptedVariable(self.name, self._dtype, key)


class SubscriptedVariable(Variable):
    def __init__(self, name, dtype, key):
        Variable.__init__(self, name, dtype)
        self.key = key
    
    def __str__(self):
        return '%s[%s]' % (self.name, self.key)
    __repr__ = __str__

    def eval(self, context):
        raise NotImplementedError
    
#        globals = context['globals']
#        period = expr_eval(self.key, context)
#        base_period = globals['period'][0]
#        period_idx = period - base_period
#        if self.name not in globals.dtype.fields:
#            raise Exception("Unknown global: %s" % self.name)
#        return globals[self.name][period_idx]


class VirtualArray(object):
    def __getattr__(self, key):
        return Variable(key)


class Where(Expr):
    func_name = 'if'

    def __init__(self, cond, iftrue, iffalse):
        self.cond = cond
        self.iftrue = iftrue
        self.iffalse = iffalse
        
    def as_string(self, context):
        cond = as_string(self.cond, context)
        
        # filter is stored as an unevaluated expression
        filter = context.get('__filter__')
        if filter is None:
            context['__filter__'] = self.cond
        else:
            context['__filter__'] = filter & self.cond

        iftrue = as_string(self.iftrue, context)
        
        if filter is None:
            context['__filter__'] = ~self.cond
        else:
            context['__filter__'] = filter & ~self.cond
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

and_re = re.compile('([ )])and([ (])')
or_re = re.compile('([ )])or([ (])')
not_re = re.compile(r'([ (=]|^)not(?=[ (])')

def parse(s, globals=None, conditional_context=None, expression=True,
          autovariables=False):
    # this prevents any function named something ending in "if"
    str_to_parse = s.replace('if(', 'where(')
    str_to_parse = and_re.sub(r'\1&\2', str_to_parse)
    str_to_parse = or_re.sub(r'\1|\2', str_to_parse)
    str_to_parse = not_re.sub(r'\1~', str_to_parse)
    
    mode = 'eval' if expression else 'exec'
    try:
        c = compile(str_to_parse, '<expr>', mode)
    except Exception, e:
        msg = e.args[0] if e.args else ''
        cls = e.__class__
        raise cls("%s\n%s" % (s, msg))

    # Instances of this class have attributes filename, lineno, offset and text
    # for easier access to the details. str() of the exception instance returns
    # only the message.    

    context = {'False': False,
               'True': True,
               'nan': float('nan')}

    if autovariables:
        varnames = c.co_names
        context.update((name, Variable(name)) for name in varnames)

    if conditional_context is not None:
        for var in c.co_names:
            if var in conditional_context:
                context.update(conditional_context[var])

    context.update(functions)
    if globals is not None:
        context.update(globals)

    context['__builtins__'] = None
    if expression:
        try:
            return eval(c, context)
        #IOError and such. Those are clearer when left unmodified.
        except EnvironmentError:  
            raise
        except Exception, e:
            msg = e.args[0] if e.args else ''
            cls = e.__class__
            raise cls("%s\n%s" % (s, msg))
    else:
        exec c in context
    
        # cleanup result
        del context['__builtins__']
        for funcname in functions.keys():
            del context[funcname]
        return context
    
        
if __name__ == '__main__':
    expr = "0.4893 * a1" \
           "+ 0.0131 * a1 ** 2" \
           "+ 0.0467 * (a1 - a2)" \
           "- 0.0189 * (a1 - a2) ** 2" \
           "- 0.9087 * (w1 & ~w2)" \
           "- 1.3286 * (~w1 & w2)" \
           "- 0.7939 * ((e1==3) & (e2==4))" \
           "- 1.4128 * ((e1==2) & (e2==3))"

    fnames = ('a1', 'w1', 'e1', 'a2', 'w2', 'e2')
    d = dict((name, Variable(name)) for name in fnames)
    symbolic_expr = eval(expr, d)
    d['w2'].value = True
    d['a2'].value = 50
    d['e2'].value = 4
#    symbolic_expr.show()
    opt_expr = symbolic_expr.simplify()
    print opt_expr
    