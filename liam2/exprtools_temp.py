# -*- coding: utf-8 -*-


from __future__ import division, print_function

import ast
import types

from expr import UnaryOp, BinaryOp, LogicalOp, ComparisonOp
from utils import add_context

import actions
import aggregates
import alignment
import charts
import groupby
import links
import matching
import exprmisc
import exprrandom
import regressions
import tfunc


functions = {}
for module in (actions, aggregates, alignment, charts, groupby, links, matching,
               exprmisc, exprrandom, regressions, tfunc):
    functions.update(module.functions)


def to_ast(expr, context):
    # print("to_ast(%s) in entity: %s" % (expr, context['__entity__']))
    if isinstance(expr, Node):
        return expr.to_ast(context)
    elif isinstance(expr, list):
        return [to_ast(e, context) for e in expr]
    elif isinstance(expr, tuple):
        return tuple([to_ast(e, context) for e in expr])
    elif isinstance(expr, dict):
        return {k: to_ast(v, context) for k, v in expr.iteritems()}
    elif isinstance(expr, slice):
        return slice(to_ast(expr.start, context),
                     to_ast(expr.stop, context),
                     to_ast(expr.step, context))
    else:
        return expr


def unaryop(opname):
    def func(self):
        return UnaryOpNode(opname, self)
    return func


def binop(opname, kind='binary', reversed=False):
    def op(self, other):
        classes = {'binary': BinaryOpNode,
                   'logical': LogicalOpNode,
                   'comparison': ComparisonOpNode}
        class_ = classes[kind]
        return class_(opname, other, self) if reversed \
                                           else class_(opname, self, other)
    return op


class Node(object):
    # make sure we do not use "normal" python logical operators (and, or, not)
    def __nonzero__(self):
        raise Exception("Improper use of boolean operators, you probably "
                        "forgot parenthesis around operands of an 'and' or "
                        "'or' expression. The complete expression cannot be "
                        "displayed but it contains: '%s'." % str(self))

    __lt__ = binop('<', 'comparison')
    __le__ = binop('<=', 'comparison')
    __eq__ = binop('==', 'comparison')
    __ne__ = binop('!=', 'comparison')
    __gt__ = binop('>', 'comparison')
    __ge__ = binop('>=', 'comparison')

    __add__ = binop('+')
    __radd__ = binop('+', reversed=True)
    __sub__ = binop('-')
    __rsub__ = binop('-', reversed=True)
    __mul__ = binop('*')
    __rmul__ = binop('*', reversed=True)

    __div__ = binop('/')
    __rdiv__ = binop('/', reversed=True)
    __truediv__ = binop('/')
    __rtruediv__ = binop('/', reversed=True)
    __floordiv__ = binop('//')
    __rfloordiv__ = binop('//', reversed=True)

    __mod__ = binop('%')
    __rmod__ = binop('%', reversed=True)
    # FIXME
    __divmod__ = binop('divmod')
    __rdivmod__ = binop('divmod', reversed=True)
    __pow__ = binop('**')
    __rpow__ = binop('**', reversed=True)

    __lshift__ = binop('<<')
    __rlshift__ = binop('<<', reversed=True)
    __rshift__ = binop('>>')
    __rrshift__ = binop('>>', reversed=True)

    __and__ = binop('&', 'logical')
    __rand__ = binop('&', 'logical', reversed=True)
    __xor__ = binop('^', 'logical')
    __rxor__ = binop('^', 'logical', reversed=True)
    __or__ = binop('|', 'logical')
    __ror__ = binop('|', 'logical', reversed=True)

    __neg__ = unaryop('-')
    __pos__ = unaryop('+')
    __abs__ = unaryop('abs')
    __invert__ = unaryop('~')

    def __getitem__(self, key):
        return SubscriptNode(self, key)

    def __getattr__(self, key):
        assert isinstance(key, str)
        return AttrNode(self, key)

    def __call__(self, *args, **kwargs):
        return CallNode(self, *args, **kwargs)


class Symbol(Node):
    # Variable
    # GlobalVariable
    # function
    # link
    # prefixinglink (other.)
    # macro (should be handled before)
    def __init__(self, name):
        self.name = name

    def to_ast(self, context):
        name = self.name
        entity_context = context[context['__entity__']]
        globals_context = context['__globals__']

        if name in entity_context:
            return entity_context[name]
        elif name in globals_context:
            return globals_context[name]
        else:
            raise NameError("name '{}' is not defined".format(name))

    def __str__(self):
        return self.name


class SubscriptNode(Node):
    # SubscriptedGlobal
    # SubscriptedArray
    def __init__(self, node, key):
        self.node = node
        self.key = key

    def to_ast(self, context):
        return to_ast(self.node, context)[to_ast(self.key, context)]

    def __str__(self):
        return '%s[%s]' % (self.node, self.key)


class AttrNode(Node):
    # links
    # array attributes
    # GlobalTable (othertable.xyz)
    def __init__(self, node, key):
        self.node = node
        assert isinstance(key, str)
        self.key = key

    def to_ast(self, context):
        # no need to use to_ast on the key since it is simply a string
        return getattr(to_ast(self.node, context), self.key)

    def __str__(self):
        return '%s.%s' % (self.node, self.key)


class UnaryOpNode(Node):
    def __init__(self, op, node):
        self.op = op
        self.node = node

    def to_ast(self, context):
        return UnaryOp(self.op, to_ast(self.node, context))


class BinaryOpNode(Node):
    ast_class = BinaryOp

    def __init__(self, op, expr1, expr2):
        self.op = op
        self.expr1 = expr1
        self.expr2 = expr2

    def to_ast(self, context):
        return self.ast_class(self.op,
                              to_ast(self.expr1, context),
                              to_ast(self.expr2, context))


class LogicalOpNode(BinaryOpNode):
    ast_class = LogicalOp


class ComparisonOpNode(BinaryOpNode):
    ast_class = ComparisonOp


class CallNode(Node):
    def __init__(self, callable_, *args, **kwargs):
        self.callable_ = callable_
        self.args = args
        self.kwargs = kwargs

    def to_ast(self, context):
        callable_ast = to_ast(self.callable_, context)
        if callable_ast is alignment.AlignmentAbsoluteValues:
            link_symbol = self.kwargs.get('link', None)
            # secondary_axis is only valid in combination with link, but
            # the error is reported to the user downstream
            if link_symbol is not None:
                link = to_ast(link_symbol, context)
                local_context = context.copy()
                local_context['__entity__'] = link._target_entity_name
                axis_symbol = self.kwargs.get('secondary_axis', None)
                if axis_symbol is not None:
                    self.kwargs['secondary_axis'] = to_ast(axis_symbol,
                                                           local_context)
                expressions = self.kwargs.get('expressions')
                if expressions is not None:
                    self.kwargs['expressions'] = to_ast(expressions,
                                                        local_context)
        # arguments of link methods (M2O.get, O2M.count, ...) need to be
        # evaluated in the context of the target entity
        link = None
        if isinstance(callable_ast, types.MethodType):
            instance = callable_ast.__self__
            if isinstance(instance, links.Link):
                link = instance
            elif isinstance(instance, links.LinkGet):
                # find the link of the deepest LinkGet in the "link chain"
                lv = instance
                while isinstance(lv.target_expr, links.LinkGet):
                    lv = lv.target_expr
                assert isinstance(lv.target_expr, links.Link)
                link = lv.target_expr
        if link is not None:
            local_context = context.copy()
            local_context['__entity__'] = link._target_entity_name
        else:
            local_context = context
        args, kwargs = to_ast((self.args, self.kwargs), local_context)
        return callable_ast(*args, **kwargs)

    def __str__(self):
        return '%s(%s, %s)' % (self.callable_, self.args, self.kwargs)

# household.get(age * sex)
#
# CallNode((AttrNode(Symbol('household'), Symbol('get')),
#           BinaryOpNode('*', Symbol('age'), Symbol('sex')),
#           ()))

# to_ast(Symbol('household')) -> Link
# to_ast(AttrNode(Symbol('household'), Symbol('get')) -> Link.get
# to_ast(example_above) -> Link.get() with args special cased
# =>
# 2 options:
#
# * initially construct LinkSymbol('household') instead of Symbol('household')
#   so that I can special case links at each step (LinkSymbol, LinkAttrNode
#   and LinkMethodCallNode)
# * special case in CallNode: if isinstance(callable_ast, instancemethod) and
#     isinstance(callable_ast.__self__, Link)


# noinspection PyPep8Naming
class BoolToBitTransformer(ast.NodeTransformer):
    def visit_BoolOp(self, node):
        # first transform children of the node
        self.generic_visit(node)

        # using a dict doesn't seem to work
        if isinstance(node.op, ast.And):
            new_op = ast.BitAnd()
        else:
            assert isinstance(node.op, ast.Or)
            new_op = ast.BitOr()
        values = node.values
        right = values.pop()
        while len(values):
            left = values.pop()
            right = ast.copy_location(ast.BinOp(left, new_op, right), node)
        return right

    # noinspection PyUnusedLocal
    def visit_Not(self, node):
        return ast.Invert()


def _parse(s, interactive=False):
    """
    low level parsing function (string -> Node)
    """
    if not isinstance(s, basestring):
        return s

    # this prevents any function named something ending in "if"
    str_to_parse = s.replace('if(', 'where(')
    tree = ast.parse(str_to_parse)
    tree = BoolToBitTransformer().visit(tree)
    body = tree.body

    # disable for now because it is not very useful yet. To be useful, I need
    # to implement:
    # * Expr.__setitem__
    # * keep the same context across several expressions in the interactive
    #   console
    if False and interactive:
        if len(body) == 0:
            to_compile = []
        else:
            # if the last statement is an expression, move it out and
            # use eval() on it instead of exec
            if isinstance(body[-1], ast.Expr):
                to_compile = [('exec', ast.Module(body[:-1])),
                              ('eval', ast.Expression(body[-1].value))]
            else:
                to_compile = [('exec', tree)]
    else:
        print("body: {}".format(body))
        print("body[0]: {}".format(body[0]))
        assert len(body) == 1 and isinstance(body[0], ast.Expr)
        to_compile = [('eval', ast.Expression(body[0].value))]

    to_eval = [(mode, compile(code, '<expr>', mode))
               for mode, code in to_compile]
    context = {'__builtins__': None}
    for _, code in to_eval:
        context.update({name: Symbol(name) for name in code.co_names})

    for mode, compiled_code in to_eval:
        if mode == 'exec':
            exec compiled_code in context

            # cleanup result. I tried different things to not get the context
            # "polluted" by builtins but could not achieve that, so I cleanup
            # after the fact.
            del context['__builtins__']
            for funcname in functions.keys():
                del context[funcname]
        else:
            assert mode == 'eval'
            return eval(compiled_code, context)


def parse(s, context, interactive=False):
    globals_context = {'False': False,
                       'True': True,
                       'nan': float('nan'),
                       'inf': float('inf')}
    globals_context.update(functions)
    globals_context.update(context.get('__globals__', {}))
    # modify in-place
    context['__globals__'] = globals_context
    try:
        node = _parse(s, interactive=interactive)
        return to_ast(node, context)
    except Exception, e:
        add_context(e, "while parsing: " + s)
        raise
