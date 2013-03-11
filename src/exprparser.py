from __future__ import division

import ast

from expr import Variable
from utils import add_context


import actions
import aggregates
import alignment
import groupby
import links
import matching
import exprmisc
import regressions
import tfunc

functions = {}
for module in (actions, aggregates, alignment, groupby, links, matching,
               exprmisc, regressions, tfunc):
    functions.update(module.functions)


#class Token(object):
#    def __init__(self, name):
#        self.name = name
#        self.args = None
#        self.kwargs = None
#        self.attr = None
#
#    def __call__(self, *args, **kwargs):
#        self.args = args
#        self.kwargs = kwargs
#
#    def __getattr__(self, key):
#        self.attr = key


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

    def visit_Not(self, node):
        return ast.Invert()


def parse(s, globals=None, conditional_context=None, interactive=False,
          autovariables=False):
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
#    if interactive:
    if False:
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
        assert len(body) == 1 and isinstance(body[0], ast.Expr)
        to_compile = [('eval', ast.Expression(body[0].value))]

    try:
        to_eval = [(mode, compile(code, '<expr>', mode))
                   for mode, code in to_compile]
    except SyntaxError:
        # SyntaxError are clearer if left unmodified since they already contain
        # the faulty string

        # Instances of this class have attributes filename, lineno, offset and
        # text for easier access to the details. str() of the exception
        # instance returns only the message.
        raise
    except Exception, e:
        raise add_context(e, s)

    context = {'False': False,
               'True': True,
               'nan': float('nan')}

    if autovariables:
        for _, code in to_eval:
            varnames = code.co_names
            context.update((name, Variable(name)) for name in varnames)
#        context.update((name, Token(name)) for name in varnames)

    #FIXME: this whole conditional context feature is a huge hack.
    # It relies on the link target not having the same fields/links
    # than the local entity (or not using them).
    # A collision will only occur rarely but it will make it all the more
    # frustrating for the user. The only solution I can see is to split the
    # parsing into two distinct phases:
    # 1) parse using a dummy evaluation context with: all names are bound
    #    to dummy objects with an overridden __getattr__ and __call__
    #    methods which simply store their args without any check.
    # 2) evaluate/convert that to expression objects and check that each
    #    field/link actually exist in the context it is used.
    # the Token class above was my first naive try at doing that, however,
    # it's not as easy as I first thought because functions need to be
    # delayed too.
    if conditional_context is not None:
        for _, code in to_eval:
            for var in code.co_names:
                if var in conditional_context:
                    context.update(conditional_context[var])

    context.update(functions)
    if globals is not None:
        context.update(globals)

    context['__builtins__'] = None
    for mode, compiled_code in to_eval:
        if mode == 'exec':
            #XXX: use "add_context" on exceptions?
            exec compiled_code in context

            # cleanup result. I tried different things to not get the context
            # "polluted" by builtins but could not achieve that, so I cleanup
            # after the fact.
            del context['__builtins__']
            for funcname in functions.keys():
                del context[funcname]
        else:
            assert mode == 'eval'
            try:
                return eval(compiled_code, context)
            # IOError and such. Those are clearer when left unmodified.
            except EnvironmentError:
                raise
            except Exception, e:
                raise add_context(e, s)

