from __future__ import division

import re

from expr import add_context, Variable


import expr
import alignment
import matching
import properties
import actions
import regressions
import links
import tfunc

functions = {}
for module in (expr, alignment, matching, properties, actions, regressions,
               links, tfunc):
    functions.update(module.functions)

and_re = re.compile('([ )])and([ (])')
or_re = re.compile('([ )])or([ (])')
not_re = re.compile(r'([ (=]|^)not(?=[ (])')


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


def parse(s, globals=None, conditional_context=None, expression=True,
          autovariables=False):
    if not isinstance(s, basestring):
        return s

    # this prevents any function named something ending in "if"
    str_to_parse = s.replace('if(', 'where(')
    str_to_parse = and_re.sub(r'\1&\2', str_to_parse)
    str_to_parse = or_re.sub(r'\1|\2', str_to_parse)
    str_to_parse = not_re.sub(r'\1~', str_to_parse)

    mode = 'eval' if expression else 'exec'
    try:
        c = compile(str_to_parse, '<expr>', mode)
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
        varnames = c.co_names
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
            raise add_context(e, s)
    else:
        exec c in context

        # cleanup result
        del context['__builtins__']
        for funcname in functions.keys():
            del context[funcname]
        return context
