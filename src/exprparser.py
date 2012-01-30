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

functions = {}
for module in (expr, alignment, matching, properties, actions, regressions,
               links):
    functions.update(module.functions)

and_re = re.compile('([ )])and([ (])')
or_re = re.compile('([ )])or([ (])')
not_re = re.compile(r'([ (=]|^)not(?=[ (])')


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
