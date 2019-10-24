# encoding: utf-8
from __future__ import absolute_import, division, print_function

import sys
import inspect


PY2 = sys.version_info[0] == 2

if PY2:
    basestring = basestring
    bytes = str
    unicode = unicode
    long = long

    import io
    # add support for encoding. Slow on Python2, but that is not a problem given what we do with it.
    open = io.open
    del io
    input = raw_input
    from StringIO import StringIO
    import cPickle as pickle
    from itertools import izip as zip
else:
    basestring = str
    bytes = bytes
    unicode = str
    long = int
    open = open
    input = input
    from io import StringIO
    import pickle
    zip = zip


def with_metaclass(meta, *bases):
    """
    Function from future/utils/__init__.py. License: MIT.

    (initially from jinja2/_compat.py. License: BSD).

    Use it like this::

        class BaseForm(object):
            pass

        class FormType(type):
            pass

        class Form(with_metaclass(FormType, BaseForm)):
            pass

    This requires a bit of explanation: the basic idea is to make a
    dummy metaclass for one level of class instantiation that replaces
    itself with the actual metaclass.  Because of internal type checks
    we also need to make sure that we downgrade the custom metaclass
    for one level to something closer to type (that's why __call__ and
    __init__ comes back from type etc.).

    This has the advantage over six.with_metaclass of not introducing
    dummy classes into the final MRO.
    """

    class metaclass(meta):
        __call__ = type.__call__
        __init__ = type.__init__

        def __new__(cls, name, this_bases, d):
            if this_bases is None:
                return type.__new__(cls, name, (), d)
            return meta(name, bases, d)

    return metaclass('temporary_class', None, {})


def csv_open(filename, mode='r'):
    assert 'b' not in mode and 't' not in mode
    if PY2:
        return open(filename, mode + 'b')
    else:
        return open(filename, mode, newline='', encoding='utf8')


if sys.version_info[:2] >= (3, 3):
    import inspect

    # we don't use inspect.getargspec/getfullargspec directly because ...
    def getargspec(func):
        sig = inspect.signature(func)
        Parameter = inspect.Parameter

        def params(sig, kind):
            return [p for p in sig.parameters.values() if p.kind == kind]

        pos_or_kw = params(sig, Parameter.POSITIONAL_OR_KEYWORD)
        args = [p.name for p in pos_or_kw]
        varargs = [p.name for p in params(sig, Parameter.VAR_POSITIONAL)]
        varargs = varargs[0] if varargs else None
        varkw = [p.name for p in params(sig, Parameter.VAR_KEYWORD)]
        varkw = varkw[0] if varkw else None
        defaults = [p.default for p in pos_or_kw if p.default is not Parameter.empty]
        defaults = defaults if defaults else None
        return inspect.ArgSpec(args, varargs, varkw, defaults)
else:
    getargspec = inspect.getargspec
