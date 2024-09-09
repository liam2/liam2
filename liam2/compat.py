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
    #     'kwonlyargs' is a list of keyword-only parameter names.
    kwonlyargs = []
    #     'kwonlydefaults' is a dictionary mapping names from kwonlyargs to defaults.
    kwonlydefaults = {}
    #     'annotations' is a dictionary mapping parameter names to annotations.
    annotations = {}
    return inspect.FullArgSpec(args, varargs, varkw, defaults,
                               kwonlyargs, kwonlydefaults, annotations)
