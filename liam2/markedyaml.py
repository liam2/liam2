# coding: utf-8
import yaml
from yaml import Loader, SafeLoader

_marked_classes = {t: type("Marked" + t.__name__.title(), (t,),
                           {'start': None, 'end': None})
                   for t in (dict, list, str)}

def _wrap(method_name):
    def wrapped(constructor, node):
        result = getattr(constructor, method_name)(node)
        result = _marked_classes[type(result)](result)
        result.start, result.end = node.start_mark, node.end_mark
        return result
    return wrapped


def make_marked_loader(name, base):
    cls = type(name, (base,), {})
    for tag, method_name in [('map', 'construct_mapping'),
                             ('seq', 'construct_sequence'),
                             ('str', 'construct_yaml_str')]:
        cls.add_constructor(u'tag:yaml.org,2002:' + tag, _wrap(method_name))
    return cls

MarkedLoader = make_marked_loader('MarkedLoader', Loader)
SafeMarkedLoader = make_marked_loader('SafeMarkedLoader', SafeLoader)

def marked_load(stream):
    return yaml.load(stream, MarkedLoader)

def safe_marked_load(stream):
    return yaml.load(stream, SafeMarkedLoader)
