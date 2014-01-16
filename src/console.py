from __future__ import print_function

import sys

import numpy as np

import config
from entities import global_variables
from expr import expr_eval, Variable
from exprtools import parse
from context import EntityContext
from registry import entity_registry
from process import Process

entity_required = \
    "current entity is not set. It is required to set one using " \
    "the 'entity' command before evaluating any query"

period_required = \
    "current period is not set. It is required to set one using " \
    "the 'period' command before evaluating any query"

help_template = """
%s
    help:            print this help
    q[uit] or exit:  quit the program
    %s
    entities:        list the available entities
    entity [name]:   set the current entity to another entity
    periods:         list the available periods for the current entity
    period [period]: set the current period
    fields [entity]: list the fields of that entity (or the current entity)
    globals:         list the available globals

    show is implicit on all commands
"""


class InvalidPeriod(ValueError):
    pass


class Console(object):
    def __init__(self, entity=None, period=None,
                 globals_def=None, globals_data=None):
        if entity is None and len(entity_registry) == 1:
            entity = entity_registry.values()[0]
        self.entity = entity
        if period is None and entity is not None:
            period = EntityContext(entity).list_periods()[-1]
        self.period = period
        self.globals_def = globals_def
        self.globals_data = globals_data

        global_context = {'__globals__': global_variables(globals_def)}
        parsing_context = global_context.copy()
        parsing_context.update((entity.name, entity.all_symbols(global_context))
                               for entity in entity_registry.itervalues())
        parsing_context['__entity__'] = entity.name
        self.parsing_context = parsing_context

    def list_entities(self):
        ent_names = [repr(k) for k in entity_registry.keys()]
        print("available entities:", ', '.join(ent_names))

    def get_entity(self, name):
        try:
            return entity_registry[name]
        except KeyError:
            print("entity '%s' does not exist" % name)
            self.list_entities()
            return None

    def _display_entity(self):
        if self.entity is None:
            print(entity_required)
        else:
            print("current entity set to", self.entity.name)

    def set_entity(self, name):
        entity = self.get_entity(name)
        if entity is not None:
            self.parsing_context['__entity__'] = entity.name
            self.entity = entity
            self._display_entity()
            if self.period is not None:
                if self.period not in self._list_periods():
                    self.period = None
                    raise InvalidPeriod("entity '%s' has no data for period %d"
                                        % (self.entity.name, self.period))

    def _list_periods(self):
        return EntityContext(self.entity).list_periods()

    def list_periods(self):
        if self.entity is None:
            raise Exception(entity_required)

        periods = self._list_periods()
        print("available periods: %s" % ', '.join(str(p) for p in periods))

    def _display_period(self):
        if self.period is None:
            print(period_required)
        else:
            print("current period set to", self.period)

    def set_period(self, period):
        try:
            period = int(period)
            if self.entity is not None and period not in self._list_periods():
                raise InvalidPeriod("entity '%s' has no data for period %d"
                                    % (self.entity.name, period))
            self.period = period
            self._display_period()
        except InvalidPeriod:
            raise
        except ValueError:
            print("invalid period")

    def list_fields(self, ent_name=None):
        if ent_name is None:
            entity = self.entity
            if entity is None:
                raise Exception(entity_required)
        else:
            entity = self.get_entity(ent_name)
            if entity is None:
                return
        print("fields:", ', '.join(name for name, _ in entity.fields))

    def list_globals(self):
        print("globals:")
        for k, v in self.globals_data.iteritems():
            if v.dtype.names:
                details = ": " + ", ".join(v.dtype.names)
            else:
                details = ""
            print("* %s%s" % (k, details))

    def execute(self, s):
        entity = self.entity
        if entity is None:
            raise Exception(entity_required)

        period = self.period
        if period is None:
            raise Exception(period_required)

        parsing_context = self.parsing_context
        entity_context = parsing_context[parsing_context['__entity__']]

        # add all currently defined temp_variables because otherwise
        # local variables (defined within a procedure) wouldn't be available
        entity_context.update((name, Variable(name))
                              for name in entity.temp_variables.keys())
        expr = parse(s, parsing_context, interactive=True)
        ctx = EntityContext(entity, {'period': period,
                                     'nan': np.nan,
                                     '__globals__': self.globals_data})
        if isinstance(expr, Process):
            expr.run(ctx)
            print("done.")
        else:
            return expr_eval(expr, ctx)

    def run(self, debugger=False):
        if debugger:
            help_text = help_template % (
                "Commands:", """
    s[tep]:          execute the next process
    r[esume]:        resume normal execution
""")
        else:
            help_text = help_template % (
                "Welcome to LIAM2 interactive console.",
                "")
        if not debugger:
            print(help_text)
        self._display_entity()
        self._display_period()
        while True:
            try:
                s = raw_input('>>> ').strip()
                if s == '':
                    continue
                elif s == 'help':
                    print(help_text)
                elif s == 'entity':
                    self._display_entity()
                elif s.startswith('entity ') and s[7:]:
                    self.set_entity(s[7:])
                elif s == 'entities':
                    self.list_entities()
                elif s == 'period':
                    self._display_period()
                elif s.startswith('period ') and s[7:]:
                    self.set_period(s[7:])
                elif s == 'periods':
                    self.list_periods()
                elif s.startswith('fields'):
                    if s[7:]:
                        self.list_fields(s[7:])
                    else:
                        self.list_fields()
                elif s == "globals":
                    self.list_globals()
                elif debugger and s in ('s', 'step'):
                    return 'step'
                elif debugger and s in ('r', 'resume'):
                    return
                elif s in ('q', 'quit', 'exit'):
                    sys.exit()
                else:
                    res = self.execute(s)
                    if res is not None:
                        print(res)
            except Exception, e:
                if config.debug:
                    import traceback
                    traceback.print_exc()
                msg = str(e)
                #noinspection PyArgumentList
                lines = msg.splitlines()
                if len(lines) > 1:
                    msg = '\n'.join(lines[1:])
                print(msg)
