import sys

import numpy as np

from expr import expr_eval, parse
import entities


class AutoflushFile(object):
    def __init__(self, f):
        self.f = f

    def write(self, s):
        self.f.write(s)
        self.f.flush()


class Console(object):
    def __init__(self, entity=None, period=None):
        self.entity = entity
        self.period = period
    
    def list_entities(self):
        ent_names = [repr(k) for k in entities.entity_registry.keys()]
        print "available entities:", ', '.join(ent_names) 

    def list_fields(self, ent_name=None):
        if ent_name is None:
            entity = self.entity
            if entity is None:
                print 'entity not set, use "entity [name]" to set ' \
                      'the current entity'
                return
        else:
            entity = self.get_entity(ent_name)
            if entity is None:
                return
        print "fields:", ', '.join(name for name, _ in entity.fields) 

    def get_entity(self, name):        
        try:
            return entities.entity_registry[name]
        except KeyError:
            print "entity '%s' does not exist" % name
            self.list_entities()
            
    def set_entity(self, name):
        entity = self.get_entity(name)
        if entity is not None:
            self.entity = entity 
            print "current entity set to", name
            
    def set_period(self, period):
        try:
            period = int(period)
            self.period = period
            print "current period set to", period
        except:
            print "invalid period"
        
    def execute(self, s):
        entity = self.entity
        if entity is None:
            raise Exception('entity not set, use "entity [name]" to set '
                            'the current entity')
        period = self.period
        if period is None:
            period = entity.array['period'][0]
            self.set_period(period)
        vars = entity.variables
        cond_context = entity.conditional_context
        expr = parse(s, vars, cond_context)
        ctx = entities.EntityContext(entity, {'period': period})
        return expr_eval(expr, ctx)
    
    def run(self, debugger=False):
        sys.stdout = AutoflushFile(sys.stdout)
        if debugger:
            help = """
Commands:
    help:            print this help
    s[tep]:          execute the next process 
    r[esume]:        resume normal execution
    
    entity [name]:   set the current entity to another entity
    period [period]: set the current period
    fields [entity]: list the fields of that entity (or the current entity)
    
    show is implicit on all commands
"""
        else:
            help = """
Welcome to LIAM interactive console.        
    help:            print this help
    q[uit] or exit:  quit the console
    
    entity [name]:   set the current entity (this is required before any query)
    period [period]: set the current period (if not set, uses the last period 
                     simulated)
    fields [entity]: list the fields of that entity (or the current entity)
    
    show is implicit on all commands
"""
        if not debugger:
            print help
        while True:
            s = raw_input('>>> ').strip()
            if s == '':
                continue
            elif s == 'help':
                print help
            elif s.startswith('entity'):
                if s[7:]:
                    self.set_entity(s[7:])
                else:
                    self.list_entities()
            elif s.startswith('period '):
                self.set_period(s[7:])
            elif s.startswith('fields'):
                if s[7:]:
                    self.list_fields(s[7:])
                else:
                    self.list_fields()
            elif debugger and s in ('s', 'step'):
                sys.stdout = sys.stdout.f
                return 'step'
            elif debugger and s in ('r', 'resume') or \
                 not debugger and s in ('q', 'quit', 'exit'):
                sys.stdout = sys.stdout.f
                return
            else:
                try:
                    res = self.execute(s)
                    if res is not None:
                        print res
                except Exception, e:
#                    import traceback
#                    traceback.print_exc()
                    msg = str(e)
                    lines = msg.splitlines()
                    if len(lines) > 1:
                        msg = '\n'.join(lines[1:])
                    print msg
