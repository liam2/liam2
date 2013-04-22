import time
import os
import operator
from collections import defaultdict
import random

import numpy as np
import yaml

from data import H5Data, Void
from entities import Entity
from registry import entity_registry
from utils import (time2str, timed, gettime, validate_dict,
                   field_str_to_type, fields_yaml_to_type)
import console
import config
import expr


def show_top_times(what, times):
    count = len(times)
    print "top %d %s:" % (count, what)
    for name, timing in times:
        print " - %s: %s" % (name, time2str(timing))
    print "total for top %d %s:" % (count, what),
    print time2str(sum(timing for name, timing in times))


def show_top_processes(process_time, count):
    process_times = sorted(process_time.iteritems(),
                           key=operator.itemgetter(1),
                           reverse=True)
    show_top_times('processes', process_times[:count])


def show_top_expr(count=None):
    show_top_times('expressions', expr.timings.most_common(count))


class Simulation(object):
    yaml_layout = {
        'globals': {
            'periodic': [{
                '*': str
            }],
            '*': {
                'fields': [{
                    '*': None  # Or(str, {'type': str, 'initialdata': bool})
                }],
                'type': str
            }
        },
        '#entities': {
            '*': {
                'fields': [{
                    '*': None
                }],
                'links': {
                    '*': {
                        '#type': str,  # Or('many2one', 'one2many')
                        '#target': str,
                        '#field': str
                    }
                },
                'macros': {
                    '*': None
                },
                'processes': {
                    '*': None
                }
            }
        },
        '#simulation': {
            'init': [{
                '*': [str]
            }],
            '#processes': [{
                '*': [None]  # Or(str, [str, int])
            }],
            'random_seed': int,
            '#input': {
                'path': str,
                '#file': str,
                'method': str
            },
            '#output': {
                'path': str,
                '#file': str
            },
            '#periods': int,
            '#start_period': int,
            'skip_shows': bool,
            'timings': bool,
            'assertions': str,
            'default_entity': str
        }
    }

    def __init__(self, globals_def, periods, start_period,
                 init_processes, init_entities, processes, entities,
                 data_source, default_entity=None):
        if 'periodic' in globals_def:
            globals_def['periodic'].insert(0, ('PERIOD', int))

        self.globals_def = globals_def
        self.periods = periods
        self.start_period = start_period
        # init_processes is a list of tuple: (process, 1)
        self.init_processes = init_processes
        self.init_entities = init_entities
        # processes is a list of tuple: (process, periodicity)
        self.processes = processes
        self.entities = entities
        self.data_source = data_source
        self.default_entity = default_entity

        self.stepbystep = False

    @classmethod
    def from_yaml(cls, fpath,
                  input_dir=None, input_file=None,
                  output_dir=None, output_file=None):
        simulation_path = os.path.abspath(fpath)
        simulation_dir = os.path.dirname(simulation_path)
        with open(fpath) as f:
            content = yaml.load(f)
        validate_dict(content, cls.yaml_layout)

        # the goal is to get something like:
        # globals_def = {'periodic': [('a': int), ...],
        #                'MIG': int}
        globals_def = {}
        for k, v in content.get('globals', {}).iteritems():
            # periodic is a special case
            if k == 'periodic':
                type_ = fields_yaml_to_type(v)
            else:
                # "fields" and "type" are synonyms
                type_def = v.get('fields') or v.get('type')
                if isinstance(type_def, basestring):
                    type_ = field_str_to_type(type_def, "array '%s'" % k)
                else:
                    if not isinstance(type_def, list):
                        raise SyntaxError("invalid structure for globals")
                    type_ = fields_yaml_to_type(type_def)
            globals_def[k] = type_

        simulation_def = content['simulation']
        seed = simulation_def.get('random_seed')
        if seed is not None:
            seed = int(seed)
            print "using fixed random seed: %d" % seed
            random.seed(seed)
            np.random.seed(seed)

        periods = simulation_def['periods']
        start_period = simulation_def['start_period']
        config.skip_shows = simulation_def.get('skip_shows', False)
        #TODO: check that the value is one of "raise", "skip", "warn"
        config.assertions = simulation_def.get('assertions', 'raise')
        config.show_timings = simulation_def.get('timings', True)

        input_def = simulation_def['input']
        input_directory = input_dir if input_dir is not None \
                                    else input_def.get('path', '')
        if not os.path.isabs(input_directory):
            input_directory = os.path.join(simulation_dir, input_directory)
        config.input_directory = input_directory

        output_def = simulation_def['output']
        output_directory = output_dir if output_dir is not None \
                                      else output_def.get('path', '')
        if not os.path.isabs(output_directory):
            output_directory = os.path.join(simulation_dir, output_directory)
        config.output_directory = output_directory

        if output_file is None:
            output_file = output_def['file']
        output_path = os.path.join(output_directory, output_file)

        for k, v in content['entities'].iteritems():
            entity_registry.add(Entity.from_yaml(k, v))

        for entity in entity_registry.itervalues():
            entity.check_links()
            entity.parse_processes(globals_def)
            entity.compute_lagged_fields()

        init_def = [d.items()[0] for d in simulation_def.get('init', {})]
        init_processes, init_entities = [], set()
        for ent_name, proc_names in init_def:
            if ent_name not in entity_registry:
                raise Exception("Entity '%s' not found" % ent_name)

            entity = entity_registry[ent_name]
            init_entities.add(entity)
            init_processes.extend([(entity.processes[proc_name], 1)
                                   for proc_name in proc_names])

        processes_def = [d.items()[0] for d in simulation_def['processes']]
        processes, entities = [], set()
        for ent_name, proc_defs in processes_def:
            entity = entity_registry[ent_name]
            entities.add(entity)
            for proc_def in proc_defs:
                # proc_def is simply a process name
                if isinstance(proc_def, basestring):
                    # use the default periodicity of 1
                    proc_name, periodicity = proc_def, 1
                else:
                    proc_name, periodicity = proc_def
                processes.append((entity.processes[proc_name], periodicity))

        method = input_def.get('method', 'h5')

        if method == 'h5':
            if input_file is None:
                input_file = input_def['file']
            input_path = os.path.join(input_directory, input_file)
            data_source = H5Data(input_path, output_path)
        elif method == 'void':
            input_path = None
            data_source = Void(output_path)
        else:
            print method, type(method)

        default_entity = simulation_def.get('default_entity')
        return Simulation(globals_def, periods, start_period,
                          init_processes, init_entities, processes, entities,
                          data_source, default_entity)

    def load(self):
        return timed(self.data_source.load, self.globals_def,
                     entity_registry)

    def run(self, run_console=False):
        start_time = time.time()
        h5in, h5out, globals_data = timed(self.data_source.run,
                                          self.globals_def,
                                          entity_registry,
                                          self.start_period - 1)
#        input_dataset = self.data_source.run(self.globals_def,
#                                             entity_registry)
#        output_dataset = self.data_sink.prepare(self.globals_def,
#                                                entity_registry)
#        output_dataset.copy(input_dataset, self.start_period - 1)
#        for entity in input_dataset:
#            indexed_array = buildArrayForPeriod(entity)

        # tell numpy we do not want warnings for x/0 and 0/0
        np.seterr(divide='ignore', invalid='ignore')

        process_time = defaultdict(float)
        period_objects = {}

        def simulate_period(period_idx, period, processes, entities,
                            init=False):
            print "\nperiod", period
            if init:
                for entity in entities:
                    print "  * %s: %d individuals" % (entity.name,
                                                      len(entity.array))
            else:
                print "- loading input data"
                for entity in entities:
                    print "  *", entity.name, "...",
                    timed(entity.load_period_data, period)
                    print "    -> %d individuals" % len(entity.array)
            for entity in entities:
                entity.array_period = period
                entity.array['period'] = period

            if processes:
                # build context for this period:
                const_dict = {'period': period,
                              'nan': float('nan'),
                              '__globals__': globals_data}

                num_processes = len(processes)
                for p_num, process_def in enumerate(processes, start=1):
                    process, periodicity = process_def

                    print "- %d/%d" % (p_num, num_processes), process.name,
                    #TODO: provide a custom __str__ method for Process &
                    # Assignment instead
                    if hasattr(process, 'predictor') and process.predictor \
                       and process.predictor != process.name:
                        print "(%s)" % process.predictor,
                    print "...",
                    if period_idx % periodicity == 0: 
                        elapsed, _ = gettime(process.run_guarded, self,
                                             const_dict)
                    else:
                        elapsed = 0
                        print "skipped (periodicity)"

                    process_time[process.name] += elapsed
                    if config.show_timings:
                        print "done (%s elapsed)." % time2str(elapsed)
                    else:
                        print "done."
                    self.start_console(process.entity, period,
                                       globals_data)

            print "- storing period data"
            for entity in entities:
                print "  *", entity.name, "...",
                timed(entity.store_period_data, period)
                print "    -> %d individuals" % len(entity.array)
#            print " - compressing period data"
#            for entity in entities:
#                print "  *", entity.name, "...",
#                for level in range(1, 10, 2):
#                    print "   %d:" % level,
#                    timed(entity.compress_period_data, level)
            period_objects[period] = sum(len(entity.array)
                                         for entity in entities)

        try:
            simulate_period(0, self.start_period - 1, self.init_processes,
                            self.entities, init=True)
            main_start_time = time.time()
            periods = range(self.start_period,
                            self.start_period + self.periods)
            for period_idx, period in enumerate(periods):
                period_start_time = time.time()
                simulate_period(period_idx, period,
                                self.processes, self.entities)
                time_elapsed = time.time() - period_start_time
                print "period %d done (%s elapsed)." % (period,
                                                        time2str(time_elapsed))

            total_objects = sum(period_objects[period] for period in periods)
            total_time = time.time() - main_start_time
            print """
==========================================
 simulation done
==========================================
 * %s elapsed
 * %d individuals on average
 * %d individuals/s/period on average
==========================================
""" % (time2str(time.time() - start_time),
       total_objects / self.periods,
       total_objects / total_time)

            show_top_processes(process_time, 10)
#            if config.debug:
#                show_top_expr()

            if run_console:
                c = console.Console(self.console_entity, periods[-1],
                                    self.globals_def, globals_data)
                c.run()

        finally:
            if h5in is not None:
                h5in.close()
            h5out.close()

    @property
    def console_entity(self):
        '''compute the entity the console should start in (if any)'''

        return entity_registry[self.default_entity] \
               if self.default_entity is not None \
               else None

    def start_console(self, entity, period, globals_data):
        if self.stepbystep:
            c = console.Console(entity, period, self.globals_def, globals_data)
            res = c.run(debugger=True)
            self.stepbystep = res == "step"
