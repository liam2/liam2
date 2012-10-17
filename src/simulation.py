import time
import os
import operator
from collections import defaultdict
import random

import numpy as np
import yaml

from data import H5Data, Void
from entities import Entity, str_to_type
from registry import entity_registry
from utils import time2str, timed, gettime, validate_dict
import console
import config


def show_top_processes(process_time, num_processes):
    process_times = sorted(process_time.iteritems(),
                           key=operator.itemgetter(1),
                           reverse=True)
    print "top %d processes:" % num_processes
    for name, p_time in process_times[:num_processes]:
        print " - %s: %s" % (name, time2str(p_time))
    print "total for top %d processes:" % num_processes,
    print time2str(sum(p_time for name, p_time
                       in process_times[:num_processes]))


class Simulation(object):
    yaml_layout = {
        'globals': {
            'periodic': [{
                '*': str
            }]
        },
        '#entities': {
            '*': {
                'fields': [{
                    '*': None
                }],
                'links': {
                    '*': {
                        '#type': str,
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
                '*': [str]
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
            'default_entity': str
        }
    }

    def __init__(self, globals_fields, periods, start_period,
                 init_processes, init_entities, processes, entities,
                 data_source, default_entity=None):
        if globals_fields is not None:
            globals_fields = [('PERIOD', int)] + globals_fields
        self.globals_fields = globals_fields
        self.periods = periods
        self.start_period = start_period
        self.init_processes = init_processes
        self.init_entities = init_entities
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
        #XXX: use validictory instead of my custom validator?
        # http://readthedocs.org/docs/validictory/
        validate_dict(content, cls.yaml_layout)

        globals_def = content.get('globals', {})
        periodic_globals = globals_def.get('periodic', [])
        # list of one-item-dicts to list of tuples
        periodic_globals = [d.items()[0] for d in periodic_globals]
        globals_fields = [(name, str_to_type[typestr])
                          for name, typestr in periodic_globals]

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
            entity.parse_processes(globals_fields)
            entity.compute_lagged_fields()

        init_def = [d.items()[0] for d in simulation_def.get('init', {})]
        init_processes, init_entities = [], set()
        for ent_name, proc_names in init_def:
            if ent_name not in entity_registry:
                raise Exception("Entity '%s' not found" % ent_name)

            entity = entity_registry[ent_name]
            init_entities.add(entity)
            init_processes.extend([entity.processes[proc_name]
                                   for proc_name in proc_names])

        processes_def = [d.items()[0] for d in simulation_def['processes']]
        processes, entities = [], set()
        for ent_name, proc_names in processes_def:
            entity = entity_registry[ent_name]
            entities.add(entity)
            processes.extend([entity.processes[proc_name]
                              for proc_name in proc_names])

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
        return Simulation(globals_fields, periods, start_period,
                          init_processes, init_entities, processes, entities,
                          data_source, default_entity)

    def load(self):
        return timed(self.data_source.load, self.globals_fields,
                     entity_registry)

    def run(self, run_console=False):
        start_time = time.time()
        h5in, h5out, periodic_globals = timed(self.data_source.run,
                                              self.globals_fields,
                                              entity_registry,
                                              self.start_period - 1)
#        input_dataset = self.data_source.run(self.globals_fields,
#                                             entity_registry)
#        output_dataset = self.data_sink.prepare(self.globals_fields,
#                                                entity_registry)
#        output_dataset.copy(input_dataset, self.start_period - 1)
#        for entity in input_dataset:
#            indexed_array = buildArrayForPeriod(entity)

        if periodic_globals is not None:
            try:
                globals_periods = periodic_globals['PERIOD']
            except ValueError:
                globals_periods = periodic_globals['period']
            globals_base_period = globals_periods[0]

        process_time = defaultdict(float)
        period_objects = {}

        def simulate_period(period, processes, entities, init=False):
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
                entity.array['period'] = period

            if processes:
                # build context for this period:
                const_dict = {'period': period,
                              'nan': float('nan')}

                # update "globals" with their value for this period
                if periodic_globals is not None:
                    globals_row = period - globals_base_period
                    if globals_row < 0:
                        #XXX: use missing values instead?
                        raise Exception('Missing globals data for period %d'
                                        % period)
                    period_globals = periodic_globals[globals_row]
                    const_dict.update((k, period_globals[k])
                                      for k in period_globals.dtype.names)
                    const_dict['__globals__'] = periodic_globals

                num_processes = len(processes)
                for p_num, process in enumerate(processes, start=1):
                    print "- %d/%d" % (p_num, num_processes), process.name,
                    #TODO: provided a custom __str__ method for Process &
                    # Assignment instead
                    if hasattr(process, 'predictor') and process.predictor \
                       and process.predictor != process.name:
                        print "(%s)" % process.predictor,
                    print "...",

                    elapsed, _ = gettime(process.run_guarded, self, const_dict)

                    process_time[process.name] += elapsed
                    print "done (%s elapsed)." % time2str(elapsed)
                    self.start_console(process.entity, period)

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
            simulate_period(self.start_period - 1, self.init_processes,
                            self.entities, init=True)
            main_start_time = time.time()
            periods = range(self.start_period,
                            self.start_period + self.periods)
            for period in periods:
                period_start_time = time.time()
                simulate_period(period, self.processes, self.entities)
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

            if run_console:
                if self.default_entity is not None:
                    entity = entity_registry[self.default_entity]
                else:
                    entity = None
                c = console.Console(entity, periods[-1])
                c.run()

        finally:
            if h5in is not None:
                h5in.close()
            h5out.close()

    def start_console(self, entity, period):
        if self.stepbystep:
            c = console.Console(entity, period)
            res = c.run(debugger=True)
            self.stepbystep = res == "step"
