import time
import os
import operator
from collections import defaultdict
import random

import numpy as np
import tables
import yaml

from data import H5Data, Void
from entities import entity_registry, str_to_type
from utils import time2str, timed, gettime
import console

# imports needed for the simulation file eval
import alignment
import matching
import properties
import actions
import regressions


input_directory = "."
output_directory = "."

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
    '''
{
    'globals': {
        'periodic': [{
            '*': str
        }]
    }, 
    '#entities': {
        '*': {
            'fields': [{
                '*': '*'
            }],
            'links': {
                '*': {'*': '*'}
            },
            'macros': {
                '*': str
            },
            'processes': {
                '*': '*'
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
        }
        '#periods': int,
        '#start_period': int,
    }
}
'''
    
    def __init__(self, fpath, console=False):
        global output_directory
        global input_directory
        
        self.console = console
        simulation_path = os.path.abspath(fpath)
        simulation_dir = os.path.dirname(simulation_path) 
        with open(fpath) as f:
            content = yaml.load(f)

        #TODO: raise exception when there are unknown keywords
        # use validictory? http://readthedocs.org/docs/validictory/
        globals_def = content.get('globals', {})
        periodic_globals = globals_def.get('periodic', [])
        # list of one-item-dicts to list of tuples
        periodic_globals = [d.items()[0] for d in periodic_globals]
        self.globals = [(name, str_to_type[typestr])
                        for name, typestr in periodic_globals]

        simulation_def = content['simulation']
        seed = simulation_def.get('random_seed')
        if seed is not None:
            seed = int(seed)
            print "using fixed random seed: %d" % seed
            random.seed(seed)
            np.random.seed(seed)
        
        self.periods = simulation_def['periods']
        self.start_period = simulation_def['start_period']
        
        output_def = simulation_def['output']
        output_directory = output_def.get('path', '')
        if not os.path.isabs(output_directory):
            output_directory = os.path.join(simulation_dir, output_directory) 
        self.output_path = os.path.join(output_directory, output_def['file'])  

        input_def = simulation_def['input']
        input_directory = input_def.get('path', '')
        if not os.path.isabs(input_directory):
            input_directory = os.path.join(simulation_dir, input_directory) 
        
        entity_registry.add_all(content['entities'])
        for entity in entity_registry.itervalues():
            entity.check_links()
            entity.parse_processes(self.globals)
        
        init_def = [d.items()[0] for d in simulation_def.get('init', {})]
        init_processes, init_entities = [], set()
        for ent_name, proc_names in init_def:
            if ent_name not in entity_registry:
                raise Exception("Entity '%s' not found" % ent_name)

            entity = entity_registry[ent_name]
            init_entities.add(entity)
            init_processes.extend([entity.processes[proc_name]
                                   for proc_name in proc_names])
        self.init_processes = init_processes
        self.init_entities = init_entities
        
        agespine_def = [d.items()[0] for d in simulation_def['processes']]
        processes, entities = [], set()
        for ent_name, proc_names in agespine_def:
            entity = entity_registry[ent_name]
            entities.add(entity)
            processes.extend([entity.processes[proc_name] for proc_name in proc_names])
        self.processes = processes
        self.entities = entities
        
        method = input_def.get('method', 'h5')
        
        if method == 'h5':
            self.input_path = os.path.join(input_directory, input_def['file'])
            self.data_source = H5Data(self.input_path, self.output_path)
        elif method == 'void':
            self.input_path = None
            self.data_source = Void(self.output_path)
        else:
            print method, type(method)
        self.stepbystep = False

    def run(self):
        start_time = time.time()

        periodic_globals = timed(self.data_source.run, 
                                 entity_registry,
                                 self.start_period)

        #FIXME: this breaks the datasource generalisation
        h5in = tables.openFile(self.input_path, mode="r")
        h5out = tables.openFile(self.output_path, mode="a",
                                title="Simulation history")
        for entity in self.entities:
            entity.locate_tables(h5in, h5out)

        if periodic_globals is not None:
            try:
                globals_periods = periodic_globals['PERIOD']
            except ValueError:
                globals_periods = periodic_globals['period']
            globals_base_period = globals_periods[0]
        
        process_time = defaultdict(float)

        def simulate_period(period, processes, entities, init=False):        
            print "\nperiod", period
            
            if init:
                for entity in entities:
                    print "  * %s: %d individuals" % (entity.name, len(entity.array))
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
                        #TODO: use missing values instead
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
                    if hasattr(process, 'predictor') and process.predictor and \
                       process.predictor != process.name:
                        print "(%s)" % process.predictor,
                    print "...",
                    
                    elapsed, _ = gettime(process.run_guarded, self, const_dict)
                    
                    process_time[process.name] += elapsed
                    print "done (%s elapsed)." % time2str(elapsed)
                    self.start_console(process.entity, period)

            if not init:
                print "- storing period data"
                for entity in entities:
                    print "  *", entity.name, "...",
                    timed(entity.store_period_data, period)
        
        try:
            simulate_period(self.start_period, self.init_processes,
                            self.entities, init=True)
    
            for period in range(self.start_period, 
                                self.start_period + self.periods):
                period_start_time = time.time()
                simulate_period(period, self.processes, self.entities)
                print "period %d done (%s elapsed)." % (period, 
                                                        time2str(time.time() 
                                                               - period_start_time)) 
            print "simulation done (%s elapsed)." % time2str(time.time() 
                                                             - start_time)
            show_top_processes(process_time, 10)
    
            if self.console:
                c = console.Console()
                c.run()

        finally:
            h5in.close()
            h5out.close()

    def start_console(self, entity, period):
        if self.stepbystep:
            c = console.Console(entity, period)
            res = c.run(debugger=True)
            self.stepbystep = res == "step"
                
        