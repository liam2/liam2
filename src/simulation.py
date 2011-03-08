import time
import os
import operator
from collections import defaultdict
import random

import numpy as np
import tables
import yaml

from data import H5Data, Void
from entities import entity_registry, str_to_type, EntityContext
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
    def __init__(self, fpath, console=False):
        global output_directory
        global input_directory
        
        self.console = console
        simulation_path = os.path.abspath(fpath)
        simulation_dir = os.path.dirname(simulation_path) 
        with open(fpath) as f:
            content = yaml.load(f)

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
        
        entity_registry.add_all(self, content['entities'])
        for entity in entity_registry.itervalues():
            entity.parse_processes()
        
        init_def = [d.items()[0] for d in simulation_def.get('init', {})]
        init_processes, init_entities = [], set()
        for ent_name, proc_names in init_def:
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

        periodic_globals, entities_data = timed(self.data_source.run, 
                                                entity_registry,
                                                self.start_period)
        
        for ent_name, (array, per_period_array) in entities_data.items():
            entity = entity_registry[ent_name] 
            entity.array = array
            entity.per_period_array = per_period_array
            print "indexing", ent_name, "..."
            #TODO: make a function out of this and use it in datamain
            # build id_to_rownum
            if len(array):
                ids = array['id']
                max_id = np.max(ids)
                id_to_rownum = np.empty(max_id + 1, dtype=int)
                id_to_rownum[:] = -1
                for idx, id in enumerate(ids):
                    id_to_rownum[id] = idx
            else:
                id_to_rownum = np.empty(0, dtype=int) 
            entity.id_to_rownum = id_to_rownum
            #TODO: index earlier periods too
            entity.period_index[array['period'][0]] = id_to_rownum
            
        h5file = tables.openFile(self.output_path, mode="a",
                                 title="Simulation history")

        for entity in self.entities:
            entity.locate_tables(h5file)
            
        globals_base_period = periodic_globals['period'][0]
        
        process_time = defaultdict(float)

        def simulate_period(period, processes, entities):        
            print "\nperiod", period
            # build context for this period:
            # update "globals" with their value for this period
            globals_row = period - globals_base_period
            period_globals = periodic_globals[globals_row]
            const_dict = dict((k, period_globals[k])
                              for k in period_globals.dtype.names)
            const_dict['nan'] = float('nan')
            
            for entity in entities:
                entity.array['period'] = period
                entity.per_period_array['period'] = period

            num_processes = len(processes)
            for p_num, process in enumerate(processes, start=1):
                print "- %d/%d" % (p_num, num_processes), process.name,
                if hasattr(process, 'predictor') and process.predictor and \
                   process.predictor != process.name:
                    print "(%s)" % process.predictor,
                print "...",
                
                elapsed, _ = gettime(process.run_guarded, const_dict)
                
                process_time[process.name] += elapsed
                print "done (%s elapsed)." % time2str(elapsed)
                self.start_console(process.entity, period)

            print "- storing period data"
            for entity in entities:
                print "  *", entity.name,
                timed(entity.store_period_data, period)
        
        simulate_period(self.start_period - 1, self.init_processes,
                        self.init_entities)

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
        
        h5file.close()

    def start_console(self, entity, period):
        if self.stepbystep: 
            c = console.Console(entity, period)
            res = c.run(debugger=True)
            self.stepbystep = res == "step"
        