from __future__ import print_function

import time
import os.path
import operator
from collections import defaultdict
import random

import numpy as np
import tables
import yaml

from data import H5Data, Void
from entities import Entity
from registry import entity_registry
from utils import (time_period, addmonth, 
                   time2str, timed, gettime, validate_dict,
                   expand_wild, multi_get, multi_set,
                   merge_dicts, merge_items,
                   field_str_to_type, fields_yaml_to_type)
import console
import config
import expr


def show_top_times(what, times):
    count = len(times)
    print("top %d %s:" % (count, what))
    for name, timing in times:
        print(" - %s: %s" % (name, time2str(timing)))
    print("total for top %d %s:" % (count, what), end=' ')
    print(time2str(sum(timing for name, timing in times)))


def show_top_processes(process_time, count):
    process_times = sorted(process_time.iteritems(),
                           key=operator.itemgetter(1),
                           reverse=True)
    show_top_times('processes', process_times[:count])


def show_top_expr(count=None):
    show_top_times('expressions', expr.timings.most_common(count))


def handle_imports(content, directory):
    import_files = content.get('import', [])
    if isinstance(import_files, basestring):
        import_files = [import_files]
    for fname in import_files[::-1]:
        import_path = os.path.join(directory, fname)
        print("importing: '%s'" % import_path)
        import_directory = os.path.dirname(import_path)
        with open(import_path) as f:
            import_content = handle_imports(yaml.load(f), import_directory)
            for wild_key in ('globals/periodic', 'globals/*/fields',
                             'entities/*/fields'):
                multi_keys = expand_wild(wild_key, import_content)
                for multi_key in multi_keys:
                    import_fields = multi_get(import_content, multi_key)
                    local_fields = multi_get(content, multi_key, [])
                    # fields are in "yaml ordered dict" format and we want
                    # simple list of items
                    import_fields = [d.items()[0] for d in import_fields]
                    local_fields = [d.items()[0] for d in local_fields]
                    # merge the lists
                    merged_fields = merge_items(import_fields, local_fields)
                    # convert them back to "yaml ordered dict"
                    merged_fields = [{k: v} for k, v in merged_fields]
                    multi_set(content, multi_key, merged_fields)
            content = merge_dicts(import_content, content)
    return content


class Simulation(object):
    yaml_layout = {
        'import': None,
        'globals': {
            'periodic': [{
                '*': str
            }],
            '*': {
                'fields': [{
                    '*': None  # Or(str, {'type': str, 'initialdata': bool, 'default': type})
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
                '*': [None] # Or(str, [str, int])
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
            'legislation': {
                '#ex_post': bool,
                '#annee': int
            },            
            'final_stat': bool,
            'time_scale': str,
            'retro': bool,
            '#periods': int,
            'start_period': int,
            'init_period': int,
            'skip_shows': bool,
            'timings': bool,
            'assertions': str,
            'default_entity': str,
            'autodump': None,
            'autodiff': None
        }
                   
    }

    def __init__(self, globals_def, periods, init_period,
                 init_processes, init_entities, processes, entities,
                 data_source, default_entity=None, legislation=None, final_stat=False,
                  time_scale='year', retro = False):
        if 'periodic' in globals_def:
            globals_def['periodic'].insert(0, ('PERIOD', int))

        self.globals_def = globals_def
        self.periods = periods
        #TODO: work on it for start with seme
        if (time_scale not in ['year','year0']) and \
            (init_period % 100 > 12 or init_period % 100 == 0 or init_period < 9999) :
            raise Exception("Non valid start period") 
        self.init_period = init_period
        self.retro = retro
        # init_processes is a list of tuple: (process, 1)
        self.init_processes = init_processes
        self.init_entities = init_entities
        # processes is a list of tuple: (process, periodicity, start)
        self.processes = processes
        self.entities = entities
        self.data_source = data_source
        self.default_entity = default_entity
        self.legislation = legislation
        self.final_stat = final_stat
        self.time_scale = time_scale
        self.longitudinal = {}
        self.retro = retro 
        self.stepbystep = False

    @classmethod
    def from_yaml(cls, fpath,
                  input_dir=None, input_file=None,
                  output_dir=None, output_file=None):
        simulation_path = os.path.abspath(fpath)
        simulation_dir = os.path.dirname(simulation_path)
        with open(fpath) as f:
            content = yaml.load(f)

        content = handle_imports(content, simulation_dir)
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
            print("using fixed random seed: %d" % seed)
            random.seed(seed)
            np.random.seed(seed)

        periods = simulation_def['periods']
        time_scale = simulation_def.get('time_scale', 'year')
        retro = simulation_def.get('retro', False)
        
        start_period = simulation_def.get('start_period',None)
        init_period = simulation_def.get('init_period',None)
        if start_period is None and init_period is None:
            raise Exception("Either start_period either init_period should be given.")
        if start_period is not None:
            if init_period is not None: 
                raise Exception("Start_period can't be given if init_period is.")
            step = time_period[time_scale]*(1 - 2*(retro))           
            init_period = addmonth(start_period, step)
        
        config.skip_shows = simulation_def.get('skip_shows', False)
        #TODO: check that the value is one of "raise", "skip", "warn"
        config.assertions = simulation_def.get('assertions', 'raise')
        config.show_timings = simulation_def.get('timings', True)
        
        autodump = simulation_def.get('autodump', None)
        if autodump is True:
            autodump = 'autodump.h5'
        if isinstance(autodump, basestring):
            # by default autodump will dump all rows
            autodump = (autodump, None)
        config.autodump = autodump

        autodiff = simulation_def.get('autodiff', None)
        if autodiff is True:
            autodiff = 'autodump.h5'
        if isinstance(autodiff, basestring):
            # by default autodiff will compare all rows
            autodiff = (autodiff, None)
        config.autodiff = autodiff

        legislation = simulation_def.get('legislation', None)
        final_stat = simulation_def.get('final_stat', None)
        
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
        if not os.path.exists(output_directory):
            print("creating directory: '%s'" % output_directory)
            os.makedirs(output_directory)
        config.output_directory = output_directory

        if output_file is None:
            output_file = output_def['file']
        output_path = os.path.join(output_directory, output_file)
    
        method = input_def.get('method', 'h5')

        #need to be before processes because in case of legislation, we need input_table for now.
        if method == 'h5':
            if input_file is None:
                input_file = input_def['file']
            input_path = os.path.join(input_directory, input_file)
            data_source = H5Data(input_path, output_path)
        elif method == 'void':
            input_path = None
            data_source = Void(output_path)
        else:
            print(method, type(method))    
    
    
    
        for k, v in content['entities'].iteritems():
            entity_registry.add(Entity.from_yaml(k, v))
            
        for entity in entity_registry.itervalues():
            entity.check_links()
            entity.parse_processes(globals_def)
            entity.compute_lagged_fields()
            
        init_def = [d.items()[0] for d in simulation_def.get('init', {})]
        init_processes, init_entities = [], set()
        for ent_name, proc_names in init_def:
            if ent_name != 'legislation':
                if ent_name not in entity_registry:
                    raise Exception("Entity '%s' not found" % ent_name)
                
                entity = entity_registry[ent_name]
                init_entities.add(entity)
                init_processes.extend([(entity.processes[proc_name], 1, 1)
                                       for proc_name in proc_names])
            else: 
#                 proc1 = ExtProcess('liam2of',['simulation',None])
                proc2 = ExtProcess('of_on_liam',['simulation',2009,'period'])
#                 proc3 = ExtProcess('merge_leg',['simulation',data_source.output_path,
#                                                 "C:/Til/output/"+"simul_leg.h5",'period'])
#                 init_processes.append((proc1, 1))
                init_processes.append((proc2, 1, 1))
#                 processes.append((proc3, 1))


        processes_def = [d.items()[0] for d in simulation_def['processes']]
        processes, entity_set = [], set()
        for ent_name, proc_defs in processes_def:
            if ent_name != 'legislation':
                entity = entity_registry[ent_name]
                entity_set.add(entity)
                for proc_def in proc_defs:
                    # proc_def is simply a process name
                    if isinstance(proc_def, basestring):
                        # use the default periodicity of 1
                        proc_name, periodicity, start = proc_def, 1, 1
                    else:
                        if len(proc_def) == 3:
                            proc_name, periodicity, start = proc_def
                        elif len(proc_def) == 2:
                            proc_name, periodicity = proc_def 
                            start = 1
                    processes.append((entity.processes[proc_name], periodicity, start))
            else: 
#                 proc1 = ExtProcess('liam2of',['simulation',None])
                proc2 = ExtProcess('of_on_liam',['simulation',proc_defs[0],'period'])
#                 proc3 = ExtProcess('merge_leg',['simulation',data_source.output_path,
#                                                 "C:/Til/output/"+"simul_leg.h5",'period'])

#                 processes.append((proc1, 1))
                processes.append((proc2, 'year',12))
#                 processes.append((proc3, 1))
        entities = sorted(entity_set, key=lambda e: e.name)

        default_entity = simulation_def.get('default_entity')
        #processes[2][0].subprocesses[0][0]
        return Simulation(globals_def, periods, init_period,
                          init_processes, init_entities, processes, entities,
                          data_source, default_entity, legislation, final_stat, time_scale, retro)

    def load(self):
        return timed(self.data_source.load, self.globals_def,
                     entity_registry)

    def run(self, run_console=False):
        start_time = time.time()
        
        h5in, h5out, globals_data = timed(self.data_source.run,
                                          self.globals_def,
                                          entity_registry,
                                          self.init_period)
        
        if config.autodump or config.autodiff:
            if config.autodump:
                fname, _ = config.autodump
                mode = 'w'
            else:  # config.autodiff
                fname, _ = config.autodiff
                mode = 'r'
            fpath = os.path.join(config.output_directory, fname)
            h5_autodump = tables.openFile(fpath, mode=mode)
            config.autodump_file = h5_autodump
        else:
            h5_autodump = None
            
#        input_dataset = self.data_source.run(self.globals_def,
#                                             entity_registry)
#        output_dataset = self.data_sink.prepare(self.globals_def,
#                                                entity_registry)
#        output_dataset.copy(input_dataset, self.init_period - 1)
#        for entity in input_dataset:
#            indexed_array = buildArrayForPeriod(entity)

        # tell numpy we do not want warnings for x/0 and 0/0
        np.seterr(divide='ignore', invalid='ignore')

        process_time = defaultdict(float)
        period_objects = {}
        
        def simulate_period(period_idx, period, periods, processes, entities,
                            init=False):
            print("\nperiod", period)
            if init:
                for entity in entities:
                    print("  * %s: %d individuals" % (entity.name,
                                                      len(entity.array)))
            else:
                print("- loading input data")
                for entity in entities:
                    print("  *", entity.name, "...", end=' ')
                    timed(entity.load_period_data, period)
                    print("    -> %d individuals" % len(entity.array))
            for entity in entities:
                entity.array_period = period
                entity.array['period'] = period

            if processes:
                # build context for this period:
                const_dict = {'period_idx': period_idx+1,
                              'periods': periods,
                              'periodicity': time_period[self.time_scale]*(1 - 2*(self.retro)),
                              'longitudinal': self.longitudinal,
                              'format_date': self.time_scale,
                              '__simulation__': self,
                              'period': period,
                              'nan': float('nan'),
                              '__globals__': globals_data}
                assert(periods[period_idx+1] == period)

                num_processes = len(processes)
                for p_num, process_def in enumerate(processes, start=1):
                    process, periodicity, start = process_def
                    print("- %d/%d" % (p_num, num_processes), process.name,
                          end=' ')
                    print("...", end=' ')
                    # TDOD: change that
                    if isinstance(periodicity, int ):
                        if period_idx % periodicity == 0:
                            elapsed, _ = gettime(process.run_guarded, self,
                                                 const_dict)
                        else:
                            elapsed = 0
                            print("skipped (periodicity)")
                    else:
                        assert (periodicity  in time_period)
                        periodicity_process = time_period[periodicity]
                        periodicity_simul = time_period[self.time_scale]
                        month_idx = period % 100  
                        # first condition, to run a process with start == 12
                        # each year even if year are yyyy01
                        #modify start if periodicity_simul is not month
                        start = int(start/periodicity_simul-0.01)*periodicity_simul + 1
                        
                        if (periodicity_process <= periodicity_simul and self.time_scale != 'year0') or \
                                 month_idx % periodicity_process == start % periodicity_process:
                            const_dict['periodicity'] = periodicity_process*(1 - 2*(self.retro))
                            elapsed, _ = gettime(process.run_guarded, self,
                                                 const_dict)                        
                        else:
                            elapsed = 0
                            print("skipped (periodicity)")
                            
                    process_time[process.name] += elapsed
                    if config.show_timings:
                        print("done (%s elapsed)." % time2str(elapsed))
                    else:
                        print("done.")
                    self.start_console(process.entity, period,
                                       globals_data)
            # update longitudinal                 
            from pandas import DataFrame, HDFStore
            person = [x for x in entities if x.name == 'person'][0] # maybe we have a get_entity or anything more nice than that #TODO: check
            id = person.array.columns['id'] 

            for varname in ['sali', 'workstate']:
                var = person.array.columns[varname]              
                if init:
                    fpath = self.data_source.input_path
                    input_file = HDFStore(fpath, mode="r")
                    if 'longitudinal' in input_file.root:
                        input_longitudinal = input_file.root.longitudinal
                        if varname in input_longitudinal:
                            self.longitudinal[varname] = input_file['/longitudinal/' + varname]
                        else:
                            self.longitudinal[varname] = DataFrame({'id':id, period:var})
                    else:
                        self.longitudinal[varname] = DataFrame({'id':id, period:var})
                else:        
                    table = DataFrame({'id':id, period:var})       
                    self.longitudinal[varname] = self.longitudinal[varname].merge(table, on='id', how='outer')
    #             pdb.set_trace()
                #self.entities[2].table

            print("- storing period data")
            for entity in entities:
                print("  *", entity.name, "...", end=' ')
                timed(entity.store_period_data, period)
                print("    -> %d individuals" % len(entity.array))
#            print " - compressing period data"
#            for entity in entities:
#                print "  *", entity.name, "...",
#                for level in range(1, 10, 2):
#                    print "   %d:" % level,
#                    timed(entity.compress_period_data, level)
            period_objects[period] = sum(len(entity.array)
                                         for entity in entities)
            
            
            

        try:
            assert(self.time_scale in time_period)
            month_periodicity = time_period[self.time_scale]
            time_direction = 1 - 2*(self.retro)
            time_step = month_periodicity*time_direction
            
            periods = [ self.init_period + int(t/12)*100 + t%12   
                        for t in range(0, (self.periods+1)*time_step, time_step)]
            if self.time_scale == 'year0':
                periods = [ self.init_period + t for t in range(0, (self.periods+1))]
            print("simulated period are going to be: ",periods)
            
            init_start_time = time.time()
            simulate_period(0, self.init_period, [None,periods[0]], self.init_processes,
                            self.entities, init=True)
            
            time_init = time.time() - init_start_time
            main_start_time = time.time()
        
            for period_idx, period in enumerate(periods[1:]):
                period_start_time = time.time()
                simulate_period(period_idx, period, periods,
                                self.processes, self.entities)

#                 if self.legislation:                
#                     if not self.legislation['ex_post']:
#           
#                         elapsed, _ = gettime(liam2of.main,period)
#                         process_time['liam2of'] += elapsed
#                         elapsed, _ = gettime(of_on_liam.main,self.legislation['annee'],[period])
#                         process_time['legislation'] += elapsed
#                         elapsed, _ = gettime(merge_leg.merge_h5,self.data_source.output_path,
#                                              "C:/Til/output/"+"simul_leg.h5",period)                            
#                         process_time['merge_leg'] += elapsed

                time_elapsed = time.time() - period_start_time
                print("period %d done" % period, end=' ')
                if config.show_timings:
                    print("(%s elapsed)." % time2str(time_elapsed))
                else:
                    print()
                    
            total_objects = sum(period_objects[period] for period in periods)
 
#             if self.legislation:           
#                 if self.legislation['ex_post']:
#                        
#                     elapsed, _ = gettime(liam2of.main)
#                     process_time['liam2of'] += elapsed
#                     elapsed, _ = gettime(of_on_liam.main,self.legislation['annee'])
#                     process_time['legislation'] += elapsed
#                     # TODO: faire un programme a part, so far ca ne marche pas pour l'ensemble
#                     # adapter n'est pas si facile, comme on veut economiser une table, 
#                     # on ne peut pas faire de append directement parce qu on met 2010 apres 2011
#                     # a un moment dans le calcul
#                     elapsed, _ = gettime(merge_leg.merge_h5,self.data_source.output_path,
#                                          "C:/Til/output/"+"simul_leg.h5",None)                            
#                     process_time['merge_leg'] += elapsed


            if self.final_stat:
                elapsed, _ = gettime(start, period)
                process_time['Stat'] += elapsed              

            total_time = time.time() - main_start_time
            time_year = 0
            if len(periods)>1:
                nb_year_approx = periods[-1]/100 - periods[1]/100
                if nb_year_approx > 0 : 
                    time_year = total_time/nb_year_approx
               
            total_time = time.time() - main_start_time
            try:
                ind_per_sec = str(int(total_objects / total_time))
            except ZeroDivisionError:
                ind_per_sec = 'inf'
            print( """
==========================================
 simulation done
==========================================
 * %s elapsed 
 * %d individuals on average 
 * %s individuals/s/period on average
 
 * %s second for init_process
 * %s time/period in average
 * %s time/year in average
==========================================
""" % (time2str(time.time() - start_time),
        total_objects / self.periods,
        ind_per_sec,
        time2str(time_init),
        time2str(total_time / self.periods),
        time2str(time_year)))
           
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
            if h5_autodump is not None:
                h5_autodump.close()

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
