import carray as ca
import numpy as np
import tables

from utils import safe_put, timed, count_occurences, size2str
from data import mergeArrays
from expr import parse, Variable, SubscriptableVariable, \
                 expr_eval, dtype, \
                 get_missing_value, hasvalue 

str_to_type = {'float': float, 'int': int, 'bool': bool}

def compress_column(a, level):
    arr = ca.carray(a, cparams=ca.cparams(level))
    print "%d -> %d (%.2f)" % (arr.nbytes, arr.cbytes,
                               float(arr.nbytes)/arr.cbytes),
    return arr

def decompress_column(a):
    return a[:]
    
class EntityContext(object):
    def __init__(self, entity, extra):
        self.entity = entity
        self.extra = extra
        self['__entity__'] = entity
        self['__weight_col__'] = entity.weight_col
        self['__on_align_overflow__'] = entity.on_align_overflow 

    def __getitem__(self, key):
        try:
            return self.extra[key]
        except KeyError:
            period = self.extra['period']
#            current_period = self.entity.array['period'][0] 
            if self._iscurrentperiod:
                try:
                    return self.entity.temp_variables[key]
                except KeyError:
                    try:
                        return self.entity.array[key]
                    except ValueError:
                        raise KeyError(key)
#            elif period == current_period - 1 and \
#                 self.entity.array_lag is not None:
#                try:
#                    return self.entity.array_lag[key]
#                except ValueError:
#                    raise KeyError(key)
            else:
                bounds = self.entity.output_rows.get(period)
                if bounds is not None: 
                    startrow, stoprow = bounds
                else:
                    startrow, stoprow = 0, 0
#                print "loading from disk...",
#                res = timed(self.entity.table.read,
#                             start=startrow, stop=stoprow, 
#                             field=key)
#                for level in range(1, 10, 2):
#                    print "   %d - compress:" % level,
#                    arr = timed(compress_column, res, level)
#                    print "decompress:",
#                    timed(decompress_column, arr)
#                return res
                return self.entity.table.read(start=startrow, stop=stoprow, 
                                              field=key)
    
    @property
    def _iscurrentperiod(self):
        current_array = self.entity.array

        #FIXME: in the rare case where there is nothing in the current array
        #       we cannot know whether the period in the context is the
        #       "current" period or a past period. For now we assume it is
        #       the current period because it is the most likely situation, but
        #       it is not correct!
        if not len(current_array):
            return True
        
        # if the current period array is the same as the context period
        return current_array['period'][0] == self.extra['period']

    def __setitem__(self, key, value):
        self.extra[key] = value

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False
    
    def keys(self):
        res = list(self.entity.array.dtype.names)
        res.extend(sorted(self.entity.temp_variables.keys()))
        return res
    
    def get(self, key, elsevalue=None):
        try:
            return self[key]
        except KeyError:
            return elsevalue
        
    def copy(self):
        return EntityContext(self.entity, self.extra.copy())
    
    def length(self):
        if self._iscurrentperiod:
            return len(self.entity.array)
        else:
            period = self.extra['period']
            bounds = self.entity.output_rows.get(period)
            if bounds is not None:
                startrow, stoprow = bounds
                return stoprow - startrow
            else:
                return 0

    @property
    def id_to_rownum(self):
        period = self.extra['period']
        if self._iscurrentperiod:
            return self.entity.id_to_rownum
        elif period in self.entity.output_index:
            return self.entity.output_index[period]
        else:
            #FIXME: yes, it's true, that if period is not in output_index, it 
            # probably means that we are before start_period and in that case,
            # input_index == output_index, but it would be cleaner to simply
            # initialise output_index correctly
            return self.entity.input_index[period]


def context_length(ctx):
    if hasattr(ctx, 'length'):
        return ctx.length()
    elif '__len__' in ctx:
        return ctx['__len__']
    else:
        usual_len = None
        for k, value in ctx.iteritems():
            if isinstance(value, np.ndarray):
                if usual_len is not None and len(value) != usual_len: 
                    raise Exception('incoherent array lengths: %s''s is %d '
                                    'while the len of others is %d' %
                                    (k, len(value), usual_len))
                usual_len = len(value)
        return usual_len


class Entity(object):
    def __init__(self, name, fields, missing_fields, links,
                 macro_strings, process_strings, weight_col=None,
                 on_align_overflow='carry'):
        self.name = name
        
        duplicate_names = [name
                           for name, num
                           in count_occurences(fname for fname, _ in fields)
                           if num > 1]
        if duplicate_names:
            raise Exception("duplicate fields in entity '%s': %s"
                            % (self.name, ', '.join(duplicate_names)))

        self.fields = [('period', int), ('id', int)] + fields 
        
        # only used in data (to check that all "required" fields are present
        # in the input file and data_main (where it will not survive its 
        # modernisation)
        
        # one potential solution would be to split the fields argument and 
        # attribute in input_fields and output_fields (regardless of whether
        # it is split in the simulation/yaml file).
        
        # however that might be just a temporary solution as we will soon need
        # more arguments to fields (default values, ranges, etc...)
        
        # another solution is to use a Field class
        # seems like the better long term solution 
        self.missing_fields = missing_fields
        self.period_individual_fnames = [name for name, _ in fields]
        self.links = links
        
        self.weight_col = weight_col
        self.on_align_overflow = on_align_overflow

        self.macro_strings = macro_strings
        self.process_strings = process_strings

        self.expectedrows = tables.parameters.EXPECTED_ROWS_TABLE
        self.table = None
        
        self.input_rows = {}
        #XXX: it might be unnecessary to keep it in memory after the initial
        # load.
        #TODO: it *is* unnecessary to keep periods which have already been
        # simulated, because (currently) when we go back in time, we always go
        # back using the output table.
        self.input_index = {}

        self.output_rows = {}
        self.output_index = {}
        
        self.base_period = None
        self.array = None
        self.array_lag = None

        self.num_tmp = 0
        self.temp_variables = {}
        self.id_to_rownum = None
        self._variables = None

    @classmethod
    def from_yaml(cls, ent_name, entity_def):
        from properties import Link
        
        # YAML "ordered dict" syntax returns a list of dict and we want a list
        # of tuples
        #FIXME: if "fields" key is present but no field is defined,
        #entity_def.get('fields', []) returns None and this breaks 
        fields_def = [d.items()[0] for d in entity_def.get('fields', [])]

        fields = []
        missing_fields = []
        for name, fielddef in fields_def:
            if isinstance(fielddef, dict):
                strtype = fielddef['type']
                if not fielddef.get('initialdata', True):
                    missing_fields.append(name) 
            else:
                strtype = fielddef
            fields.append((name, str_to_type[strtype]))

        link_defs = entity_def.get('links', {})
        links = dict((name, Link(name, l['type'], l['field'], l['target']))
                     for name, l in link_defs.iteritems())

        #TODO: add option for on_align_overflow
        return Entity(ent_name, fields, missing_fields, links,
                      entity_def.get('macros', {}),
                      entity_def.get('processes', {}),
                      entity_def.get('weight'))
        
    @staticmethod
    def collect_predictors(items):
        predictors = []
        for k, v in items:
            if k is None:
                continue
            if isinstance(v, basestring):
                predictors.append(k)
            elif isinstance(v, dict):
                predictors.append(v['predictor'])
        return predictors
        
    @property        
    def variables(self):
        if self._variables is None:
            global_predictors = \
                self.collect_predictors(self.process_strings.iteritems()) 
            all_fields = set(global_predictors)
            stored_fields = set(self.period_individual_fnames)
            temporary_fields = all_fields - stored_fields
            
            vars = dict((name, Variable(name, type_))
                        for name, type_ in self.fields)
            vars.update((name, Variable(name)) for name in temporary_fields)
            vars.update(self.links)
            self._variables = vars
        return self._variables
    
    def check_links(self):
        for name, link in self.links.iteritems():
            target_name = link._target_entity
            if target_name not in entity_registry:
                raise Exception("Target of '%s' link in entity '%s' is an "
                                "unknown entity (%s)"  % (name, self.name,
                                                          target_name)) 

    @property
    def conditional_context(self):
        cond_context = {}
        for name, link in self.links.iteritems():
            target_name = link._target_entity
            target_entity = entity_registry[target_name]
            if target_entity is not self:
                cond_context[name] = target_entity.variables
        return cond_context

    def parse_processes(self, globals):
        from properties import Assignment, Process, ProcessGroup
        vars = dict((name, SubscriptableVariable(name, type_))
                    for name, type_ in globals)
        vars.update(self.variables)
        
        cond_context = self.conditional_context
        vars.update((k, parse(v, vars, cond_context))
                    for k, v in self.macro_strings.iteritems())

        def parse_expressions(items, vars):
            processes = []
            for k, v in items:
                if isinstance(v, basestring):
                    expr = parse(v, vars, cond_context)
                    if not isinstance(expr, Process):
                        process = Assignment(expr)
                    else:
                        process = expr
#                    process = parse_expression(v, vars)
                elif isinstance(v, list):
                    # v should be a list of dict
                    group_expressions = []
                    for element in v:
                        if isinstance(element, dict):
                            group_expressions.append(element.items()[0])
                        else:
                            group_expressions.append((None, element))
#                    group_expressions = [d.items()[0] for d in v]
                    group_predictors = \
                        self.collect_predictors(group_expressions)
                    group_context = vars.copy()
                    group_context.update((name, Variable(name))
                                         for name in group_predictors)
                    sub_processes = parse_expressions(group_expressions,
                                                      group_context)
                    process = ProcessGroup(k, sub_processes)
                elif isinstance(v, dict):
                    expr = parse(v['expr'], vars, cond_context)
                    process = Assignment(expr)
                    process.predictor = v['predictor']
#                    process = parse_expression(v['expr'], vars)
                else:
                    raise Exception("unknown expression type for %s: %s"
                                    % (k, type(v)))
                processes.append((k, process))
            return processes
            
        processes = dict(parse_expressions(self.process_strings.iteritems(),
                                           vars))

        fnames = set(self.period_individual_fnames)
        def attach_processes(items):
            for k, v in items:
                if isinstance(v, ProcessGroup):
                    v.entity = self
                    attach_processes(v.subprocesses)
                elif isinstance(v, Assignment):
                    predictor = v.predictor if v.predictor is not None else k
                    if predictor in fnames:
                        kind = 'period_individual'
                    else:
                        kind = None
                    v.attach(k, self, kind)
                else:
                    v.attach(k, self)
        attach_processes(processes.iteritems())
        self.processes = processes

    def locate_tables(self, h5in, h5out):
        self.input_table = \
            getattr(h5in.root.entities, self.name) if h5in is not None else None
        self.table = getattr(h5out.root.entities, self.name)

    def load_period_data(self, period):
        rows = self.input_rows.get(period)
        if rows is None:
            # nothing needs to be done in that case
            return

        start, stop = rows
        input_array = self.input_table.read(start, stop)

        self.array, self.id_to_rownum = \
            mergeArrays(self.array, input_array, result_fields='array1')

    def store_period_data(self, period):
#        temp_mem = 0
#        for v in self.temp_variables.itervalues():
#            if isinstance(v, np.ndarray) and v.shape:
#                temp_mem += v.dtype.itemsize * len(v)

        # erase all temporary variables which have been computed this period
        self.temp_variables = {}

#        main_mem = self.array.dtype.itemsize * len(self.array)
#        print "mem used: %s (main: %s / temp: %s)" \
#              % (size2str(temp_mem + main_mem),
#                 size2str(main_mem),
#                 size2str(temp_mem))

        if period in self.output_rows:
            raise Exception("trying to modify already simulated rows")
        else:
            #TODO: only store variables which are effectively used in lag
            # expressions
#            self.array_lag = self.array.copy()
            startrow = self.table.nrows
            self.table.append(self.array)
            self.output_rows[period] = (startrow, self.table.nrows)
            self.output_index[period] = self.id_to_rownum
        self.table.flush()

    def compress_period_data(self, level):
        print "%d -> %d (%f)" % ca.ctable(self.array, cparams=ca.cparams(level))._get_stats()

    def fill_missing_values(self, ids, values, context, filler='auto'):
        if filler is 'auto':
            filler = get_missing_value(values)
        result = np.empty(context_length(context), dtype=values.dtype)
        result.fill(filler)
        if len(ids):
            safe_put(result, context.id_to_rownum[ids], values)
        return result 

    def value_for_period(self, expr, period, context, fill='auto'):
        sub_context = EntityContext(self, {'period': period}) 
        result = expr_eval(expr, sub_context)
         
        if isinstance(result, np.ndarray) and result.shape:
            ids = expr_eval(Variable('id'), sub_context)
            if fill is None:
                return ids, result
            else:
                # expand values to the current "outer" context
                return self.fill_missing_values(ids, result, context, fill)
        else:
            return result

    def duration(self, bool_expr, context):
        value = expr_eval(bool_expr, context)

        baseperiod = self.base_period
        period = context['period'] - 1
        
        # using a full int so that the "store" type check works 
        result = value.astype(np.int)
        res_size = len(self.array)
        last_period_true = np.empty(res_size, dtype=np.int)
        last_period_true.fill(period + 1)

        id_to_rownum = context.id_to_rownum        
        still_running = value
        while np.any(still_running) and period >= baseperiod:
            ids, values = self.value_for_period(bool_expr, period, context,
                                                fill=None)
            missing = np.ones(res_size, dtype=bool)
            period_value = np.zeros(res_size, dtype=bool)
            if len(ids):
                value_rows = id_to_rownum[ids]
                safe_put(missing, value_rows, False)
                safe_put(period_value, value_rows, values)
            
            value = still_running & period_value
            result += value * (last_period_true - period)
            
            still_running &= period_value | missing
            last_period_true[period_value] = period
            period -= 1
        return result

    def tavg(self, expr, context):
        baseperiod = self.base_period
        period = context['period'] - 1
        
        res_size = len(self.array)
        
        num_values = np.zeros(res_size, dtype=np.int)
        last_period_wh_value = np.empty(res_size, dtype=np.int)
        last_period_wh_value.fill(context['period']) # current period

        sum_values = np.zeros(res_size, dtype=np.float)        
        id_to_rownum = context.id_to_rownum
        while period >= baseperiod:
            ids, values = self.value_for_period(expr, period, context,
                                                fill=None)

            # filter out lines which are present because there was a value for
            # that individual at that period but not for that column
            acceptable_rows = hasvalue(values)
            acceptable_ids = ids[acceptable_rows]
            if len(acceptable_ids):
                acceptable_values = values[acceptable_rows]
                
                value_rows = id_to_rownum[acceptable_ids]

                has_value = np.zeros(res_size, dtype=bool)
                safe_put(has_value, value_rows, True)

                period_value = np.zeros(res_size, dtype=np.float)
                safe_put(period_value, value_rows, acceptable_values)
                
                num_values += has_value * (last_period_wh_value - period)
                sum_values += period_value
                last_period_wh_value[has_value] = period
            period -= 1
        return sum_values / num_values

    def tsum(self, expr, context):
        baseperiod = self.base_period
        period = context['period'] - 1
        
        typemap = {bool: int, int: int, float: float}
        res_type = typemap[dtype(expr, context)]
        res_size = len(self.array)

        sum_values = np.zeros(res_size, dtype=res_type)
        id_to_rownum = context.id_to_rownum
        while period >= baseperiod:
            ids, values = self.value_for_period(expr, period, context,
                                                fill=None)
            
            # filter out lines which are present because there was a value for
            # that individual at that period but not for that column
            acceptable_rows = hasvalue(values)
            acceptable_ids = ids[acceptable_rows]
            if len(acceptable_ids):
                acceptable_values = values[acceptable_rows]

                value_rows = id_to_rownum[acceptable_ids]

                period_value = np.zeros(res_size, dtype=np.float)
                safe_put(period_value, value_rows, acceptable_values)

                sum_values += period_value
            period -= 1
        return sum_values
    
    def __repr__(self):
        return "<Entity '%s'>" % self.name


class EntityRegistry(dict):
    def add(self, entity):
        self[entity.name] = entity

    def add_all(self, entities_def):
        for k, v in entities_def.iteritems():
            self.add(Entity.from_yaml(k, v))
        
    
entity_registry = EntityRegistry()
