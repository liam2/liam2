import numpy as np
import tables

from expr import parse, Variable, SubscriptableVariable, \
                 VirtualArray, expr_eval, \
                 get_missing_value, get_missing_record, hasvalue 

str_to_type = {'float': float, 'int': int, 'bool': bool}

class EntityContext(object):
    def __init__(self, entity, extra):
        self.entity = entity
        self.extra = extra
        self['__entity__'] = entity

    def __getitem__(self, key):
        try:
            return self.extra[key]
        except KeyError:
            period = self.extra['period']
            array_period = self.entity.array['period'][0]
            if array_period == period:
                try:
                    return self.entity.temp_variables[key]
                except KeyError:
                    try:
                        return self.entity.array[key]
                    except ValueError:
                        raise KeyError(key)
            else:
                bounds = self.entity.output_rows.get(period)
                if bounds is not None: 
                    startrow, stoprow = bounds
                else:
                    startrow, stoprow = 0, 0
        
                return self.entity.table.read(start=startrow, stop=stoprow, 
                                              field=key)

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
        period = self.extra['period']
        array_period = self.entity.array['period'][0]
        if array_period == period:
            return len(self.entity.array)
        else:
            bounds = self.entity.output_rows.get(period)
            if bounds is not None: 
                startrow, stoprow = bounds
                return stoprow - startrow
            else:
                return 0

    @property
    def id_to_rownum(self):
        period = self.extra['period']
        array_period = self.entity.array['period'][0]
        if array_period == period:
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
    def __init__(self, simulation, name, entity_def):
        self.simulation = simulation
        self.name = name
        
        # YAML "ordered dict" syntax returns a list of dict and we want a list
        # of tuples
        fields = [d.items()[0] for d in entity_def.get('fields', [])]

        self.fields = [('period', int), ('id', int)]
        self.missing_fields = []
        for name, fielddef in fields:
            if isinstance(fielddef, dict):
                strtype = fielddef['type']
                if not fielddef.get('initialdata', True):
                    self.missing_fields.append(name)
            else:
                strtype = fielddef
            self.fields.append((name, str_to_type[strtype]))
    
        per_period_fields = [d.items()[0] 
                             for d in entity_def.get('per_period_fields', [])]
        per_period_fields = [(name, str_to_type[strtype])
                             for name, strtype in per_period_fields]
        self.per_period_fields = [('period', int)] + per_period_fields
        #TODO: handle them correctly
        self.pp_missing_fields = []
         
        self.period_individual_fnames = [name for name, _ in self.fields]
        self.period_fnames = [name for name, _ in self.per_period_fields]

        from properties import Link

        link_defs = entity_def.get('links', {})
        self.links = dict((name, Link(name, link['type'], link['field'], 
                                      link['target']))
                          for name, link in link_defs.iteritems())

        self.macro_strings = entity_def.get('macros', {})
        
        self.process_strings = entity_def.get('processes', {})

        self.expectedrows = tables.parameters.EXPECTED_ROWS_TABLE
        self.table = None
        
        self.input_rows = {}
        #XXX: it might be unnecessary to keep it in memory after the initial
        # load.
        #TODO: it *is* unnecessary to keep periods which have already been
        # simulated.
        self.input_index = {}

        self.output_rows = {}
        self.output_index = {}
        
        self.base_period = None
        self.array = None

        self.per_period_table = None
        # this will only ever be a one line array
        self.per_period_array = None
        
        self.num_tmp = 0
        self.temp_variables = {}
        self.id_to_rownum = None
        self._variables = None

    def collect_predictors(self, items):
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
            stored_fields = set(self.period_individual_fnames) | \
                            set(self.period_fnames)
            temporary_fields = all_fields - stored_fields
            
            vars = dict((name, Variable(name, type_))
                        for name, type_ in self.fields)
            vars.update((name, Variable(name, type_))
                        for name, type_ in self.per_period_fields)
            vars.update((name, Variable(name)) for name in temporary_fields)
            vars.update(self.links)
            self._variables = vars
        return self._variables
    
    @property
    def conditional_context(self):
        cond_context = {}
        for name, link in self.links.iteritems():
            target_name = link._target_entity
            target_entity = entity_registry[target_name]
            if target_entity is not self:
                cond_context[name] = target_entity.variables
        return cond_context
        
    def parse_processes(self):
        from properties import Assignment, Process, ProcessGroup
        vars = dict((name, SubscriptableVariable(name, type_))
                    for name, type_ in self.simulation.globals)
        vars.update(self.variables)
        vars['__parent__'] = VirtualArray()
        
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
        period_fnames = set(self.period_fnames)
        def attach_processes(items):
            for k, v in items:
                if isinstance(v, ProcessGroup):
                    v.entity = self
                    attach_processes(v.subprocesses)
                elif isinstance(v, Assignment):
                    predictor = v.predictor if v.predictor is not None else k
                    if predictor in fnames:
                        kind = 'period_individual'
                    elif predictor in period_fnames:
                        kind = 'period'
                    else:
                        kind = None
                    v.attach(k, self, kind)
                else:
                    v.attach(k, self)
        attach_processes(processes.iteritems())
        self.processes = processes
        
    def fieldtype(self, name):
        #TODO: store them as a map
        for fname, ftype in self.fields:
            if fname == name:
                return ftype
        return None

    def getcolumn(self, name):
        if isinstance(name, Variable):
            name = name.name

        if name in self.period_individual_fnames:
            return self.array[name]
        elif name in self.period_fnames:
            return self.per_period_array[name]
        else:
            return self.temp_variables[name]
    
    def locate_tables(self, h5in, h5out):
        input_entities = h5in.root.entities
        self.input_table = getattr(input_entities, self.name)

        output_entities = h5out.root.entities
        self.table = getattr(output_entities, self.name)
        self.per_period_table = getattr(output_entities, 
                                        self.name + "_per_period")
        
    def load_period_data(self, period):
        input_table = self.input_table
        output_dtype = self.array.dtype
        output_names = set(output_dtype.names)
        input_names = set(input_table.dtype.names)
        common_fields = output_names & input_names
         
        rows = self.input_rows.get(period)
        if rows is None:
            # nothing needs to be done in that case
            return
        
        start, stop = rows
        #TODO: chunking instead of reading the whole array in one pass 
        # *might* be a good idea to preserve some memory
        input_array = input_table.read(start, stop)
        
        # compute union of ids present in last period and those loaded from 
        # the input file
        #XXX: try with union(list of ids) instead of working with
        # booleans, that way we wouldn't need to keep the input_index
        # in memory after the initial transfer from input to output
        max_id = max(input_array['id'][-1], self.array['id'][-1])
        present_last_period = self.id_to_rownum != -1
        present_last_period.resize(max_id + 1)
        present_in_input = self.input_index[period] != -1
        present_in_input.resize(max_id + 1)
        is_present = present_in_input
        is_present |= present_last_period

        # compute new id_to_rownum
        id_to_rownum = np.empty(max_id + 1, dtype=int)
        id_to_rownum.fill(-1)

        rownum = 0
        for id, present in enumerate(is_present):
            if present:
                id_to_rownum[id] = rownum
                rownum += 1
        
        # allocate resulting array
        output_array = np.empty(rownum, dtype=output_dtype)
        
        # 1) fill all with missing
        missing_row = get_missing_record(self.array)
        output_array.fill(missing_row)
        
        # 2) copy data from last period
        target_rownums = id_to_rownum[self.array['id']]
#        #TODO: factorize a "safe_put" function
        output_array[target_rownums] = self.array
        if target_rownums[-1] != len(output_array) - 1:
            output_array[-1] = missing_row
        
        # 3) copy data from input file
        rownums = id_to_rownum[input_array['id']]
        output_array_to_modify = output_array[rownums]
        
        # Note that all rows which correspond to rownums == -1 have wrong
        # values (they have the value of the last row) but it is not 
        # necessary to correct them since they will not be copied back
        # into output_array.
        # np.putmask(output_array_to_modify, rownums == -1, missing_row)

        for fname in common_fields:
            output_array_to_modify[fname] = input_array[fname]

        # backup last row
        last_row = output_array[-1]
        output_array[rownums] = output_array_to_modify
        # restore last row if it was erroneously modified (because of a -1
        # in rownums)
        if rownums[-1] != len(output_array) - 1:
            output_array[-1] = last_row

        self.array = output_array
        self.id_to_rownum = id_to_rownum

    def store_period_data(self, period):
        # erase all temporary variables which have been computed this period
        self.temp_variables = {}

        if period in self.output_rows:
            raise Exception("trying to modify already simulated rows")
        else:
            startrow = self.table.nrows
            self.table.append(self.array)
            self.output_rows[period] = (startrow, self.table.nrows)
            self.output_index[period] = self.id_to_rownum
        self.table.flush()

        self.per_period_table.append(self.per_period_array)              
        self.per_period_table.flush()

    def fill_missing_values(self, ids, values, context, filler='auto'):
        if filler is 'auto':
            filler = get_missing_value(values)
        result_len = context_length(context)
        result = np.empty(result_len, dtype=values.dtype)
        result.fill(filler)
        
        if len(ids):
            rownums = context.id_to_rownum[ids]
            np.put(result, rownums, values)

            # Fix the value of the last individual because it was incorrectly
            # set to the value of someone dead (ie his id correspond to -1 in
            # id_to_rownum). This assumes "ids" are sorted
            if rownums[-1] != result_len - 1 and result[-1] != filler:
                result[-1] = filler
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
            value_rows = id_to_rownum[ids]

            #XXX: use fill missing?
            missing = np.ones(res_size, dtype=bool)
            missing[value_rows] = False
            if value_rows[-1] != res_size - 1 and not missing[-1]:
                missing[-1] = True

            #XXX: use fill missing?
            period_value = np.zeros(res_size, dtype=bool)
            period_value[value_rows] = values
            if value_rows[-1] != res_size - 1 and period_value[-1]:
                period_value[-1] = False
            
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

                #XXX: use fill missing?
                has_value = np.zeros(res_size, dtype=bool)
                has_value[value_rows] = True
                if value_rows[-1] != res_size - 1 and has_value[-1]:
                    has_value[-1] = False

                #XXX: use fill missing?
                period_value = np.zeros(res_size, dtype=np.float)
                period_value[value_rows] = acceptable_values
                if value_rows[-1] != res_size - 1 and period_value[-1] != 0:
                    period_value[-1] = 0
                
                num_values += has_value * (last_period_wh_value - period)
                sum_values += period_value
                last_period_wh_value[has_value] = period
            period -= 1
        return sum_values / num_values

    def tsum(self, expr, context):
        baseperiod = self.base_period
        period = context['period'] - 1
        
        res_size = len(self.array)

        #TODO: use int for bool & int, float for float
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
                
                #XXX: use fill missing?
                period_value = np.zeros(res_size, dtype=np.float)
                period_value[value_rows] = acceptable_values
                if value_rows[-1] != res_size - 1 and period_value[-1] != 0:
                    period_value[-1] = 0
                
                sum_values += period_value
            period -= 1
        return sum_values
    
    def __repr__(self):
        return "<Entity '%s'>" % self.name

    def __getitem__(self, key):
        return self.getcolumn(key)


class EntityRegistry(dict):
    def add(self, entity):
        self[entity.name] = entity

    def add_all(self, simulation, entities_def):
        for k, v in entities_def.iteritems():
            self.add(Entity(simulation, k, v))
    
entity_registry = EntityRegistry()
