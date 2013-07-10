from __future__ import print_function

import warnings

#import carray as ca
import numpy as np
import tables

import config
from context import EntityContext, context_length
from data import mergeArrays, get_fields, ColumnArray
from expr import (Variable, GlobalVariable, GlobalTable, GlobalArray,
                  expr_eval, get_missing_value)
from exprparser import parse
from process import Assignment, Compute, Process, ProcessGroup
from registry import entity_registry
from utils import (safe_put, count_occurences, field_str_to_type, size2str,
                   UserDeprecationWarning)


#def compress_column(a, level):
#    arr = ca.carray(a, cparams=ca.cparams(level))
#    print "%d -> %d (%.2f)" % (arr.nbytes, arr.cbytes,
#                               float(arr.nbytes) / arr.cbytes),
#    return arr
#
#
#def decompress_column(a):
#    return a[:]

class Entity(object):
    '''
    fields is a list of tuple (name, type, options)
    '''
    def __init__(self, name, fields=None, missing_fields=None, links=None,
                 macro_strings=None, process_strings=None,
                 array=None):
        self.name = name

        # we should have exactly one of either array or fields defined
        assert ((fields is None and array is not None) or
                (fields is not None and array is None))

        if array is not None:
            fields = get_fields(array)
            array_period = np.min(array['period'])

        duplicate_names = [name
                           for name, num
                           in count_occurences(fname for fname, _ in fields)
                           if num > 1]
        if duplicate_names:
            raise Exception("duplicate fields in entity '%s': %s"
                            % (self.name, ', '.join(duplicate_names)))
        fnames = [name for name, _ in fields]
        if 'id' not in fnames:
            fields.insert(0, ('id', int))
        if 'period' not in fnames:
            fields.insert(0, ('period', int))
        self.fields = fields

        # only used in data (to check that all "required" fields are present
        # in the input file)

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

        self.macro_strings = macro_strings
        self.process_strings = process_strings

        self.expectedrows = tables.parameters.EXPECTED_ROWS_TABLE
        self.table = None
        self.input_table = None

        self.indexed_input_table = None
        self.indexed_output_table = None

        self.input_rows = {}
        #TODO: it is unnecessary to keep periods which have already been
        # simulated, because (currently) when we go back in time, we always go
        # back using the output table.
        self.input_index = {}

        self.output_rows = {}
        self.output_index = {}

        self.base_period = None
        # we need a separate field, instead of using array['period'] to be able
        # to get the period even when the array is empty.
        self.array_period = None
        self.array = None

        self.lag_fields = []
        self.array_lag = None

        self.num_tmp = 0
        self.temp_variables = {}
        self.id_to_rownum = None
        self._variables = None

    @classmethod
    def from_yaml(cls, ent_name, entity_def):
        from links import Many2One, One2Many

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
            fields.append((name,
                           field_str_to_type(strtype, "field '%s'" % name)))

        link_defs = entity_def.get('links', {})
        str2class = {'one2many': One2Many, 'many2one': Many2One}
        links = dict((name,
                      str2class[l['type']](name, l['field'], l['target']))
                     for name, l in link_defs.iteritems())

        return Entity(ent_name, fields, missing_fields, links,
                      entity_def.get('macros', {}),
                      entity_def.get('processes', {}))

    @classmethod
    def from_table(cls, table):
        return Entity(table.name, get_fields(table), missing_fields=[],
                      links={}, macro_strings={}, process_strings={})

    @staticmethod
    def collect_predictors(items):
        predictors = []
        for k, v in items:
            if k is None:
                continue
            # no need to test for bool since bool is a subclass of int
            if isinstance(v, (basestring, int, float)):
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

            variables = dict((name, Variable(name, type_))
                             for name, type_ in self.fields)
            variables.update((name, Variable(name))
                             for name in temporary_fields)
            variables.update(self.links)
            self._variables = variables
        return self._variables

    def global_variables(self, globals_def):
        #FIXME: these should be computed once somewhere else, not for each
        # entity. I guess they should have a class of their own
        variables = {}
        for name, global_type in globals_def.iteritems():
            if isinstance(global_type, list):
                # add namespace for table
                variables[name] = GlobalTable(name, global_type)
                if name == 'periodic':
                    # special case to add periodic variables in the global
                    # namespace
                    variables.update(
                        (name, GlobalVariable('periodic', name, type_))
                        for name, type_ in global_type)
            else:
                assert isinstance(global_type, type)
                variables[name] = GlobalArray(name, global_type)
        return variables

    def check_links(self):
        for name, link in self.links.iteritems():
            target_name = link._target_entity_name
            if target_name not in entity_registry:
                raise Exception("Target of '%s' link in entity '%s' is an "
                                "unknown entity (%s)" % (name, self.name,
                                                         target_name))

    def get_cond_context(self, entities_visited=None):
        '''returns the conditional context: {link: variables}'''

        if entities_visited is None:
            entities_visited = set()
        else:
            entities_visited = entities_visited.copy()
        entities_visited.add(self)

        linked_entities = {}
        for k, link in self.links.items():
            entity = link._target_entity()
            if entity not in entities_visited:
                linked_entities[k] = entity

        cond_context = {}
        # use a set of entities to compute the conditional context only once
        # per target entity
        for entity in set(linked_entities.values()):
            cond_context.update(entity.get_cond_context(entities_visited))

        # entities linked directly take priority over (override) farther ones
        cond_context.update((k, entity.variables)
                            for k, entity in linked_entities.items())
        return cond_context
    conditional_context = property(get_cond_context)

    def all_variables(self, globals_def):
        from links import PrefixingLink

        variables = self.global_variables(globals_def).copy()
        variables.update(self.variables)
        cond_context = self.conditional_context
        macros = dict((k, parse(v, variables, cond_context))
                      for k, v in self.macro_strings.iteritems())
        variables.update(macros)
        variables['other'] = PrefixingLink(macros, self.links, '__other_')
        return variables

    def parse_expressions(self, items, variables, cond_context):
        processes = []
        for k, v in items:
            if isinstance(v, (bool, int, float)):
                process = Assignment(v)
            elif isinstance(v, basestring):
                expr = parse(v, variables, cond_context)
                if not isinstance(expr, Process):
                    if k is None:
                        process = Compute(expr)
                    else:
                        process = Assignment(expr)
                else:
                    process = expr
            elif isinstance(v, list):
                # v is a procedure
                # it should be a list of dict (assignment) or string (action)
                group_expressions = []
                for element in v:
                    if isinstance(element, dict):
                        group_expressions.append(element.items()[0])
                    else:
                        group_expressions.append((None, element))
                group_predictors = \
                    self.collect_predictors(group_expressions)
                group_context = variables.copy()
                group_context.update((name, Variable(name))
                                     for name in group_predictors)
                sub_processes = self.parse_expressions(group_expressions,
                                                       group_context,
                                                       cond_context)
                process = ProcessGroup(k, sub_processes)
            elif isinstance(v, dict):
                warnings.warn("Using the 'predictor' keyword is deprecated. "
                              "If you need several processes to "
                              "write to the same variable, you should "
                              "rather use procedures.",
                              UserDeprecationWarning)
                expr = parse(v['expr'], variables, cond_context)
                process = Assignment(expr)
                process.predictor = v['predictor']
            else:
                raise Exception("unknown expression type for %s: %s"
                                % (k, type(v)))
            processes.append((k, process))
        return processes

    def parse_processes(self, globals_def):
        processes = self.parse_expressions(self.process_strings.iteritems(),
                                           self.all_variables(globals_def),
                                           self.conditional_context)
        processes = dict(processes)

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

    def compute_lagged_fields(self):
        from tfunc import Lag
        from links import LinkValue
        lag_vars = set()
        for p in self.processes.itervalues():
            for expr in p.expressions():
                for node in expr.allOf(Lag):
                    for v in node.allOf(Variable):
                        if not isinstance(v, GlobalVariable):
                            lag_vars.add(v.name)
                    for lv in node.allOf(LinkValue):
                        lag_vars.add(lv.link._link_field)
                        target_entity = lv.link._target_entity()
                        if target_entity == self:
                            target_vars = lv.target_expression.allOf(Variable)
                            lag_vars.update(v.name for v in target_vars)

        if lag_vars:
            # make sure we have an 'id' column, and that it comes first
            # (makes debugging easier). 'id' is always necessary for lag
            # expressions to be able to "expand" the vector of values to the
            # "current" individuals.
            lag_vars.discard('id')
            lag_vars = ['id'] + sorted(lag_vars)

        field_type = dict(self.fields)
        self.lag_fields = [(v, field_type[v]) for v in lag_vars]

    def load_period_data(self, period):
        if self.lag_fields:
            #TODO: use ColumnArray here
            #XXX: do we need np.empty? (but watch for alias problems)
            self.array_lag = np.empty(len(self.array),
                                      dtype=np.dtype(self.lag_fields))
            for field, _ in self.lag_fields:
                self.array_lag[field] = self.array[field]

        rows = self.input_rows.get(period)
        if rows is None:
            # nothing needs to be done in that case
            return

        start, stop = rows

        # It would be nice to use ColumnArray.from_table and adapt mergeArrays
        # to produce a ColumnArray in all cases, but it is not a huge priority
        # for now
        input_array = self.input_table.read(start, stop)

        self.array, self.id_to_rownum = \
            mergeArrays(self.array, input_array, result_fields='array1')
        # this can happen, depending on the layout of columns in input_array,
        # but the usual case (in retro) is that self.array is a superset of
        # input_array, in which case mergeArrays returns a ColumnArray
        if not isinstance(self.array, ColumnArray):
            self.array = ColumnArray(self.array)

    def store_period_data(self, period):
        if config.debug:
            temp_mem = sum(v.nbytes for v in self.temp_variables.itervalues()
                           if isinstance(v, np.ndarray))
            main_mem = self.array.nbytes
            print("mem used: %s (main: %s / temp: %s)" \
                  % (size2str(temp_mem + main_mem),
                     size2str(main_mem),
                     size2str(temp_mem)))

        # erase all temporary variables which have been computed this period
        self.temp_variables = {}

        if period in self.output_rows:
            raise Exception("trying to modify already simulated rows")
        else:
            startrow = self.table.nrows
            self.array.append_to_table(self.table)
            self.output_rows[period] = (startrow, self.table.nrows)
            self.output_index[period] = self.id_to_rownum
        self.table.flush()

#    def compress_period_data(self, level):
#        compressed = ca.ctable(self.array, cparams=ca.cparams(level))
#        print "%d -> %d (%f)" % compressed._get_stats()

    def fill_missing_values(self, ids, values, context, filler='auto'):
        '''
        ids: ids present in past period
        context: current period context
        '''

        if filler is 'auto':
            filler = get_missing_value(values)
        result = np.empty(context_length(context), dtype=values.dtype)
        result.fill(filler)
        if len(ids):
            id_to_rownum = context.id_to_rownum
            # if there was more objects in the past than in the current
            # period. Currently, remove() keeps old ids, so this never
            # happens, but if we ever change remove(), we'll need to add
            # such a check everywhere we use id_to_rownum
#            invalid_ids = ids > len(id_to_rownum)
#            if np.any(invalid_ids):
#                fix ids
            rows = id_to_rownum[ids]
            safe_put(result, rows, values)
        return result

    def value_for_period(self, expr, period, context, fill='auto'):
        sub_context = EntityContext(self,
                                    {'period': period,
                                     '__globals__': context['__globals__']})
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

    def __repr__(self):
        return "<Entity '%s'>" % self.name
