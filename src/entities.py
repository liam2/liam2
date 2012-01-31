import carray as ca
import numpy as np
import tables

from utils import safe_put, count_occurences
from data import mergeArrays, get_fields
from registry import entity_registry
from expr import Variable, SubscriptableVariable, \
                 expr_eval, dtype, \
                 get_missing_value, hasvalue
from exprparser import parse
from context import EntityContext, context_length

str_to_type = {'float': float, 'int': int, 'bool': bool}


def compress_column(a, level):
    arr = ca.carray(a, cparams=ca.cparams(level))
    print "%d -> %d (%.2f)" % (arr.nbytes, arr.cbytes,
                               float(arr.nbytes) / arr.cbytes),
    return arr


def decompress_column(a):
    return a[:]


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

        self.weight_col = weight_col
        self.on_align_overflow = on_align_overflow

        self.macro_strings = macro_strings
        self.process_strings = process_strings

        self.expectedrows = tables.parameters.EXPECTED_ROWS_TABLE
        self.table = None
        self.input_table = None

        self.indexed_input_table = None
        self.indexed_output_table = None

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
        from links import Link

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
            #FIXME: what about int and float literals?
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

            variables = dict((name, Variable(name, type_))
                             for name, type_ in self.fields)
            variables.update((name, Variable(name))
                             for name in temporary_fields)
            variables.update(self.links)
            self._variables = variables
        return self._variables

    def check_links(self):
        for name, link in self.links.iteritems():
            target_name = link._target_entity
            if target_name not in entity_registry:
                raise Exception("Target of '%s' link in entity '%s' is an "
                                "unknown entity (%s)" % (name, self.name,
                                                         target_name))

    @property
    def conditional_context(self):
        cond_context = {}
        for name, link in self.links.iteritems():
            # we need both one2many and many2one links (for .get)
            target_name = link._target_entity
            target_entity = entity_registry[target_name]
            if target_entity is not self:
                cond_context[name] = target_entity.variables
        return cond_context

    def parse_processes(self, globals):
        from properties import Assignment, Compute, Process, ProcessGroup
        from links import PrefixingLink
        variables = dict((name, SubscriptableVariable(name, type_))
                         for name, type_ in globals)
        variables.update(self.variables)
        cond_context = self.conditional_context
        macros = dict((k, parse(v, variables, cond_context))
                      for k, v in self.macro_strings.iteritems())
        variables['other'] = PrefixingLink(macros, self.links, '__other_')

        variables.update(macros)

        def parse_expressions(items, variables):
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
                    # v should be a list of dict
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
                    sub_processes = parse_expressions(group_expressions,
                                                      group_context)
                    process = ProcessGroup(k, sub_processes)
                elif isinstance(v, dict):
                    expr = parse(v['expr'], variables, cond_context)
                    process = Assignment(expr)
                    process.predictor = v['predictor']
                else:
                    raise Exception("unknown expression type for %s: %s"
                                    % (k, type(v)))
                processes.append((k, process))
            return processes

        processes = dict(parse_expressions(self.process_strings.iteritems(),
                                           variables))

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
            self.array_lag = self.array.copy()
            startrow = self.table.nrows
            self.table.append(self.array)
            self.output_rows[period] = (startrow, self.table.nrows)
            self.output_index[period] = self.id_to_rownum
        self.table.flush()

    def compress_period_data(self, level):
        compressed = ca.ctable(self.array, cparams=ca.cparams(level))
        print "%d -> %d (%f)" % compressed._get_stats()

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

    def __repr__(self):
        return "<Entity '%s'>" % self.name
