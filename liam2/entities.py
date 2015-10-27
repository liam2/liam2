# encoding: utf-8
from __future__ import print_function

import collections
import sys
import warnings

# import bcolz
import numpy as np
import tables

import config
from data import (merge_arrays, get_fields, ColumnArray, index_table,
                  build_period_array)
from expr import (Variable, VariableMethodHybrid, GlobalVariable, GlobalTable,
                  GlobalArray, Expr, MethodSymbol, normalize_type)
from exprtools import parse
from process import Assignment, ProcessGroup, While, Function, Return
from utils import (count_occurrences, field_str_to_type, size2str,
                   WarnOverrideDict, split_signature, argspec,
                   UserDeprecationWarning)


max_vars = 0

# def compress_column(a, level):
#    arr = bcolz.carray(a, cparams=bcolz.cparams(level))
#    print "%d -> %d (%.2f)" % (arr.nbytes, arr.cbytes,
#                               float(arr.nbytes) / arr.cbytes),
#    return arr
#
#
# def decompress_column(a):
#    return a[:]

def global_symbols(globals_def):
    # FIXME: these should be computed once somewhere else, not for each
    # entity. I guess they should have a class of their own
    symbols = {}
    for name, global_def in globals_def.iteritems():
        global_type = global_def.get('fields')
        if isinstance(global_type, list):
            # add namespace for table
            symbols[name] = GlobalTable(name, global_type)
            if name == 'periodic':
                # special case to add periodic variables in the global
                # namespace
                symbols.update(
                    (name, GlobalVariable('periodic', name, type_))
                    for name, type_ in global_type)
        else:
            global_type = global_def['type']
            assert isinstance(global_type, type), "not a type: %s" % global_type
            symbols[name] = GlobalArray(name, global_type)
    return symbols


# This is an awful workaround for the fact that tables.Array does not support
# fancy indexes with negative indices.
# See https://github.com/PyTables/PyTables/issues/360
class DiskBackedArray(object):
    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, item):
        # load the array entirely in memory before indexing it
        return self.arr[:][item]

    def __getattr__(self, item):
        return getattr(self.arr, item)


class Field(object):
    def __init__(self, name, dtype, input=True, output=True, default=None):
        self.name = name
        self.dtype = dtype
        self.input = input
        self.output = output
        self.default = default


class FieldCollection(list):
    def __init__(self, iterable=None):
        list.__init__(self, iterable)
        for f in self:
            assert isinstance(f, Field)

    @property
    def in_input(self):
        return FieldCollection(f for f in self if f.input)

    @property
    def in_output(self):
        return FieldCollection(f for f in self if f.output)

    @property
    def names(self):
        for f in self:
            yield f.name

    @property
    def name_types(self):
        return [(f.name, f.dtype) for f in self]

    @property
    def dtype(self):
        return np.dtype(list(self.name_types))


class Entity(object):
    """
    fields is a list of tuple (name, type)
    """

    def __init__(self, name, fields=None, links=None, macro_strings=None,
                 process_strings=None, array=None):
        self.name = name

        # we should have exactly one of either array or fields defined
        assert ((fields is None and array is not None) or
                (fields is not None and array is None))

        if array is not None:
            if fields is None:
                fields = get_fields(array)
            array_period = np.min(array['period'])
        else:
            array_period = None

        if not isinstance(fields, FieldCollection):
            def fdef2field(name, fielddef):
                initialdata = True
                output = True
                if isinstance(fielddef, Field):
                    return fielddef
                elif isinstance(fielddef, (dict, str)):
                    if isinstance(fielddef, dict):
                        strtype = fielddef['type']
                        initialdata = fielddef.get('initialdata', True)
                        output = fielddef.get('output', True)
                    elif isinstance(fielddef, str):
                        strtype = fielddef
                    dtype = field_str_to_type(strtype, "field '%s'" % name)
                else:
                    assert isinstance(fielddef, type)
                    dtype = normalize_type(fielddef)
                return Field(name, dtype, initialdata, output)

            fields = FieldCollection(fdef2field(name, fdef)
                                     for name, fdef in fields)

        duplicate_names = [name
                           for name, num
                           in count_occurrences(fields.names)
                           if num > 1]
        if duplicate_names:
            raise Exception("duplicate fields in entity '%s': %s"
                            % (self.name, ', '.join(duplicate_names)))

        fnames = set(fields.names)
        if 'id' not in fnames:
            fields.insert(0, Field('id', int))
        if 'period' not in fnames:
            fields.insert(0, Field('period', int))
        self.fields = fields
        self.links = links

        if macro_strings is None:
            macro_strings = {}
        self.macro_strings = macro_strings

        self.process_strings = process_strings
        self.processes = None

        self.expectedrows = tables.parameters.EXPECTED_ROWS_TABLE
        self.table = None
        self.input_table = None

        self.indexed_input_table = None
        self.indexed_output_table = None

        self.input_rows = {}
        # TODO: it is unnecessary to keep periods which have already been
        # simulated, because (currently) when we go back in time, we always go
        # back using the output table... but periods before the start_period
        # are only present in input_index
        self.input_index = {}

        self.output_rows = {}
        self.output_index = {}
        self.output_index_node = None

        self.base_period = None
        # we need a separate field, instead of using array['period'] to be able
        # to get the period even when the array is empty.
        self.array_period = array_period
        self.array = array

        self.lag_fields = []
        self.array_lag = None

        self.num_tmp = 0
        self.temp_variables = {}
        self.id_to_rownum = None
        if array is not None:
            rows_per_period, index_per_period = index_table(array)
            self.input_rows = rows_per_period
            self.output_rows = rows_per_period
            self.input_index = index_per_period
            self.output_index = index_per_period
            self.id_to_rownum = index_per_period[array_period]
        self._variables = None
        self._methods = None

    @classmethod
    def from_yaml(cls, ent_name, entity_def):
        from links import Many2One, One2Many, One2One

        # YAML "ordered dict" syntax returns a list of dict and we want a list
        # of tuples
        # FIXME: if "fields" key is present but no field is defined,
        # entity_def.get('fields', []) returns None and this breaks
        fields_def = [d.items()[0] for d in entity_def.get('fields', [])]

        link_defs = entity_def.get('links', {})
        str2class = {'one2many': One2Many, 'many2one': Many2One, 'one2one': One2One}
        links = dict((name,
                      str2class[l['type']](name, l['field'], l['target']))
                     for name, l in link_defs.iteritems())

        return Entity(ent_name, fields_def, links,
                      entity_def.get('macros', {}),
                      entity_def.get('processes', {}))

    # noinspection PyProtectedMember
    def attach_and_resolve_links(self, entities):
        for link in self.links.itervalues():
            link._attach(self)
            link._resolve_target(entities)

    @property
    def local_var_names(self):
        return set(self.temp_variables.keys()) - set(self.variables.keys())

    @classmethod
    def from_table(cls, table):
        return Entity(table.name, get_fields(table), links={}, macro_strings={},
                      process_strings={})

    @staticmethod
    def collect_predictors(items):
        # this excludes lists (functions) and dict (while, ...)
        return [k for k, v in items
                if k is not None and isinstance(v, (basestring, int, float))]

    @property
    def variables(self):
        if self._variables is None:
            if self.process_strings:
                processes = self.process_strings.items()
            else:
                processes = []

            # names of all processes (hybrid or not) of the entity
            process_names = set(k for k, v in processes if k is not None)

            # names of all entity variables (temporary or not) which are set
            # globally
            all_predictors = set(self.collect_predictors(processes))

            field_names = set(self.fields.names)

            # normal fields (non-callable/no hybrid variable-function for them)
            variables = dict((name, Variable(self, name, type_))
                             for name, type_ in self.fields.name_types
                             if name in field_names - process_names)

            # callable fields (fields with a process of the same name)
            variables.update((name, VariableMethodHybrid(self, name, type_))
                             for name, type_ in self.fields.name_types
                             if name in field_names & process_names)
            # global temporaries (they are all callable).
            variables.update((name, VariableMethodHybrid(self, name))
                             for name in all_predictors - field_names)
            variables.update(self.links)
            self._variables = variables
        return self._variables

    @staticmethod
    def ismethod(v):
        keys = ('args', 'code', 'return')
        return (isinstance(v, list) or
                isinstance(v, dict) and any(key in v for key in keys))

    @property
    def methods(self):
        if self._methods is None:
            pstrings = self.process_strings
            items = pstrings.iteritems() if pstrings is not None else ()
            # variable-method hybrids are handled by the self.variable property
            stored_fields = set(self.fields.in_output.names)
            methodnames = [k for k, v in items
                           if self.ismethod(v) and k not in stored_fields]
            # factorial(n) -> factorial
            methodnames = [split_signature(name)[0] if '(' in name else name
                           for name in methodnames]
            self._methods = [(name, MethodSymbol(name, self))
                             for name in methodnames]
        return self._methods

    def all_symbols(self, global_context):
        from links import PrefixingLink

        symbols = WarnOverrideDict(self.variables.copy())
        local_context = global_context.copy()
        local_context[self.name] = symbols
        local_context['__entity__'] = self.name
        macros = dict((k, parse(v, local_context))
                      for k, v in self.macro_strings.iteritems())
        symbols.update(macros)
        symbols['other'] = PrefixingLink(self, macros, self.links, '__other_')
        symbols.update(self.methods)
        return symbols

    def parse_expr(self, k, v, context):
        if isinstance(v, (bool, int, float)):
            return Assignment(k, self, v)
        elif isinstance(v, basestring):
            return Assignment(k, self, parse(v, context))
        else:
            # lets be explicit about it
            return None

    @staticmethod
    def get_group_context(context, varnames):
        ent_name = context['__entity__']
        entity = context['__entities__'][ent_name]
        group_context = context.copy()
        entity_context = group_context[ent_name].copy()
        entity_context.update((name, Variable(entity, name))
                              for name in varnames)
        group_context[ent_name] = entity_context
        return group_context

    def parse_process_group(self, k, items, context, purge=True):
        # items is a list of [dict (assignment) or string (action)]
        if items is None:
            raise ValueError("no processes in '%s'" % k)
        group_expressions = [elem.items()[0] if isinstance(elem, dict)
                             else (None, elem)
                             for elem in items]
        group_predictors = self.collect_predictors(group_expressions)
        group_context = self.get_group_context(context, group_predictors)
        sub_processes = self.parse_expressions(group_expressions, group_context)
        return ProcessGroup(k, self, sub_processes, purge)

    # Once we can make it an error for non-function processes/statements,
    # we should probably split this method into parse_functions and
    # parse_function_body.
    def parse_expressions(self, items, context, functions_only=False):
        """
        items -- a list of tuples (name, process_string)
        context -- parsing context
                   a dict of all symbols available for all entities
        functions_only -- whether non-functions processes are allowed
        """
        processes = []
        for k, v in items:
            if k == 'while':
                if isinstance(v, dict):
                    raise SyntaxError("""
This syntax for while is not supported anymore:
  - while:
      cond: {cond_expr}
      code:
          - ...
Please use this instead:
  - while {cond_expr}:
      - ...
""".format(cond_expr=v['cond']))
                else:
                    raise ValueError("while is a reserved keyword")
            elif k is not None and k.startswith('while '):
                if not isinstance(v, list):
                    raise SyntaxError("while is a reserved keyword")
                cond = parse(k[6:].strip(), context)
                assert isinstance(cond, Expr)
                code = self.parse_process_group("while_code", v, context,
                                                purge=False)
                process = While(k, self, cond, code)
            elif k == 'return':
                e = SyntaxError("return is a reserved keyword. To return "
                                "from a function, use 'return expr' "
                                "instead of 'return: expr'")
                e.liam2context = "while parsing: return: {}".format(v)
                raise e
            elif k is None and v.startswith('return'):
                assert len(v) == 6 or v[6] == ' '
                if len(v) > 6:
                    result_def = v[7:].strip()
                else:
                    result_def = None
                result_expr = parse(result_def, context)
                process = Return(None, self, result_expr)
            else:
                process = self.parse_expr(k, v, context)
                if process is not None and functions_only:
                    if k in self.fields.names:
                        msg = """defining a process outside of a function is
deprecated because it is ambiguous. You should:
 * wrap the '{name}: {expr}' assignment inside a function like this:
        compute_{name}:  # you can name it any way you like but simply \
'{name}' is not recommended !
            - {name}: {expr}
 * update the simulation.processes list to use 'compute_{name}' (the function \
name) instead of '{name}'.
"""
                    else:
                        msg = """defining a process outside of a function is \
deprecated because it is ambiguous.
1) If '{name}: {expr}' is an assignment ('{name}' stores the result of \
'{expr}'), you should:
 * wrap the assignment inside a function, for example, like this:
        compute_{name}:  # you can name it any way you like but simply \
'{name}' is not recommended !
            - {name}: {expr}
 * update the simulation.processes list to use 'compute_{name}' (the function \
name) instead of '{name}'.
 * add '{name}' in the entities fields with 'output: False'
2) otherwise if '{expr}' is an expression which does not return any value, you \
can simply transform it into a function, like this:
        {name}:
            - {expr}
"""
                    warnings.warn(msg.format(name=k, expr=v),
                                  UserDeprecationWarning)
                if process is None:
                    if self.ismethod(v):
                        if isinstance(v, dict):
                            args = v.get('args', '')
                            code = v.get('code', '')
                            result = v.get('return', '')
                            oldargs = "\n      args: {}".format(args) \
                                if args else ''
                            oldcode = "\n      code:\n          - ..." \
                                if code else ''
                            newcode = "\n      - ..." if code else ''
                            oldresult = "\n      return: " + result \
                                if result else ''
                            newresult = "\n      - return " + result \
                                if result else ''
                            template = """
This syntax for defining functions with arguments or a return value is not
supported anymore:
  {funcname}:{oldargs}{oldcode}{oldresult}

Please use this instead:
  {funcname}({newargs}):{newcode}{newresult}"""
                            msg = template.format(funcname=k, oldargs=oldargs,
                                                  oldcode=oldcode,
                                                  oldresult=oldresult,
                                                  newargs=args, newcode=newcode,
                                                  newresult=newresult)
                            raise SyntaxError(msg)

                        assert isinstance(v, list)
                        # v should be a list of dicts (assignments) or
                        # strings (actions)
                        if "(" in k:
                            k, args = split_signature(k)
                            argnames = argspec(args).args
                            code_def, result_def = v, None
                        else:
                            argnames, code_def, result_def = [], v, None
                        method_context = self.get_group_context(context,
                                                                argnames)
                        code = self.parse_process_group(k + "_code", code_def,
                                                        method_context,
                                                        purge=False)
                        # TODO: use code.predictors instead (but it currently
                        # fails for some reason) or at least factor this out
                        # with the code in parse_process_group
                        group_expressions = [elem.items()[0]
                                             if isinstance(elem, dict)
                                             else (None, elem)
                                             for elem in code_def]
                        group_predictors = \
                            self.collect_predictors(group_expressions)
                        method_context = self.get_group_context(
                            method_context, group_predictors)
                        result_expr = parse(result_def, method_context)
                        assert result_expr is None or \
                               isinstance(result_expr, Expr)
                        process = Function(k, self, argnames, code, result_expr)
                    elif isinstance(v, dict) and 'predictor' in v:
                        raise ValueError("Using the 'predictor' keyword is "
                                         "not supported anymore. "
                                         "If you need several processes to "
                                         "write to the same variable, you "
                                         "should rather use functions.")
                    else:
                        raise Exception("unknown expression type for %s: %s"
                                        % (k, type(v)))
            processes.append((k, process))
        return processes

    def parse_processes(self, context):
        processes = self.parse_expressions(self.process_strings.iteritems(),
                                           context, functions_only=True)
        self.processes = dict(processes)
        # self.ssa()

    # def resolve_method_calls(self):
    #     for p in self.processes.itervalues():
    #         for expr in p.expressions():
    #             for node in expr.all_of(MethodCallToResolve):
    #                 # replace node in the parent node by the "resolved" node
    #                 # TODO: mimic ast.NodeTransformer
    #                 node.resolve()

    def ssa(self):
        fields_versions = collections.defaultdict(int)
        for p in self.processes.itervalues():
            if isinstance(p, ProcessGroup):
                p.ssa(fields_versions)

    def compute_lagged_fields(self):
        from tfunc import Lag
        from links import LinkGet

        lag_vars = set()
        for p in self.processes.itervalues():
            for expr in p.expressions():
                for node in expr.all_of(Lag):
                    for v in node.all_of(Variable):
                        if not isinstance(v, GlobalVariable):
                            lag_vars.add(v.name)
                    for lv in node.all_of(LinkGet):
                        # noinspection PyProtectedMember
                        lag_vars.add(lv.link._link_field)
                        # noinspection PyProtectedMember
                        target_entity = lv.link._target_entity
                        if target_entity == self:
                            target_vars = lv.target_expr.all_of(Variable)
                            lag_vars.update(v.name for v in target_vars)

        if lag_vars:
            # make sure we have an 'id' column, and that it comes first
            # (makes debugging easier). 'id' is always necessary for lag
            # expressions to be able to "expand" the vector of values to the
            # "current" individuals.
            lag_vars.discard('id')
            lag_vars = ['id'] + sorted(lag_vars)

        field_type = dict(self.fields.name_types)
        self.lag_fields = [(v, field_type[v]) for v in lag_vars]

    def build_period_array(self, start_period):
        self.array, self.id_to_rownum = \
            build_period_array(self.input_table,
                               list(self.fields.name_types),
                               self.input_rows,
                               self.input_index, start_period)
        assert isinstance(self.array, ColumnArray)
        self.array_period = start_period

    def load_period_data(self, period):
        if self.lag_fields:
            # TODO: use ColumnArray here
            # XXX: do we need np.empty? (but watch for alias problems)
            self.array_lag = np.empty(len(self.array),
                                      dtype=np.dtype(self.lag_fields))
            for field, _ in self.lag_fields:
                self.array_lag[field] = self.array[field]

        # if not self.indexed_input_table.has_period(period):
        #     # nothing needs to be done in that case
        #     return
        #
        # input_array = self.indexed_input_table.read(period)

        rows = self.input_rows.get(period)
        if rows is None:
            # nothing needs to be done in that case
            return

        start, stop = rows

        # It would be nice to use ColumnArray.from_table and adapt merge_arrays
        # to produce a ColumnArray in all cases, but it is not a huge priority
        # for now
        input_array = self.input_table.read(start, stop)

        self.array, self.id_to_rownum = \
            merge_arrays(self.array, input_array, result_fields='array1')
        # this can happen, depending on the layout of columns in input_array,
        # but the usual case (in retro) is that self.array is a superset of
        # input_array, in which case merge_arrays returns a ColumnArray
        if not isinstance(self.array, ColumnArray):
            self.array = ColumnArray(self.array)

    def purge_locals(self):
        """purge all local variables"""
        global max_vars

        temp_vars = self.temp_variables
        local_var_names = self.local_var_names
        num_locals = len(local_var_names)
        if config.debug and num_locals:
            local_vars = [v for k, v in temp_vars.iteritems()
                          if k in local_var_names]
            max_vars = max(max_vars, num_locals)
            temp_mem = sum(sys.getsizeof(v) +
                           (v.nbytes if isinstance(v, np.ndarray) else 0)
                           for v in local_vars)
            avgsize = sum(v.dtype.itemsize if isinstance(v, np.ndarray) else 0
                          for v in local_vars) / num_locals
            if config.log_level in ("functions", "processes"):
                print(("purging {} variables (max {}), will free {} of memory "
                       "(avg field size: {} b)".format(num_locals, max_vars,
                                                       size2str(temp_mem),
                                                       avgsize)))
        for var in local_var_names:
            del temp_vars[var]

    def flush_index(self, period):
        # keep an in-memory copy of the index for the current period
        self.output_index[period] = self.id_to_rownum

        # also flush it to disk
        h5file = self.output_index_node._v_file
        h5file.create_array(self.output_index_node, "_%d" % period,
                            self.id_to_rownum, "Period %d index" % period)

        # if an old index exists (this is not the case for the first period!),
        # point to the one on the disk, instead of the one in memory,
        # effectively clearing the one in memory
        idxname = '_%d' % (period - 1)
        if idxname in self.output_index_node:
            prev_disk_array = getattr(self.output_index_node, idxname)
            # DiskBackedArray is a workaround for pytables#360 (see above)
            self.output_index[period - 1] = DiskBackedArray(prev_disk_array)

    def store_period_data(self, period):
        if config.debug and config.log_level in ("functions", "processes"):
            temp_mem = sum(v.nbytes for v in self.temp_variables.itervalues()
                           if isinstance(v, np.ndarray))
            main_mem = self.array.nbytes
            print("mem used: %s (main: %s / temp: %s)"
                  % (size2str(temp_mem + main_mem),
                     size2str(main_mem),
                     size2str(temp_mem)))

        # erase all temporary variables which have been computed this period
        self.temp_variables = {}

        if period in self.output_rows:
            raise Exception("trying to modify already simulated rows")

        startrow = self.table.nrows
        self.array.append_to_table(self.table)
        self.output_rows[period] = (startrow, self.table.nrows)
        self.flush_index(period)
        self.table.flush()

    #     def compress_period_data(self, level):
    #     compressed = bcolz.ctable(self.array, cparams=bcolz.cparams(level))
    #     print "%d -> %d (%f)" % compressed._get_stats()

    def optimize_processes(self):
        """
        Common subexpression elimination
        """
        # XXX:
        # * we either need to do SSA first, or for each assignment process,
        #   "forget" all expressions containing the assigned variable
        #   doing it using SSA seems cleaner, but in the end it shouldn't
        #   change much. If we do not do SSA, we will need to "forget" about
        #   expressions which contain an assigned variable at *each step* of
        #   the process, including when simply counting the number of occurrence
        #   of expressions. In that case we also need to iterate on the
        #   processes in the same order than the simulation!
        # * I don't know if it is a good idea to optimize cross-functions.
        #   On one hand it offers much more possibilities for optimizations
        #   but, on the other hand the optimization pass might just take too
        #   much time... If we do not do it globally, we should move the method
        #   to ProcessGroup instead. But let's try it cross-functions first.
        # * cross-functions might get tricky when we take function calls
        #   into account.

        # TODO:
        # * it will be simpler and better to do this in two passes: first
        #   find duplicated expr and number of occurrences of each expr, then
        #   proceed with the factorization
        expr_count = collections.Counter()
        for p in self.processes.itervalues():
            for expr in p.expressions():
                for subexpr in expr.traverse():
                    if isinstance(subexpr, Expr) and \
                            not isinstance(subexpr, Variable):
                        expr_count[subexpr] += 1
        print()
        print("most common expressions")
        print("=" * 20)
        print(expr_count.most_common(100))

        # if count(larger) <= count(smaller) <= count(larger) + 1: kill smaller
        # if count(smaller) > count(larger) + 1: do both (larger uses smaller)

        # seen = {}
        # for p in self.processes.itervalues():
        #     for expr in p.expressions():
        #         for subexpr in expr.traverse():
        #             if subexpr in seen:
        #                 original = seen[subexpr]
        #                 # 1) add an assignment process before the process of
        #                 # the "original" expression to compute a temporary
        #                 # variable
        #                 # 2) modify "original" expr to use the temp var
        #                 # 3) modify the current expr to use the temp var
        #             else:
        #                 seen[subexpr] = subexpr

    def __repr__(self):
        return "<Entity '%s'>" % self.name

    def __str__(self):
        return self.name
