from __future__ import division, print_function

import collections

import numpy as np

import config
from diff_h5 import diff_array
from data import append_carray_to_table, ColumnArray
from expr import Expr, Variable, type_to_idx, idx_to_type, expr_eval
from context import EntityContext
import utils


class BreakpointException(Exception):
    pass


class Process(object):
    def __init__(self):
        self.name = None
        self.entity = None

    def attach(self, name, entity):
        self.name = name
        self.entity = entity

    def run_guarded(self, simulation, const_dict):
        try:
            context = EntityContext(self.entity, const_dict.copy())
            self.run(context)
        except BreakpointException:
            simulation.stepbystep = True

    def run(self, context):
        raise NotImplementedError()

    def expressions(self):
        raise NotImplementedError()

    def __str__(self):
        return "<process '%s'>" % self.name


class Compute(Process):
    """these processes only compute an expression and do not store their
       result (but they usually have side-effects). No class inherits from
       this but we use it when a user does not store anywhere the result of
       an expression (with a side effect) which *does* return a value.
       new() is a good example for this"""

    def __init__(self, expr):
        super(Compute, self).__init__()
        self.expr = expr

    def run(self, context):
        expr_eval(self.expr, context)

    def expressions(self):
        if isinstance(self.expr, Expr):
            yield self.expr


class Assignment(Process):
    def __init__(self, expr):
        super(Assignment, self).__init__()
        self.predictor = None
        self.kind = None  # period_individual, period, individual, global
        self.expr = expr

    def attach(self, name, entity, kind=None):
        super(Assignment, self).attach(name, entity)
        if self.predictor is None:
            self.predictor = name
        self.kind = kind

    def run(self, context):
        value = expr_eval(self.expr, context)
        self.store_result(value)

    def store_result(self, result):
        if result is None:
            return
        if self.name is None:
            raise Exception('trying to store None key')

        if isinstance(result, np.ndarray):
            res_type = result.dtype.type
        else:
            res_type = type(result)

        if self.kind == 'period_individual':
            # we cannot store/cache self.entity.array[self.name] because the
            # array object can change (eg when enlarging it due to births)
            target = self.entity.array
        else:
            target = self.entity.temp_variables

        #TODO: assert type for temporary variables too
        if self.kind is not None:
            target_type_idx = type_to_idx[target[self.predictor].dtype.type]
            res_type_idx = type_to_idx[res_type]
            if res_type_idx > target_type_idx:
                raise Exception(
                    "trying to store %s value into '%s' field which is of "
                    "type %s" % (idx_to_type[res_type_idx].__name__,
                                 self.predictor,
                                 idx_to_type[target_type_idx].__name__))

        # the whole column is updated
        target[self.predictor] = result

    def expressions(self):
        if isinstance(self.expr, Expr):
            yield self.expr


max_vars = 0


class ProcessGroup(Process):
    def __init__(self, name, subprocesses):
        super(ProcessGroup, self).__init__()
        self.name = name
        self.subprocesses = subprocesses
        self.calls = collections.Counter()

    @property
    def _modified_fields(self):
        fnames = [v.predictor for _, v in self.subprocesses
                  if isinstance(v, Assignment)]
        if not fnames:
            return []

        fnames.insert(0, 'id')
        temp = self.entity.temp_variables
        array = self.entity.array
        length = len(array)

        fields = [(k, temp[k] if k in temp else array[k])
                  for k in utils.unique(fnames)]
        return [(k, v) for k, v in fields
                if isinstance(v, np.ndarray) and v.shape == (length,)]

    def _tablename(self, period):
        self.calls[(period, self.name)] += 1
        num_calls = self.calls[(period, self.name)]
        if num_calls > 1:
            return '{}_{}'.format(self.name, num_calls)
        else:
            return self.name

    def _autodump(self, period):
        fields = self._modified_fields
        if not fields:
            return

        fname, numrows = config.autodump
        h5file = config.autodump_file
        name = self._tablename(period)
        dtype = np.dtype([(k, v.dtype) for k, v in fields])
        table = h5file.createTable('/{}'.format(period), name, dtype,
                                   createparents=True)

        fnames = [k for k, _ in fields]
        print("writing {} to {}/{}/{} ...".format(', '.join(fnames),
                                                  fname, period, name))

        context = EntityContext(self.entity, {'period': period})
        append_carray_to_table(context, table, numrows)
        print("done.")

    def _autodiff(self, period, numdiff=10, raiseondiff=False):
        fields = self._modified_fields
        if not fields:
            return

        fname, numrows = config.autodiff
        h5file = config.autodump_file
        tablepath = '/{}/{}'.format(period, self._tablename(period))
        print("comparing with {}{} ...".format(fname, tablepath))
        if tablepath in h5file:
            table = h5file.getNode(tablepath)
            disk_array = ColumnArray.from_table(table, stop=numrows)
            diff_array(disk_array, ColumnArray(fields), numdiff, raiseondiff)
        else:
            print("  SKIPPED (could not find table)")

    def run_guarded(self, simulation, const_dict):
        global max_vars

        period = const_dict['period']

        print()
        for k, v in self.subprocesses:
            print("    *", end=' ')
            if k is not None:
                print(k, end=' ')
            utils.timed(v.run_guarded, simulation, const_dict)
#            print "done."
            simulation.start_console(v.entity, period,
                                     const_dict['__globals__'])
        if config.autodump is not None:
            self._autodump(period)

        if config.autodiff is not None:
            self._autodiff(period)

        # purge all local variables
        temp_vars = self.entity.temp_variables
        all_vars = self.entity.variables
        local_var_names = set(temp_vars.keys()) - set(all_vars.keys())
        num_locals = len(local_var_names)
        if config.debug and num_locals:
            local_vars = [v for k, v in temp_vars.iteritems()
                          if k in local_var_names and
                             isinstance(v, np.ndarray)]
            max_vars = max(max_vars, num_locals)
            temp_mem = sum(v.nbytes for v in local_vars)
            avgsize = sum(v.dtype.itemsize for v in local_vars) / num_locals
            print(("purging {} variables (max {}), will free {} of memory "
                  "(avg field size: {} b)".format(num_locals, max_vars,
                                                  utils.size2str(temp_mem),
                                                  avgsize)))

        for var in local_var_names:
            del temp_vars[var]

    def ssa(self, fields_versions):
        procedure_vars = set(k for k, p in self.subprocesses if k is not None)
        global_vars = set(self.entity.variables.keys())
        local_vars = procedure_vars - global_vars

        local_versions = collections.defaultdict(int)
        for k, p in self.subprocesses:
            # mark all variables in the expression with their current version
            for expr in p.expressions():
                for node in expr.all_of(Variable):
                    versions = (local_versions if node.name in local_vars
                                else fields_versions)
                    #FIXME: for .version to be meaningful, I need to have
                    # a different variable instance each time the variable
                    # is used.
                    #>>> the best solution AFAIK is to parse the expressions
                    # in the same order as the "agespine".
                    That way we will be able to type all temporary variables
                    directly, and it would also solve the conditional
                    context hack. There is no problem with temporary variables
                    having different types over their lifetimes as these
                    will actually be different variables (because their
                    version will be different). There is no problem with
                    a variable being different in two control flow
                    "branches", because we do not have that case: if() coerce
                    types.
                    note that even if a branch is never "taken" (an if
                    condition that is always True or always False or a forloop
                    without any iteration), the type of the expressions
                    that variables are assigned to in that branch will
                    influence the type of the variable in subsequent code.
                    XXX: what if I have a user-defined function/procedure that
                    I call from two different places with an argument of a
                    different type? ideally, it should generate two distinct
                    procedures, but I am not there yet. Having a check on the
                    second call that the argument passed is of the same type
                    than the signature type (which was inferred from the
                    first call) seems enough for now.
                    node.version = versions[node.name]
                    node.used += 1
            # on assignment, increase the variable version
            if isinstance(p, Assignment):
                #XXX: is this always == k?
                target = p.predictor
                versions = (local_versions if target in local_vars
                            else fields_versions)
                versions[target] += 1

    def expressions(self):
        for _, p in self.subprocesses:
            for e in p.expressions():
                yield e
