import numpy as np

from expr import Expr, type_to_idx, idx_to_type, expr_eval
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
    '''these processes only compute an expression and do not store their
       result (but they usually have side-effects)'''

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

        if isinstance(result, dict):
            indices = result.get('indices')
            filter_values = result.get('filter')
            assert filter_values is None or indices is None
            result = result['values']
        else:
            indices = None
            filter_values = None

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

        if indices is None and filter_values is None:
            # the whole column is updated
            target[self.predictor] = result
        else:
            if isinstance(target, np.ndarray):
                hasfield = self.predictor in target.dtype.fields
            else:
                hasfield = self.predictor in target

            if not hasfield:
                assert self.kind is None, \
                       "found a missing field which is not a temporary " \
                       "variable"
                #XXX: I'm not sure we should do this at all and in any case
                # this step should be delayed further because it could
                # be unnecessary.
                target[self.predictor] = np.zeros(len(self.entity.array),
                                                  dtype=res_type)
            if indices is not None:
                np.put(target[self.predictor], indices, result)
            elif filter_values is not None:
                np.putmask(target[self.predictor], filter_values, result)

    def expressions(self):
        if isinstance(self.expr, Expr):
            yield self.expr


class ProcessGroup(Process):
    def __init__(self, name, subprocesses):
        super(ProcessGroup, self).__init__()
        self.name = name
        self.subprocesses = subprocesses

    def run_guarded(self, simulation, const_dict):
        print
        for k, v in self.subprocesses:
            print "    *",
            if k is not None:
                print k,
            utils.timed(v.run_guarded, simulation, const_dict)
#            print "done."
            simulation.start_console(v.entity, const_dict['period'],
                                     const_dict['__globals__'])
        # purge all local variables
        temp_vars = self.entity.temp_variables
        all_vars = self.entity.variables
        local_vars = set(temp_vars.keys()) - set(all_vars.keys())
        for var in local_vars:
            del temp_vars[var]

    def expressions(self):
        for _, p in self.subprocesses:
            for e in p.expressions():
                yield e
