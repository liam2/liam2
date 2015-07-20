from __future__ import division, print_function

import collections

import numpy as np

import config
from diff_h5 import diff_array
from data import append_carray_to_table, ColumnArray
from expr import Expr, Variable, type_to_idx, idx_to_type, expr_eval, expr_cache
from context import EntityContext
import utils


class BreakpointException(Exception):
    pass


class ReturnException(Exception):
    def __init__(self, result):
        self.result = result


class Process(object):
    def __init__(self, name, entity):
        self.name = name
        self.entity = entity

    def run_guarded(self, context):
        try:
            # purge extra
            context.entity_data.extra = {}
            self.run(context)
        except BreakpointException:
            # XXX: store this directly in the (evaluation) context instead of
            # in the simulation?
            context.simulation.stepbystep = True

    def run(self, context):
        raise NotImplementedError()

    def expressions(self):
        raise NotImplementedError()

    def __repr__(self):
        return "<process '%s'>" % self.name


class Return(Process):
    def __init__(self, name, entity, result_expr):
        super(Return, self).__init__(name, entity)
        self.result_expr = result_expr

    def run_guarded(self, context):
        raise ReturnException(expr_eval(self.result_expr, context))

    def expressions(self):
        if isinstance(self.result_expr, Expr):
            yield self.result_expr


class Assignment(Process):
    def __init__(self, name, entity, expr):
        super(Assignment, self).__init__(name, entity)
        self.expr = expr
        self.temporary = name not in entity.fields.in_output.names

    def run(self, context):
        value = expr_eval(self.expr, context)
        # Assignment to a field with a name == None is valid: it simply means
        # the result must not be stored. This happens when a user does not
        # store anywhere the result of an expression (it usually has side
        # effects -- csv, new, remove, ...).
        if self.name is not None:
            self.store_result(value, context)

    def store_result(self, result, context):
        if isinstance(result, np.ndarray):
            res_type = result.dtype.type
        else:
            res_type = type(result)

        if self.temporary:
            target = self.entity.temp_variables
        else:
            # we cannot store/cache self.entity.array[self.name] because the
            # array object can change (eg when enlarging it due to births)
            target = self.entity.array

            # TODO: assert type for temporary variables too
            target_type_idx = type_to_idx[target[self.name].dtype.type]
            res_type_idx = type_to_idx[res_type]
            if res_type_idx > target_type_idx:
                raise Exception(
                    "trying to store %s value into '%s' field which is of "
                    "type %s" % (idx_to_type[res_type_idx].__name__,
                                 self.name,
                                 idx_to_type[target_type_idx].__name__))

        # the whole column is updated
        target[self.name] = result

        # invalidate cache
        period = context.period
        if isinstance(period, np.ndarray):
            assert np.isscalar(period) or not period.shape
            period = int(period)
        expr_cache.invalidate(period, context.entity_name,
                              Variable(self.entity, self.name))

    def expressions(self):
        if isinstance(self.expr, Expr):
            yield self.expr


class While(Process):
    """this class implements while loops"""

    def __init__(self, name, entity, cond, code):
        """
        cond -- an Expr returning a (single) boolean, it means the condition
                value must be the same for all individuals
        code -- a ProcessGroup
        """
        Process.__init__(self, name, entity)
        self.cond = cond
        assert isinstance(code, ProcessGroup)
        self.code = code

    def run_guarded(self, context):
        while expr_eval(self.cond, context):
            self.code.run_guarded(context)
            # FIXME: this is a bit brutal :) This is necessary because
            # otherwise test_while loops indefinitely (because "values" is
            # never incremented)
            expr_cache.clear()

    def expressions(self):
        if isinstance(self.cond, Expr):
            yield self.cond
        for e in self.code.expressions():
            yield e


class ProcessGroup(Process):
    def __init__(self, name, entity, subprocesses, purge=True):
        super(ProcessGroup, self).__init__(name, entity)
        self.subprocesses = subprocesses
        self.calls = collections.Counter()
        self.purge = purge
        self.versions = {}

    def run_guarded(self, context):
        period = context.period

        if config.log_level == "processes":
            print()

        try:
            for k, v in self.subprocesses:
                if config.log_level == "processes":
                    print("    *", end=' ')
                    if k is not None:
                        print(k, end=' ')
                    utils.timed(v.run_guarded, context)
                else:
                    v.run_guarded(context)
                    #            print "done."
                context.simulation.start_console(context)
        finally:
            if config.autodump is not None:
                self._autodump(context)

            if config.autodiff is not None:
                self._autodiff(period)

            if self.purge:
                self.entity.purge_locals()

    @property
    def predictors(self):
        return [v.name for _, v in self.subprocesses
                if isinstance(v, Assignment) and v.name is not None]

    @property
    def _modified_fields(self):
        fnames = self.predictors
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

    def _autodump(self, context):
        fields = self._modified_fields
        if not fields:
            return

        period = context.period
        fname, numrows = config.autodump
        h5file = config.autodump_file
        name = self._tablename(period)
        dtype = np.dtype([(k, v.dtype) for k, v in fields])
        table = h5file.create_table('/{}'.format(period), name, dtype,
                                    createparents=True)

        fnames = [k for k, _ in fields]
        print("writing {} to {}/{}/{} ...".format(', '.join(fnames),
                                                  fname, period, name))

        entity_context = EntityContext(context, self.entity, {'period': period})
        append_carray_to_table(entity_context, table, numrows)
        print("done.")

    def _autodiff(self, period, showdiffs=10, raiseondiff=False):
        fields = self._modified_fields
        if not fields:
            return

        fname, numrows = config.autodiff
        h5file = config.autodump_file
        tablepath = '/p{}/{}'.format(period, self._tablename(period))
        print("comparing with {}{} ...".format(fname, tablepath))
        if tablepath in h5file:
            table = h5file.getNode(tablepath)
            disk_array = ColumnArray.from_table(table, stop=numrows)
            diff_array(disk_array, ColumnArray(fields), showdiffs, raiseondiff)
        else:
            print("  SKIPPED (could not find table)")

    def expressions(self):
        for _, p in self.subprocesses:
            for e in p.expressions():
                yield e

    def ssa(self, fields_versions):
        function_vars = set(k for k, p in self.subprocesses if k is not None)
        global_vars = set(self.entity.variables.keys())
        local_vars = function_vars - global_vars

        local_versions = collections.defaultdict(int)
        for k, p in self.subprocesses:
            # mark all variables in the expression with their current version
            for expr in p.expressions():
                for node in expr.all_of(Variable):
                    versions = (local_versions if node.name in local_vars
                                else fields_versions)
                    # FIXME: for .version to be meaningful, I need to have
                    # a different variable instance each time the variable
                    # is used.
                    #>>> the best solution AFAIK is to parse the expressions
                    # in the same order as the "agespine".
                    # That way we will be able to type all temporary variables
                    # directly, and it would also solve the conditional
                    # context hack. There is no problem with temporary variables
                    # having different types over their lifetimes as these
                    # will actually be different variables (because their
                    # version will be different). There is no problem with
                    # a variable being different in two control flow
                    # "branches", because we do not have that case: if() coerce
                    # types.
                    # note that even if a branch is never "taken" (an if
                    # condition that is always True or always False or a forloop
                    # without any iteration), the type of the expressions
                    # that variables are assigned to in that branch will
                    # influence the type of the variable in subsequent code.
                    # XXX: what if I have a user-defined function that
                    # I call from two different places with an argument of a
                    # different type? ideally, it should generate two distinct
                    # functions, but I am not there yet. Having a check on the
                    # second call that the argument passed is of the same type
                    # than the signature type (which was inferred from the
                    # first call) seems enough for now.
                    node.version = versions[node.name]
                    node.used += 1
            # on assignment, increase the variable version
            if isinstance(p, Assignment):
                # XXX: is this always == k?
                target = p.predictor
                versions = (local_versions if target in local_vars
                            else fields_versions)
                versions[target] += 1


class Function(Process):
    """this class implements user-defined functions"""

    def __init__(self, name, entity, argnames, code=None, result=None):
        """
        args -- a list of strings
        code -- a ProcessGroup (or None)
        result -- an Expr (or None)
        """
        Process.__init__(self, name, entity)

        assert isinstance(argnames, list)
        assert all(isinstance(a, basestring) for a in argnames)
        self.argnames = argnames

        assert code is None or isinstance(code, ProcessGroup)
        self.code = code

        assert result is None or isinstance(result, Expr)
        self.result = result

    def run_guarded(self, context, *args, **kwargs):
        # XXX: wouldn't some form of cascading context make all this junk much
        # cleaner? Context(globalvars, localvars) (globalvars contain both
        # entity fields and global temporaries)

        backup = self.backup_and_purge_locals()

        if len(args) != len(self.argnames):
            raise TypeError("takes exactly %d arguments (%d given)" %
                            (len(self.argnames), len(args)))

        for name in self.argnames:
            if name in self.entity.fields.names:
                raise ValueError("function '%s' cannot have an argument named "
                                 "'%s' because there is a field with the "
                                 "same name" % (self.name, name))

        # contextual filter should not transfer to the called function (even
        # if that would somewhat make sense) because in many cases the
        # columns used in the contextual filter are not available within the
        # called function. This is only relevant for functions called within
        # an if() expression.
        context = context.clone(filter_expr=None)

        # add arguments to the local namespace
        for name, value in zip(self.argnames, args):
            # backup the variable if it existed in the caller namespace
            if name in self.entity.temp_variables:
                # we can safely assign to backup without checking if that name
                # was already assigned because it is not possible for a variable
                # to be both in entity.temp_variables and in backup (they are
                # removed from entity.temp_variables).
                backup[name] = self.entity.temp_variables.pop(name)

            # cannot use context[name] = value because that would store the
            # value in .extra, which is wiped at the start of each process
            # and we need it to be available across all processes of the
            # function
            self.entity.temp_variables[name] = value
        try:
            self.code.run_guarded(context)
            result = expr_eval(self.result, context)
        except ReturnException as r:
            result = r.result
        self.purge_and_restore_locals(backup)
        return result

    def expressions(self):
        if self.code is not None:
            for e in self.code.expressions():
                yield e
        if self.result is not None:
            yield self.result

    def backup_and_purge_locals(self):
        # backup and purge local variables
        backup = {}
        for name in self.entity.local_var_names:
            backup[name] = self.entity.temp_variables.pop(name)
        return backup

    def purge_and_restore_locals(self, backup):
        # purge the local from the function we just ran
        self.entity.purge_locals()
        # restore local variables for our caller
        for k, v in backup.iteritems():
            self.entity.temp_variables[k] = v
