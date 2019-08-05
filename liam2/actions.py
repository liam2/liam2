# encoding: utf-8
from __future__ import absolute_import, division, print_function

import os
import csv

import numpy as np
import larray as la

from liam2 import config
from liam2.compat import csv_open
from liam2.context import EvaluationContext, EntityContext
from liam2.expr import FunctionExpr, expr_cache, expr_eval
from liam2.process import BreakpointException
from liam2.partition import filter_to_indices
from liam2.utils import FileProducer, merge_dicts, PrettyTable, ndim, isnan


class Show(FunctionExpr):
    def evaluate(self, context):
        if config.skip_shows:
            if config.log_level == "processes":
                print("show skipped", end=' ')
        else:
            super(Show, self).evaluate(context)

    def compute(self, context, *args):
        print(' '.join(str(v) for v in args), end=' ')


class QuickShow(Show):
    funcname = 'qshow'

    def compute(self, context, *args):
        titles = [str(expr) for expr in self.args]
        print('\n'.join('%s: %s' % (title, value)
                        for title, value in zip(titles, args)),
              end=' ')


class CSV(FunctionExpr, FileProducer):
    kwonlyargs = merge_dicts(FileProducer.argspec.kwonlydefaults, mode='w')
    ext = '.csv'
    fname_required = True

    def compute(self, context, *args, **kwargs):
        table = (la.Array, PrettyTable)
        sequence = (list, tuple)
        if (len(args) > 1 and
                not any(isinstance(arg, table + sequence) for arg in args)):
            args = (args,)

        mode = kwargs.pop('mode', 'w')
        if mode not in ('w', 'a'):
            raise ValueError("csv() mode argument must be either "
                             "'w' (overwrite) or 'a' (append)")

        fname = self._get_fname(kwargs)

        entity = context.entity
        period = context.period
        fname = fname.format(entity=entity.name, period=period)
        if config.log_level in ("functions", "processes"):
            print("writing to", fname, "...", end=' ')

        file_path = os.path.abspath(os.path.join(config.output_directory, fname))
        with csv_open(file_path, mode) as f:
            writer = csv.writer(f)
            for arg in args:
                # make sure the result is at least two-dimensional
                if isinstance(arg, la.Array):
                    arg = arg.dump()
                elif isinstance(arg, PrettyTable):
                    pass
                else:
                    dims = ndim(arg)
                    if dims == 0:
                        arg = [[arg]]
                    if dims == 1:
                        arg = [arg]
                writer.writerows(arg)


def shrink_array_dict(d, not_removed_indices, old_array_axes, new_array_axes):
    assert isinstance(not_removed_indices, np.ndarray)
    old_id_axis = old_array_axes.id
    len_before = len(old_id_axis)
    for name, value in d.items():
        # This is brittle but there is nothing I can do about it now. Ideally, we should disallow
        # storing expressions which do not result in a scalar or per-individual value
        # (eg expressions using global arrays) in entity.temporary_variables
        # the problem is that users currently do not have any other choice in this regard.
        # globals are not writable/there are no globals.temporary variables nor global processes nor global macros
        # see issue #250.
        if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == len_before:
            # TODO: we should make sure this case never happens !
            d[name] = value[not_removed_indices]
        elif isinstance(value, la.Array) and value.axes == old_array_axes:
            d[name] = la.Array(value.data[not_removed_indices], new_array_axes)
        # this case can currently actually happen (see above comment and issue #250)
        # elif isinstance(value, la.Array):
        #   pass


class RemoveIndividuals(FunctionExpr):
    def compute(self, context, filter=None):
        assert isinstance(context, EvaluationContext)

        entity = context.entity
        old_array_axes = entity.array.axes
        len_before = len(entity.array)

        filter_value = filter
        if isinstance(filter_value, la.Array):
            filter_value = filter_value.data

        if filter_value is None:
            # this is inefficient, but remove without filter is not common enough to bother
            filter_value = np.ones(old_array_axes.shape, dtype=bool)
        elif not filter_value.any():
            return

        assert isinstance(filter_value, np.ndarray) and filter_value.ndim == 1
        not_removed = ~filter_value
        not_removed_indices = filter_to_indices(not_removed)

        # Shrink array & temporaries. 99% of the function time is spent here.
        entity.array.keep(not_removed_indices)
        shrink_array_dict(entity.temp_variables, not_removed_indices, old_array_axes, entity.array.axes)

        # update id_to_rownum
        already_removed = entity.id_to_rownum == -1
        already_removed_indices = filter_to_indices(already_removed)
        already_removed_indices_shifted = \
            already_removed_indices - np.arange(len(already_removed_indices))

        id_to_rownum = np.arange(len_before)
        id_to_rownum -= filter_value.cumsum()
        # XXX: use np.putmask(id_to_rownum, filter_value, -1)
        id_to_rownum[filter_value] = -1
        entity.id_to_rownum = np.insert(id_to_rownum,
                                        already_removed_indices_shifted,
                                        -1)

        # this version is cleaner and slightly faster but the result is also
        # slightly wrong: it eliminates ids for dead/removed individuals at
        # the end of the array and this cause bugs in time-related functions
#        ids = entity.array['id']
#        id_to_rownum = np.full(np.max(ids) + 1, -1, dtype=int)
#        id_to_rownum[ids] = np.arange(len(ids), dtype=int)
#        entity.id_to_rownum = id_to_rownum
        if config.log_level == "processes":
            print("%d %s(s) removed (%d -> %d)"
                  % (filter_value.sum(), entity.name, len_before,
                     len(entity.array)),
                  end=' ')

        # TODO: in the case of remove(), we should update (take a subset of) all
        # the cache keys matching the entity, but with the current code,
        # it is most likely not worth it because the cache probably contains
        # mostly stuff we will never use.
        expr_cache.invalidate(context.period, context.entity_name)


class Breakpoint(FunctionExpr):
    def compute(self, context, period=None):
        if period is None or period == context.period:
            raise BreakpointException()


class Assert(FunctionExpr):
    # subclasses should have msg in their no_eval attribute. We delay evaluating it because it can potentially be
    # costly to compute. e.g. when using msg=dump()
    no_eval = ('msg',)

    def __init__(self, *args, **kwargs):
        assert self.argspec.args[-1] == 'msg', \
               "%s.compute MUST have 'msg' as its last argument" % self.__class__.__name__
        FunctionExpr.__init__(self, *args, **kwargs)

    def evaluate(self, context):
        if config.assertions == "skip":
            if config.log_level == "processes":
                print("assertion skipped", end=' ')
        else:
            args, kwargs = self._eval_args(context)
            if config.log_level == "processes":
                print("assertion", end=' ')
            failure = self.compute(context, *args, **kwargs)
            if failure:
                # evaluate msg. It MUST be the last argument.
                msg = expr_eval(args[-1], context)
                if msg is None:
                    msg = failure
                else:
                    if isinstance(msg, tuple):
                        msg = ' '.join(str(v) for v in msg)
                    msg = '{}: {}'.format(failure, msg)
                if config.assertions == "warn":
                    # if config.log_level == "processes":
                    print("FAILED:", msg, end=' ')
                else:
                    raise AssertionError(msg)
            else:
                if config.log_level == "processes":
                    print("ok", end=' ')

    # any (direct) subclass MUST have a compute method with "msg" as its the last argument.
    def compute(self, context, msg=None):
        raise NotImplementedError()


class AssertTrue(Assert):
    funcname = 'assertTrue'

    def compute(self, context, value, msg=None):
        if not value:
            return str(self.args[0]) + " is not True"


class AssertFalse(Assert):
    funcname = 'assertFalse'

    def compute(self, context, value, msg=None):
        if value:
            return str(self.args[0]) + " is not False"


class ComparisonAssert(Assert):
    inv_op = None

    def compute(self, context, v1, v2, msg=None):
        result = self.compare(v1, v2)
        if isinstance(result, tuple):
            result, details = result
        else:
            details = ''
        if not result:
            op = self.inv_op
            # use %r to print values. At least for floats on python2, this yields to a better precision.
            return "%s %s %s (%r %s %r)%s" % (self.args[0], op, self.args[1],
                                              v1, op, v2, details)

    def compare(self, v1, v2):
        raise NotImplementedError()


class AssertEqual(ComparisonAssert):
    funcname = 'assertEqual'
    inv_op = "!="

    def compare(self, v1, v2):
        # even though np.array_equal also works on scalars, we don't use it
        # systematically because it does not work on list of strings
        if (isinstance(v1, (np.ndarray, la.Array)) or
                isinstance(v2, (np.ndarray, la.Array))):
            v1, v2 = np.asarray(v1), np.asarray(v2)
            if v1.shape != v2.shape:
                return False, ' (shape differ: %s vs %s)' % (v1.shape, v2.shape)
            result = np.array_equal(v1, v2)
            nan_v1, nan_v2 = isnan(v1), isnan(v2)
            if (not result and np.any(nan_v1 | nan_v2) and
                    np.array_equal(nan_v1, nan_v2)):
                return False, ' but arrays contain NaNs, did you meant to ' \
                              'use assertNanEqual instead?'
            else:
                return result
        else:
            return v1 == v2


class AssertNanEqual(ComparisonAssert):
    funcname = 'assertNanEqual'
    inv_op = "!="

    def compare(self, v1, v2):
        both_nan = np.isnan(v1) & np.isnan(v2)
        return np.all(both_nan | (v1 == v2))


class AssertEquiv(ComparisonAssert):
    funcname = 'assertEquiv'
    inv_op = "is not equivalent to"

    def compare(self, v1, v2):
        return np.array_equiv(v1, v2)


class AssertIsClose(ComparisonAssert):
    funcname = 'assertIsClose'
    inv_op = "is not close to"

    def compare(self, v1, v2):
        return np.allclose(v1, v2)


class AssertRaises(Assert):
    funcname = 'assertRaises'
    no_eval = ('expr', 'msg')

    def compute(self, context, exception, expr, msg=None):
        expected_exception = eval(exception)
        try:
            expr_eval(expr, context)
            return "expression did not raise any exception"
        except expected_exception as e:
            if config.debug:
                print("exception raised (as expected): %r" % e, end=' ')
            return False
        except Exception as e:
            return "expression raised another exception (%r)" % e


functions = {
    'csv': CSV,
    # can't use "print" in python 2.x because it's a keyword, not a function
    'show': Show,
    'qshow': QuickShow,
    'remove': RemoveIndividuals,
    'breakpoint': Breakpoint,
    'assertTrue': AssertTrue,
    'assertFalse': AssertFalse,
    'assertEqual': AssertEqual,
    'assertNanEqual': AssertNanEqual,
    'assertEquiv': AssertEquiv,
    'assertIsClose': AssertIsClose,
    'assertRaises': AssertRaises
}
