from __future__ import print_function

import os
import csv

import numpy as np

import config
from expr import FunctionExpr
from process import BreakpointException
from partition import filter_to_indices
from utils import LabeledArray, FileProducer, merge_dicts, PrettyTable


class Show(FunctionExpr):
    def evaluate(self, context):
        if config.skip_shows:
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
        table = (LabeledArray, PrettyTable)
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
        print("writing to", fname, "...", end=' ')

        file_path = os.path.join(config.output_directory, fname)
        with open(file_path, mode + 'b') as f:
            writer = csv.writer(f)
            sequence = (list, tuple)
            for arg in args:
                if isinstance(arg, LabeledArray):
                    arg = arg.as_table()
                elif isinstance(arg, sequence):
                    arg = [arg]
                elif isinstance(arg, PrettyTable):
                    pass
                else:
                    arg = [[arg]]
                writer.writerows(arg)


class RemoveIndividuals(FunctionExpr):
    def compute(self, context, filter):
        filter_value = filter
        if not np.any(filter_value):
            return

        not_removed = ~filter_value

        entity = context.entity
        len_before = len(entity.array)

        # Shrink array & temporaries. 99% of the function time is spent here.
        entity.array.keep(not_removed)
        temp_variables = entity.temp_variables
        for name, temp_value in temp_variables.items():
            if isinstance(temp_value, np.ndarray) and temp_value.shape:
                temp_variables[name] = temp_value[not_removed]

        # update id_to_rownum
        already_removed = entity.id_to_rownum == -1
        already_removed_indices = filter_to_indices(already_removed)
        already_removed_indices_shifted = \
            already_removed_indices - np.arange(len(already_removed_indices))

        id_to_rownum = np.arange(len_before)
        id_to_rownum -= filter_value.cumsum()
        #XXX: use np.putmask(id_to_rownum, filter_value, -1)
        id_to_rownum[filter_value] = -1
        entity.id_to_rownum = np.insert(id_to_rownum,
                                        already_removed_indices_shifted,
                                        -1)
        # this version is cleaner and slightly faster but the result is also
        # slightly wrong: it eliminates ids for dead/removed individuals at
        # the end of the array and this cause bugs in time-related functions
#        ids = entity.array['id']
#        id_to_rownum = np.empty(np.max(ids) + 1, dtype=int)
#        id_to_rownum.fill(-1)
#        id_to_rownum[ids] = np.arange(len(ids), dtype=int)
#        entity.id_to_rownum = id_to_rownum

        print("%d %s(s) removed (%d -> %d)" % (filter_value.sum(), entity.name,
                                               len_before, len(entity.array)),
              end=' ')


class Breakpoint(FunctionExpr):
    def compute(self, context, period=None):
        if period is None or period == context.period:
            raise BreakpointException()


class Assert(FunctionExpr):
    def eval_assertion(self, context, *args):
        raise NotImplementedError()

    def compute(self, context, *args):
        if config.assertions == "skip":
            print("assertion skipped", end=' ')
        else:
            print("assertion", end=' ')
            failure = self.eval_assertion(context, *args)
            if failure:
                if config.assertions == "warn":
                    print("FAILED:", failure, end=' ')
                else:
                    raise AssertionError(failure)
            else:
                print("ok", end=' ')


class AssertTrue(Assert):
    def eval_assertion(self, context, value):
        if not value:
            return str(self.args[0]) + " is not True"


class AssertFalse(Assert):
    def eval_assertion(self, context, value):
        if value:
            return str(self.args[0]) + " is not False"


class ComparisonAssert(Assert):
    inv_op = None

    def eval_assertion(self, context, v1, v2):
        result = self.compare(v1, v2)
        if isinstance(result, tuple):
            result, details = result
        else:
            details = ''
        if not result:
            op = self.inv_op
            return "%s %s %s (%s %s %s)%s" % (self.args[0], op, self.args[1],
                                              v1, op, v2, details)

    def compare(self, v1, v2):
        raise NotImplementedError()


#TODO: move to utils
def isnan(a):
    """
    isnan is equivalent to np.isnan, except that it returns False instead of
    raising a TypeError if the argument is an array of non-numeric.
    """
    if isinstance(a, np.ndarray):
        return np.issubsctype(a, np.floating) and np.isnan(a)
    else:
        return np.isnan(a)


class AssertEqual(ComparisonAssert):
    inv_op = "!="

    def compare(self, v1, v2):
        # even though np.array_equal also works on scalars, we don't use it
        # systematically because it does not work on list of strings
        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
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
    inv_op = "!="

    def compare(self, v1, v2):
        both_nan = np.isnan(v1) & np.isnan(v2)
        return np.all(both_nan | (v1 == v2))


class AssertEquiv(ComparisonAssert):
    inv_op = "is not equivalent to"

    def compare(self, v1, v2):
        return np.array_equiv(v1, v2)


class AssertIsClose(ComparisonAssert):
    inv_op = "is not close to"

    def compare(self, v1, v2):
        return np.allclose(v1, v2)


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
    'assertIsClose': AssertIsClose
}
