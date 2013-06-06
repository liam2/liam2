from __future__ import print_function

import os
import csv

import numpy as np

import config
from expr import Expr, expr_eval
from exprbases import TableExpression
from process import Process, BreakpointException
from partition import filter_to_indices
from utils import LabeledArray


class Show(Process):
    def __init__(self, *args, **kwargs):
        Process.__init__(self)
        self.args = args
        self.print_exprs = kwargs.pop('print_exprs', False)
        if kwargs:
            kwarg, _ = kwargs.popitem()
            raise TypeError("'%s' is an invalid keyword argument for show()"
                            % kwarg)

    def expressions(self):
        for arg in self.args:
            if isinstance(arg, Expr):
                yield arg

    def run(self, context):
        if config.skip_shows:
            print("show skipped", end=' ')
        else:
            values = [expr_eval(expr, context) for expr in self.args]
            if self.print_exprs:
                titles = [str(expr) for expr in self.args]
                print('\n'.join('%s: %s' % (title, value)
                                for title, value in zip(titles, values)), end=' ')
            else:
                print(' '.join(str(v) for v in values), end=' ')

    def __str__(self):
        #TODO: the differentiation shouldn't be needed. I guess I should
        # have __repr__ defined for all expressions
        str_args = [str(arg) if isinstance(arg, Expr) else repr(arg)
                    for arg in self.args]
        return 'show(%s)' % ', '.join(str_args)


class QuickShow(Show):
    def __init__(self, *args):
        Show.__init__(self, *args, print_exprs=True)

    def __str__(self):
        return Show.__str__(self).replace('show(', 'qshow(')


class CSV(Process):
    def __init__(self, *args, **kwargs):
        Process.__init__(self)
        if (len(args) > 1 and
            not any(isinstance(arg, (TableExpression, list, tuple))
                    for arg in args)):
            args = (args,)
        self.args = args
        suffix = kwargs.pop('suffix', '')
        fname = kwargs.pop('fname', None)
        mode = kwargs.pop('mode', 'w')
        if kwargs:
            kwarg, _ = kwargs.popitem()
            raise TypeError("'%s' is an invalid keyword argument for csv()"
                            % kwarg)

        if fname is not None and suffix:
            raise ValueError("csv() can't have both 'suffix' and 'fname' "
                             "arguments")
        if fname is None:
            suffix = "_" + suffix if suffix else ""
            fname = "{entity}_{period}" + suffix + ".csv"
        self.fname = fname
        if mode not in ('w', 'a'):
            raise ValueError("csv() mode argument must be either "
                             "'w' (overwrite) or 'a' (append)")
        self.mode = mode

    def expressions(self):
        for arg in self.args:
            if isinstance(arg, (list, tuple)):
                for expr in arg:
                    if isinstance(expr, Expr):
                        yield expr
            elif isinstance(arg, Expr):
                yield arg

    def run(self, context):
        entity = context['__entity__']
        period = context['period']
        fname = self.fname.format(entity=entity.name, period=period)
        print("writing to", fname, "...", end=' ')
        file_path = os.path.join(config.output_directory, fname)

        with open(file_path, self.mode + 'b') as f:
            dataWriter = csv.writer(f)
            for arg in self.args:
                #XXX: use py3.4 singledispatch?
                if isinstance(arg, TableExpression):
                    data = expr_eval(arg, context)
                    if isinstance(data, LabeledArray):
                        data = data.as_table()
                elif isinstance(arg, (list, tuple)):
                    data = [[expr_eval(expr, context) for expr in arg]]
                else:
                    data = [[expr_eval(arg, context)]]
                dataWriter.writerows(data)


class RemoveIndividuals(Process):
    def __init__(self, filter):
        Process.__init__(self)
        self.filter = filter

    def expressions(self):
        yield self.filter

    def run(self, context):
        filter_value = expr_eval(self.filter, context)

        if not np.any(filter_value):
            return

        not_removed = ~filter_value

        entity = context['__entity__']
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
        already_removed_indices_shifted = already_removed_indices - \
                                  np.arange(len(already_removed_indices))

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
                                               len_before, len(entity.array)), end=' ')


class Breakpoint(Process):
    def __init__(self, period=None):
        Process.__init__(self)
        self.period = period

    def run(self, context):
        if self.period is None or self.period == context['period']:
            raise BreakpointException()

    def __str__(self):
        if self.period is not None:
            return 'breakpoint(%d)' % self.period
        else:
            return ''

    def expressions(self):
        return ()


class Assert(Process):
    def eval_assertion(self, context):
        raise NotImplementedError()

    def run(self, context):
        if config.assertions == "skip":
            print("assertion skipped", end=' ')
        else:
            print("assertion", end=' ')
            failure = self.eval_assertion(context)
            if failure:
                if config.assertions == "warn":
                    print("FAILED:", failure, end=' ')
                else:
                    raise AssertionError(failure)
            else:
                print("ok", end=' ')


class AssertTrue(Assert):
    def __init__(self, expr):
        Process.__init__(self)
        self.expr = expr

    def eval_assertion(self, context):
        if not expr_eval(self.expr, context):
            return str(self.expr)

    def expressions(self):
        if isinstance(self.expr, Expr):
            yield self.expr


class ComparisonAssert(Assert):
    def __init__(self, expr1, expr2):
        Process.__init__(self)
        self.expr1 = expr1
        self.expr2 = expr2

    def eval_assertion(self, context):
        v1 = expr_eval(self.expr1, context)
        v2 = expr_eval(self.expr2, context)
        if not self.compare(v1, v2):
            op = self.inv_op
            return "%s %s %s (%s %s %s)" % (self.expr1, op, self.expr2,
                                            v1, op, v2)

    def compare(self, v1, v2):
        raise NotImplementedError()

    def expressions(self):
        if isinstance(self.expr1, Expr):
            yield self.expr1
        if isinstance(self.expr2, Expr):
            yield self.expr2


class AssertEqual(ComparisonAssert):
    inv_op = "!="

    def compare(self, v1, v2):
        # even though np.array_equal also works on scalars, we don't use it
        # systematically because it does not work on list of strings
        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
            return np.array_equal(v1, v2)
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
    # can't use "print" in python 2.x because it's a keyword, not a function
#    'print': Print,
    'csv': CSV,
    'show': Show,
    'qshow': QuickShow,
    'remove': RemoveIndividuals,
    'breakpoint': Breakpoint,
    'assertTrue': AssertTrue,
    'assertEqual': AssertEqual,
    'assertNanEqual': AssertNanEqual,
    'assertEquiv': AssertEquiv,
    'assertIsClose': AssertIsClose
}
