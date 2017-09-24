# encoding: utf-8
from __future__ import print_function

import ast
from collections import defaultdict, deque, namedtuple
import itertools
from itertools import izip, product
import math
import operator
from textwrap import wrap
import re
import sys
import time
import warnings

import numpy as np
import numexpr as ne
try:
    from PyQt4 import QtGui, QtCore
    QtAvailable = True
except ImportError:
    QtGui, QtCore = None, None
    QtAvailable = False

import config


def make_hashable(obj):
    if isinstance(obj, (list, tuple)):
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, slice):
        return (slice, obj.start, obj.stop, obj.step)
    else:
        return obj


class classproperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()


class ExceptionOnGetAttr(object):
    """
    ExceptionOnGetAttr can be used when an optional part is missing
    so that an exception is only raised if the object is actually used.
    """
    def __init__(self, exception):
        self.exception = exception

    def __getattr__(self, key):
        raise self.exception


class UserDeprecationWarning(UserWarning):
    pass


def deprecated(f, old=None, new=None, msg=None):
    assert old is not None or msg is not None
    if msg is None:
        if new is None:
            new = f.__name__
        msg = "%s is deprecated, please use %s instead" % (old, new)

    def func(*args, **kwargs):
        # TODO: when we will be able to link expressions to line numbers in the
        # model, we should use warnings.warn_explicit instead
        warnings.warn(msg, UserDeprecationWarning)
        return f(*args, **kwargs)
    func.__name__ = "__deprecated_" + f.__name__
    return func


def removed(f, old=None, new=None, msg=None):
    assert old is not None or msg is not None
    if msg is None:
        if new is None:
            new = f.__name__
        msg = "%s does not exist anymore, please use %s instead" % (old, new)

    def func(*args, **kwargs):
        raise SyntaxError(msg)
    func.__name__ = "__removed_" + f.__name__
    return func


def find_first(char, s, depth=0):
    """
    returns the position of the first occurrence of the 'ch' character at
    'depth' depth in the 's' string.
    returns -1 if not found
    raises ValueError if the string contains imbalanced parentheses or brackets
    """
    match = {'(': ')', '[': ']'}
    opening, closing = set(match), set(match.values())
    stack = []
    for i, c in enumerate(s):
        if c == char and len(stack) == depth:
            return i
        if c in opening:
            stack.append(match[c])
        elif c in closing:
            if not stack or c != stack.pop():
                raise ValueError("syntax error: imbalanced parentheses or "
                                 "brackets in string: %s" % s)
    if stack:
        raise ValueError("syntax error: missing parenthesis or "
                         "bracket in string: %s" % s)
    return -1


def englishenum(iterable):
    """
    Returns an "english enumeration" of the strings in the iterable.
    >>> englishenum(['a', 'b', 'c'])
    'a, b, and c'
    >>> englishenum('abc')
    'a, b, and c'
    >>> englishenum('ab')
    'a and b'
    >>> englishenum('a')
    'a'
    """
    l = list(iterable)
    if len(l) == 2:
        return '%s and %s' % tuple(l)
    elif len(l) > 2:
        l[-1] = 'and ' + l[-1]
    return ', '.join(l)


class AutoFlushFile(object):
    def __init__(self, f):
        self.f = f

    def write(self, s):
        self.f.write(s)
        self.f.flush()

    def __getattr__(self, key):
        return getattr(self.f, key)


def time2str(seconds, precise=True):
    minutes = seconds // 60
    hours = minutes // 60
    seconds %= 60
    minutes %= 60
    l = []
    if hours > 0:
        l.append("%d hour%s" % (hours, 's' if hours > 1 else ''))
    if minutes > 0:
        l.append("%d minute%s" % (minutes, 's' if minutes > 1 else ''))
    if precise:
        if seconds >= 0.005:
            l.append("%.2f second%s" % (seconds, 's' if seconds > 1 else ''))
        elif not l:
            l = ["%d ms" % (seconds * 1000)]
    else:
        if int(seconds) or not l:
            l.append("%d second%s" % (seconds, 's' if seconds > 1 else ''))

    return ' '.join(l)


def size2str(value):
    """
    >>> size2str(0)
    '0 bytes'
    >>> size2str(100)
    '100 bytes'
    >>> size2str(1023)
    '1023 bytes'
    >>> size2str(1024)
    '1.00 Kb'
    >>> size2str(2000)
    '1.95 Kb'
    >>> size2str(10000000)
    '9.54 Mb'
    >>> size2str(1.27 * 1024 ** 3)
    '1.27 Gb'
    """
    units = ["bytes", "Kb", "Mb", "Gb", "Tb", "Pb"]
    scale = int(math.log(value, 1024)) if value else 0
    fmt = "%.2f %s" if scale else "%d %s"
    return fmt % (value / 1024.0 ** scale, units[scale])


# def mem_usage():
#    pid = os.getpid()
#    proc = psutil.Process(pid)
#    return proc.get_memory_info()[0]


# def mem_usage_str():
#    return size2str(mem_usage())


def gettime(func, *args, **kwargs):
    start = time.time()
    res = func(*args, **kwargs)
    return time.time() - start, res


def timed(func, *args, **kwargs):
    elapsed, res = gettime(func, *args, **kwargs)
    if config.show_timings:
        print("done (%s elapsed)." % time2str(elapsed))
    else:
        print("done.")
    return res


def prod(values):
    return reduce(operator.mul, values, 1)


def ndim(arraylike):
    """
    Computes the number of dimensions of arbitrary structures, including
    sequence of arrays and array of sequences.
    """
    n = 0
    while isinstance(arraylike, (list, tuple, np.ndarray)):
        if len(arraylike) == 0:
            raise ValueError('Cannot compute ndim of array with empty dim')
        # XXX: check that other elements have the same length?
        arraylike = arraylike[0]
        n += 1
    return n


def safe_put(a, ind, v):
    """
    np.put but where values corresponding to -1 indices are ignored,
    instead of being copied to the last position
    assumes indices in ind are sorted !
    """
    from data import ColumnArray

    if not len(a) or not len(ind):
        return
    # backup last value, in case it gets overwritten
    last_value = a[-1]
    if isinstance(v, ColumnArray):
        for fname in a.dtype.names:
            safe_put(a[fname], ind, v[fname])
    else:
        a.put(ind, v)
    # if the last value was erroneously modified (because of one -1 in ind)
    # this assumes indices are sorted
    if ind[-1] != len(a) - 1:
        # restore its previous value
        a[-1] = last_value


def safe_take(a, indices, missing_value):
    """
    like np.take but out-of-bounds indices return the missing value
    """
    indexed = a.take(indices, mode='clip')
    return ne.evaluate('where((idx < 0) | (idx >= maxidx), missing, indexed)',
                       {'idx': indices, 'maxidx': len(a),
                        'missing': missing_value, 'indexed': indexed})


# we provide our own version of fromiter because it swallows any exception
# occurring within the iterable if the count argument is used
try:
    from cutils import fromiter
except ImportError:
    def fromiter(iterable, dtype, count=-1):
        if count == -1:
            return np.fromiter(iterable, dtype)
        else:
            buf = np.empty(count, dtype=dtype)
            i = 0
            for e in iterable:
                buf[i] = e
                i += 1
                if i == count:
                    break
            if i < count:
                raise ValueError("iterator too short")
            return buf


# this is a workaround because nansum(bool_array) fails on numpy 1.7.0
# see https://github.com/numpy/numpy/issues/2978
# as a bonus, this version is also faster
def nansum(a, axis=None):
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.inexact):
        return np.nansum(a, axis)
    else:
        return np.sum(a, axis)


def expand(value, shape):
    """Expands value to shape if necessary

    Parameters
    ----------
    value : scalar or ndarray
        value to expand
    shape : int or tuple of int
        shape to expand to if value is a scalar or 0d array

    Returns
    -------
    expanded array
    """
    if np.isscalar(shape):
        shape = (shape,)
    if isinstance(value, np.ndarray) and value.shape:
        # assert value.shape == shape, "%s != %s" % (value.shape, shape)
        # FIXME: this assertion fails because we are sloppy in
        # AlignmentAbsoluteValues.align_link (target_context is not filtered)
        return value
    else:
        return np.full(shape, value)


# TODO: provide a cython version for this (using fused types)
# ctypedef fused np_numeric:
#     np.int32_t
#     np.int64_t
#     np.float32_t
#     np.float64_t
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def bool isconstant(np.ndarray[np_numeric, ndim=1] a):
#    cdef:
#        np.float64_t value
#        Py_ssize_t i, n=a.size
#    value = a[0]
#    for i in range(1, n):
#        if a[i] != value:
#            return False
#    return True
def isconstant(a, filter_value=None):
    if filter_value is None:
        return a.max() == a.min()
    else:
        value = a[0]
        return np.all(filter_value & (a == value))


def _make_aggregate(func):
    def method(self, axis=None):
        if axis == 1:
            result = np.empty(len(self.data), dtype=self.data[0].dtype)
            for i, a in enumerate(self.data):
                result[i] = func(a)
            return result
        else:
            raise NotImplementedError("axis != 1")
    return method


class IrregularNDArray(object):
    """
    A wrapper for collections of arrays (eg list of arrays or arrays of
    arrays) to make them act somewhat like a 2D (numpy) array. This makes it
    possible to have irregular lengths in the second dimension.
    """
    def __init__(self, data):
        self.data = data

    prod = _make_aggregate(np.prod)
    sum = _make_aggregate(np.sum)
    min = _make_aggregate(np.min)
    max = _make_aggregate(np.max)

    def __getattr__(self, key):
        return getattr(self.data, key)

    def __getitem__(self, key):
        return self.data[key]


class Axis(object):
    def __init__(self, name, labels):
        self.name = name
        self.labels = labels

    def __len__(self):
        return len(self.labels)


class LabeledArray(np.ndarray):
    # noinspection PyNoneFunctionAssignment
    def __new__(cls, input_array, dim_names=None, pvalues=None,
                row_totals=None, col_totals=None):
        obj = np.asarray(input_array).view(cls)
        ndim = obj.ndim
        if dim_names is not None and len(dim_names) != ndim:
            raise Exception('number of dimension names (%d) does not match '
                            'number of dimensions (%d)'
                            % (len(dim_names), ndim))
        if pvalues is not None:
            if len(pvalues) != obj.ndim:
                raise Exception('number of label vectors (%d) does not match '
                                'number of dimensions (%d)' % (len(pvalues),
                                                               ndim))
            label_shape = tuple(len(pv) for pv in pvalues)
            if label_shape != obj.shape:
                raise Exception('sizes of label vectors (%s) do not match '
                                'array shape (%s)' % (label_shape, obj.shape))
        if row_totals is not None:
            height = prod(obj.shape[:-1])
            if len(row_totals) != height:
                raise Exception('size of row totals vector (%s) does not '
                                'match array shape (%s)' % (len(row_totals),
                                                            height))
        if col_totals is not None:
            width = obj.shape[-1] if row_totals is None else obj.shape[-1] + 1
            if len(col_totals) != width:
                raise Exception('size of col totals vector (%s) does not '
                                'match array shape (%s)' % (len(col_totals),
                                                            width))
        obj.dim_names = dim_names
        obj.pvalues = pvalues
        obj.row_totals = row_totals
        obj.col_totals = col_totals
        return obj

    @property
    def axes(self):
        if self.dim_names is None or self.pvalues is None:
            return []
        else:
            return [Axis(name, labels)
                    for name, labels in zip(self.dim_names, self.pvalues)]

    def __getitem__(self, key):
        obj = np.ndarray.__getitem__(self, key)
        # I am unsure under which conditions obj is not a LabeledArray, but
        # it *can* happen.
        if obj.ndim > 0 and isinstance(obj, LabeledArray):
            if isinstance(key, (tuple, list)):
                # complete the key if needed
                if len(key) < self.ndim:
                    key = key + (slice(None),) * (self.ndim - len(key))

                # handle fancy indexing (for an nd array)
                if any(isinstance(dim_key, np.ndarray) and dim_key.shape
                       for dim_key in key):
                    obj.pvalues = None
                    obj.dim_names = None
                else:
                    # int key => dimension disappears & pvalues are discarded
                    # slice key => dimension (and pvalues) stays
                    if self.pvalues is not None:
                        pvalues = self.pvalues
                        obj.pvalues = [pv[dim_key]
                                       for pv, dim_key in zip(pvalues, key)
                                       if isinstance(dim_key, slice)]
                        # convert empty list to None (if all dim keys were int)
                        if not obj.pvalues:
                            obj.pvalues = None
                    if self.dim_names is not None:
                        names = self.dim_names
                        obj.dim_names = [name
                                         for name, dim_key in zip(names, key)
                                         if isinstance(dim_key, slice)]
                        # convert empty list to None (if all dim keys were int)
                        if not obj.dim_names:
                            obj.dim_names = None
            elif isinstance(key, slice):
                obj.pvalues = [self.pvalues[0][key]] + [self.pvalues[1:]]
            # handle fancy indexing (for a 1d array)
            elif isinstance(key, np.ndarray):
                obj.dim_names = None
                obj.pvalues = None
            else:
                # assert isinstance(key, int), \
                #        "key: '%s' is of type %s" % (key, type(key))
                # key is "int-like"
                if self.dim_names is not None:
                    obj.dim_names = self.dim_names[1:]
                if self.pvalues is not None:
                    obj.pvalues = self.pvalues[1:]

            # sanity checks
            if obj.dim_names is not None:
                assert len(obj.dim_names) == obj.ndim, \
                       "len(dim_names) (%d) != ndim (%d)" \
                       % (len(obj.dim_names), obj.ndim)
            if obj.pvalues is not None:
                assert len(obj.pvalues) == obj.ndim, \
                       "len(pvalues) (%d) != ndim (%d)" \
                       % (len(obj.pvalues), obj.ndim)
        return obj

    # deprecated since Python 2.0 but we need to define it to catch "simple"
    # slices because ndarray is a "builtin" type
    def __getslice__(self, i, j):
        obj = np.ndarray.__getslice__(self, i, j)
        if self.pvalues is not None:
            obj.pvalues = [self.pvalues[0][slice(i, j)]] + self.pvalues[1:]
        obj.col_totals = None
        obj.row_totals = None
        return obj

    def transpose(self, *args):
        res_data = np.asarray(self)
        if not args:
            args = list(range(self.ndim))[::-1]
        res_data = res_data.transpose(args)
        res_dim_names = [self.dim_names[i] for i in args] if self.dim_names is not None else None
        res_pvalues = [self.pvalues[i] for i in args] if self.pvalues is not None else None
        return LabeledArray(res_data, res_dim_names, res_pvalues)

    # noinspection PyAttributeOutsideInit
    def __array_finalize__(self, obj):
        # We are in the middle of the LabeledArray.__new__ constructor,
        # and our special attributes will be set when we return to that
        # constructor, so we do not need to set them here.
        if obj is None:
            return

        # obj is our "template" object (on which we have asked a view on).
        if isinstance(obj, LabeledArray) and self.shape == obj.shape:
            # obj.view(LabeledArray)
            # labeled_arr[:3]
            self.dim_names = obj.dim_names
            self.pvalues = obj.pvalues
            self.row_totals = obj.row_totals
            self.col_totals = obj.col_totals
        else:
            self.dim_names = None
            self.pvalues = None
            self.row_totals = None
            self.col_totals = None

    def as_table(self):
        if not self.ndim:
            return []

        # gender |      |
        #  False | True | total
        #     20 |   16 |    35

        #   dead | gender |      |
        #        |  False | True | total
        #  False |     20 |   15 |    35
        #   True |      0 |    1 |     1
        #  total |     20 |   16 |    36

        # agegroup | gender |  dead |      |
        #          |        | False | True | total
        #        5 |  False |    20 |   15 |    xx
        #        5 |   True |     0 |    1 |    xx
        #       10 |  False |    25 |   10 |    xx
        #       10 |   True |     1 |    1 |    xx
        #          |  total |    xx |   xx |    xx
        width = self.shape[-1]
        height = prod(self.shape[:-1])
        if self.dim_names is not None:
            result = [self.dim_names +
                      [''] * (width - 1),
                      # 2nd line
                      [''] * (self.ndim - 1) +
                      list(self.pvalues[-1])]
            if self.row_totals is not None:
                result[0].append('')
                result[1].append('total')
        else:
            result = []
        data = np.asarray(self).ravel()

        if self.pvalues is not None:
            categ_values = list(product(*self.pvalues[:-1]))
        else:
            categ_values = [[] for _ in range(height)]
        row_totals = self.row_totals
        for y in range(height):
            # this is a bit wasteful because it creates LabeledArrays for each
            # line, but the waste is insignificant compared to the time to
            # compute the array in the first place
            line = list(categ_values[y]) + \
                   list(data[y * width:(y + 1) * width])
            if row_totals is not None:
                line.append(row_totals[y])
            result.append(line)
        if self.col_totals is not None and self.ndim > 1:
            result.append([''] * (self.ndim - 2) + ['total'] + self.col_totals)
        return result

    def __repr__(self):
        if not self.ndim:
            return str(np.asscalar(self))
        else:
            return '\n' + table2str(self.as_table(), 'nan') + '\n'
    # explicitly defining __str__ is needed here because it exists on ndarray
    __str__ = __repr__

    # def __array_prepare__(self, arr, context=None):
    #     print 'In __array_prepare__:'
    #     print '   self is %s' % repr(self)
    #     print '   arr is %s' % repr(arr)
    #     print '   context is %s' % repr(context)
    #     res = np.ndarray.__array_prepare__(self, arr, context)
    #     print '   result is %s' % repr(res)
    #     return res

    def __array_wrap__(self, out_arr, context=None):
        # print 'In __array_wrap__:'
        # print '   self is %s' % repr(self)
        # print '   arr is %s' % repr(out_arr)
        # print '   context is %s' % repr(context)
        res = np.ndarray.__array_wrap__(self, out_arr, context)
        res.col_totals = None
        res.row_totals = None
        # print '   result is %s' % repr(res)
        return res


def aslabeledarray(data):
    sequence = (tuple, list)
    if isinstance(data, LabeledArray):
        return data
    elif (isinstance(data, sequence) and len(data) and
          isinstance(data[0], LabeledArray)):
        arraydata = np.asarray(data)
        # TODO: check that all arrays have the same axes
        dim_names = [None] + data[0].dim_names
        dim_labels = [range(len(data))] + data[0].pvalues
        return LabeledArray(arraydata, dim_names, dim_labels)
    else:
        arraydata = np.asarray(data)
        dim_names = [None for _ in arraydata.shape]
        dim_labels = [range(d) for d in arraydata.shape]
        return LabeledArray(arraydata, dim_names, dim_labels)


class ProgressBar(object):
    def __init__(self, maximum=100):
        pass

    def update(self, value):
        raise NotImplementedError()

    def destroy(self):
        pass


class TextProgressBar(ProgressBar):
    def __init__(self, maximum=100):
        ProgressBar.__init__(self)
        self.percent = 0
        self.maximum = maximum

    def update(self, value):
        # update progress bar
        percent_done = (value * 100) / self.maximum
        to_display = percent_done - self.percent
        if to_display:
            chars_to_write = list("." * to_display)
            offset = 9 - (self.percent % 10)
            while offset < to_display:
                chars_to_write[offset] = '|'
                offset += 10
            sys.stdout.write(''.join(chars_to_write))
        self.percent = percent_done


def loop_wh_progress(func, sequence, *args, **kwargs):
    pb = TextProgressBar(len(sequence))
    for i, value in enumerate(sequence, start=1):
        try:
            func(i, value, *args, **kwargs)
            pb.update(i)
        except StopIteration:
            break
    pb.destroy()


def count_occurrences(seq):
    counter = defaultdict(int)
    for e in seq:
        counter[e] += 1
    return counter.items()


def skip_comment_cells(lines):
    def notacomment(v):
        return not v.startswith('#')
    for line in lines:
        stripped_line = list(itertools.takewhile(notacomment, line))
        if stripped_line:
            yield stripped_line


def strip_rows(lines):
    """
    returns an iterator of lines with leading and trailing blank (empty or
    which contain only space) cells.
    """
    def isblank(s):
        return s == '' or s.isspace()
    for line in lines:
        leading_dropped = list(itertools.dropwhile(isblank, line))
        rev_line = list(itertools.dropwhile(isblank,
                                            reversed(leading_dropped)))
        yield list(reversed(rev_line))


def format_value(value, missing):
    if isinstance(value, float):
        # nans print as "-1.#J", let's use something nicer
        if value != value:
            return missing
        else:
            return '%.2f' % value
    elif isinstance(value, np.ndarray) and value.shape:
        # prevent numpy default wrapping
        return str(list(value)).replace(',', '')
    else:
        return str(value)


def get_col_width(table, index):
    return max(len(row[index]) for row in table)


def longest_word(s):
    return max(len(w) for w in s.split()) if s else 0


def get_min_width(table, index):
    return max(longest_word(row[index]) for row in table)


def table2str(table, missing):
    """table is a list of lists"""
    if not table:
        return ''
    numcols = max(len(row) for row in table)
    # pad rows that have too few columns
    for row in table:
        if len(row) < numcols:
            row.extend([''] * (numcols - len(row)))
    formatted = [[format_value(value, missing) for value in row]
                 for row in table]
    colwidths = [get_col_width(formatted, i) for i in xrange(numcols)]

    total_width = sum(colwidths)
    sep_width = (len(colwidths) - 1) * 3
    if total_width + sep_width > 80:
        minwidths = [get_min_width(formatted, i) for i in xrange(numcols)]
        available_width = 80.0 - sep_width - sum(minwidths)
        ratio = available_width / total_width
        colwidths = [minw + max(int(width * ratio), 0)
                     for minw, width in izip(minwidths, colwidths)]

    lines = []
    for row in formatted:
        wrapped_row = [wrap(value, width)
                       for value, width in izip(row, colwidths)]
        maxheight = max(len(value) for value in wrapped_row)
        newlines = [[] for _ in range(maxheight)]
        for value, width in izip(wrapped_row, colwidths):
            for i in range(maxheight):
                chunk = value[i] if i < len(value) else ''
                newlines[i].append(chunk.rjust(width))
        lines.extend(newlines)
    return '\n'.join(' | '.join(row) for row in lines)


class PrettyTable(object):
    def __init__(self, iterable, missing=None):
        self.data = list(iterable)
        self.missing = missing

    def __iter__(self):
        if self.missing is not None:
            return iter(self.convert_nans())
        else:
            return iter(self.data)

    def convert_nans(self):
        missing = self.missing
        for row in self.data:
            formatted = [str(value) if value == value else missing
                         for value in row]
            yield formatted

    def __repr__(self):
        missing = self.missing
        if missing is None:
            missing = 'nan'
        else:
            missing = str(missing)
        return '\n' + table2str(self.data, missing) + '\n'


# copied from itertools recipes
def unique(iterable):
    """
    List all elements once, preserving order. Remember all elements ever seen.
    """
    # unique('AAAABBBCCDAABBB') --> A B C D
    seen = set()
    seen_add = seen.add
    for element in iterable:
        if element not in seen:
            seen_add(element)
            yield element


def duplicates(iterable):
    """
    List duplicated elements once, preserving order. Remember all elements ever
    seen.
    """
    # duplicates('AAAABBBCCDAABBB') --> A B C
    counts = defaultdict(int)
    for element in iterable:
        counts[element] += 1
        if counts[element] == 2:
            yield element


def unique_duplicate(iterable):
    """
    List all elements once and duplicated elements, preserving order.
    Remember all elements ever seen.
    """
    # unique_duplicate('AAAABBBCCDAABBB') --> [A, B, C, D], [A, B, C]
    counts = {}
    uniques = []
    dupes = []
    append_uniques = uniques.append
    append_dupes = dupes.append
    for element in iterable:
        count = counts[element] = counts.get(element, 0) + 1
        if count == 2:
            append_dupes(element)
        elif count == 1:
            append_uniques(element)
    return uniques, dupes


# adapted from pseudo code of itertools.tee
def split_columns_as_iterators(iterable):
    iterator = iter(iterable)
    header = iterator.next()
    numcolumns = len(header)
    # deque (used as a FIFO queue) for each column (so that each iterator does
    # not need to advance at the same speed. However in that case the memory
    # consumption can be high.
    deques = [deque() for _ in range(numcolumns)]

    def gen(deq):
        while True:
            if not deq:                # when the local queue is empty
                row = next(iterator)       # fetch a new row and
                if len(row) != numcolumns:
                    raise Exception("all rows do not have the same number of "
                                    "columns")
                # dispatch it to the other queues
                for queue, value in zip(deques, row):
                    queue.append(value)
            yield deq.popleft()
    return tuple(gen(d) for d in deques)


def countlines(filepath):
    with open(filepath) as f:
        return sum(1 for _ in f)


# dict tools
# ----------

class WarnOverrideDict(dict):
    def update(self, other=None, **kwargs):
        # copy the items to not lose them in case it is an exhaustible
        # iterable
        # also converts list and tuple to dict
        if not isinstance(other, dict):
            other = dict(other)
        self._intersect_warn(self, other)
        self._intersect_warn(self, kwargs)
        self._intersect_warn(other, kwargs)
        dict.update(self, other, **kwargs)

    @staticmethod
    def _intersect_warn(d1, d2):
        intersect = set(d1.keys()) & set(d2.keys())
        if intersect:
            print("Warning: name collision for:",
                  ",".join("%s (%s vs %s)" % (k, type(d1[k]), d2[k])
                           for k in sorted(intersect)))


def merge_dicts(*args, **kwargs):
    """
    Returns a new dictionary which is the result of recursively merging all
    the dictionaries passed as arguments and the keyword arguments.
    Later dictionaries overwrite earlier ones. kwargs overwrite all.
    """
    result = args[0].copy()
    for arg in args[1:] + (kwargs,):
        for k, v in arg.iteritems():
            if isinstance(v, dict) and k in result:
                v = merge_dicts(result[k], v)
            result[k] = v
    return result


def merge_items(*args):
    """
    Returns a new list which is the result of merging all the lists of (key,
    value) pairs passed as arguments. Earlier lists take precedence.
    Order is preserved.
    """
    result = args[0][:]
    keys_seen = set(k for k, _ in args[0])
    for other_items in args[1:]:
        new_items = [(k, v) for k, v in other_items if k not in keys_seen]
        result.extend(new_items)
        keys_seen |= set(k for k, _ in new_items)
    return result


def expand_wild_tuple(keys, d):
    """
    expands a multi-level tuple key containing wildcards (*) with the keys
    actually present in a multi-level dictionary. See expand_wild.
    """
    assert isinstance(keys, (tuple, list))
    if not keys:
        return [()]
    if not isinstance(d, dict):
        return []
    if keys[0] == '*':
        sub_keys = d.keys()
    elif keys[0] in d:
        sub_keys = [keys[0]]
    else:
        sub_keys = []
    return [(k,) + sub
            for k in sub_keys
            for sub in expand_wild_tuple(keys[1:], d.get(k))]


def expand_wild(wild_key, d):
    """
    expands a multi-level string key (separated by '/') containing wildcards
    (*) with the keys actually present in a multi-level dictionary.

    >>> res = expand_wild('a/*/c', {'a': {'one': {'c': 0}, 'two': {'c': 0}}})
    >>> sorted(res) # expand_wild result ordering is random
    ['a/one/c', 'a/two/c']

    >>> d = {'a': {'one': {'c': 0}, 'two': {'d': 0}}}
    >>> res = expand_wild('a/*/*', d)
    >>> sorted(res) # expand_wild result ordering is random
    ['a/one/c', 'a/two/d']
    >>> expand_wild('a/one/c', d)
    set(['a/one/c'])
    >>> expand_wild('a/one/d', d)
    set([])
    """
    return {'/'.join(r) for r in expand_wild_tuple(wild_key.split('/'), d)}


def multi_get(d, key, default=None):
    """
    equivalent to dict.get, but d and key can be multi-level (separated by '/')
    """
    keys = key.split('/')
    for k in keys:
        if k in d:
            d = d[k]
        else:
            return default
    return d


def multi_set(d, key, value):
    """
    equivalent to d[key] = value, but d and key can be multi-level
    (separated by '/'). Creates dictionaries on missing keys.
    """
    keys = key.split('/')
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def invert_dict(d):
    return dict((v, k) for k, v in d.iteritems())


# validate functions
# ------------------

def validate_keys(d, required=(), optional=(), context='',
                  extra_allowed=False):
    required_keys = set(required)
    optional_keys = set(optional)
    used_keys = set(d.keys())
    if extra_allowed:
        invalid_keys = set()
    else:
        valid_keys = required_keys | optional_keys
        invalid_keys = used_keys - valid_keys
    missing_keys = required_keys - used_keys
    if invalid_keys:
        kind, keys = 'invalid', invalid_keys
    elif missing_keys:
        kind, keys = 'missing', missing_keys
    else:
        kind, keys = '', []

    if keys:
        if context:
            template = "%%s keyword(s) in %s: '%%s'" % context
        else:
            template = "%s keyword(s): '%s'"
        raise SyntaxError(template % (kind, "', '".join(keys)))


def validate_list(l, target, context):
    assert len(target) == 1
    target_element = target[0]
    for v in l:
        validate_value(v, target_element, context)


def validate_dict(d, target, context=''):
    assert isinstance(target, dict)
    if not isinstance(d, dict):
        raise Exception("invalid structure for '%s': it should be a map and "
                        "it is a %s" % (context, type(d).__name__))
    targets = target.keys()
    required = set(k[1:] for k in targets if k.startswith('#'))
    optional = set(k for k in targets if not k.startswith('#'))
    anykey = '*' in optional
    if anykey:
        optional.remove('*')
    # in case we have a * in there, we should only make sure that required keys
    # are present, otherwise we have to also check if provided keys are valid
    validate_keys(d, required, optional, context, extra_allowed=anykey)
    for k, v in d.iteritems():
        if k in required:
            section_def = target['#' + k]
        elif k in optional:
            section_def = target[k]
        elif anykey:
            section_def = target['*']
        else:
            # this shouldn't happen unless target is an empty dictionary
            assert not target
            raise KeyError('empty section def at: %s' % context)
        if section_def is None or isinstance(section_def, type):
            target_type = section_def
        else:
            target_type = type(section_def)

        if target_type is not None:
            subcontext = context + ' -> ' + k if context else k
            if isinstance(v, target_type):
                validate_value(v, section_def, subcontext)
            else:
                raise Exception("invalid structure for '%s'" % subcontext)


def validate_value(v, target, context):
    if target is None:
        # None is meant as "any" object
        return
    if isinstance(target, dict):
        validate_dict(v, target, context)
    elif isinstance(target, list):
        validate_list(v, target, context)
    else:
        if not isinstance(v, target):
            raise Exception("invalid structure for '%s'" % context)

# fields handling
# ---------------
str_to_type = {'float': float, 'int': int, 'bool': bool}


def field_str_to_type(str_type, context):
    """
    Converts a (field) string type to its Python type.
    """
    if str_type not in str_to_type:
        raise SyntaxError("'%s' is not a valid type for %s."
                          % (str_type, context))
    return str_to_type[str_type]


def fields_str_to_type(str_fields_list):
    """
    Converts a field list (list of tuple) with string types to a list with
    Python types.
    """
    return [(name, field_str_to_type(type_, "field '%s'" % name))
            for name, type_ in str_fields_list]


def fields_yaml_to_type(dict_fields_list):
    """
    Transform a list of (one item) dict with str types to a list of tuple with
    Python types
    """
    return fields_str_to_type([d.items()[0] for d in dict_fields_list])


# exceptions handling
# -------------------

def add_context(exception, s):
    if isinstance(exception, SyntaxError) and exception.offset is not None:
        # most SyntaxError are clearer if left unmodified since they already
        # contain the faulty string but some do not (eg non-keyword arg after
        # keyword arg).
        # SyntaxError instances have 'filename', 'lineno', 'offset' and 'text'
        # attributes.
        return
    encoding = sys.getdefaultencoding()
    exception.liam2context = s.encode(encoding, 'replace')


# we cannot do this in classes __new__ (args are verified in metaclass.__call__)
class ExplainTypeError(type):
    def __call__(cls, *args, **kwargs):
        try:
            # noinspection PyArgumentList
            return type.__call__(cls, *args, **kwargs)
        except TypeError, e:
            funcname = cls.funcname
            msg = e.args[0].replace('__init__()', funcname)

            def repl_py2(matchobj):
                needed, given = int(matchobj.group(1)), int(matchobj.group(3))
                word = matchobj.group(2)
                return '%d %s (%d given)' % (needed - 1, word, given - 1)

            def repl_py3_toomany_from_to(matchobj):
                nfrom, nto, given = [int(matchobj.group(n)) for n in (1, 2, 4)]
                word = matchobj.group(3)
                verb = 'were' if given != 2 else 'was'
                return 'from %d to %d positional %s but %d %s given' \
                       % (nfrom - 1, nto - 1, word, given - 1, verb)

            def repl_py3_toomany(matchobj):
                needed, given = int(matchobj.group(1)), int(matchobj.group(3))
                word = matchobj.group(2)
                verb = 'were' if given != 2 else 'was'
                return 'takes %d positional %s but %d %s given' \
                       % (needed - 1, word, given - 1, verb)

            def repl_py3_missing(matchobj):
                missing = int(matchobj.group(1))
                return 'missing %d positional argument' % (missing - 1)

            # Python2 style
            msg = re.sub('(\d+) (arguments?) \((\d+) given\)', repl_py2, msg)
            # Python3 style for too many args in the presence of default values
            msg = re.sub('from (\d+) to (\d+) positional (arguments?) but '
                         '(\d+) were given',
                         repl_py3_toomany_from_to, msg)
            # Python3 style for too many args with no default values
            # "takes" is included to not match from_to again
            msg = re.sub('takes (\d+) positional (arguments?) but '
                         '(\d+) were given',
                         repl_py3_toomany, msg)
            # Python3 style for missing
            msg = re.sub('missing (\d+) positional argument',
                         repl_py3_missing, msg)
            raise TypeError(msg)


# function signatures
# -------------------

# TODO: use python3 signature objects
FullArgSpec = namedtuple('FullArgSpec',
                         'args, varargs, varkw, defaults, kwonlyargs, '
                         'kwonlydefaults, annotations')


class NiceArgSpec(FullArgSpec):
    def __str__(self):
        if self.defaults:
            ndef = len(self.defaults)
            l = list(self.args[:-ndef])
            l.extend([k + '=' + repr(v)
                      for k, v in zip(self.args[-ndef:], self.defaults)])
        else:
            l = list(self.args)
        if self.varargs:
            l.append('*' + self.varargs)
        if self.varkw:
            l.append('**' + self.varkw)
        if self.kwonlyargs:
            l.extend([k + '=' + repr(self.kwonlydefaults[k])
                      for k in self.kwonlyargs])
        return ', '.join(l)


def _argspec(*args, **kwonlyargs):
    """
    args = argument names. Arguments with a default value must be given as a
    ('name', value) tuple. varargs and varkw argument names, if any, should be
    prefixed with '*' and '**' respectively and must be the last positional
    arguments.
    >>> _argspec('a', 'b', ('c', 1), '*d', '**e', f=None)
    ... # doctest: +NORMALIZE_WHITESPACE
    FullArgSpec(args=['a', 'b', 'c'], varargs='d', varkw='e', defaults=(1,),
                kwonlyargs=['f'], kwonlydefaults={'f': None}, annotations={})
    >>> _argspec('a', '*', '**b', c=None)
    ... # doctest: +NORMALIZE_WHITESPACE
    FullArgSpec(args=['a'], varargs=None, varkw='b', defaults=None,
                kwonlyargs=['c'], kwonlydefaults={'c': None}, annotations={})
    >>> _argspec('a', 'b', ('c', 1), d=None)
    ... # doctest: +NORMALIZE_WHITESPACE
    FullArgSpec(args=['a', 'b', 'c'], varargs=None, varkw=None, defaults=(1,),
                kwonlyargs=['d'], kwonlydefaults={'d': None}, annotations={})
    """
    def lastitem_startswith(l, s):
        return l and isinstance(l[-1], basestring) and l[-1].startswith(s)
    args = list(args)
    varkw = args.pop()[2:] if lastitem_startswith(args, '**') else None
    varargs = args.pop()[1:] if lastitem_startswith(args, '*') else None
    if not varargs:
        varargs = None
    defaults = tuple(a[1] for a in args if isinstance(a, tuple))
    if not defaults:
        defaults = None
    else:
        assert all(isinstance(arg, tuple) for arg in args[-len(defaults):])
        assert all(isinstance(arg, tuple) for arg in args[-len(defaults):])
    args = [a[0] if isinstance(a, tuple) else a for a in args]
    return NiceArgSpec(args, varargs, varkw, defaults,
                       kwonlyargs=sorted(kwonlyargs.keys()),
                       kwonlydefaults=kwonlyargs,
                       annotations={})


def argspec(*args, **kwonlyargs):
    """
    >>> argspec('a, b, c=1, *d, **e, f=None')
    ... # doctest: +NORMALIZE_WHITESPACE
    FullArgSpec(args=['a', 'b', 'c'], varargs='d', varkw='e', defaults=(1,),
                kwonlyargs=['f'], kwonlydefaults={'f': None}, annotations={})
    >>> a = argspec('*, f=None', g=None)
    >>> a.kwonlyargs
    ['f', 'g']
    >>> sorted(a.kwonlydefaults.items())
    [('f', None), ('g', None)]
    >>> argspec('a, *, **b, c=None')
    ... # doctest: +NORMALIZE_WHITESPACE
    FullArgSpec(args=['a'], varargs=None, varkw='b', defaults=None,
                kwonlyargs=['c'], kwonlydefaults={'c': None}, annotations={})
    >>> argspec('a, b, c=1, d=None')
    ... # doctest: +NORMALIZE_WHITESPACE
    FullArgSpec(args=['a', 'b', 'c', 'd'], varargs=None, varkw=None,
                defaults=(1, None), kwonlyargs=[], kwonlydefaults={},
                annotations={})
    >>> argspec('')
    ... # doctest: +NORMALIZE_WHITESPACE
    FullArgSpec(args=[], varargs=None, varkw=None, defaults=None, kwonlyargs=[],
                kwonlydefaults={}, annotations={})
    >>> str(argspec('a, b, c=1, *d, **e, f=None'))
    'a, b, c=1, *d, **e, f=None'
    """
    if len(args) == 1 and isinstance(args[0], basestring):
        str_args = args[0].split(',') if args[0] else []
        str_args = [a.strip().split('=') for a in str_args]
        args = [(a[0], ast.literal_eval(a[1])) if len(a) > 1 else a[0]
                for a in str_args]

        def star(a):
            return isinstance(a, basestring) and '*' in a
        if any(star(a) for a in args):
            while not star(args[-1]):
                k, v = args.pop()
                assert k not in kwonlyargs, "several kwonlyargs named %s" % k
                kwonlyargs[k] = v
    return _argspec(*args, **kwonlyargs)


def split_signature(signature):
    signature = signature.strip()
    assert signature.count("(") == 1
    assert signature.count(")") == 1
    assert signature[-1] == ')'
    pos = signature.find('(')
    name = signature[:pos]
    args = signature[pos + 1:-1]
    return name, args


# miscellaneous tools
# -------------------

class FileProducer(object):
    argspec = argspec(suffix='', fname=None)

    # default extension if only suffix is defined
    ext = None
    # do we need to return a file name if neither suffix nor fname are defined?
    fname_required = False

    def _get_fname(self, kwargs):
        """
        Returns a filename depending on the given kwargs.
        Note that kwargs are **popped in-place** !
        """
        suffix = kwargs.pop('suffix', '')
        fname = kwargs.pop('fname', None)

        if fname is not None and suffix:
            raise ValueError("%s() can't have both 'suffix' and 'fname' "
                             "arguments" % self.__class__.__name__.lower())
        if fname is None and (suffix or self.fname_required):
            suffix = "_" + suffix if suffix else ""
            fname = "{entity}_{period}" + suffix + self.ext
        return fname


def isnan(a):
    """
    isnan is equivalent to np.isnan, except that it returns False instead of
    raising a TypeError if the argument is an array of non-numeric.
    """
    if isinstance(a, np.ndarray):
        return np.issubsctype(a, np.floating) and np.isnan(a)
    else:
        return np.isnan(a)


def array_nan_equal(a, b):
    # np.array_equal is not implemented on strings in numpy < 1.9
    if np.issubdtype(a.dtype, np.str) and np.issubdtype(b.dtype, np.str):
        try:
            return (a == b).all()
        except ValueError:
            return False
    else:
        return np.all((a == b) | (np.isnan(a) & np.isnan(b)))
