import sys
import time
import operator
import itertools
from itertools import izip
from textwrap import wrap
from collections import defaultdict, deque

import numpy as np


def time2str(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    hours = minutes // 60
    minutes = minutes % 60
    l = []
    if hours > 0:
        l.append("%d hour%s" % (hours, 's' if hours > 1 else ''))
    if minutes > 0:
        l.append("%d minute%s" % (minutes, 's' if minutes > 1 else ''))
    if seconds >= 0.005:
        l.append("%.2f second%s" % (seconds, 's' if seconds > 1 else ''))
    if not l:
        l = ["%d ms" % (seconds * 1000)]
    return ' '.join(l)


def size2str(value):
    unit = "bytes"
    if value > 1024.0:
        value /= 1024.0
        unit = "Kb"
        if value > 1024.0:
            value /= 1024.0
            unit = "Mb"
        return "%.2f %s" % (value, unit)
    else:
        return "%d %s" % (value, unit)


def gettime(func, *args, **kwargs):
    start = time.time()
    res = func(*args, **kwargs)
    return time.time() - start, res


def timed(func, *args, **kwargs):
    elapsed, res = gettime(func, *args, **kwargs)
    print "done (%s elapsed)." % time2str(elapsed)
    return res


def prod(values):
    return reduce(operator.mul, values, 1)


def safe_put(a, ind, v):
    if not len(a) or not len(ind):
        return
    # backup last value, in case it gets overwritten
    last_value = a[-1]
    np.put(a, ind, v)
    # if the last value was erroneously modified (because of a -1 in ind)
    # this assumes indices are sorted
    if ind[-1] != len(a) - 1:
        # restore its previous value
        a[-1] = last_value


def loop_wh_progress(func, sequence):
    len_todo = len(sequence)
    write = sys.stdout.write
    last_percent_done = 0
    for i, value in enumerate(sequence, start=1):
        try:
            func(i, value)
        except StopIteration:
            break

        # update progress bar
        percent_done = (i * 100) / len_todo
        to_display = percent_done - last_percent_done
        if to_display:
            chars_to_write = list("." * to_display)
            offset = 9 - (last_percent_done % 10)
            while offset < to_display:
                chars_to_write[offset] = '|'
                offset += 10
            write(''.join(chars_to_write))
        last_percent_done = percent_done


def count_occurences(seq):
    counter = defaultdict(int)
    for e in seq:
        counter[e] += 1
    return counter.items()


def skip_comment_cells(lines):
    notacomment = lambda v: not v.startswith('#')
    for line in lines:
        yield list(itertools.takewhile(notacomment, line))


def strip_rows(lines):
    '''
    returns an iterator of lines with leading and trailing blank cells
    removed
    '''
    isblank = lambda s: s == ''
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
        # prevent numpy's default wrapping
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
    table_len = len(table[0])
    formatted = [[format_value(value, missing) for value in row]
                  for row in table]
    colwidths = [get_col_width(formatted, i) for i in xrange(table_len)]

    total_width = sum(colwidths)
    sep_width = (len(colwidths) - 1) * 3
    if total_width + sep_width > 80:
        minwidths = [get_min_width(formatted, i) for i in xrange(table_len)]
        available_width = 80.0 - sep_width - sum(minwidths)
        ratio = available_width / total_width
        colwidths = [minw + max(int(width * ratio), 0)
                     for minw, width in izip(minwidths, colwidths)]

    lines = []
    for row in formatted:
        wrapped_row = [wrap(value, width)
                       for value, width in izip(row, colwidths)]
        maxlines = max(len(value) for value in wrapped_row)
        newlines = [[] for _ in range(maxlines)]
        for value, width in izip(wrapped_row, colwidths):
            for i in range(maxlines):
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

    def __str__(self):
        missing = self.missing
        if missing is None:
            missing = 'nan'
        else:
            missing = str(missing)
        return '\n' + table2str(self.data, missing) + '\n'
    __repr__ = __str__


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

    def gen(mydeque):
        while True:
            if not mydeque:                # when the local queue is empty
                row = next(iterator)       # fetch a new row and
                if len(row) != numcolumns:
                    raise Exception("all rows do not have the same number of "
                                    "columns")
                # dispatch it to the other queues
                for queue, value in zip(deques, row):
                    queue.append(value)
            yield mydeque.popleft()
    return tuple(gen(d) for d in deques)


def merge_dicts(*args, **kwargs):
    result = args[0].copy()
    for arg in args[1:] + (kwargs,):
        for k, v in arg.iteritems():
            if isinstance(v, dict) and k in result:
                v = merge_dicts(result[k], v)
            result[k] = v
    return result


def merge_items(*args):
    result = args[0][:]
    keys_seen = set(k for k, _ in args[0])
    for other_items in args[1:]:
        new_items = [(k, v) for k, v in other_items if k not in keys_seen]
        result.extend(new_items)
        keys_seen |= set(k for k, _ in new_items)
    return result


def invert_dict(d):
    return dict((v, k) for k, v in d.iteritems())


def countlines(filepath):
    with open(filepath) as f:
        return sum(1 for _ in f)


#--------------------#
# validate functions #
#--------------------#

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
    if isinstance(v, dict):
        validate_dict(v, target, context)
    elif isinstance(v, list):
        validate_list(v, target, context)
    # otherwise that type (int, str) is not validated further


# fields handling
# ---------------
str_to_type = {'float': float, 'int': int, 'bool': bool}


def field_str_to_type(str_type, context):
    """
    Converts a (field) string type to its Python type.
    """
    if str_type not in str_to_type:
        raise SyntaxError("'%s' is not a valid type for %s." % context)
    return str_to_type[str_type]


def fields_str_to_type(str_fields_list):
    """
    Converts a field list (list of tuple) with string types to a list with
    Python types.
    """
    return [(name, field_str_to_type(type_, "field '%s'" % name))
            for name, type_ in str_fields_list]


def fields_yaml_to_type(dict_fields_list):
    '''
    Transform a list of (one item) dict with str types to a list of tuple with
    Python types
    '''
    return fields_str_to_type([d.items()[0] for d in dict_fields_list])
