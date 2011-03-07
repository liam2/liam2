import sys, time, itertools
from itertools import izip
from textwrap import wrap

import numpy as np

def time2str(seconds):
    if seconds > 60:
        minutes = seconds / 60
        seconds = seconds % 60
        minutes_plural = 's' if minutes > 1 else ''
        seconds_plural = 's' if seconds < 0.005 or seconds > 1 else ''
        return "%d minute%s %.2f second%s" % (minutes, minutes_plural, 
                                              seconds, seconds_plural)
    else:
        seconds_plural = 's' if seconds < 0.005 or seconds > 1 else ''
        return "%.2f second%s" % (seconds, seconds_plural)


def gettime(func, *args, **kwargs):
    start = time.time()
    res = func(*args, **kwargs)
    return time.time() - start, res

def timed(func, *args, **kwargs):
    elapsed, res = gettime(func, *args, **kwargs)
    print "done (%s elapsed)." % time2str(elapsed)
    return res

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
            decal = 9 - (last_percent_done % 10)
            #FIXME: this breaks for very small numbers (ie if doing more than 
            # 10% in one pass)
            if decal < to_display:
                chars_to_write[decal] = '|'
            write(''.join(chars_to_write))
        last_percent_done = percent_done

def skip_comment_cells(lines):
    notacomment = lambda v: not v.startswith('#')
    for line in lines:
        yield list(itertools.takewhile(notacomment, line))

def strip_rows(lines):
    'returns an iterator of lines with leading and trailing blank cells removed'

    isblank = lambda s: s == ''
    for line in lines:
        leading_dropped = list(itertools.dropwhile(isblank, line))
        rev_line = list(itertools.dropwhile(isblank, reversed(leading_dropped)))
        yield list(reversed(rev_line))


def format_value(value):
    if isinstance(value, float):
        # nans print as "-1.#J", let's use something nicer
        if value != value:
            return 'nan'
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

def table2str(table):
    formatted = [[format_value(value) for value in row]
                  for row in table] 
    colwidths = [get_col_width(formatted, i) for i in xrange(len(table[0]))]

    total_width = sum(colwidths)
    sep_width = (len(colwidths) - 1) * 3 
    if total_width + sep_width > 80:
        minwidths = [get_min_width(formatted, i) for i in xrange(len(table[0]))]
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
    def __init__(self, iterable):
        self.data = list(iterable)
    
    def __iter__(self):
        return iter(self.data)
    
    def __str__(self):
        return '\n' + table2str(self.data) + '\n'
    __repr__ = __str__