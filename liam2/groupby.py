# encoding: utf-8
from __future__ import absolute_import, division, print_function

import numpy as np
import larray as la

from liam2.context import context_length
from liam2.expr import expr_eval, collect_variables, not_hashable
from liam2.exprbases import TableExpression
from liam2.utils import expand, prod
from liam2.aggregates import Count
from liam2.partition import partition_nd


class GroupByArray(la.Array):
    def __init__(self, data, axes=None, row_totals=None, col_totals=None):
        super(GroupByArray, self).__init__(data, axes)
        data = self.data
        if row_totals is not None:
            height = prod(data.shape[:-1])
            if len(row_totals) != height:
                raise Exception('size of row totals vector (%s) does not '
                                'match array shape (%s)' % (len(row_totals),
                                                            height))
        if col_totals is not None:
            width = data.shape[-1] if row_totals is None else data.shape[-1] + 1
            if len(col_totals) != width:
                raise Exception('size of col totals vector (%s) does not '
                                'match array shape (%s)' % (len(col_totals),
                                                            width))
        self.row_totals = row_totals
        self.col_totals = col_totals

    # used by __repr__ return table2str(self.dump(...))
    def dump(self, header=True, wide=True, value_name='value', light=False, axes_names=True, na_repr='as_is',
             maxlines=-1, edgeitems=5, _axes_display_names=False):
        r"""dump(self, header=True, wide=True, value_name='value', light=False, axes_names=True, na_repr='as_is',
                 maxlines=-1, edgeitems=5)

        Dump array as a 2D nested list. This is especially useful when writing to an Excel sheet via open_excel().

        Parameters
        ----------
        header : bool
            Whether or not to output axes names and labels.
        wide : boolean, optional
            Whether or not to write arrays in "wide" format. If True, arrays are exported with the last axis
            represented horizontally. If False, arrays are exported in "narrow" format: one column per axis plus one
            value column. Not used if header=False. Defaults to True.
        value_name : str, optional
            Name of the column containing the values (last column) when `wide=False` (see above).
            Not used if header=False. Defaults to 'value'.
        light : bool, optional
            Whether or not to hide repeated labels. In other words, only show a label if it is different from the
            previous one. Defaults to False.
        axes_names : bool or 'except_last', optional
            Assuming header is True, whether or not to include axes names. If axes_names is 'except_last',
            all axes names will be included except the last. Defaults to True.
        na_repr : any scalar, optional
            Replace missing values (NaN floats) by this value. Defaults to 'as_is' (do not do any replacement).
        maxlines : int, optional
            Maximum number of lines to show. Defaults to -1 (all lines are shown).
        edgeitems : int, optional
            If number of lines to display is greater than `maxlines`, only the first and last `edgeitems` lines are
            displayed. Only active if `maxlines` is not -1. Defaults to 5.

        Returns
        -------
        2D nested list of builtin Python values or None for 0d arrays

        Examples
        --------
        >>> arr = la.ndtest((2, 2, 2))

        >>> gbarr = GroupByArray(arr.data, arr.axes, row_totals=[1, 5, 9, 13], col_totals=[12, 16, 28])
        >>> gbarr.dump()                               # doctest: +NORMALIZE_WHITESPACE
        [['a',    'b\\c', 'c0', 'c1', 'total'],
         ['a0',     'b0',    0,    1,       1],
         ['a0',     'b1',    2,    3,       5],
         ['a1',     'b0',    4,    5,       9],
         ['a1',     'b1',    6,    7,      13],
         ['',    'total',   12,   16,      28]]
        """
        table = super(GroupByArray, self).dump(header=header, wide=wide, value_name=value_name, light=light,
                                               axes_names=axes_names, na_repr=na_repr, maxlines=maxlines,
                                               edgeitems=edgeitems, _axes_display_names=_axes_display_names)
        if table is None:
            return

        assert isinstance(table, list)
        # we modify the table inplace
        row_totals = self.row_totals
        if row_totals is not None:
            if header:
                table[0].append('total')
                first_data_line = 1
            else:
                first_data_line = 0
            for line, total in zip(table[first_data_line:], row_totals):
                line.append(total)
        if self.col_totals is not None and self.ndim > 1:
            table.append([''] * (self.ndim - 2) + ['total'] + self.col_totals)
        return table


class GroupBy(TableExpression):
    funcname = 'groupby'
    no_eval = ('expressions', 'expr')
    kwonlyargs = {'expr': None, 'filter': None, 'percent': False,
                  'pvalues': None, 'axes': None, 'totals': True}

    # noinspection PyNoneFunctionAssignment
    def compute(self, context, *expressions, **kwargs):
        if not expressions:
            raise TypeError("groupby() takes at least 1 argument")

        # TODO: allow lists/tuples of arguments to group by the combinations
        # of keys
        for e in expressions:
            if isinstance(e, (bool, int, float)):
                raise TypeError("groupby() does not work with constant "
                                "arguments")
            if isinstance(e, (tuple, list)):
                raise TypeError("groupby() takes expressions as arguments, "
                                "not a list of expressions")

        # On python 3, we could clean up this code (keyword only arguments).
        expr = kwargs.pop('expr', None)
        if expr is None:
            expr = Count()

#        by = kwargs.pop('by', None)
        filter_value = kwargs.pop('filter', None)
        percent = kwargs.pop('percent', False)
        possible_values = kwargs.pop('pvalues', None)
        axes = kwargs.pop('axes', None)
        if possible_values is not None and axes is not None:
            raise ValueError("cannot use both possible_values and axes arguments in groupby")

        totals = kwargs.pop('totals', True)

        expr_vars = collect_variables(expr)
        expr_vars_names = [v.name for v in expr_vars]

        if filter_value is not None:
            all_vars = expr_vars.copy()
            for e in expressions:
                all_vars |= collect_variables(e)
            all_vars_names = [v.name for v in all_vars]

            # FIXME: use the actual filter_expr instead of not_hashable
            filtered_context = context.subset(filter_value, all_vars_names, not_hashable)
        else:
            filtered_context = context

        filtered_columns = [expr_eval(e, filtered_context) for e in expressions]
        filtered_columns = [expand(c, context_length(filtered_context)) for c in filtered_columns]

        if axes is not None:
            possible_values = [axis.labels for axis in axes]

        # We pre-filtered columns instead of passing the filter to partition_nd
        # because it is a bit faster this way. The indices are still correct,
        # because we use them on a filtered_context.
        groups, possible_values = partition_nd(filtered_columns, True, possible_values)
        if not groups:
            return la.Array([])

        if axes is None:
            axes = la.AxisCollection([la.Axis(axis_labels, name=str(e))
                                      for axis_labels, e in zip(possible_values, expressions)])

        shape = axes.shape

        def eval_expr_on_subset(indices):
            # we use not_hashable to avoid storing the subset in the cache
            return expr_eval(expr, filtered_context.subset(indices, expr_vars_names, not_hashable))

        # evaluate the expression on each group

        # groups is a (flat) list of list.
        # the first variable is the outer-most "loop",
        # the last one the inner most.
        data1d = [eval_expr_on_subset(indices) for indices in groups]

        if percent:
            totals = True

        if totals:
            width = shape[-1]
            height = prod(shape[:-1])
            rows_indices = [np.concatenate([groups[y * width + x]
                                            for x in range(width)])
                            for y in range(height)]
            cols_indices = [np.concatenate([groups[y * width + x]
                                            for y in range(height)])
                            for x in range(width)]
            # add grand total
            # None means "all indices" in this case (equivalent to but faster than np.concatenate(cols_indices))
            cols_indices.append(None)

            # evaluate the expression on each "combined" group (ie compute totals)
            row_totals = [eval_expr_on_subset(indices) for indices in rows_indices]
            col_totals = [eval_expr_on_subset(indices) for indices in cols_indices]
        else:
            row_totals = None
            col_totals = None

        if percent:
            # convert to np.float64 to get +-inf if total_value is int(0)
            # instead of Python's built-in behaviour of raising an exception.
            # This can happen at least when using the default expr (count())
            # and the filter yields empty groups
            total_value = np.float64(col_totals[-1])
            data1d = [100.0 * value / total_value for value in data1d]
            row_totals = [100.0 * value / total_value for value in row_totals]
            col_totals = [100.0 * value / total_value for value in col_totals]

#        if self.by or self.percent:
#            if self.percent:
#                total_value = data1d[-1]
#                divisors = [total_value for _ in data1d]
#            else:
#                num_by = len(self.by)
#                inc = prod(len_pvalues[-num_by:])
#                num_groups = len(groups)
#                num_categories = prod(len_pvalues[:-num_by])
#
#                categories_groups_idx = [range(cat_idx, num_groups, inc)
#                                         for cat_idx in range(num_categories)]
#
#                divisors = ...
#
#            data1d = [100.0 * value / divisor
#                    for value, divisor in zip(data1d, divisors)]

        # convert to a 1d array. We don't simply use data1d = np.array(data1d),
        # because if data1d is a list of ndarray (for example if we use
        # groupby(a, expr=id), *and* all the ndarrays have the same length,
        # the result is a 2d array instead of an array of ndarrays like we
        # need (at this point).
        arr1d = np.empty(len(data1d), dtype=type(data1d[0]))
        arr1d[:] = data1d
        data1d = arr1d

        # and reshape it
        data = data1d.reshape(shape)
        return GroupByArray(data, axes, row_totals, col_totals)


functions = {
    'groupby': GroupBy
}
