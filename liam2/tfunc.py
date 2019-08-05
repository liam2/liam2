# encoding: utf-8
from __future__ import absolute_import, division, print_function

import numpy as np
import larray as la

from liam2.context import context_length
from liam2.expr import expr_eval, getdtype, hasvalue, FunctionExpr, always, firstarg_dtype, get_default_value
from liam2.utils import safe_put


class TimeFunction(FunctionExpr):
    no_eval = ('expr',)

    @staticmethod
    def fill_missing_values(ids, values, context, filler='auto'):
        """
        ids: ids present in past period
        values: values in past period
        context: current period context
        """

        if filler is 'auto':
            filler = get_default_value(values)
        result = np.full(context_length(context), filler, dtype=values.dtype)
        if len(ids):
            id_to_rownum = context.id_to_rownum
            # if there was more objects in the past than in the current
            # period. Currently, remove() keeps old ids, so this never
            # happens, but if we ever change remove(), we'll need to add
            # such a check everywhere we use id_to_rownum
            # invalid_ids = ids > len(id_to_rownum)
            # if np.any(invalid_ids):
            #     fix ids
            rows = id_to_rownum[ids]
            safe_put(result, rows, values)
        return result

    @staticmethod
    def value_for_period(expr, period, context, fill='auto'):
        sub_context = context.clone(fresh_data=True, period=period)
        result = expr_eval(expr, sub_context)
        if isinstance(result, (np.ndarray, la.Array)) and result.shape:
            ids = sub_context['id']
            if fill is None:
                return ids, result
            else:
                # expand values to the current "outer" context
                return TimeFunction.fill_missing_values(ids, result, context,
                                                        fill)
        else:
            return result


class ValueForPeriod(TimeFunction):
    funcname = 'value_for_period'

    def compute(self, context, expr, period, missing='auto'):
        return self.value_for_period(expr, period, context, missing)

    dtype = firstarg_dtype


# TODO: this should be a compound expression:
# Lag(expr, numperiods, missing)
# ->
# ValueForPeriod(expr, Subtract(Variable('period'), numperiods), missing)
class Lag(TimeFunction):
    def compute(self, context, expr, num_periods=1, missing='auto'):
        period = context.period - num_periods
        return self.value_for_period(expr, period, context, missing)

    dtype = firstarg_dtype


class Duration(TimeFunction):
    no_eval = ('bool_expr',)

    def compute(self, context, bool_expr):
        entity = context.entity

        baseperiod = entity.base_period
        period = context.period - 1
        value = expr_eval(bool_expr, context)

        # using a full int so that the "store" type check works
        result = value.astype(np.int)
        res_size = len(entity.array)
        last_period_true = np.full(res_size, period + 1, dtype=np.int)

        id_to_rownum = context.id_to_rownum
        still_running = value.copy()
        while np.any(still_running) and period >= baseperiod:
            ids, values = self.value_for_period(bool_expr, period, context,
                                                fill=None)
            missing = np.ones(res_size, dtype=bool)
            period_value = np.zeros(res_size, dtype=bool)
            if len(ids):
                value_rows = id_to_rownum[ids]
                safe_put(missing, value_rows, False)
                safe_put(period_value, value_rows, values)

            value = still_running & period_value
            result += value * (last_period_true - period)

            still_running &= period_value | missing
            last_period_true[period_value] = period
            period -= 1
        return result

    # TODO: move the check to __init__ and use dtype = always(int)
    def dtype(self, context):
        assert getdtype(self.args[0], context) == bool
        return int


class TimeAverage(TimeFunction):
    funcname = 'tavg'

    def compute(self, context, expr):
        entity = context.entity

        baseperiod = entity.base_period
        period = context.period - 1

        res_size = len(entity.array)

        num_values = np.zeros(res_size, dtype=np.int)
        # current period
        last_period_wh_value = np.full(res_size, context.period, dtype=np.int)

        sum_values = np.zeros(res_size, dtype=np.float)
        id_to_rownum = context.id_to_rownum
        while period >= baseperiod:
            ids, values = self.value_for_period(expr, period, context,
                                                fill=None)

            # filter out lines which are present because there was a value for
            # that individual at that period but not for that column
            acceptable_rows = hasvalue(values)
            acceptable_ids = ids[acceptable_rows]
            if len(acceptable_ids):
                acceptable_values = values[acceptable_rows]

                value_rows = id_to_rownum[acceptable_ids]

                has_value = np.zeros(res_size, dtype=bool)
                safe_put(has_value, value_rows, True)

                period_value = np.zeros(res_size, dtype=np.float)
                safe_put(period_value, value_rows, acceptable_values)

                num_values += has_value * (last_period_wh_value - period)
                sum_values += period_value
                last_period_wh_value[has_value] = period
            period -= 1
        return sum_values / num_values

    dtype = always(float)


class TimeSum(TimeFunction):
    funcname = 'tsum'

    def compute(self, context, expr):
        entity = context.entity

        baseperiod = entity.base_period
        period = context.period - 1

        typemap = {bool: int, int: int, float: float}
        res_type = typemap[getdtype(expr, context)]
        res_size = len(entity.array)

        sum_values = np.zeros(res_size, dtype=res_type)
        id_to_rownum = context.id_to_rownum
        while period >= baseperiod:
            ids, values = self.value_for_period(expr, period, context,
                                                fill=None)

            # filter out lines which are present because there was a value for
            # that individual at that period but not for that column
            acceptable_rows = hasvalue(values)
            acceptable_ids = ids[acceptable_rows]
            if len(acceptable_ids):
                acceptable_values = values[acceptable_rows]

                value_rows = id_to_rownum[acceptable_ids]

                period_value = np.zeros(res_size, dtype=np.float)
                safe_put(period_value, value_rows, acceptable_values)

                sum_values += period_value
            period -= 1
        return sum_values

    dtype = firstarg_dtype


functions = {
    'value_for_period': ValueForPeriod,
    'lag': Lag,
    'duration': Duration,
    'tavg': TimeAverage,
    'tsum': TimeSum
}
