from __future__ import print_function

import numpy as np

from utils import safe_put
from expr import (expr_eval, getdtype, hasvalue, FunctionExpr, always,\
                  firstarg_dtype)


class TimeFunction(FunctionExpr):
    no_eval = ('expr',)


class ValueForPeriod(TimeFunction):
    func_name = 'value_for_period'

    def compute(self, context, expr, period, missing='auto'):
        entity = context.entity
        return entity.value_for_period(expr, period, context, missing)

    dtype = firstarg_dtype


#TODO: this should be a compound expression:
# Lag(expr, numperiods, missing)
# ->
# ValueForPeriod(expr, Subtract(Variable('period'), numperiods), missing)
class Lag(TimeFunction):
    func_name = 'lag'

    def compute(self, context, expr, num_periods=1, missing='auto'):
        entity = context.entity
        period = context.period - num_periods
        return entity.value_for_period(expr, period, context, missing)

    dtype = firstarg_dtype


class Duration(TimeFunction):
    func_name = 'duration'
    no_eval = ('bool_expr',)

    def compute(self, context, bool_expr):
        entity = context.entity

        baseperiod = entity.base_period
        period = context.period - 1
        value = expr_eval(bool_expr, context)

        # using a full int so that the "store" type check works
        result = value.astype(np.int)
        res_size = len(entity.array)
        last_period_true = np.empty(res_size, dtype=np.int)
        last_period_true.fill(period + 1)

        id_to_rownum = context.id_to_rownum
        still_running = value.copy()
        while np.any(still_running) and period >= baseperiod:
            ids, values = entity.value_for_period(bool_expr, period, context,
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

    #TODO: move the check to __init__ and use dtype = always(int)
    def dtype(self, context):
        assert getdtype(self.args[0], context) == bool
        return int


class TimeAverage(TimeFunction):
    func_name = 'tavg'

    def compute(self, context, expr):
        entity = context.entity

        baseperiod = entity.base_period
        period = context.period - 1

        res_size = len(entity.array)

        num_values = np.zeros(res_size, dtype=np.int)
        last_period_wh_value = np.empty(res_size, dtype=np.int)
        last_period_wh_value.fill(context.period)  # current period

        sum_values = np.zeros(res_size, dtype=np.float)
        id_to_rownum = context.id_to_rownum
        while period >= baseperiod:
            ids, values = entity.value_for_period(expr, period, context,
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
    func_name = 'tsum'

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
            ids, values = entity.value_for_period(expr, period, context,
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
