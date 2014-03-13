from __future__ import print_function

import numpy as np

from utils import safe_put
from expr import expr_eval, getdtype, hasvalue
from exprbases import AbstractExprCall


class ValueForPeriod(AbstractExprCall):
    func_name = 'value_for_period'
    #TODO: update Expr._eval_args to take __eval_args__ into account
    __eval_args__ = ('period', 'missing')
    #AND/OR
    __no_eval_args__ = ('expr',)

    def __init__(self, expr, period, missing='auto'):
        AbstractExprCall.__init__(self, expr, period, missing)

    def _compute(self, context, expr, period, missing):
        entity = context.entity
        return entity.value_for_period(expr, period, context, missing)


class Lag(AbstractExprCall):
    func_name = 'lag'

    # for traversal, cache
    __fields__ = ('args',)
    __fields__ = ('args', 'kwargs')
    # might be better to store them all in args so that cache key are
    # independent as to how args are passed >> for the cache, yes
    # but for NumpyFunction...
    # >>> in fact, I cannot store everything in args only, because of
    #     kwargs-only args (like in groupby)
    # >>> but all "normal" arguments (whether they have a default value or not
    #     that are **passed as kwargs or not** should be stored in "args"
    # => should rename AbstractExprCall.kwargs into kwonlyargs
    # arg_names
    # kwargs_names -> extra_arg_names

    # kwargs
    __fields__ = ('args', 'kwargs')
    __eval_args__ = ('num_periods', 'missing')
    #AND/OR
    __no_eval_args__ = ('expr',)

    def __init__(self, expr, num_periods=1, missing='auto'):
        AbstractExprCall.__init__(self, expr)
        # self.num_periods = num_periods
        # self.missing = missing

    # def _compute(self, context, expr, num_periods, missing):
    #     entity = context.entity
    #     period = context.period - num_periods
    #     return entity.value_for_period(expr, period, context, missing)

    def evaluate(self, context):
        expr, num_periods, missing = self.args
        entity = context.entity
        period = context.period - expr_eval(num_periods, context)
        missing = expr_eval(missing, context)
        return entity.value_for_period(expr, period, context, missing)

    # def evaluate(self, context):
    #     entity = context.entity
    #     period = context.period - expr_eval(self.num_periods, context)
    #     return entity.value_for_period(self.expr, period, context, self.missing)

    def dtype(self, context):
        return getdtype(self.expr, context)


class Duration(AbstractExprCall):
    func_name = 'duration'

    def evaluate(self, context):
        entity = context.entity

        baseperiod = entity.base_period
        period = context.period - 1
        bool_expr = self.expr
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

    def dtype(self, context):
        assert getdtype(self.expr, context) == bool
        return int


class TimeAverage(AbstractExprCall):
    func_name = 'tavg'

    def evaluate(self, context):
        entity = context.entity

        baseperiod = entity.base_period
        period = context.period - 1
        expr = self.expr

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


class TimeSum(AbstractExprCall):
    func_name = 'tsum'

    def evaluate(self, context):
        entity = context.entity

        baseperiod = entity.base_period
        period = context.period - 1
        expr = self.expr

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

functions = {
    'value_for_period': ValueForPeriod,
    'lag': Lag,
    'duration': Duration,
    'tavg': TimeAverage,
    'tsum': TimeSum,
}
