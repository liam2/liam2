import numpy as np

from utils import safe_put
from expr import expr_eval, dtype, hasvalue
from properties import FunctionExpression


class ValueForPeriod(FunctionExpression):
    func_name = 'value_for_period'

    def __init__(self, expr, period, missing='auto'):
        FunctionExpression.__init__(self, expr)
        self.period = period
        self.missing = missing

    def eval(self, context):
        entity = context['__entity__']
        return entity.value_for_period(self.expr, self.period, context,
                                       self.missing)


class Lag(FunctionExpression):
    func_name = 'lag'

    def __init__(self, expr, num_periods=1, missing='auto'):
        FunctionExpression.__init__(self, expr)
        self.num_periods = num_periods
        self.missing = missing

    def eval(self, context):
        entity = context['__entity__']
        period = context['period'] - self.num_periods
        return entity.value_for_period(self.expr, period, context,
                                       self.missing)

    def dtype(self, context):
        return dtype(self.expr, context)


class Duration(FunctionExpression):
    func_name = 'duration'

    def eval(self, context):
        entity = context['__entity__']
#        return entity.duration(self.expr, context)

#    def duration(self, bool_expr, context):
        bool_expr = self.expr
        value = expr_eval(bool_expr, context)

        baseperiod = self.base_period
        period = context['period'] - 1

        # using a full int so that the "store" type check works
        result = value.astype(np.int)
        res_size = len(self.array)
        last_period_true = np.empty(res_size, dtype=np.int)
        last_period_true.fill(period + 1)

        id_to_rownum = context.id_to_rownum
        still_running = value
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
        assert dtype(self.expr, context) == bool
        return int


class TimeAverage(FunctionExpression):
    func_name = 'tavg'

    def eval(self, context):
        entity = context['__entity__']
#        return entity.tavg(self.expr, context)

#    def tavg(self, expr, context):
        baseperiod = entity.base_period
        period = context['period'] - 1
        expr = self.expr

        res_size = len(entity.array)

        num_values = np.zeros(res_size, dtype=np.int)
        last_period_wh_value = np.empty(res_size, dtype=np.int)
        last_period_wh_value.fill(context['period'])  # current period

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


class TimeSum(FunctionExpression):
    func_name = 'tsum'

    def eval(self, context):
        entity = context['__entity__']

        baseperiod = entity.base_period
        period = context['period'] - 1
        expr = self.expr

        typemap = {bool: int, int: int, float: float}
        res_type = typemap[dtype(expr, context)]
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
