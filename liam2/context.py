from __future__ import print_function

import numpy as np


class EvaluationContext(object):
    def __init__(self, simulation=None, entities=None, global_tables=None,
                 period=None, periods=None, periodicity=None,
                 period_idx=None, format_date=None,
                 entity_name=None, filter_expr=None,
                 entities_data=None,
                 longitudinal=None):
        """
        :param simulation: Simulation
        :param entities: dict of entities {name: entity}
        :param global_tables: dict of ndarrays (structured or not)
        :param period: int of the current period
        :param periods: list of alls periods
        :param periodicity: number of months between two periods
        :param period_idx: idx of current period in periods
        :param period_idx: string for the period format
        :param entity_name: name (str) of the current entity
        :param filter_expr: contextual filter expression (Expr)
        :param entities_data: dict of data for entities (dict of
                              EntityContext or dict of dict)
        :return:
        """
        self.simulation = simulation
        self.entities = entities
        self.global_tables = global_tables
        self.period = period
        self.periods = periods
        self.periodicity = periodicity
        self.period_idx = period_idx
        self.format_date = format_date
        self.entity_name = entity_name
        self.filter_expr = filter_expr
        if entities_data is None:
            entities_data = {name: EntityContext(self, entity)
                             for name, entity in entities.iteritems()}
        self.entities_data = entities_data
        self.longitudinal = longitudinal


    def copy(self, fresh_data=False):
        entities_data = None if fresh_data else self.entities_data.copy()
        # FIXME: when switching entities, filter should not come along, or maybe
        # filter should be a per-entity dict?
        return EvaluationContext(self.simulation, self.entities,
                                 self.global_tables, self.period,
                                 self.periods, self.periodicity,
                                 self.period_idx, self.format_date,
                                 self.entity_name, self.filter_expr,
                                 entities_data, self.longitudinal)

    def clone(self, fresh_data=False, **kwargs):
        res = self.copy(fresh_data=fresh_data)
        for k, v in kwargs.iteritems():
            allowed_kwargs = ('simulation', 'entities', 'global_tables',
                              'period', 'periods', 'periodicity',
                              'period_idx', 'format_date',
                              'entity_name', 'filter_expr',
                              'entities_data', 'entity_data', 'longitudinal')
            assert k in allowed_kwargs, "%s is not a valid kwarg" % k
            setattr(res, k, v)
        return res

    @property
    def entity(self):
        return self.entities.get(self.entity_name)

    @entity.setter
    def entity(self, value):
        assert value.name in self.entities, '%s not in %s' % (value,
                                                              self.entities)
        self.entity_name = value.name

    @property
    def id_to_rownum(self):
        entity_context = self.entity_data
        if hasattr(entity_context, 'id_to_rownum'):
            return entity_context.id_to_rownum
        else:
            # fall back on the entity itself
            return self.entity.id_to_rownum

    @property
    def entity_data(self):
        return self.entities_data[self.entity_name]

    @entity_data.setter
    def entity_data(self, value):
        self.entities_data[self.entity_name] = value

    def __getitem__(self, key):
        if key in ['period', 'periods', 'periodicity',
                   'period_idx', 'format_date', 'longitudinal']:
            return getattr(self, key)
        else:
            return self.entity_data[key]

    def get(self, key, elsevalue=None):
        try:
            return self[key]
        except KeyError:
            return elsevalue

    def __setitem__(self, key, value):
        # XXX: how do we set a new global?
        self.entity_data[key] = value

    def __contains__(self, key):
        from expr import Variable
        if isinstance(key, Variable):
            entity, name = key.entity, key.name
            if entity is None:
                # FIXME: this is wrong (but currently needed because some
                # Variable are created without entity)
                return True
            else:
                return name in self.entities_data[entity.name]
        else:
            return key in self.entity_data

    def length(self):
        return context_length(self.entity_data)

    def keys(self, extra=True):
        entity_data = self.entity_data
        if isinstance(entity_data, EntityContext):
            return entity_data.keys(extra)
        else:
            return entity_data.keys()

    def update(self, other, **kwargs):
        self.entity_data.update(other, **kwargs)

    def __len__(self):
        return self.length()

    def subset(self, index=None, keys=None, filter_expr=None):
        """
        returns a copy of the context with only a subset of the current entity.
        The main use case is to take a subset of rows. Since this is a
        costly operation, the user can also provide keys so that only the
        columns he needs are filtered.
        :param index: indices to take (list or ndarray)
        :param keys: list of column names or None (take all)
        :param filter_expr: expression used to compute the index. This is
        only used to compute the cache key
        :return:
        """
        data_subset = context_subset(self.entity_data, index, keys)
        return self.clone(entity_data=data_subset, filter_expr=filter_expr)

    def empty(self, length=None):
        """
        returns a copy of the context with the same length but no data.
        """
        return self.clone(entity_data=empty_context(length))


class EntityContext(object):
    def __init__(self, eval_ctx, entity, extra=None):
        self.eval_ctx = eval_ctx
        self.entity = entity
        if extra is None:
            extra = {}
        self.extra = extra

    def __getitem__(self, key):

        if key in ['period', 'periods', 'periodicity',
                   'period_idx', 'format_date', 'longitudinal']:
            return getattr(self.eval_ctx, key)

        try:
            return self.extra[key]
        except KeyError:
            period = self.eval_ctx.period
            array_period = self.entity.array_period
            if period == array_period:
                try:
                    return self.entity.temp_variables[key]
                except KeyError:
                    try:
                        return self.entity.array[key]
                    except ValueError:
                        raise KeyError(key)
            else:
                # FIXME: lags will break if used from a context subset (eg in
                # new() or groupby(): all individuals will be returned instead
                # of only the "filtered" ones.
                if (self.entity.array_lag is not None and
                    array_period is not None and
                    period == array_period - 1 and
                        key in self.entity.array_lag.dtype.fields):
                    return self.entity.array_lag[key]

                bounds = self.entity.output_rows.get(period)
                if bounds is not None:
                    startrow, stoprow = bounds
                else:
                    startrow, stoprow = 0, 0
                return self.entity.table.read(start=startrow, stop=stoprow,
                                              field=key)

    # is the current array period the same as the context period?
    @property
    def is_array_period(self):
        return self.entity.array_period == self.eval_ctx.period

    def __setitem__(self, key, value):
        self.extra[key] = value

    def __delitem__(self, key):
        del self.extra[key]

    def __contains__(self, key):
        entity = self.entity
        # entity.array can be None! (eg. with "explore")
        keyinarray = (self.is_array_period and
                      (key in entity.temp_variables or
                       key in entity.array.dtype.fields))
        keyintable = (entity.table is not None and
                      key in entity.table.dtype.fields)
        return key in self.extra or keyinarray or keyintable

    def keys(self, extra=True):
        res = list(self.entity.array.dtype.names)
        res.extend(sorted(self.entity.temp_variables.keys()))
        if extra:
            res.extend(sorted(self.extra.keys()))
        return res

    def get(self, key, elsevalue=None):
        try:
            return self[key]
        except KeyError:
            return elsevalue

    def copy(self):
        return EntityContext(self.entity, self.extra.copy())

    def length(self):
        if self.is_array_period:
            return len(self.entity.array)
        else:
            period = self.eval_ctx.period
            bounds = self.entity.output_rows.get(period)
            if bounds is not None:
                startrow, stoprow = bounds
                return stoprow - startrow
            else:
                return 0

    def __len__(self):
        return self.length()

    def list_periods(self):
        return self.entity.output_index.keys()

    @property
    def id_to_rownum(self):
        period = self.eval_ctx.period
        if self.is_array_period:
            return self.entity.id_to_rownum
        elif period in self.entity.output_index:
            return self.entity.output_index[period]
        else:
            # FIXME: yes, it's true, that if period is not in output_index, it
            # probably means that we are before start_period and in that case,
            # input_index == output_index, but it would be cleaner to simply
            # initialise output_index correctly
            return self.entity.input_index[period]


def new_context_like(context, length=None):
    if length is None:
        length = context_length(context)
    #FIXME: nan should come from somewhere else
    return {'period': context['period'],
            'periods': context['periods'],
            'period_idx': context['period_idx'],
            '__len__': length,
            '__entity__': context['__entity__'],
            '__globals__': context['__globals__'],
            'nan': float('nan')}

def empty_context(length):
    return {'__len__': length}


def context_subset(context, index=None, keys=None):
    # if keys is None, take all fields
    if keys is None:
        keys = context.keys()
    # tuples are not valid numpy indexes (I don't know why)
    if isinstance(index, list):
        if not index:
            index = np.array([], dtype=int)
        else:
            index = np.array(index)
    if index is None:
        length = context_length(context)
    elif np.issubdtype(index.dtype, int):
        length = len(index)
    else:
        assert np.issubdtype(index.dtype, bool)
        assert len(index) == context_length(context), \
               "boolean index has length %d instead of %d" % \
               (len(index), context_length(context))
        length = np.sum(index)
    result = empty_context(length)
    for key in keys:
        value = context[key]
        if index is not None and isinstance(value, np.ndarray) and value.shape:
            value = value[index]
        result[key] = value
    return result


def context_keep(context, keys):
    keep = set(keys)
    for key in context.keys():
        if key not in keep and not key.startswith('__'):
            del context[key]


def context_delete(context, rownum):
    result = {}
    # this copies everything including __len__, period, nan, ...
    for key in context.keys():
        value = context[key]
        # globals are left unmodified
        if key != '__globals__':
            if isinstance(value, np.ndarray) and value.shape:
                # axis=0 so that if value.ndim > 1, treat it as if it was an
                # array of arrays
                value = np.delete(value, rownum, axis=0)
        result[key] = value
    result['__len__'] -= 1
    return result


def context_length(ctx):
    if hasattr(ctx, 'length'):
        return ctx.length()
    elif '__len__' in ctx:
        return ctx['__len__']
    else:
        usual_len = None
        for k, value in ctx.iteritems():
            if isinstance(value, np.ndarray):
                if usual_len is not None and len(value) != usual_len:
                    raise Exception('incoherent array lengths: %s''s is %d '
                                    'while the len of others is %d' %
                                    (k, len(value), usual_len))
                usual_len = len(value)
        return usual_len
