# encoding: utf-8
from __future__ import absolute_import, division, print_function

import numpy as np

from liam2.utils import unique


class EvaluationContext(object):
    def __init__(self, simulation=None, entities=None, global_tables=None,
                 period=None, entity_name=None, filter_expr=None,
                 entities_data=None):
        """
        :param simulation: Simulation
        :param entities: dict of entities {name: entity}
        :param global_tables: dict of ndarrays (structured or not)
        :param period: int of the current period
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
        self.entity_name = entity_name
        self.filter_expr = filter_expr
        if entities_data is None:
            entities_data = {name: EntityContext(self, entity)
                             for name, entity in entities.items()}
        self.entities_data = entities_data

    def copy(self, fresh_data=False):
        # FIXME: when fresh_data is False, entities_data should clone each
        #        EntityContext and set their eval_ctx attribute to the newly created
        #        EvaluationContext (res), otherwise each EntityContext still points to
        #        the old EvaluationContext (which obviously can have some different
        #        attributes, eg period).

        # The problem is that the fix currently breaks many models because in
        # some (all?) cases EvaluableExpression are stored in a context (extra)
        # that is not the same than the context where it is used (and thus the
        # temporary field is not found if extra is not exactly the same object).
        # entities_data = None if fresh_data else {}
        entities_data = None if fresh_data else self.entities_data.copy()
        # FIXME: when switching entities, filter should not come along, or maybe
        #        filter should be a per-entity dict?
        res = EvaluationContext(self.simulation, self.entities,
                                self.global_tables, self.period,
                                self.entity_name, self.filter_expr,
                                entities_data)
        # res.entities_data = {name: ent_ctx.clone(eval_ctx=res)
        #                      for name, ent_ctx
        #                      in self.entities_data.items()}
        return res

    def clone(self, fresh_data=False, **kwargs):
        res = self.copy(fresh_data=fresh_data)
        allowed_kwargs = {'simulation', 'entities', 'global_tables',
                          'period', 'entity_name', 'filter_expr',
                          'entities_data', 'entity_data'}
        for k, v in kwargs.items():
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
        if key == 'period':
            return self.period
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
        from liam2.expr import Variable
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
            assert isinstance(entity_data, dict)
            return list(entity_data.keys())

    def update(self, other, **kwargs):
        self.entity_data.update(other, **kwargs)

    def __len__(self):
        return self.length()

    def subset(self, index, keys=None, filter_expr=None):
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
        if key == 'period':
            return self.eval_ctx.period

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
        period = self.eval_ctx.period
        array_period = entity.array_period
        # entity.array can be None! (eg. with "explore")
        keyinarray = (self.is_array_period and
                      (key in entity.temp_variables or
                       key in entity.array.dtype.fields))
        # we need to check explicitly whether the key is in array_lag because
        # with output=None it can contain more fields than table.
        keyinlagarray = (entity.array_lag is not None and
                         array_period is not None and
                         period == array_period - 1 and
                         key in entity.array_lag.dtype.fields)
        keyintable = (entity.table is not None and
                      key in entity.table.dtype.fields)
        return key in self.extra or keyinarray or keyinlagarray or keyintable

    def keys(self, extra=True):
        res = list(self.entity.array.dtype.names)
        res.extend(sorted(self.entity.temp_variables.keys()))
        if extra:
            res.extend(sorted(self.extra.keys()))
        # in theory, this should not be needed because we present defining a function argument with the same
        # name than an entity field
        assert list(unique(res)) == res
        return res

    def get(self, key, elsevalue=None):
        try:
            return self[key]
        except KeyError:
            return elsevalue

    def copy(self):
        return EntityContext(self.eval_ctx, self.entity, self.extra.copy())

    def clone(self, **kwargs):
        res = self.copy()
        allowed_kwargs = {'eval_ctx', 'entity', 'extra'}
        for k, v in kwargs.items():
            assert k in allowed_kwargs, "%s is not a valid kwarg" % k
            setattr(res, k, v)
        return res

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
        return sorted(self.entity.output_index.keys())

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


def empty_context(length):
    return {'__len__': length}


def context_subset(context, index=None, keys=None):
    # if keys is None, take all fields
    if keys is None:
        keys = list(context.keys())

    # tuples are not valid numpy indexes
    if isinstance(index, list):
        # forcing to int, even for empty lists
        index = np.array(index, dtype=int)

    if index is None:
        length = context_length(context)
    elif np.issubdtype(index.dtype, np.integer):
        length = len(index)
    else:
        assert np.issubdtype(index.dtype, np.bool_)
        assert len(index) == context_length(context), \
            "boolean index has length %d instead of %d" % \
            (len(index), context_length(context))
        length = index.sum()

    result = empty_context(length)
    for key in keys:
        value = context[key]
        if index is not None and isinstance(value, np.ndarray) and value.shape:
            value = value[index]
        result[key] = value
    return result


def context_keep(context, keys):
    keep = set(keys)
    for key in list(context.keys()):
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
        for k, value in ctx.items():
            if isinstance(value, np.ndarray):
                if usual_len is not None and len(value) != usual_len:
                    raise Exception('incoherent array lengths: %s''s is %d '
                                    'while the len of others is %d' %
                                    (k, len(value), usual_len))
                usual_len = len(value)
        return usual_len
