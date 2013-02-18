import numpy as np


class EntityContext(object):
    def __init__(self, entity, extra=None):
        self.entity = entity
        if extra is None:
            extra = {}
        self.extra = extra
        self['__entity__'] = entity
#        self['__weight_col__'] = entity.weight_col
#        self['__on_align_overflow__'] = entity.on_align_overflow

    def __getitem__(self, key):
        try:
            return self.extra[key]
        except KeyError:
            period = self.extra['period']
            array_period = self.entity.array_period
            if period == array_period:
                try:
                    return self.entity.temp_variables[key]
                except KeyError:
                    try:
                        return self.entity.array[key]
                    except ValueError:
                        raise KeyError(key)
            elif array_period is not None and period == array_period - 1 and \
                 self.entity.array_lag is not None:
                try:
                    return self.entity.array_lag[key]
                except ValueError:
                    raise KeyError(key)
            else:
                bounds = self.entity.output_rows.get(period)
                if bounds is not None:
                    startrow, stoprow = bounds
                else:
                    startrow, stoprow = 0, 0
#                print "loading from disk...",
#                res = timed(self.entity.table.read,
#                             start=startrow, stop=stoprow,
#                             field=key)
#                for level in range(1, 10, 2):
#                    print "   %d - compress:" % level,
#                    arr = timed(compress_column, res, level)
#                    print "decompress:",
#                    timed(decompress_column, arr)
#                return res
                return self.entity.table.read(start=startrow, stop=stoprow,
                                              field=key)

    # is the current array period the same as the context period?
    @property
    def _is_array_period(self):
        return self.entity.array_period == self.extra['period']

    def __setitem__(self, key, value):
        self.extra[key] = value

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

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
        if self._is_array_period:
            return len(self.entity.array)
        else:
            period = self.extra['period']
            bounds = self.entity.output_rows.get(period)
            if bounds is not None:
                startrow, stoprow = bounds
                return stoprow - startrow
            else:
                return 0

    def list_periods(self):
        return self.entity.output_index.keys()

    @property
    def id_to_rownum(self):
        period = self.extra['period']
        if self._is_array_period:
            return self.entity.id_to_rownum
        elif period in self.entity.output_index:
            return self.entity.output_index[period]
        else:
            #FIXME: yes, it's true, that if period is not in output_index, it
            # probably means that we are before start_period and in that case,
            # input_index == output_index, but it would be cleaner to simply
            # initialise output_index correctly
            return self.entity.input_index[period]


def new_context_like(context, length=None):
    #FIXME: nan should come from somewhere else
    if length is None:
        length = context_length(context)
    return {'period': context['period'],
            '__len__': length,
            '__entity__': context['__entity__'],
            '__globals__': context['__globals__'],
            'nan': float('nan')}


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
        assert len(index) == context_length(context), \
               "boolean index has length %d instead of %d" % \
               (len(index), context_length(context))
        length = np.sum(index)
    result = new_context_like(context, length=length)
    for key in keys:
        value = context[key]
        if index is not None and isinstance(value, np.ndarray) and value.shape:
            value = value[index]
        result[key] = value
    return result


def context_delete(context, rownum):
    result = {}
    # this copies everything including __len__, period, nan, ...
    for key in context.keys():
        value = context[key]
        # globals are left unmodified
        if key != '__globals__':
            if isinstance(value, np.ndarray) and value.shape:
                value = np.delete(value, rownum)
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
