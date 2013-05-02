from itertools import izip, groupby
from operator import itemgetter
import warnings

import numpy as np
import numexpr as ne

from expr import Variable, dtype, expr_eval, missing_values, get_missing_value
from exprbases import EvaluableExpression
from context import EntityContext, context_length
from registry import entity_registry


class Link(object):
    def __init__(self, name, link_field, target_entity_name):
        # the leading underscores are necessary to not collide with
        # user-defined fields via __getattr__.
        self._name = name
        self._link_field = link_field
        self._target_entity_name = target_entity_name

    def __str__(self):
        return self._name

    def _target_entity(self):
        return entity_registry[self._target_entity_name]

    def _target_context(self, context):
        target_entity = self._target_entity()
        return EntityContext(target_entity,
                             {'period': context['period'],
                             '__globals__': context['__globals__']})


class Many2One(Link):
    def get(self, key, missing_value=None):
        return LinkValue(self, key, missing_value)

    __getattr__ = get


class One2Many(Link):
    def count(self, target_filter=None):
        return CountLink(self, target_filter)

    def sum(self, target_expr, target_filter=None):
        return SumLink(self, target_expr, target_filter)

    def avg(self, target_expr, target_filter=None):
        return AvgLink(self, target_expr, target_filter)

    def min(self, target_expr, target_filter=None):
        return MinLink(self, target_expr, target_filter)

    def max(self, target_expr, target_filter=None):
        return MaxLink(self, target_expr, target_filter)


class PrefixingLink(object):
    def __init__(self, macros, links, prefix):
        self.macros = macros
        self.links = links
        self.prefix = prefix

    def __getattr__(self, key):
        if key in self.macros:
            raise Exception("Using macros with the 'other' link is not "
                            "supported yet")
#            macro = self.macros[key]
#            variables = macro.collect_variables(entity=entity)
#            renames = dict((name, self.prefix + name) for name in variables)
#            return macro.rename_variables(renames)
        if key in self.links:
            link = self.links[key]
            return link.__class__(link._name,
                                  self.prefix + link._link_field,
                                  link._target_entity_name)
        return Variable(self.prefix + key)


class LinkExpression(EvaluableExpression):
    '''abstract base class for all function which handle links (both many2one
       and one2many'''

    def __init__(self, link):
        self.link = link

    def target_context(self, context):
        return self.link._target_context(context)

    #XXX: I think this is not enough. Implement Visitor pattern instead?
    def traverse(self, context):
        yield self


class LinkValue(LinkExpression):
    def __init__(self, link, target_expression, missing_value=None):
        '''
        links can be either a Link instance, a string, or a list of either
        target_expression can be any expression (it will be evaluated on the
                          target rows)
        '''
        LinkExpression.__init__(self, link)
        if isinstance(target_expression, basestring):
            target_expression = Variable(target_expression)
        self.target_expression = target_expression
        self.missing_value = missing_value

    def collect_variables(self, context):
        #XXX: don't we also need the fields within the target expression?
        return set([self.link._link_field])

    def dtype(self, context):
        target_context = self.target_context(context)
        return dtype(self.target_expression, target_context)

    def get(self, key, missing_value=None):
        # in this case, target_expression must be a Variable with a link name,
        # however given that we have no context, we do not know the current
        # entity and cannot make a strong assertion here.
        #XXX: we could add an _entity fields to the Link class though
        # assert self.target_expression in entity.links
        assert isinstance(self.target_expression, Variable)
        target_entity = self.link._target_entity()
        target_link = target_entity.links[self.target_expression.name]
        return LinkValue(self.link,
                         LinkValue(target_link, key, missing_value))

    __getattr__ = get

    def evaluate(self, context):
        target_ids = expr_eval(Variable(self.link._link_field), context)
        target_context = self.target_context(context)

        id_to_rownum = target_context.id_to_rownum

        missing_int = missing_values[int]
        target_rows = id_to_rownum[target_ids]

        target_values = expr_eval(self.target_expression, target_context)
        missing_value = self.missing_value
        if missing_value is None:
            missing_value = get_missing_value(target_values)

        result_values = target_values[target_rows]

        # it is a bit faster with numexpr (mixed_links: 0.22s -> 0.17s)
        return ne.evaluate("where((ids != mi) & (rows != mi), values, mv)",
                           {'ids': target_ids, 'rows': target_rows,
                            'values': result_values, 'mi': missing_int,
                            'mv': missing_value})

    def __str__(self):
        return '%s.%s' % (self.link, self.target_expression)
    __repr__ = __str__


class AggregateLink(LinkExpression):
    def __init__(self, link, target_filter=None):
        LinkExpression.__init__(self, link)
        self.target_filter = target_filter

    def evaluate(self, context):
        assert isinstance(context, EntityContext), \
               "one2many aggregates in groupby are currently not supported"
        link = self.link
        assert isinstance(link, One2Many)

        # eg (in household entity):
        # persons: {type: one2many, target: person, field: hh_id}
        target_context = self.target_context(context)

        # this is a one2many, so the link column is on the target side
        link_column = expr_eval(Variable(link._link_field), target_context)

        missing_int = missing_values[int]

        if self.target_filter is not None:
            target_filter = expr_eval(self.target_filter, target_context)
            source_ids = link_column[target_filter]
        else:
            target_filter = None
            source_ids = link_column

        id_to_rownum = context.id_to_rownum
        if len(id_to_rownum):
            source_rows = id_to_rownum[source_ids]
            # filter out missing values: those where the value of the link
            # points to nowhere (-1)
            #XXX: use np.putmask(source_rows, source_ids == missing_int,
            #                    missing_int)
            source_rows[source_ids == missing_int] = missing_int
        else:
            assert np.all(source_ids == missing_int)
            # we need to make a copy because eval_rows modifies the array
            # in place in some cases (countlink and descendants)
            #TODO: document this fact in eval_rows
            source_rows = source_ids.copy()

        return self.eval_rows(source_rows, target_filter, context)

    def eval_rows(self, source_rows, target_filter, context):
        raise NotImplementedError

    def collect_variables(self, context):
        # no variable at all because collect_variable is only interested in
        # the columns of the *current entity* and since we are only working
        # with one2many relationships, the link column is always on the other
        # side.
        return set()


class CountLink(AggregateLink):
    func_name = 'countlink'

    def eval_rows(self, source_rows, target_filter, context):
        # We can't use a negative value because that is not allowed by
        # bincount, and using a value too high will uselessly increase the size
        # of the array returned by bincount
        idx_for_missing = context_length(context)

        missing_int = missing_values[int]

        # filter out missing values: those where the object pointed to does not
        # exist anymore (the id corresponds to -1 in id_to_rownum)
        #XXX: use np.putmask(source_rows, source_ids == missing_int,
        #                    missing_int)
        source_rows[source_rows == missing_int] = idx_for_missing

        counts = self.count(source_rows, target_filter, context)
        counts.resize(idx_for_missing)
        return counts

    def count(self, source_rows, target_filter, context):
        #XXX: the test is probably not needed anymore with numpy 1.6.2+
        if len(source_rows):
            return np.bincount(source_rows)
        else:
            return np.array([], dtype=int)

    def dtype(self, context):
        return int

    def __str__(self):
        if self.target_filter is not None:
            target_filter = ", target_filter=%s" % self.target_filter
        else:
            target_filter = ""
        return '%s(%s%s)' % (self.func_name, self.link._name, target_filter)


class SumLink(CountLink):
    func_name = 'sumlink'

    def __init__(self, link, target_expr, target_filter=None):
        CountLink.__init__(self, link, target_filter)
        self.target_expr = target_expr

    def count(self, source_rows, target_filter, context):
        target_context = self.target_context(context)
        value_column = expr_eval(self.target_expr, target_context)
        if isinstance(value_column, np.ndarray) and value_column.shape:
            if target_filter is not None:
                value_column = value_column[target_filter]
            assert len(source_rows) == len(value_column), \
                   "%d != %d" % (len(source_rows), len(value_column))

            res = np.bincount(source_rows, value_column)

            # we need to explicitly convert to the type of the value field
            # because bincount always return floats when its weight argument
            # is used.
            return res.astype(value_column.dtype)
        else:
            # suming a scalar value
            return np.bincount(source_rows) * value_column

    def dtype(self, context):
        target_context = self.target_context(context)
        expr_dype = dtype(self.target_expr, target_context)
        #TODO: merge this typemap with tsum's
        typemap = {bool: int, int: int, float: float}
        return typemap[expr_dype]

    def __str__(self):
        if self.target_filter is not None:
            target_filter = ", target_filter=%s" % self.target_filter
        else:
            target_filter = ""
        return '%s(%s, %s%s)' % (self.func_name, self.link._name,
                                 self.target_expr, target_filter)


class AvgLink(SumLink):
    func_name = 'avglink'

    def count(self, source_rows, target_filter, context):
        sums = super(AvgLink, self).count(source_rows, target_filter, context)
        count = np.bincount(source_rows)

        # this is slightly sub optimal if the value column contains integers
        # as the data is converted from float to int then back to float
        return sums.astype(float) / count

    def dtype(self, context):
        return float


class MinLink(AggregateLink):
    func_name = 'minlink'
    aggregate_func = min

    def __init__(self, link, target_expr, target_filter=None):
        AggregateLink.__init__(self, link, target_filter)
        self.target_expr = target_expr

    def dtype(self, context):
        target_context = self.target_context(context)
        return dtype(self.target_expr, target_context)

    def eval_rows(self, source_rows, target_filter, context):
        target_context = self.target_context(context)
        value_column = expr_eval(self.target_expr, target_context)
        if target_filter is not None:
            value_column = value_column[target_filter]
        assert len(source_rows) == len(value_column)

        result = np.empty(context_length(context), dtype=value_column.dtype)
        result.fill(get_missing_value(value_column))

        id_sort_indices = np.argsort(source_rows)
        sorted_rownum = source_rows[id_sort_indices]
        sorted_values = value_column[id_sort_indices]
        groups = groupby(izip(sorted_rownum, sorted_values), key=itemgetter(0))
        aggregate_func = self.aggregate_func
        for rownum, values in groups:
            if rownum == -1:
                continue
            # Note that v[n] is faster than using an itemgetter, even with map
            result[rownum] = aggregate_func(v[1] for v in values)
        return result

    def __str__(self):
        if self.target_filter is not None:
            target_filter = ", target_filter=%s" % self.target_filter
        else:
            target_filter = ""
        return '%s(%s, %s%s)' % (self.func_name, self.link._name,
                                 self.target_expr, target_filter)


class MaxLink(MinLink):
    func_name = 'maxlink'
    aggregate_func = max


class UserDeprecationWarning(UserWarning):
    pass


def deprecated(class_):
    func_name = class_.__name__.lower()
    method_name = func_name[:-4]

    def func(*args, **kwargs):
        #TODO: when we will be able to link expressions to line numbers in the
        # model, we should use warnings.warn_explicit instead
        msg = "%s(link, ...) is deprecated, please use " \
              "link.%s(...) instead" % (func_name, method_name)
        warnings.warn(msg, UserDeprecationWarning)
        expr = class_(*args, **kwargs)
#        print "Warning: %s\n     at: %s" % (msg, expr)
        return expr
    func.__name__ = func_name
    return func

# all the these functions are deprecated
functions = {
    'countlink': deprecated(CountLink),
    'sumlink': deprecated(SumLink),
    'avglink': deprecated(AvgLink),
    'minlink': deprecated(MinLink),
    'maxlink': deprecated(MaxLink),
}
