from itertools import izip, groupby
from operator import itemgetter

import numpy as np

from expr import Variable, dtype, expr_eval, \
                 missing_values, get_missing_value
from context import EntityContext, context_length
from registry import entity_registry
from properties import EvaluableExpression


class Link(object):
    def __init__(self, name, link_type, link_field, target_entity):
        # the leading underscores are necessary to not collide with
        # user-defined fields via __getattr__.
        self._name = name
        self._link_type = link_type
        self._link_field = link_field
        self._target_entity = target_entity

    def get(self, key, missing_value=None):
        if self._link_type == 'one2many':
            raise SyntaxError("To use the '%s' link (which is a one2many "
                              "link), you have to use link functions (e.g. "
                              "countlink)" % self._name)
        return LinkValue(self, key, missing_value)

    __getattr__ = get

    def __str__(self):
        return self._name


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
            return Link(link._name,
                        link._link_type,
                        self.prefix + link._link_field,
                        link._target_entity)
        return Variable(self.prefix + key)


class LinkExpression(EvaluableExpression):
    '''abstract base class for all function which handle links (both many2one
       and one2many'''
    def __init__(self, link):
        self.link = link

    def get_link(self, context):
        # use the context as the first entity, so that it works with subsets of
        # the entity population
        link = self.link
        if isinstance(link, basestring):
            link = context['__entity__'].links[link]
        return link

    def target_entity(self, context):
        link = self.get_link(context)
        return entity_registry[link._target_entity]

    def target_context(self, context):
        target_entity = self.target_entity(context)
        return EntityContext(target_entity,
                             {'period': context['period'],
                             '__globals__': context['__globals__']})

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
        link = self.get_link(context)
        #XXX: don't we also need the fields within the target expression?
        return set([link._link_field])

    def dtype(self, context):
        target_context = self.target_context(context)
        return dtype(self.target_expression, target_context)

    def get(self, key, missing_value=None):
        # in this case, target_expression must be a Variable with a
        # link name, however given that we have no context, we
        # don't know the current entity and
        # can't make a strong assertion here
        # assert self.target_expression in entity.links
        assert isinstance(self.target_expression, Variable)
        return LinkValue(self.link,
                         LinkValue(self.target_expression.name, key,
                                   missing_value))

    __getattr__ = get

    def evaluate(self, context):
        link = self.get_link(context)
        target_ids = expr_eval(Variable(link._link_field), context)
        target_context = self.target_context(context)

        id_to_rownum = target_context.id_to_rownum

        missing_int = missing_values[int]
        target_rows = id_to_rownum[target_ids]

        target_values = expr_eval(self.target_expression, target_context)
        missing_value = self.missing_value
        if missing_value is None:
            missing_value = get_missing_value(target_values)

        valid_link = (target_ids != missing_int) & (target_rows != missing_int)
        return np.where(valid_link, target_values[target_rows], missing_value)

    def __str__(self):
        return '%s.%s' % (self.link, self.target_expression)
    __repr__ = __str__


class AggregateLink(LinkExpression):
    def __init__(self, link, target_filter=None):
        LinkExpression.__init__(self, link)
        self.target_filter = target_filter

    def evaluate(self, context):
        assert isinstance(context, EntityContext), \
               "aggregates in groupby is currently not supported"
        link = self.get_link(context)
        assert link._link_type == 'one2many'

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
            # in place
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
        if target_filter is not None:
            value_column = value_column[target_filter]
        assert len(source_rows) == len(value_column)
        res = np.bincount(source_rows, value_column)

        # we need to explicitly convert to the type of the value field because
        # bincount always return floats when its weight argument is used.
        return res.astype(value_column.dtype)

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

        # silence x/0 and 0/0
        #TODO: this should be either done globally or not at all, or as a last
        # resort restored before returning from this function.
#        np.seterr(invalid='ignore', divide='ignore')

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


functions = {
    'countlink': CountLink,
    'sumlink': SumLink,
    'avglink': AvgLink,
    'minlink': MinLink,
    'maxlink': MaxLink,
}
