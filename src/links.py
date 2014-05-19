from __future__ import print_function

from itertools import izip, groupby
from operator import itemgetter

import numpy as np
import numexpr as ne

from expr import (Expr, Variable, getdtype, expr_eval, missing_values,
                  get_missing_value, always, FunctionExpr)
from context import context_length
from utils import deprecated


class Link(object):
    def __init__(self, name, link_field, target_entity_name,
                 target_entity=None):
        # the leading underscores are necessary to not collide with
        # user-defined fields via __getattr__.
        self._name = name
        self._link_field = link_field
        self._target_entity_name = target_entity_name
        self._target_entity = target_entity
        self._entity = None

    def _attach(self, entity):
        self._entity = entity

    def _resolve_target(self, entities):
        target_name = self._target_entity_name
        try:
            self._target_entity = entities[target_name]
        except KeyError:
            raise Exception("Target of '%s' link in entity '%s' is an "
                            "unknown entity (%s)" % (self.name,
                                                     self._entity.name,
                                                     target_name))

    def _target_context(self, context):
        # we need a "fresh" context (fresh_data=True) so that if we come from
        #  a subset context (dict instead of EntityContext, eg in a new()
        # child), the target_expression must be evaluated on an unfiltered
        # target context (because, even if the current context is filtered,
        # there is no restriction on where the link column ids can point to (
        # they can point to ids outside the filter)
        return context.clone(fresh_data=True,
                             entity_name=self._target_entity_name)

    def __str__(self):
        return self._name

    def __repr__(self):
        return "%s(%s, %s, %s)" % (self.__class__.__name__, self._name,
                                   self._link_field, self._target_entity_name)


class Many2One(Link):
    def get(self, key, missing_value=None):
        if isinstance(key, basestring):
            key = Variable(key)
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
            #noinspection PyProtectedMember
            return link.__class__(link._name,
                                  self.prefix + link._link_field,
                                  link._target_entity_name,
                                  link._target_entity)
        return Variable(self.prefix + key)


class LinkExpression(FunctionExpr):
    """
    Abstract base class for all functions which handle links (both many2one
    and one2many)
    """
    def target_context(self, context):
        #noinspection PyProtectedMember
        #TODO: implement this
        # return self.args.link._target_context(context)
        assert isinstance(self.link, Link)
        return self.link._target_context(context)

    #XXX: I think this is not enough. Implement Visitor pattern instead?
    def traverse(self, context):
        yield self


class LinkValue(LinkExpression):
    funcname = "linkvalue"
    no_eval = ('target_expression',)

    def traverse(self, context):
        #XXX: don't we also need the fields within the target expression?
        #noinspection PyProtectedMember
        yield Variable(self.link._link_field)
        yield self

    def dtype(self, context):
        target_context = self.target_context(context)
        return getdtype(self.target_expression, target_context)

    @property
    def link(self):
        return self.args[0]

    @property
    def target_expression(self):
        return self.args[1]

    @property
    def missing_value(self):
        return self.args[2]

    def get(self, key, missing_value=None):
        # partner.mother.household.region.get(households.count()))

        # partner is
        #   ManyToOne
        # partner.mother is (after __init__)
        #   LinkValue(Link('partner'), Variable('mother'))
        # partner.mother.household is
        #   LinkValue(Link('partner'),
        #             LinkValue(Link('mother'), Variable('household')))
        # partner.mother.household.region is
        #   LinkValue(Link('partner'),
        #             LinkValue(Link('mother'),
        #                       LinkValue(Link('household'),
        #                                 Variable('region'))))
        lv = self
        link_chain = [lv.link]
        # find the deepest LinkValue
        while isinstance(lv.target_expression, LinkValue):
            lv = lv.target_expression
            link_chain.append(lv.link)
        expr = lv.target_expression

        # at this point, expr must be a Variable with a link name,
        # however given that we have no context, we do not know the *current*
        # entity (we do know the target entity) and cannot make a strong
        # assertion here.
        #XXX: we could add an _entity field to the Link class though
        # assert expr.name in entity.links
        assert isinstance(expr, Variable), "%s is not a Variable (%s)" \
                                           % (expr, type(expr))
        #noinspection PyProtectedMember
        deepest_link = lv.link._target_entity.links[expr.name]

        # add one more link to the chain. Previously, we modified
        # lv.target_expression inplace and it was easier but this relied on the
        # fact that we cannot currently store partial links in variables,
        # eg: "p: partner" then "x: p.household" and this could be supported
        # some day.
        result = deepest_link.get(key, missing_value)
        for link in link_chain[::-1]:
            result = LinkValue(link, result)
        return result

    __getattr__ = get

    def compute(self, context, link, target_expression, missing_value=None):
        """
        link must be a Link instance
        target_expression can be any expression (it will be evaluated on the
                          target rows)
        """
        assert isinstance(link, Link)
        assert isinstance(target_expression, Expr), str(type(target_expression))

        #noinspection PyProtectedMember
        target_ids = expr_eval(Variable(link._link_field), context)
        target_context = self.target_context(context)

        id_to_rownum = target_context.id_to_rownum

        missing_int = missing_values[int]
        target_rows = id_to_rownum[target_ids]

        target_values = expr_eval(target_expression, target_context)
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
    no_eval = ('target_filter',)

    @property
    def link(self):
        return self.args[0]

    @property
    def target_filter(self):
        return self.args[1]

    def compute(self, context, link, target_filter=None):
        # assert isinstance(context, EntityContext), \
        #         "one2many aggregates in groupby are currently not supported"
        assert isinstance(link, One2Many), "%s (%s)" % (link, type(link))

        # eg (in household entity):
        # persons: {type: one2many, target: person, field: hh_id}
        target_context = self.target_context(context)

        # this is a one2many, so the link column is on the target side
        #noinspection PyProtectedMember
        link_column = expr_eval(Variable(link._link_field), target_context)

        missing_int = missing_values[int]

        if target_filter is not None:
            target_filter = expr_eval(target_filter, target_context)
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
        raise NotImplementedError()


class CountLink(AggregateLink):
    funcname = 'countlink'

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

    dtype = always(int)

    #FIXME: str() should return the new syntax: link.count() instead of
    # countlink(link)
    def __str__(self):
        if self.target_filter is not None:
            target_filter = ", target_filter=%s" % self.target_filter
        else:
            target_filter = ""
        #noinspection PyProtectedMember
        return '%s(%s%s)' % (self.funcname, self.link._name, target_filter)


class SumLink(CountLink):
    funcname = 'sumlink'
    no_eval = ('target_expr', 'target_filter')

    def compute(self, context, link, target_expr, target_filter=None):
        return super(SumLink, self).compute(context, link, target_filter)

    @property
    def target_expr(self):
        return self.args[1]

    @property
    def target_filter(self):
        return self.args[2]

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
            # summing a scalar value
            return np.bincount(source_rows) * value_column

    def dtype(self, context):
        target_context = self.target_context(context)
        expr_dype = getdtype(self.target_expr, target_context)
        #TODO: merge this typemap with the one in tsum
        typemap = {bool: int, int: int, float: float}
        return typemap[expr_dype]

    def __str__(self):
        if self.target_filter is not None:
            target_filter = ", target_filter=%s" % self.target_filter
        else:
            target_filter = ""
        #noinspection PyProtectedMember
        return '%s(%s, %s%s)' % (self.funcname, self.link._name,
                                 self.target_expr, target_filter)


class AvgLink(SumLink):
    funcname = 'avglink'

    def count(self, source_rows, target_filter, context):
        sums = super(AvgLink, self).count(source_rows, target_filter, context)
        count = np.bincount(source_rows)

        # this is slightly sub optimal if the value column contains integers
        # as the data is converted from float to int then back to float
        return sums.astype(float) / count

    dtype = always(float)


class MinLink(AggregateLink):
    funcname = 'minlink'
    aggregate_func = min

    def __init__(self, link, target_expr, target_filter=None):
        AggregateLink.__init__(self, link, target_filter)
        self.target_expr = target_expr

    def dtype(self, context):
        target_context = self.target_context(context)
        return getdtype(self.target_expr, target_context)

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
        #FIXME: this is the old syntax
        if self.target_filter is not None:
            target_filter = ", target_filter=%s" % self.target_filter
        else:
            target_filter = ""
        #noinspection PyProtectedMember
        return '%s(%s, %s%s)' % (self.funcname, self.link._name,
                                 self.target_expr, target_filter)


class MaxLink(MinLink):
    funcname = 'maxlink'
    aggregate_func = max


def deprecated_functions():
    # all these functions are deprecated
    to_deprecate = [
        ('countlink', CountLink),
        ('sumlink', SumLink),
        ('avglink', AvgLink),
        ('minlink', MinLink),
        ('maxlink', MaxLink)
    ]
    funcs = {}
    for name, func in to_deprecate:
        funcs[name] = deprecated(func,
                                 "%s(link, ...) is deprecated, please "
                                 "use link.%s(...) instead"
                                 % (name, name[:-4]))
    return funcs

functions = deprecated_functions()
