# encoding: utf-8
from __future__ import print_function, division

from itertools import izip, groupby
from operator import itemgetter

import numpy as np
import numexpr as ne

from expr import (Expr, Variable, getdtype, expr_eval, missing_values,
                  get_missing_value, always, FunctionExpr)
from context import context_length
from utils import removed

# TODO: merge this typemap with the one in tsum
counting_typemap = {bool: int, int: int, float: float}


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
                            "unknown entity (%s)" % (self._name,
                                                     self._entity.name,
                                                     target_name))

    def _target_context(self, context):
        # we need a "fresh" context (fresh_data=True) so that if we come from
        #  a subset context (dict instead of EntityContext, eg in a new()
        # child), the target_expr must be evaluated on an unfiltered
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
    def get(self, key, *args, **kwargs):
        if isinstance(key, basestring):
            entity = self._target_entity

            # We could use entity.variables instead but since local variables
            # are not in there (and links can currently point to them), we need
            # to special case that and it does not make things any simpler.
            if key in entity.links:
                key = entity.links[key]
            else:
                key = Variable(entity, key)

        return LinkGet(self, key, *args, **kwargs)

    __getattr__ = get


class One2Many(Link):
    def count(self, *args, **kwargs):
        return Count(self, *args, **kwargs)

    def sum(self, *args, **kwargs):
        return Sum(self, *args, **kwargs)

    def avg(self, *args, **kwargs):
        return Avg(self, *args, **kwargs)

    def min(self, *args, **kwargs):
        return Min(self, *args, **kwargs)

    def max(self, *args, **kwargs):
        return Max(self, *args, **kwargs)

class One2One(Link):

    def get(self, key, missing_value=None):
        return LinkValue(self, key, missing_value)

    __getattr__ = get

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
    def __init__(self, entity, macros, links, prefix):
        self.entity = entity
        self.macros = macros
        self.links = links
        self.prefix = prefix

    def __getattr__(self, key):
        if key in self.macros:
            raise Exception("Using macros with the 'other' link is not "
                            "supported yet")
#            macro = self.macros[key]
#            variables = macro.collect_variables()
#            renames = dict((name, self.prefix + name) for name in variables)
#            return macro.rename_variables(renames)
        if key in self.links:
            link = self.links[key]
            # noinspection PyProtectedMember
            return link.__class__(link._name,
                                  self.prefix + link._link_field,
                                  link._target_entity_name,
                                  link._target_entity)
        return Variable(self.entity, self.prefix + key)


class LinkExpression(FunctionExpr):
    """
    Abstract base class for all functions which handle links (both many2one
    and one2many)
    """
    def target_context(self, context):
        # noinspection PyProtectedMember
        # TODO: implement this
        # return self.args.link._target_context(context)
        assert isinstance(self.link, Link)
        return self.link._target_context(context)

    # XXX: I think this is not enough. Implement Visitor pattern instead?
    def traverse(self):
        yield self

    def dtype(self, context):
        target_context = self.target_context(context)
        return getdtype(self.target_expr, target_context)

    @property
    def link(self):
        return self.args[0]

    @property
    def target_expr(self):
        return self.args[1]

    def __repr__(self):
        args, kwargs = self._original_args
        link, args = args[0], args[1:]
        # noinspection PyProtectedMember
        return self.format(link._name + "." + self.funcname, args, kwargs)


class LinkGet(LinkExpression):
    funcname = "get"
    no_eval = ('target_expr',)

    def traverse(self):
        # XXX: don't we also need the fields within the target expression?
        # noinspection PyProtectedMember
        yield Variable(self.link._entity, self.link._link_field)
        yield self

    @property
    def missing_value(self):
        return self.args[2]

    def get(self, key, missing_value=None):
        # partner.mother.household.region.get(households.count()))

        # partner is
        #   ManyToOne
        # partner.mother is (after __init__)
        #   LinkGet(Link('partner'), Link('mother'))
        # partner.mother.household is
        #   LinkGet(Link('partner'),
        #           LinkGet(Link('mother'), Link('household')))
        # partner.mother.household.region is
        #   LinkGet(Link('partner'),
        #           LinkGet(Link('mother'),
        #                   LinkGet(Link('household'), Link('region'))))
        lv = self
        link_chain = [lv.link]
        # find the deepest LinkGet
        while isinstance(lv.target_expr, LinkGet):
            lv = lv.target_expr
            link_chain.append(lv.link)
        assert isinstance(lv.target_expr, Link)

        # add one more link to the chain. Previously, we modified
        # lv.target_expr inplace and it was easier but this relied on the
        # fact that we cannot currently store partial links in variables,
        # eg: "p: partner" then "x: p.household" and this could be supported
        # some day.
        result = lv.target_expr.get(key, missing_value)
        for link in link_chain[::-1]:
            result = LinkGet(link, result)
        return result

    __getattr__ = get

    def compute(self, context, link, target_expr, missing_value=None):
        """
        link must be a Link instance
        target_expr can be any expression (it will be evaluated on the
                          target rows)
        """
        assert isinstance(link, Link)
        assert isinstance(target_expr, Expr), str(type(target_expr))

        # noinspection PyProtectedMember
        target_ids = context[link._link_field]
        target_context = self.target_context(context)

        id_to_rownum = target_context.id_to_rownum

        missing_int = missing_values[int]
        target_rows = id_to_rownum[target_ids]

        target_values = expr_eval(target_expr, target_context)
        if missing_value is None:
            missing_value = get_missing_value(target_values)

        result_values = target_values[target_rows]

        # it is a bit faster with numexpr (mixed_links: 0.22s -> 0.17s)
        return ne.evaluate("where((ids != mi) & (rows != mi), values, mv)",
                           {'ids': target_ids, 'rows': target_rows,
                            'values': result_values, 'mi': missing_int,
                            'mv': missing_value})

    def __repr__(self):
        if (self.missing_value is None and
                isinstance(self.target_expr, Variable)):
            return '%s.%s' % (self.link, self.target_expr)
        else:
            return super(LinkGet, self).__repr__()


class Aggregate(LinkExpression):
    no_eval = ('target_expr', 'target_filter')

    @property
    def target_filter(self):
        return self.args[2]

    def compute(self, context, link, target_expr, target_filter=None):
        # assert isinstance(context, EntityContext), \
        #         "one2many aggregates in groupby are currently not supported"
        assert isinstance(link, One2Many), "%s (%s)" % (link, type(link))

        # eg (in household entity):
        # persons: {type: one2many, target: person, field: hh_id}
        target_context = link._target_context(context)

        # this is a one2many, so the link column is on the target side
        # noinspection PyProtectedMember
        source_ids = target_context[link._link_field]
        expr_value = expr_eval(target_expr, target_context)
        filter_value = expr_eval(target_filter, target_context)
        if filter_value is not None:
            source_ids = source_ids[filter_value]
            # intentionally not using np.isscalar because of some corner
            # cases, eg. None and np.array(1.0)
            if isinstance(expr_value, np.ndarray) and expr_value.shape:
                expr_value = expr_value[filter_value]

        missing_int = missing_values[int]

        id_to_rownum = context.id_to_rownum
        if len(id_to_rownum):
            source_rows = id_to_rownum[source_ids]
            # filter out missing values: those where the value of the link
            # points to nowhere (-1)
            # XXX: use np.putmask(source_rows, source_ids == missing_int,
            #                    missing_int)
            source_rows[source_ids == missing_int] = missing_int
        else:
            assert np.all(source_ids == missing_int)
            # we need to make a copy because eval_rows modifies the array
            # in place in some cases (countlink and descendants)
            # TODO: document this fact in eval_rows
            source_rows = source_ids.copy()

        if isinstance(expr_value, np.ndarray) and expr_value.shape:
            assert len(source_rows) == len(expr_value), \
                "%d != %d" % (len(source_rows), len(expr_value))

        return self.eval_rows(source_rows, expr_value, context)

    def eval_rows(self, source_rows, expr_value, context):
        raise NotImplementedError()


class Sum(Aggregate):
    def eval_rows(self, source_rows, expr_value, context):
        # We can't use a negative value because that is not allowed by
        # bincount, and using a value too high will uselessly increase the size
        # of the array returned by bincount
        idx_for_missing = context_length(context)

        missing_int = missing_values[int]

        # filter out missing values: those where the object pointed to does not
        # exist anymore (the id corresponds to -1 in id_to_rownum)
        # XXX: use np.putmask(source_rows, source_ids == missing_int,
        #                    missing_int)
        source_rows[source_rows == missing_int] = idx_for_missing

        counts = self.count(source_rows, expr_value)
        counts.resize(idx_for_missing)
        return counts

    def count(self, source_rows, expr_value):
        if isinstance(expr_value, np.ndarray) and expr_value.shape:
            res = np.bincount(source_rows, expr_value)

            # we need to explicitly convert to the type of the value field
            # because bincount always return floats when its weight argument
            # is used.
            return res.astype(expr_value.dtype)
        else:
            # summing a scalar value
            counts = np.bincount(source_rows)
            # Optimization for countlink. Not using != 1 because it would
            # return a bad type (int) when expr_value is 1.0.
            return counts * expr_value if expr_value is not 1 else counts

    def dtype(self, context):
        return counting_typemap[super(Sum, self).dtype(context)]


class Count(Sum):
    @property
    def target_expr(self):
        return 1

    @property
    def target_filter(self):
        return self.args[1]

    def compute(self, context, link, target_filter=None):
        return super(Count, self).compute(context, link, 1, target_filter)


class Avg(Sum):
    def count(self, source_rows, expr_value):
        sums = super(Avg, self).count(source_rows, expr_value)
        count = np.bincount(source_rows)
        return sums / count

    dtype = always(float)


class Min(Aggregate):
    aggregate_func = min

    def eval_rows(self, source_rows, expr_value, context):
        result = np.empty(context_length(context), dtype=expr_value.dtype)
        result.fill(get_missing_value(expr_value))

        id_sort_indices = np.argsort(source_rows)
        sorted_rownum = source_rows[id_sort_indices]
        sorted_values = expr_value[id_sort_indices]
        groups = groupby(izip(sorted_rownum, sorted_values), key=itemgetter(0))
        aggregate_func = self.aggregate_func
        for rownum, values in groups:
            if rownum == -1:
                continue
            # Note that v[n] is faster than using an itemgetter, even with map
            result[rownum] = aggregate_func(v[1] for v in values)
        return result


class Max(Min):
    aggregate_func = max


def removed_functions():
    # all these functions do not exist anymore, but we need to give a hint
    # to users upgrading
    to_remove = [
        ('countlink', Count),
        ('sumlink', Sum),
        ('avglink', Avg),
        ('minlink', Min),
        ('maxlink', Max)
    ]
    return {name: removed(func, "%s(link, ...)" % name,
                          "link.%s(...)" % name[:-4])
            for name, func in to_remove}

functions = removed_functions()
