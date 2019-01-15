# encoding: utf-8
from __future__ import absolute_import, division, print_function

import random

import numpy as np
import larray as la

from liam2.expr import expr_eval, always, expr_cache
from liam2.exprbases import FilteredExpression
from liam2.context import context_length, context_delete, context_subset, context_keep
from liam2.utils import loop_wh_progress

try:
    from liam2.cpartition import group_indices_nd
except ImportError:
    group_indices_nd = None


def group_context(used_variables, setfilter, context):
    """
    return a dict of the form:
    {
    'field1': array([field1_value_for_group1, field1_value_for_group2, ...]),
    'field2': array([field2_value_for_group1, field2_value_for_group2, ...]),
    '__ids__': array([array_of_ids_for_group1, array_of_ids_for_group2, ...]),
    '__len__': number_of_groups
    }
    """
    names = sorted(used_variables)
    if isinstance(setfilter, la.Array):
        setfilter = setfilter.data
    idcol = context['id']
    assert isinstance(idcol, la.Array)
    idcol = idcol.data
    columns = [context[name] for name in names]
    columns = [col.data if isinstance(col, la.Array) else col
               for col in columns]

    if group_indices_nd is None:
        raise Exception('aglo="byvalue" is not available when C extensions cannot be used')

    # group_indices_nd returns a dict {value_or_tuple: array_of_indices}
    # XXX: we cannot use partition_nd as-is because it computes unique labels per dimension, but we might want to
    #      factorize a common function.
    groups_indices = group_indices_nd(columns, setfilter)
    num_groups = len(groups_indices)

    # [(field1_value_for_group1, field2_value_for_group1, ..., fieldM_value_for_group1),
    #  ...,
    #  (field1_value_for_groupN, field2_value_for_group1, ..., fieldM_value_for_groupN)]
    combined_labels_present = list(groups_indices.keys())

    # transform that to a list per field:

    # [[field1_value_for_group1, field1_value_for_group2, ..., field1_value_for_groupN]
    #  ...,
    #  [fieldM_value_for_group1, fieldM_value_for_group2, ..., fieldM_value_for_groupN]
    keylists = zip(*combined_labels_present) if len(columns) > 1 else [combined_labels_present]

    # we need Arrays with an id axis (using wildcard axis because ids are bogus anyway -- those are *group* ids)
    keyarrays = [la.Array(c, la.Axis(len(c), 'id')) for c in keylists]

    # we want a 1d array of arrays, not the 2d array that np.array(groups_indices.values())
    # produces if we have a list of arrays with all the same length
    ids_by_group = np.empty(num_groups, dtype=object)
    ids_by_group[:] = [idcol[group_indices] for group_indices in groups_indices.values()]

    result = dict(zip(names, keyarrays))
    result['__ids__'] = ids_by_group
    result['__len__'] = num_groups
    return result


class Matching(FilteredExpression):
    """
    Base class for matching functions
    """
    dtype = always(int)


class RankMatching(Matching):
    """
    Matching based on rank/order
    * set 1 is ranked by decreasing orderby1
    * set 2 is ranked by decreasing orderby2
    * individuals in the nth position in each list are matched together.
    """
    funcname = 'rank_matching'
    no_eval = ('set1filter', 'set2filter')

    def compute(self, context, set1filter, set2filter, orderby1, orderby2):
        set1filterexpr = self._getfilter(context, set1filter)
        set1filtervalue = expr_eval(set1filterexpr, context)
        set2filterexpr = self._getfilter(context, set2filter)
        set2filtervalue = expr_eval(set2filterexpr, context)
        set1len = set1filtervalue.sum()
        set2len = set2filtervalue.sum()
        numtomatch = min(set1len, set2len)
        print("matching with %d/%d individuals" % (set1len, set2len))
        result = np.full(context_length(context), -1, dtype=int)
        if not numtomatch:
            return result

        filtered_orderby1 = orderby1[set1filtervalue]
        if isinstance(filtered_orderby1, la.Array):
            sorted_set1_indices = filtered_orderby1.indicesofsorted().i[-numtomatch:]
        else:
            sorted_set1_indices = filtered_orderby1.argsort()[-numtomatch:]
        filtered_orderby2 = orderby2[set2filtervalue]
        if isinstance(filtered_orderby2, la.Array):
            sorted_set2_indices = filtered_orderby2.indicesofsorted().i[-numtomatch:]
        else:
            sorted_set2_indices = filtered_orderby2.argsort()[-numtomatch:]

        set1ids = context['id'][set1filtervalue]
        set2ids = context['id'][set2filtervalue]

        id_to_rownum = context.id_to_rownum
        sorted_set1_ids = set1ids.i[sorted_set1_indices]
        sorted_set2_ids = set2ids.i[sorted_set2_indices]
        # cannot use sorted_setX_indices because those are "local" indices
        result[id_to_rownum[sorted_set1_ids]] = sorted_set2_ids
        result[id_to_rownum[sorted_set2_ids]] = sorted_set1_ids
        return result


class SequentialMatching(Matching):
    """
    Matching based on searching for the best match one by one.

    In general that kind of matching does not provide the best matching,
    because it does not optimize the *overall* distance function (it does not
    necessarily return the match with the lowest sum of distances between all
    matched pairs).

    - orderby gives the way individuals of set 1 are sorted before matching.
      The first individual will be matched with the highest scoring individual
      from set 2. The next individual in set 1 will be matched with the highest
      scoring individual among the remaining individuals in set 2.

    - orderby can be :
        - an expression (usually a variable name). It is supposed to be
          a "difficulty" because it is better (according to a general
          objective score) to match hard-to-match people first.
        - the string 'EDtM', in which case, the (reduced) "Euclidean Distance to
          the Mean" is used to order individuals.
    """
    funcname = 'matching'
    no_eval = ('set1filter', 'set2filter', 'score', 'orderby')

    def traverse(self):
        # FIXME: we should not override the parent traverse method, so that all
        # "child" expressions are traversed too.
        # This is not done currently, because it would traverse score_expr.
        # This is a problem because traverse is used by collect_variables and
        # the presence of variables is checked in expr.expr_eval() before
        # the evaluate method is called and the context is completed during
        # evaluation (__other_xxx is added during evaluation).
        yield self

    def compute(self, context, set1filter, set2filter, score, orderby,
                pool_size=None, algo='onebyone'):
        global matching_ctx

        if pool_size is not None:
            assert isinstance(pool_size, int)
            assert pool_size > 0

        set1filterexpr = self._getfilter(context, set1filter)
        set1filtervalue = expr_eval(set1filterexpr, context)
        set2filterexpr = self._getfilter(context, set2filter)
        set2filtervalue = expr_eval(set2filterexpr, context)
        set1len = set1filtervalue.sum()
        set2len = set2filtervalue.sum()
        print("matching with %d/%d individuals" % (set1len, set2len), end='')

        # TODO: instead of filtering "v.name not in global_tables", we should keep the whole Variable instance and use
        #       that in context.subset, context_keep, et. al. But adding support for Variable in all those
        #       functions would be some significant work.
        varnames = {v.name for v in score.collect_variables() if v.name not in context.global_tables}
        used_variables1 = {n for n in varnames if not n.startswith('__other_')}
        used_variables2 = {n[8:] for n in varnames if n.startswith('__other_')}

        if isinstance(orderby, str):
            assert orderby == 'EDtM'
            orderby_vars = used_variables1
        else:
            orderby_vars = {v.name for v in orderby.collect_variables()}

        if algo == 'onebyone':
            all_vars = {'id'} | used_variables1 | orderby_vars
            set1 = context.subset(set1filtervalue, all_vars, set1filterexpr)
            set2 = context.subset(set2filtervalue, {'id'} | used_variables2,
                                  set2filterexpr)

            # subset creates a dict for the current entity, so .entity_data is a dict
            set1 = set1.entity_data
            set2 = set2.entity_data

            # FIXME
            # TypeError: reshape() takes 2 positional arguments but 3 were given
            # this was meant to make set1['__ids__'][sorted_idx] an array of length 1
            # so that "onebyone" and "byvalue" branches can use the same code within the loop
            set1['__ids__'] = set1['id'].data.reshape(set1len, 1)
            set2['__ids__'] = set2['id'].data.reshape(set2len, 1)

            print()
        else:
            # optimized matching by grouping sets by values, which usually
            # means smaller sets and improved running time.
            assert algo == 'byvalue'

            # if orderby contains variables that are not used in the score
            # expression, this will effectively add variables in the
            # matching context AND group by those variables. This is correct
            # because otherwise (if we did not group by them), we could have
            # groups containing individuals with different values of the
            # ordering variables (ie the ordering would not be respected).
            set1 = group_context(used_variables1 | orderby_vars,
                                 set1filtervalue, context)
            set2 = group_context(used_variables2, set2filtervalue, context)

            # we cannot simply take the [:min(set1len, set2len)] indices like in
            # the non-optimized case and iterate over that because we don't know
            # how many groups we will need to match.
            print(" (%d/%d groups)"
                  % (context_length(set1), context_length(set2)))

        if isinstance(orderby, str):
            orderbyvalue = np.zeros(context_length(set1))
            for name in used_variables1:
                column = set1[name]
                orderbyvalue += (column - column.mean()) ** 2 / column.var()
        else:
            orderbyvalue = expr_eval(orderby, context.clone(entity_data=set1))

        # Delete variables which are not in the score expression (but in the
        # orderby expr or possibly "id") because they are no longer needed and
        # would slow things down.
        context_keep(set1, used_variables1)
        context_keep(set2, used_variables2)

        if isinstance(orderbyvalue, la.Array):
            # TODO: this should be the only case supported
            sorted_set1_indices = orderbyvalue.data.argsort()[::-1]
        else:
            sorted_set1_indices = orderbyvalue.argsort()[::-1]

        result = np.full(context_length(context), -1, dtype=int)
        id_to_rownum = context.id_to_rownum

        # prefix all keys except __len__
        matching_ctx = {'__other_' + k if k != '__len__' else k: v
                        for k, v in set2.items()}

        def match_cell(idx, sorted_idx, pool_size):
            global matching_ctx

            set2_size = context_length(matching_ctx)
            if not set2_size:
                raise StopIteration

            if pool_size is not None and set2_size > pool_size:
                pool_indices = random.sample(range(set2_size), pool_size)
                local_ctx = context_subset(matching_ctx, pool_indices)
            else:
                local_ctx = matching_ctx.copy()

            # local_var_names = {'__ids__'} | used_variables1
            local_ctx['__ids__'] = set1['__ids__'][sorted_idx]
            # using .data instead of .i is marginally faster
            local_ctx.update((k, set1[k].data[sorted_idx] if isinstance(set1[k], la.Array) else set1[k][sorted_idx])
                             for k in used_variables1)

            eval_ctx = context.clone(entity_data=local_ctx)
            set2_scores = expr_eval(score, eval_ctx)
            if isinstance(set2_scores, la.Array):
                assert set2_scores.ndim == 1 and set2_scores.axes[0].name == 'id'
                cell2_idx = set2_scores.data.argmax()
            else:
                cell2_idx = set2_scores.argmax()

            cell1ids = local_ctx['__ids__']
            if np.isscalar(cell1ids):
                # this is the "onebyone" case
                cell1ids = [cell1ids]

            cell2ids = local_ctx['__other___ids__'][cell2_idx]
            if np.isscalar(cell2ids):
                cell2ids = [cell2ids]

            if pool_size is not None and set2_size > pool_size:
                # transform pool-local index to set/matching_ctx index
                cell2_idx = pool_indices[cell2_idx]

            cell1size = len(cell1ids)
            cell2size = len(cell2ids)
            nb_match = min(cell1size, cell2size)

            # we could introduce a random choice here but it is not
            # much necessary. In that case, it should be done in group_context
            ids1 = cell1ids[:nb_match]
            ids2 = cell2ids[:nb_match]

            result[id_to_rownum[ids1]] = ids2
            result[id_to_rownum[ids2]] = ids1

            if nb_match == cell2size:
                matching_ctx = context_delete(matching_ctx, cell2_idx)
            else:
                # other variables do not need to be modified since the cell
                # only got smaller and was not deleted
                matching_ctx['__other___ids__'][cell2_idx] = cell2ids[nb_match:]

            # FIXME: the expr gets cached for the full matching_ctx at the
            # beginning and then when another women with the same values is
            # found, it thinks it can reuse the expr but it breaks because it
            # has not the correct length.

            # the current workaround is to invalidate the whole cache for the
            # current entity but this is not the right way to go.
            # * disable the cache for matching?
            # * use a local cache so that methods after matching() can use
            # what was in the cache before matching(). Shouldn't the cache be
            # stored inside the context anyway?
            # expr_cache.invalidate(context.period, context.entity_name)

            if nb_match < cell1size:
                set1['__ids__'][sorted_idx] = cell1ids[nb_match:]
                match_cell(idx, sorted_idx, pool_size)
        loop_wh_progress(match_cell, sorted_set1_indices, pool_size)
        return result


functions = {
    'matching': SequentialMatching,
    'rank_matching': RankMatching,
}
