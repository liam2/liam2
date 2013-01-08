import random

import numpy as np

from expr import expr_eval, collect_variables, traverse_expr, Variable, \
                 missing_values, get_missing_value

from alignment import groupby
from context import context_length, context_subset, context_delete
from context import EntityContext
from properties import EvaluableExpression, GroupCount
from registry import entity_registry
from utils import loop_wh_progress


class AlignOther(EvaluableExpression):
    def __init__(self, link, target_expressions, target_filter, filter, need,
                 orderby):
        """
        filter is a local filter (eg filter on hh, eg is_candidate)
        target_filter, is the filter for targets,
        eg hh.is_candidate
        XXX: we might want to compute target_filter automatically as I am
        unsure if it makes sense to use a filter different from
        [reverse_link].local_filter
        """

        self.link = link
        self.target_expressions = target_expressions
        self.target_filter_expr = target_filter
        self.filter_expr = filter
        self.need_expr = need
        self.orderby_expr = orderby

    def traverse(self, context):
        for expr in self.target_expressions:
            for node in traverse_expr(expr, context):
                yield node
        for node in traverse_expr(self.filter_expr, context):
            yield node
        for node in traverse_expr(self.filter_expr, context):
            yield node
        for node in traverse_expr(self.need_expr, context):
            yield node
        for node in traverse_expr(self.orderby_expr, context):
            yield node
        yield self

    def collect_variables(self, context):
        expr_vars = collect_variables(self.filter_expr, context)
        expr_vars |= collect_variables(self.need_expr, context)
        expr_vars |= collect_variables(self.orderby_expr, context)
        return expr_vars

    #TODO: somehow merge this with LinkExpression or move these two functions
    # to the Link class
    def target_entity(self, context):
        return entity_registry[self.link._target_entity]

    def target_context(self, context):
        target_entity = self.target_entity(context)
        return EntityContext(target_entity,
                             {'period': context['period'],
                             '__globals__': context['__globals__']})

    def evaluate(self, context):
        need = expr_eval(self.need_expr, context)
        orderby = expr_eval(self.orderby_expr, context)

        target_context = self.target_context(context)
        target_columns = [expr_eval(e, target_context)
                          for e in self.target_expressions]
        # this is a one2many, so the link column is on the target side
        link = self.link
        link_column = expr_eval(Variable(link._link_field), target_context)
        if self.target_filter_expr is not None:
            target_filter_value = expr_eval(self.target_filter_expr,
                                            target_context)
            filtered_columns = [col[target_filter_value]
                                  if isinstance(col, np.ndarray) and col.shape
                                  else [col]
                                for col in target_columns]
            link_column = link_column[target_filter_value]
        else:
            filtered_columns = target_columns

        id_to_rownum = context.id_to_rownum
        if self.filter_expr is not None:
            filter_value = expr_eval(self.filter_expr, context)
            filtered_context = context_subset(context, filter_value)
            #FIXME: move this somewhere else
            id_to_rownum = filtered_context.id_to_rownum
            orderby = orderby[filter_value]
        else:
            filtered_context = context


        missing_int = missing_values[int]

        #TODO: implement "source" filter (we don't want any target filter, like
        # we do in "normal" AggregateLink functions but we do want to filter
        # the source (ie only take source_ids which point to an id in the
        # filtered subset)
        source_ids = link_column

        if len(id_to_rownum):
            source_rows = id_to_rownum[source_ids]
            # filter out missing values: those where the value of the link
            # points to nowhere (-1)
            #XXX: use np.putmask(source_rows, source_ids == missing_int,
            #                    missing_int)
            source_rows[source_ids == missing_int] = missing_int
        else:
            raise Exception("blarf")

        # fetch the list of values for linked individuals of each local
        # individual. e.g. (gender, age) for each person in each household
        # we store it as a distinct list for each column, eg:
        # hh = [([15, 26, 12], [True, False, True]), ([23], [True])]
        hh = np.empty(context_length(filtered_context), dtype=object)
        # we can use .fill([]) because it reuses the same list for all hh
        numcol = len(filtered_columns)
        col_range = range(numcol)
        for i in range(len(hh)):
            hh[i] = tuple([] for _ in col_range)

        #XXX: we might want to do this in two passes, like in
        # groupby._group_labels (in the first pass, only count the number of
        # different values) so that we can create arrays directly
        #XXX: we might want to use group_labels directly
        hh_rows = id_to_rownum[link_column]
        for person_row, hh_row in enumerate(hh_rows):
            #TODO: we need int indices, not bool
            #TODO: convert all those lists to ndarrays it is 10x faster
            # In [82]: i2
            # Out[82]: ([0, 2, 4, 1, 2], [0, 1, 1, 0, 0])

            # In [83]: i2a2
            # Out[83]: (array([0, 2, 4, 1, 2]), array([0, 1, 1, 0, 0]))

            # In [84]: timeit a[i2]
            # 10000 loops, best of 3: 19.9 us per loop

            # In [85]: timeit a[i2a2]
            # 100000 loops, best of 3: 2.12 us per loop

            for target_list, source_col in zip(hh[hh_row], filtered_columns):
                target_list.append(source_col[person_row])

        #FIXME: pvalues should come from need "indexed columns"/possible values
        pvalues = [range(n) for n in need.shape]
        #FIXME: target_context should be filtered too
        num_candidates = groupby(filtered_columns, GroupCount(),
                                 target_context, pvalues)

#        print "candidates"
#        print num_candidates
        #TODO: account for fractional persons
        #frac_taken = frac_need > 0.5
        need = need.astype(int)
        still_needed = need.copy()

        still_available = num_candidates.copy()

        rel_need = still_needed.astype(float) / still_available

        unfillable_bins = still_needed > still_available
        overfilled_bins = still_needed <= 0

        #FIXME: add an argument for this or compute it automatically
        age_axis = 1
        still_needed_by_sex = need.sum(axis=age_axis)
        print "needed by sex", still_needed_by_sex
        still_needed_total = need.sum()
        print "total needed", still_needed_total

        sorted_indices = orderby.argsort()[::-1]

        aligned_indices = []
        for sorted_idx in sorted_indices:
            if still_needed_total <= 0:
                print "total reached"
                break
            persons_in_hh = hh[sorted_idx]
            if not persons_in_hh:
                continue

            # Keep the highest relative need index for the family
            hh_rel_need = np.nanmax(rel_need[persons_in_hh])
            num_excedent = overfilled_bins[persons_in_hh].sum()
            num_unfillable = unfillable_bins[persons_in_hh].sum()

#            print "hh: relneed %f, exc %d / unf %d" % (hh_rel_need,
#                                                       num_excedent,
#                                                       num_unfillable),
            # if either excedent or unfillable are not zero, adjust rel_need:
            if (num_excedent != 0) or (num_unfillable != 0):
                if num_unfillable > num_excedent:
                    hh_rel_need = 1.0
                elif num_unfillable == num_excedent:
                    hh_rel_need = 0.5
                else:  # num_unfillable < num_excedent
                    hh_rel_need = 0.0

            # Run through the random selection process, using rel_need as the
            # probability
            if random.random() < hh_rel_need:
                aligned_indices.append(sorted_idx)

                # update all counters
                still_needed_total -= len(persons_in_hh)

                # update grids (only the age/gender present in the family)

                # Note that we have to loop explicitly on individuals, instead
                # of using xxx[persons_in_hh] += 1 because that syntax does not
                # work as expected when there are more than one family member
                # in a bin (it does not increment the bin several times)
                for values in zip(*persons_in_hh):
                    sn = still_needed[values] - 1
                    still_needed[values] = sn

                    sa = still_available[values] - 1
                    still_available[values] = sa

                    #FIXME: we assume sex is the first dimension
                    snbs = still_needed_by_sex[values[0]] - 1
                    still_needed_by_sex[values[0]] = snbs

                    # unfillable stays unchanged in this case
                    overfilled_bins[values] = sn <= 0 or snbs <= 0

                    rel_need[values] = float(sn) / sa
            else:
                for values in zip(*persons_in_hh):
                    sa = still_available[values] - 1
                    still_available[values] = sa

                    sn = still_needed[values]

                    unfillable_bins[values] = sn > sa

                    rel_need[values] = float(sn) / sa

        return {'values': True, 'indices': aligned_indices}

    def dtype(self, context):
        return bool

functions = {
    'align_other': AlignOther
}
