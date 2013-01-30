import random

import numpy as np

from expr import (expr_eval, collect_variables, traverse_expr, Variable,
                  missing_values)
from links import Link, LinkValue
from groupby import groupby
from context import context_length, EntityContext
from properties import EvaluableExpression
from registry import entity_registry


class AlignOther(EvaluableExpression):
    #TODO: allow specifying possible_values manually, like in align()
    def __init__(self, link, need, orderby, filter=None, expressions=None):
        """
        filter is a local filter (eg filter on hh, eg is_candidate)
        """

        self.link = link
        self.target_expressions = expressions
        self.filter_expr = filter
        self.need_expr = need
        #XXX: make orderby optional? (defaults to 'id'? -- not sure it makes
        # sense)
        self.orderby_expr = orderby
        self.last_error = None

    def traverse(self, context):
        if self.target_expressions is not None:
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

    #TODO: somehow merge these two functions with LinkExpression or move them
    # to the Link class
    def target_entity(self, context):
        return entity_registry[self.link._target_entity]

    def target_context(self, context):
        target_entity = self.target_entity(context)
        return EntityContext(target_entity,
                             {'period': context['period'],
                             '__globals__': context['__globals__']})

    def evaluate(self, context):
        #TODO: remove the target_expressions argument, as dim_names should
        # always match the expressions, so we could just as well use dim_names
        # *as* the expressions, like we do in normal alignment
        # As a related note, I don't like to have expressions in the datafiles,
        # but we want to allow expressions anyway, so we will need an optional
        # expr argument. And this seems like a better solution for normal
        # alignment too!

        need = expr_eval(self.need_expr, context)
        if self.target_expressions is None:
            self.target_expressions = [Variable(name)
                                       for name in need.dim_names]

        orderby = expr_eval(self.orderby_expr, context)

        target_context = self.target_context(context)
        target_columns = [expr_eval(e, target_context)
                          for e in self.target_expressions]
        # this is a one2many, so the link column is on the target side
        link_column = expr_eval(Variable(self.link._link_field),
                                target_context)

        if self.filter_expr is not None:
            reverse_link = Link("reverse", "many2one", self.link._link_field,
                                context['__entity__'].name)
            target_filter = LinkValue(reverse_link, self.filter_expr, False)
            target_filter_value = expr_eval(target_filter, target_context)
            filtered_columns = [col[target_filter_value]
                                  if isinstance(col, np.ndarray) and col.shape
                                  else [col]
                                for col in target_columns]
            link_column = link_column[target_filter_value]
        else:
            filtered_columns = target_columns

        numcol = len(target_columns)
        col_range = range(numcol)

        id_to_rownum = context.id_to_rownum
        missing_int = missing_values[int]
        source_ids = link_column

        if len(id_to_rownum):
            source_rows = id_to_rownum[source_ids]
            # filter out missing values: those where the value of the link
            # points to nowhere (-1)
            #XXX: use np.putmask(source_rows, source_ids == missing_int,
            #                    missing_int)
            source_rows[source_ids == missing_int] = missing_int
        else:
            assert np.all(source_ids == missing_int)
            source_rows = []

        # fetch the list of linked individuals for each local individual.
        # e.g. the list of person ids for each household
        hh = np.empty(context_length(context), dtype=object)
        # we can't use .fill([]) because it reuses the same list for all hh
        for i in range(len(hh)):
            hh[i] = []

        # Even though this is highly sub-optimal, the time taken to create
        # those lists of ids is very small compared to the total time taken
        # for align_other (0.2s vs 4.26), so I shouldn't care too much about
        # it for now
        #XXX: we might want to do this in two passes, like in
        # groupby._group_labels (in the first pass, only count the number of
        # different values) so that we can create arrays directly
        #XXX: we might want to use group_labels directly
        #XXX: we could use partition_nd on link_column.
        # It would be a bit overkill because in this case we know the
        # possible values in advance but, in practice, creating a version
        # of partition_nd with known labels would only save one "if"
        # per object and compared to the hash lookup, it is probably a
        # very small gain. However, a version of partition_nd with known
        # labels *as* a ndarray (ala id_to_rownum) might be worth it.
        for target_row, source_row in enumerate(source_rows):
            if source_row == -1:
                continue
            hh[source_row].append(target_row)

        #XXX: what if need does not specify values present in f_columns?
        num_candidates = groupby(filtered_columns, need.pvalues)

        # handle the "fractional people problem"
        int_need = need.astype(int)
        frac_need = need - int_need
        need = int_need

        # the sum of fractional objects number of extra objects we want aligned
        extra_wanted = int(round(np.sum(frac_need)))
        # search cutoff that yield np.sum(frac_need >= cutoff) == extra_wanted
        sorted_frac_need = frac_need.flatten()
        sorted_frac_need.sort()
        cutoff = sorted_frac_need[-extra_wanted]
        extra = frac_need >= cutoff
        if np.sum(extra) > extra_wanted:
            # This case can only happen when several bins have the same
            # frac_need. In this case we could try to be even closer to our
            # target by randomly selecting X out of the Y bins which have a
            # frac_need equal to the cutoff.
            assert np.sum(frac_need == cutoff) > 1
        need += extra

        # another, much simpler, option to handle fractional people is to
        # always use 0.5 as a cutoff point:
#        need = (need + 0.5).astype(int)

        # a third option is to use random numbers:
#        int_need = need.astype(int)
#        frac_need = need - int_need
#        need = int_need + (np.random.rand(need.shape) < frac_need)
        print "need", np.sum(need)

        if self.last_error is not None:
            print "adding %d individuals from last period error" \
                  % np.sum(self.last_error)
            need += self.last_error

        still_needed = need.copy()
        still_available = num_candidates.copy()

        rel_need = still_needed.astype(float) / still_available

        unfillable_bins = still_needed > still_available
        overfilled_bins = still_needed <= 0

        #FIXME: add an argument to specify which column(s) to sum on
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
            persons_in_hh_indices = hh[sorted_idx]
            num_persons_in_hh = len(persons_in_hh_indices)
            if num_persons_in_hh == 0:
                continue

            #FIXME: we need to store the indices of the values, not the values
            # themselves. These indices should be consistent with the "need"
            # and "num_candidates" matrices (and all their derived matrices:
            # rel_need, unfillable_bins, ...). In our current example, this is
            # luckily the same thing because age starts at 0 and has no gaps
            # in values
            persons_in_hh = tuple(np.empty(num_persons_in_hh, dtype=int)
                                  for _ in col_range)
            prange = range(num_persons_in_hh)
            for hh_col, fcol in zip(persons_in_hh, filtered_columns):
                for i in prange:
                    hh_col[i] = fcol[persons_in_hh_indices[i]]

            # Keep the highest relative need index for the family
            hh_rel_need = np.nanmax(rel_need[persons_in_hh])
            num_excedent = overfilled_bins[persons_in_hh].sum()
            if num_excedent == 0:
                #FIXME: we assume sex is the first dimension
                gender = persons_in_hh[0]
                sex_counts = np.bincount(gender, minlength=2)
                if np.any(still_needed_by_sex - sex_counts <= 0):
                    num_excedent = 1

            num_unfillable = unfillable_bins[persons_in_hh].sum()

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
                still_needed_total -= num_persons_in_hh

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
                    overfilled_bins[values] = sn <= 0

                    rel_need[values] = float(sn) / sa
            else:
                for values in zip(*persons_in_hh):
                    sa = still_available[values] - 1
                    still_available[values] = sa
                    sn = still_needed[values]

                    unfillable_bins[values] = sn > sa

                    rel_need[values] = float(sn) / sa
        print "missing %d persons" % np.sum(still_needed)
        self.last_error = still_needed
        return {'values': True, 'indices': aligned_indices}

    def dtype(self, context):
        return bool

functions = {
    'align_other': AlignOther
}
