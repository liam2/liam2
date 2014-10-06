from __future__ import print_function

import random

import numpy as np


#noinspection PyNoneFunctionAssignment
def align_link_nd(scores, need, num_candidates, hh, fcols_labels,
                  secondary_axis=None):
    # need and num_candidates are LabeledArray, but we don't need the extra
    # functionality from this point on
    need = np.asarray(need)
    num_candidates = np.asarray(num_candidates)
    print("total needed", need.sum())

    still_needed = need.copy()
    still_available = num_candidates.copy()

    rel_need = still_needed.astype(np.float64) / still_available

    unfillable_bins = still_needed > still_available
    overfilled_bins = still_needed <= 0

    if secondary_axis is not None:
        assert secondary_axis < need.ndim
        other_axes = list(range(need.ndim))
        other_axes.pop(secondary_axis)
        other_axes = tuple(other_axes)
        # requires np 1.7+
        still_needed_by_sec_axis = need.sum(axis=other_axes)
        print("needed by secondary axis", still_needed_by_sec_axis)
    else:
        still_needed_by_sec_axis = None

    still_needed_total = need.sum()

    col_range = list(range(len(fcols_labels)))
    aligned = np.zeros(len(hh), dtype=bool)
    sorted_indices = scores.argsort()[::-1]
    for sorted_idx in sorted_indices:
        if still_needed_total <= 0:
            print("total reached")
            break
        persons_in_hh_indices = hh[sorted_idx]
        num_persons_in_hh = len(persons_in_hh_indices)

        # this will usually happen when the household is not a candidate
        # and thus no person in the household is a candidate either
        if num_persons_in_hh == 0:
            continue

        persons_in_hh = tuple(np.empty(num_persons_in_hh, dtype=int)
                              for _ in col_range)
        prange = list(range(num_persons_in_hh))
        for hh_col, fcol_labels in zip(persons_in_hh, fcols_labels):
            for i in prange:
                hh_col[i] = fcol_labels[persons_in_hh_indices[i]]

        # Keep the highest relative need index for the family
        hh_rel_need = np.nanmax(rel_need[persons_in_hh])

        # count number of objects in the family belonging to over-filled bins
        surplus = overfilled_bins[persons_in_hh].sum()
        if secondary_axis is not None and surplus == 0:
            hh_axis_values = persons_in_hh[secondary_axis]
            axis_num_pvalues = len(still_needed_by_sec_axis)
            hh_counts_by_sec_axis = np.bincount(hh_axis_values,
                                                minlength=axis_num_pvalues)
            if np.any(hh_counts_by_sec_axis >= still_needed_by_sec_axis):
                surplus = 1

        # count number of objects in the family belonging to unfillable bins
        num_unfillable = unfillable_bins[persons_in_hh].sum()

        # if either surplus or unfillable are not zero, adjust rel_need:
        if (surplus != 0) or (num_unfillable != 0):
            if num_unfillable > surplus:
                hh_rel_need = 1.0
            elif num_unfillable == surplus:
                hh_rel_need = 0.5
            else:  # num_unfillable < surplus
                hh_rel_need = 0.0

        # Run through the random selection process, using rel_need as the
        # probability
        if random.random() < hh_rel_need:
            aligned[sorted_idx] = True

            # update all counters
            still_needed_total -= num_persons_in_hh

            # update grids (only the bins present in the family)

            # Note that we have to loop explicitly on individuals, instead
            # of using xxx[persons_in_hh] += 1 because that syntax does not
            # work as expected when there are more than one family member
            # in a bin (it does not increment the bin several times)
            for values in zip(*persons_in_hh):
                sn = still_needed[values] - 1
                still_needed[values] = sn

                sa = still_available[values] - 1
                still_available[values] = sa

                if secondary_axis is not None:
                    still_needed_by_sec_axis[values[secondary_axis]] -= 1

                # unfillable stays unchanged in this case
                overfilled_bins[values] = sn <= 0

                # using np.float64 to workaround an issue with numpy 1.8
                # https://github.com/numpy/numpy/issues/4636
                rel_need[values] = np.float64(sn) / sa
        else:
            for values in zip(*persons_in_hh):
                sa = still_available[values] - 1
                still_available[values] = sa
                sn = still_needed[values]

                unfillable_bins[values] = sn > sa

                rel_need[values] = np.float64(sn) / sa
    print("missing %d individuals" % np.sum(still_needed))
    return aligned, still_needed
