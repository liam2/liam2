# encoding: utf-8
from __future__ import absolute_import, division, print_function

import random

import numpy as np
import larray as la


# noinspection PyNoneFunctionAssignment
def align_link_nd(scores, need, num_candidates, hh, fcols_labels,
                  secondary_axis=None):
    # need, num_candidates and fcols_labels are LArray, but we don't need the extra
    # functionality from this point on
    need = np.asarray(need)
    num_candidates = np.asarray(num_candidates)
    fcols_labels = [np.asarray(fcol_labels) for fcol_labels in fcols_labels]
    print("total needed", need.sum())

    still_needed = need.copy()
    still_available = num_candidates.copy()

    rel_need = still_needed.astype(np.float64) / still_available

    unfillable_bins = still_needed > still_available
    filled_bins = still_needed <= 0

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

        persons_in_hh = tuple(fcol_labels[persons_in_hh_indices] for fcol_labels in fcols_labels)

        # Keep the highest relative need index for the family
        hh_rel_need = np.nanmax(rel_need[persons_in_hh])

        # count number of objects in the family belonging to already filled bins
        surplus = filled_bins[persons_in_hh].sum()
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
                filled_bins[values] = sn <= 0

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


# To get a baseline when trying to find an alternative algorithm, I devised this small snippet, which
# find the optimal solution. It runs in O(2^N) which is impractical for real usage, but can serve as a
# comparison for the real algo with very small datasets.
'''
from itertools import chain, combinations

# from python itertools documentation
def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    >>> len(list(powerset(range(10))))
    1024
    >>> len(list(powerset(range(16))))
    65536
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def dist(arr, need):
    return abs(need - arr.sum(0)).sum()


def best_dist(arr, need):
    """this works in O(2^n) so do NOT run this on an array > ~16 elements"""
    if len(arr) > 16:
        raise ValueError("this works in O(2^n) so do NOT run this on an array > ~16 elements")

    indices = range(len(arr))
    combinations = list(powerset(indices))
    idx, dist_ = min(enumerate(dist(arr[list(comb)], need) for comb in combinations),
                     key=lambda kv: kv[1])
    return dist_, combinations[idx]


arr = LArray([[2, 5, 3],
              [1, 0, 0],
              [0, 2, 0],
              [0, 0, 5],
              [1, 1, 1],
              [5, 0, 2],
              [3, 2, 1],
              [1, 2, 3],
              [5, 3, 3]]).rename(0, 'hh').set_axes(1, 'age=0,1,2')
need = LArray([10, 9, 9], arr.age)
best_dist(arr.data, need.data)
Out[18]: (1, (0, 5, 6, 7))
'''
