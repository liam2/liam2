from __future__ import print_function

import random

import numpy as np


def align_link_nd(scores, need, num_candidates, hh, fcols_labels):
    # need and num_candidates are LabeledArray, but we don't need the extra
    # functionality from this point on
    need = np.asarray(need)
    num_candidates = np.asarray(num_candidates)
    print("total needed", need.sum())

    still_needed = need.copy()
    still_available = num_candidates.copy()

    rel_need = still_needed.astype(float) / still_available

    unfillable_bins = still_needed > still_available
    overfilled_bins = still_needed <= 0

    #FIXME: add an argument to specify which column(s) to sum on
    age_axis = 1
    still_needed_by_sex = need.sum(axis=age_axis)
    print("needed by sex", still_needed_by_sex)
    still_needed_total = need.sum()

    col_range = range(len(fcols_labels))
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
        prange = range(num_persons_in_hh)
        for hh_col, fcol_labels in zip(persons_in_hh, fcols_labels):
            for i in prange:
                hh_col[i] = fcol_labels[persons_in_hh_indices[i]]

        # Keep the highest relative need index for the family
        hh_rel_need = np.nanmax(rel_need[persons_in_hh])
        num_excedent = overfilled_bins[persons_in_hh].sum()
        if num_excedent == 0:
            #FIXME: we assume sex is the first dimension
            gender = persons_in_hh[0]
            sex_counts = np.bincount(gender, minlength=2)
            if np.any(sex_counts >= still_needed_by_sex):
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
            aligned[sorted_idx] = True

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
    print("missing %d individuals" % np.sum(still_needed))
    return aligned, still_needed
