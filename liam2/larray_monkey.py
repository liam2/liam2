# encoding: utf-8
from __future__ import absolute_import, division, print_function

import numpy as np
import larray as la
from larray.core.expr import ExprNode
from larray.core.array import make_numpy_broadcastable
from larray.core.group import _range_to_slice
from larray.util.misc import _isnoneslice

from liam2.compat import long


# This monkey-patches larray to allow targeting a boolean axis of an array using a boolean array because this is
# currently a very common use case in LIAM2: when we have a global array with a True/False gender axis, and we want
# to get the value of that array for all individuals depending on their gender.

# This is ugly but I consider this is the lesser evil. If we did not do this, it would require users to systematically
# specify the axis explicitly: global_array[X.gender[gender]] instead of global_array[gender]
# See https://github.com/larray-project/larray/issues/794#issuecomment-547974634 for details.

# Note that the only other practical option at this point was to make a local copy of the whole larray library,
# but in that case it is a lot less clear what is actually changed.

# The cleanest option would have been to define our own AxisCollection variant inheriting from la.AxisCollection but
# this would not work in the current version of larray as any operation on a LIAM2AxisCollection would return an
# AxisCollection and not its LIAM2 subclass. Also, the la.Array class uses AxisCollection explicitly and there is no
# mechanism to customize that (and I don't think it is worth the trouble at this point to implement such a mechanism),
# so we would have to basically re-implement the whole library

# Now the real, future-proof solution is to implement support for categoricals, so that we can have expressions like:
# where(gender == 'male', age * 5.1, age * 5.2)
# we are not there yet though.

# For some cases, it would be even better to encourage our users to avoid using constants in their code but rather
# use many (possibly autoindexed) array constants. I am unsure it would solve all cases of where(gender, ...) though.
def _key_to_igroups(self, key):
    """
    Translates any key to an IGroups tuple.

    Parameters
    ----------
    key : scalar, list/array of scalars, Group or tuple or dict of them
        any key supported by Array.__get|setitem__

    Returns
    -------
    tuple
        tuple of IGroup, each IGroup having a real axis from this array.
        The order of the IGroups is *not* guaranteed to be the same as the order of axes.

    See Also
    --------
    Axis.index
    """
    if isinstance(key, dict):
        # key axes could be strings or axis references and we want real axes
        key = tuple(self[axis][axis_key] for axis, axis_key in key.items())
    elif not isinstance(key, tuple):
        # convert scalar keys to 1D keys
        key = (key,)

    # handle ExprNode
    key = tuple(axis_key.evaluate(self) if isinstance(axis_key, ExprNode) else axis_key
                for axis_key in key)

    nonboolkey = []
    for axis_key in key:
        if isinstance(axis_key, np.ndarray) and np.issubdtype(axis_key.dtype, np.bool_):
            if axis_key.shape != self.shape:
                raise ValueError("boolean key with a different shape ({}) than array ({})"
                                 .format(axis_key.shape, self.shape))
            axis_key = la.Array(axis_key, self)

        if isinstance(axis_key, la.Array) and np.issubdtype(axis_key.dtype, np.bool_):
            bool_axes_names = [axis.name for axis in self if np.issubdtype(axis.dtype, np.bool_)]
            if bool_axes_names:
                # a "filter" key has always somme axes in common with the array (it should be a subset of the array
                # axes), so if there is no common axis, it is not a filter key.

                # TOCHECK: we might want to check for extra_key_axes too?
                common_axes = axis_key.axes & self
                could_be_a_filter = len(common_axes) >= 1
                if could_be_a_filter:
                    raise ValueError("boolean subset key ({}) is ambiguous because it can be interpreted "
                                     "either as a filter on the array or as a key on a boolean axis ({})"
                                     .format(axis_key, ', '.join(bool_axes_names)))
                nonboolkey.append(axis_key)
            else:
                extra_key_axes = axis_key.axes - self
                if extra_key_axes:
                    raise ValueError("boolean subset key contains more axes ({}) than array ({})"
                                     .format(axis_key.axes, self))

                # nonzero (currently) returns a tuple of IGroups containing 1D Arrays (one IGroup per axis)
                nonboolkey.extend(axis_key.nonzero())
        else:
            nonboolkey.append(axis_key)
    key = tuple(nonboolkey)

    # drop slice(None) and Ellipsis since they are meaningless because of guess_axis.
    # XXX: we might want to raise an exception when we find Ellipses or (most) slice(None) because except for
    #      a single slice(None) a[:], I don't think there is any point.
    key = [axis_key for axis_key in key
           if not _isnoneslice(axis_key) and axis_key is not Ellipsis]

    # translate all keys to IGroup
    return tuple(self._translate_axis_key(axis_key) for axis_key in key)


la.AxisCollection._key_to_igroups = _key_to_igroups


# Monkey patch to fix support for "long" required for Python 2, especially on Linux.
# Changed lines are marked with "MONKEY PATCH CHANGED LINE"
def _key_to_raw_and_axes(self, key, collapse_slices=False, translate_key=True):
    r"""
    Transforms any key (from Array.__getitem__) to a raw numpy key, the resulting axes, and potentially a tuple
    of indices to transpose axes back to where they were.

    Parameters
    ----------
    key : scalar, list/array of scalars, Group or tuple or dict of them
        any key supported by Array.__getitem__
    collapse_slices : bool, optional
        Whether or not to convert ranges to slices. Defaults to False.

    Returns
    -------
    raw_key: tuple
    res_axes: AxisCollection
    transposed_indices: tuple or None
    """

    if translate_key:
        key = self._translated_key(key)
    assert isinstance(key, tuple) and len(key) == self.ndim

    # scalar array
    if not self.ndim:
        return key, la.AxisCollection([]), None

    # transform ranges to slices if needed
    if collapse_slices:
        # isinstance(np.ndarray, collections.Sequence) is False but it behaves like one
        seq_types = (tuple, list, np.ndarray)
        # TODO: we should only do this if there are no Array key (with axes corresponding to the range)
        # otherwise we will be translating them back to a range afterwards
        key = [_range_to_slice(axis_key, len(axis)) if isinstance(axis_key, seq_types) else axis_key
               for axis_key, axis in zip(key, self)]

    # transform non-Array advanced keys (list and ndarray) to Array
    def to_la_ikey(axis, axis_key):
        # MONKEY PATCH CHANGED LINE
        if isinstance(axis_key, (int, long, np.integer, slice, la.Array)):
            return axis_key
        else:
            # MONKEY PATCH CHANGED LINE
            assert isinstance(axis_key, (list, np.ndarray)), \
                "unsupported key type: {} ({})".format(axis_key, type(axis_key))
            res_axis = axis.subaxis(axis_key)
            # TODO: for perf reasons, we should bypass creating an actual Array by returning axes and key_data
            # but then we will need to implement a function similar to make_numpy_broadcastable which works on axes
            # and rawdata instead of arrays
            return la.Array(axis_key, res_axis)

    key = tuple(to_la_ikey(axis, axis_key) for axis, axis_key in zip(self, key))

    # transform slice keys to Array too IF they refer to axes present in advanced key (so that those axes
    # broadcast together instead of being duplicated, which is not what we want)
    def get_axes(value):
        return value.axes if isinstance(value, la.Array) else la.AxisCollection([])

    def slice_to_sequence(axis, axis_key):
        if isinstance(axis_key, slice) and axis in la_key_axes:
            # TODO: sequence assumes the axis in the la_key is in the same order. It will be easier to solve when
            # make_numpy_broadcastable automatically aligns all arrays
            start, stop, step = axis_key.indices(len(axis))
            return la.sequence(axis.subaxis(axis_key), initial=start, inc=step)
        else:
            return axis_key

    # XXX: can we avoid computing this twice? (here and in make_numpy_broadcastable)
    la_key_axes = la.AxisCollection.union(*[get_axes(k) for k in key])
    key = tuple(slice_to_sequence(axis, axis_key) for axis, axis_key in zip(self, key))

    # start with the simple (slice) keys
    # scalar keys are ignored since they do not produce any resulting axis
    res_axes = la.AxisCollection([axis.subaxis(axis_key)
                                  for axis, axis_key in zip(self, key)
                                  if isinstance(axis_key, slice)])
    transpose_indices = None

    # if there are only simple keys, do not bother going via the "advanced indexing" code path
    # MONKEY PATCH CHANGED LINE
    if all(isinstance(axis_key, (int, long, np.integer, slice)) for axis_key in key):
        bcasted_adv_keys = key
    else:
        # Now that we know advanced indexing comes into play, we need to compute were the subspace created by the
        # advanced indexes will be inserted. Note that there is only ever a SINGLE combined subspace (even if it
        # has multiple axes) because all the non slice indexers MUST broadcast together to a single
        # "advanced indexer"

        # to determine where the "subspace" axes will be inserted, a scalar key counts as "advanced" indexing
        adv_axes_indices = [i for i, axis_key in enumerate(key)
                            if not isinstance(axis_key, slice)]
        diff = np.diff(adv_axes_indices)
        if np.any(diff > 1):
            # insert advanced indexing subspace in front
            adv_key_subspace_pos = 0

            # If all (non scalar) adv_keys are 1D and have a different axis name, we will index the cross product.
            # In that case, store their original order so that we can transpose them back to where they were.
            # MONKEY PATCH CHANGED LINE
            adv_keys = [axis_key for axis_key in key if not isinstance(axis_key, (int, long, np.integer, slice))]
            if all(axis_key.ndim == 1 for axis_key in adv_keys):
                # we can only handle the non-anonymous axes case since anonymous axes will not broadcast to the
                # cross product anyway
                if len(set(axis_key.axes[0].name for axis_key in adv_keys)) == len(adv_keys):
                    # 0, 1, 2, 3, 4, 5 <- original axes indices
                    # A  X  A  S  S  A <- key (A = adv, X = scalar/remove, S = slice)
                    # 0, 2, 5, 3, 4    <- result
                    # 0, 2, 3, 4, 5    <- desired result
                    # 0, 1, 3, 4, 2    <- what I need to feed to transpose to get the correct result
                    adv_axes_indices = [i for i, axis_key in enumerate(key)
                                        # MONKEY PATCH CHANGED LINE
                                        if not isinstance(axis_key, (int, long, np.integer, slice))]
                    # not taking scalar axes since they will disappear
                    slice_axes_indices = [i for i, axis_key in enumerate(key)
                                          if isinstance(axis_key, slice)]
                    result_axes_indices = adv_axes_indices + slice_axes_indices
                    transpose_indices = tuple(np.array(result_axes_indices).argsort())
        else:
            # the advanced indexing subspace keep its position (insert at position of first concerned axis)
            adv_key_subspace_pos = adv_axes_indices[0]

        # scalar/slice keys are ignored by make_numpy_broadcastable, which is exactly what we need
        bcasted_adv_keys, adv_key_dest_axes = make_numpy_broadcastable(key)

        # insert advanced indexing subspace
        res_axes[adv_key_subspace_pos:adv_key_subspace_pos] = adv_key_dest_axes

    # transform to raw numpy arrays
    raw_broadcasted_key = tuple(k.data if isinstance(k, la.Array) else k
                                for k in bcasted_adv_keys)
    return raw_broadcasted_key, res_axes, transpose_indices


la.AxisCollection._key_to_raw_and_axes = _key_to_raw_and_axes
