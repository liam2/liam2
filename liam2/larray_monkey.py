# encoding: utf-8
from __future__ import absolute_import, division, print_function

import numpy as np
import larray as la
from larray.core.expr import ExprNode
from larray.util.misc import _isnoneslice


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
