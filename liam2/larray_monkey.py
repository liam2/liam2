import numpy as np
import larray as la
from larray.core.expr import ExprNode

from larray.util.misc import (duplicates, _isnoneslice, has_duplicates)


# The code below is a modified copy of the AxisCollection._key_to_axis_indices_dict method from larray 0.34.6.
# https://github.com/larray-project/larray/blob/d01381da6711988e3117fd208986b6bb6ca3e700/larray/core/axis.py#L2860

# The added lines (marked by "MONKEY PATCH START/END OF ADDED LINES") are necessary to allow targeting a boolean axis
# of an array using a boolean array (instead of considering that array as a filter) because this is
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
def _key_to_axis_indices_dict(self, key):
    """
    Translate any label-based key to an {axis: indices} dict.

    Parameters
    ----------
    key : scalar, list/array of scalars, Group or tuple or dict of them
        any key supported by Array.__get|setitem__

    Returns
    -------
    dict
        dict {axis: indices}, with axis from this array.
        The order of the dict axes/keys is *not* guaranteed to be the same as the order of axes.

    See Also
    --------
    Axis.index
    """
    from larray import Array

    if isinstance(key, dict):
        # key axes could be strings or axis references and we want real axes
        key = tuple(self[axis][axis_key] for axis, axis_key in key.items())
    elif not isinstance(key, tuple):
        # convert scalar keys to 1D keys
        key = (key,)

    filtered_key = []
    for axis_key in key:
        # handle ExprNode
        if isinstance(axis_key, ExprNode):
            axis_key = axis_key.evaluate(self)

        # handle boolean filter keys
        if isinstance(axis_key, np.ndarray) and np.issubdtype(axis_key.dtype, np.bool_):
            if axis_key.shape != self.shape:
                raise ValueError(f"boolean key with a different shape ({axis_key.shape}) than array ({self.shape})")
            filtered_key.extend(Array(axis_key, self).nonzero())
        elif isinstance(axis_key, Array) and np.issubdtype(axis_key.dtype, np.bool_):
            # MONKEY PATCH START OF ADDED LINES
            bool_axes_names = [axis.name for axis in self if np.issubdtype(axis.dtype, np.bool_)]
            if bool_axes_names:
                # a "filter" key has always somme axes in common with the array (it should be a subset of the array
                # axes), so if there is no common axis, it is not a filter key.

                # TOCHECK: we might want to check for extra_key_axes too?
                common_axes = axis_key.axes & self
                could_be_a_filter = len(common_axes) >= 1
                if could_be_a_filter:
                    raise ValueError(f"boolean subset key ({axis_key}) is ambiguous because it can be interpreted "
                                     f"either as a filter on the array or as a key on a boolean axis "
                                     f"({', '.join(bool_axes_names)})")
                filtered_key.append(axis_key)
            else:
                # MONKEY PATCH END OF ADDED LINES
                extra_key_axes = axis_key.axes - self
                if extra_key_axes:
                    raise ValueError(f"boolean subset key contains more axes ({axis_key.axes}) than array ({self})")

                # TODO: factorize with check_compatible
                for i, subset_axis in enumerate(axis_key.axes):
                    array_axis = self.get_by_pos(subset_axis, i)
                    if not array_axis.iscompatible(subset_axis):
                        msg = f"""boolean subset array has incompatible axes with array:
    array axes: {self}
    subset array axes: {axis_key.axes}
    incompatible axes:
    array axis:
        {array_axis!r}
    subset array axis:
        {subset_axis!r}
    """
                        raise ValueError(msg)

                # nonzero (currently) returns a tuple of IGroups containing 1D Arrays (one IGroup per axis)
                filtered_key.extend(axis_key.nonzero())
        # drop slice(None) and Ellipsis since they are meaningless because of guess_axis.
        # XXX: we might want to raise an exception when we find Ellipses or (most) slice(None) because except for
        #      a single slice(None) a[:], I don't think there is any point.
        elif not _isnoneslice(axis_key) and axis_key is not Ellipsis:
            filtered_key.append(axis_key)

    key = tuple(filtered_key)

    # translate all keys to (axis, indices) pairs
    key_items = tuple(self._translate_axis_key(axis_key) for axis_key in filtered_key)

    assert all(isinstance(axis, la.Axis) for axis, axis_key in key_items)

    # even keys given as dict can contain duplicates (if the same axis was
    # given under different forms, e.g. name and AxisReference).
    if has_duplicates(axis for axis, axis_key in key_items):
        dupe_axes = duplicates(axis for axis, axis_key in key_items)
        dupe_axes_str = ', '.join(str(axis) for axis in dupe_axes)
        raise ValueError(f"key has several values for axis: {dupe_axes_str}\nkey: {key}")

    # ((axis, indices), (axis, indices), ...) -> dict
    return dict(key_items)


la.AxisCollection._key_to_axis_indices_dict = _key_to_axis_indices_dict
