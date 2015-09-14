from groupby import *

import numpy as np


def assertResultEq(d1, d2):
    d1k, d2k = sorted(d1.keys()), sorted(d2.keys())
    assert d1k == d2k, "result keys differ: got: %s vs expected: %s" % (d1, d2)
    for k in d1k:
        v1, v2 = d1[k], d2[k]
        assert np.array_equal(v1, v2), \
               "%s value differs. we get: %s\nexpected: %s" % (k, v1, v2)
    print ".",


def assertGroupByEq(arrays, filter, result):
    assertResultEq(group_indices_nd(arrays, filter), result)

#a = np.random.randint(10, size=1e6)
#b = np.random.randint(2, size=1e6)
#bb = b.astype(np.bool)
#b8 = b.astype(np.int8)
#c = np.random.randint(20, size=1e6)
#f = np.random.randint(10, size=1e6) * 0.1
#
#comb = np.empty(1e6, dtype=np.dtype([('a', int), ('b', bool)]))
#comb['b'] = bb
#comb['a'] = a
#
#nca = comb['a']
#ncbb = comb['b']


i1 = np.array([1, 2, 3], dtype=np.int32)
i2 = np.array([1, 1, 1], dtype=np.int32)
b1 = np.array([True, True, False])
b2 = np.array([True, True, True])
f1 = np.array([1.0, 2.0, 3.0])
f2 = np.array([np.nan, 2.0, 3.0])
f3 = np.array([np.nan, 2.0, np.nan])
f4 = np.array([1.0, 2.0, 1.0])

fltr1 = np.array([True, False, True])
all_true = np.array([True, True, True])

#-------------#
# 1 dimension #
#-------------#

# bool
assertGroupByEq([all_true], True, {True: np.array([0, 1, 2])})
assertGroupByEq([all_true], all_true, {True: np.array([0, 1, 2])})
assertGroupByEq([all_true], fltr1, {True: np.array([0, 2])})

assertGroupByEq([b1], True, {False: np.array([2]), True: np.array([0, 1])})
assertGroupByEq([b1], all_true, {False: np.array([2]), True: np.array([0, 1])})
assertGroupByEq([b1], fltr1, {False: np.array([2]), True: np.array([0])})

# int32
assertGroupByEq([i1], True, {1: np.array([0]), 2: np.array([1]),
                             3: np.array([2])})
assertGroupByEq([i1], all_true, {1: np.array([0]), 2: np.array([1]),
                                 3: np.array([2])})
assertGroupByEq([i1], fltr1, {1: np.array([0]), 3: np.array([2])})

assertGroupByEq([i2], True, {1: np.array([0, 1, 2])})
assertGroupByEq([i2], all_true, {1: np.array([0, 1, 2])})
assertGroupByEq([i2], fltr1, {1: np.array([0, 2])})

# float
assertGroupByEq([f1], True, {1.0: np.array([0]), 2.0: np.array([1]),
                             3.0: np.array([2])})
assertGroupByEq([f1], all_true, {1.0: np.array([0]), 2.0: np.array([1]),
                                 3.0: np.array([2])})
assertGroupByEq([f1], fltr1, {1.0: np.array([0]), 3.0: np.array([2])})

assertGroupByEq([f2], True, {2.0: np.array([1]), 3.0: np.array([2])})
assertGroupByEq([f2], all_true, {2.0: np.array([1]), 3.0: np.array([2])})
assertGroupByEq([f2], fltr1, {3.0: np.array([2])})

assertGroupByEq([f3], True, {2.0: np.array([1])})
assertGroupByEq([f3], all_true, {2.0: np.array([1])})
assertGroupByEq([f3], fltr1, {})

assertGroupByEq([f4], True, {1.0: np.array([0, 2]), 2.0: np.array([1])})
assertGroupByEq([f4], all_true, {1.0: np.array([0, 2]), 2.0: np.array([1])})
assertGroupByEq([f4], fltr1, {1.0: np.array([0, 2])})

#--------------#
# 2 dimensions #
#--------------#

# int32 - boolean
assertGroupByEq([i1, b1], True,
                {(1, True): np.array([0]), (2, True): np.array([1]),
                 (3, False): np.array([2])})
assertGroupByEq([b1, i1], True,
                {(True, 1): np.array([0]), (True, 2): np.array([1]),
                 (False, 3): np.array([2])})

assertGroupByEq([i1, b1], fltr1,
                {(1, True): np.array([0]), (3, False): np.array([2])})
assertGroupByEq([b1, i1], fltr1,
                {(True, 1): np.array([0]), (False, 3): np.array([2])})

assertGroupByEq([i1, b2], fltr1,
                {(1, True): np.array([0]), (3, True): np.array([2])})
assertGroupByEq([b2, i1], fltr1,
                {(True, 1): np.array([0]), (True, 3): np.array([2])})

# float - boolean
assertGroupByEq([f1, b2], fltr1,
                {(1.0, True): np.array([0]), (3.0, True): np.array([2])})

assertGroupByEq([f2, b2], fltr1, {(3.0, True): np.array([2])})
assertGroupByEq([b2, f2], fltr1, {(True, 3.0): np.array([2])})

assertGroupByEq([f3, b2], fltr1, {})
assertGroupByEq([b2, f3], fltr1, {})

#-------------#
# speed tests #
#-------------#
i1 = np.random.randint(10, size=1e6)
i2 = np.random.randint(2, size=1e6)
b1 = i2.astype(np.bool)
b2 = i2.astype(np.int8)

all_true = np.ones(1e6, dtype=np.bool)
all_false = np.zeros(1e6, dtype=np.bool)
fltr = i1 > 5
