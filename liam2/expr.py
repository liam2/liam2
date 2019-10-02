# encoding: utf-8
from __future__ import absolute_import, division, print_function

import inspect
import types
from collections import Counter

import numpy as np
import larray as la

from liam2.compat import basestring, PY2, zip, with_metaclass, getargspec
from liam2.cache import Cache
from liam2.config import debug
from liam2.context import EntityContext, EvaluationContext
from liam2.utils import (ExplainTypeError, safe_take, IrregularNDArray, NiceArgSpec, englishenum, make_hashable,
                         add_context, array_nan_equal)


try:
    import numexpr
#    numexpr.set_num_threads(1)
    evaluate = numexpr.evaluate
except ImportError:
    numexpr = None

    def make_global_context():
        context = dict((name, getattr(np, name))
                       for name in ('where', 'exp', 'log', 'abs'))
        context.update([('False', False), ('True', True)])
        return context
    eval_context = make_global_context()

    # noinspection PyUnusedLocal
    def evaluate(expr, globals_dict, locals_dict=None, **kwargs):
        complete_globals = {}
        complete_globals.update(globals_dict)
        if locals_dict is not None:
            if isinstance(locals_dict, np.ndarray):
                for fname in locals_dict.dtype.fields:
                    complete_globals[fname] = locals_dict[fname]
            else:
                complete_globals.update(locals_dict)
        complete_globals.update(eval_context)
        return eval(expr, complete_globals, {})

expr_cache = Cache()
timings = Counter()

type_to_idx = {
    bool: 0, np.bool_: 0,
    int: 1, np.int8: 1, np.int16: 1, np.int32: 1, np.int64: 1, np.intc: 1,
        np.long: 1, np.longlong: 1,
    float: 2, np.float64: 2,
    object: 3, np.object_: 3
}
idx_to_type = [bool, int, float, object]

missing_values = {
    # int: -2147483648,
    # for links, we need to have abs(missing_int) < len(a) !
    # XXX: we might want to use different missing values for links and for
    #     "normal" ints
    int: -1,
    float: float('nan'),
    # bool: -1
    bool: False
}


def normalize_type(type_):
    return idx_to_type[type_to_idx[type_]]


def get_default_value(column, default_value=None):
    normalized_type = normalize_type(column.dtype.type)
    if default_value is None:
        default_value = missing_values[normalized_type]
    assert isinstance(default_value, normalized_type), \
        "%s is not of type %s" % (default_value, normalized_type)
    return default_value


def get_default_vector(num, dtype, default_value=None):
    res = np.empty(num, dtype=dtype)
    res.fill(get_default_value(res, default_value))
    return res


def get_default_record(dtype, default_values=None):
    default_values = default_values if default_values is not None else {}
    row = np.empty(1, dtype=dtype)
    for fname in dtype.names:
        row[fname] = get_default_value(row[fname], default_values.get(fname))
    return row


def get_default_array(length, dtype, default_values=None):
    fill_value = get_default_record(dtype, default_values)
    return np.full(length, fill_value, dtype=dtype)


def hasvalue(column):
    missing_value = get_default_value(column)
    if np.isnan(missing_value):
        return ~np.isnan(column)
    else:
        return column != missing_value


def coerce_types(context, *args):
    dtype_indices = [type_to_idx[getdtype(arg, context)] for arg in args]
    return idx_to_type[max(dtype_indices)]


def as_simple_expr(expr, context):
    if isinstance(expr, Expr):
        return expr.as_simple_expr(context)
    elif isinstance(expr, list):
        return [as_simple_expr(e, context) for e in expr]
    elif isinstance(expr, tuple):
        return tuple([as_simple_expr(e, context) for e in expr])
    else:
        return expr


def as_string(expr):
    if isinstance(expr, Expr):
        return expr.as_string()
    elif isinstance(expr, list):
        return [as_string(e) for e in expr]
    elif isinstance(expr, tuple):
        return tuple([as_string(e) for e in expr])
    else:
        return repr(expr)


def traverse_expr(expr):
    if isinstance(expr, Expr):
        for node in expr.traverse():
            yield node
    elif isinstance(expr, (tuple, list)):
        for e in expr:
            for node in traverse_expr(e):
                yield node
    else:
        yield expr


def gettype(value):
    if isinstance(value, (np.ndarray, la.LArray)):
        type_ = value.dtype.type
    elif isinstance(value, (tuple, list)):
        type_ = type(value[0])
    else:
        type_ = type(value)
    return normalize_type(type_)


def getdtype(expr, context):
    if isinstance(expr, Expr):
        return expr.dtype(context)
    else:
        return gettype(expr)


def always(type_):
    def dtype(self, context):
        return type_
    return dtype


def firstarg_dtype(self, context):
    return getdtype(self.args[0], context)


def ispresent(values):
    dt = values.dtype
    if np.issubdtype(dt, np.floating):
        return np.isfinite(values)
    elif np.issubdtype(dt, np.integer):
        return values != missing_values[int]
    elif np.issubdtype(dt, np.bool_):
        # return values != missing_values[bool]
        return True
    else:
        raise Exception('%s is not a supported type for ispresent' % dt)


def collect_variables(expr):
    if isinstance(expr, Expr):
        return expr.collect_variables()
    elif isinstance(expr, (tuple, list)):
        all_vars = [collect_variables(e) for e in expr]
        return set.union(*all_vars) if all_vars else set()
    else:
        return set()


def expr_eval(expr, context):
    try:
        if isinstance(expr, Expr):
            # assert isinstance(expr.__fields__, tuple)

            globals_data = context.global_tables
            if globals_data is not None:
                globals_names = set(globals_data.keys())
                if 'periodic' in globals_data:
                    globals_names |= set(globals_data['periodic'].dtype.names)
            else:
                globals_names = set()

            # FIXME: systematically checking for the presence of variables has a
            # non-negligible cost (especially in matching), even when caching
            # collect_variables result (it is much better than before though).
            # TODO: also check for globals
            # print("vars", expr.collect_variables())
            for var in expr.collect_variables():
                if var.name not in globals_names and var not in context:
                    raise Exception("variable '%s' is unknown (it is either "
                                    "not defined or not computed yet)" % var)
            return expr.evaluate(context)

            # there are several flaws with this approach:
            # 1) I don't get action times (csv et al)
            # 2) these are cumulative times (they include child expr/processes)
            #    we might want to store the timings in a tree (based on call
            #    stack depth???) so that I could rebuild both cumulative and
            #    "real" timings.
            # 3) the sum of timings is wrong since children/nested expr times
            #    count both for themselves and for all their parents
    #        time, res = gettime(expr.evaluate, context)
    #        timings[expr.__class__.__name__] += time
    #        return res
        elif isinstance(expr, list):
            return [expr_eval(e, context) for e in expr]
        elif isinstance(expr, tuple):
            return tuple([expr_eval(e, context) for e in expr])
        elif isinstance(expr, slice):
            return slice(expr_eval(expr.start, context),
                         expr_eval(expr.stop, context),
                         expr_eval(expr.step, context))
        else:
            return expr
    except Exception as e:
        add_context(e, "when evaluating: " + str(expr))
        raise


def binop(opname, kind='binary', reverse=False):
    def op(self, other):
        classes = {'binary': BinaryOp,
                   'division': DivisionOp,
                   'logical': LogicalOp,
                   'comparison': ComparisonOp}
        class_ = classes[kind]
        return class_(opname, other, self) \
            if reverse else class_(opname, self, other)
    return op


class Expr(object):
    # XXX: I wonder if those couldn't be computed automatically by using
    # isinstance(v, Expr)
    __children__ = ()
    num_tmp = 0

    def __init__(self):
        raise NotImplementedError()

    @property
    def children(self):
        d = self.__dict__
        return tuple(d[k] for k in self.__children__)

    @property
    def value(self):
        d = self.__dict__
        # ignore _ keys, for example "_variables"
        keys = sorted(k for k in d.keys() if not k.startswith('_'))
        children = set(self.__children__)
        return tuple(d[k] for k in keys if k not in children)

    # makes sure we do not use "normal" python logical operators
    # (and, or, not)
    def __nonzero__(self):
        raise Exception("Improper use of boolean operators, you probably "
                        "forgot parenthesis around operands of an 'and' or "
                        "'or' expression. The complete expression cannot be "
                        "displayed but it contains: '%s'." % str(self))

    def evaluate(self, context):
        # period = context.period
        #
        # if isinstance(period, np.ndarray):
        #     assert np.isscalar(period) or not period.shape
        #     period = int(period)

        # cache_key = (self, period, context.entity_name, context.filter_expr)
        # try:
        #     cached_result = expr_cache.get(cache_key, None)
        #     # FIXME: lifecycle functions should invalidate all variables!
        #     if cached_result is not None:
        #         return cached_result
        # except TypeError:
        #     # The cache_key failed to hash properly, so the expr is not
        #     # cacheable. It *should* be because of a not_hashable expr
        #     # somewhere within cache_key[3].
        #     cache_key = None

        simple_expr = self.as_simple_expr(context)
        if isinstance(simple_expr, Variable) and simple_expr.name in context:
            return context[simple_expr.name]

        # check for labeled arrays, to work around the fact that numexpr
        # does not preserve ndarray subclasses.

        # avoid checking for arrays types in the past, because that is a
        # costly operation (context[var_name] fetches the column from disk
        # in that case). This probably prevents us from doing stuff like
        # align(lag(groupby() / groupby())), but it is a limitation I can
        # live with to avoid hitting the disk twice for each disk access.

        # TODO: I should rewrite this whole mess when my "dtype" method
        # supports ndarrays and la.LArray so that I can get the dtype from
        # the expression instead of from actual values.
        expr_axes = None
        numexpr_eval = True
        assert isinstance(context, EvaluationContext), type(context)
        local_ctx = context.entity_data
        if isinstance(local_ctx, EntityContext) and local_ctx.is_array_period:
            for var in simple_expr.collect_variables():
                assert var.entity is None or var.entity is context.entity, \
                    "should not have happened (as_simple_expr should " \
                    "have transformed non-local variables)"

                # var_name should always be in the context at this point
                # because missing temporaries should have been already caught
                # in expr_eval
                value = context[var.name]
                # value = local_ctx[var.name]

                # check that LArrays (if any) have all the same axes and bypass numexpr otherwise
                if isinstance(value, la.LArray):
                    if expr_axes is None:
                        expr_axes = value.axes
                    else:
                        if value.axes != expr_axes:
                            numexpr_eval = False

        # TODO: when numexpr_eval is False, we should bypass the string roundtrip
        s = simple_expr.as_string()
        constants = {'nan': float('nan'), 'inf': float('inf')}
        if numexpr_eval:
            try:
                res = evaluate(s, local_ctx, constants, truediv='auto')
                if expr_axes is not None:
                    # This relies on the fact that currently all the expression we evaluate through numexpr preserve
                    # array shapes, but if we ever use numexpr reduction capabilities, we will be in trouble
                    res = la.LArray(res, expr_axes)
            except Exception:
                if debug:
                    print("evaluate failed")
                    print("s:", s)
                    print("local context:", local_ctx)
                    print("constants:", constants)
                raise
        else:
            res = eval(s, constants, local_ctx)

        if isinstance(res, np.ndarray) and not res.shape:
            # convert to scalar (equivalent to the now deprecated np.asscalar(res))
            res = res.item()

        # if cache_key is not None:
        #     expr_cache[cache_key] = res
        #     if cached_result is not None:
        #         assert np.array_equal(res, cached_result), \
        #             "%s != %s" % (res, cached_result)
        return res

    def as_simple_expr(self, context):
        """
        evaluate any construct that is not supported by numexpr and
        create temporary variables for them
        """
        raise NotImplementedError()

    def as_string(self):
        raise NotImplementedError()

    def __getitem__(self, key):
        # TODO: we should be able to know at "compile" time if this is a
        # scalar or a vector and disallow getitem in case of a scalar
        return SubscriptedExpr(self, key)

    def __getattr__(self, key):
        if key == '_variables':
            raise AttributeError("%s (of type '%s') has no attribute '%s'"
                                     % (self, self.__class__.__name__, key))
        else:
            return ExprAttribute(self, key)
        # if key in {'data', 'dtype', 'itemsize', 'nbytes', 'ndim', 'shape', 'size',
        #            'axes', 'points', 'row_totals', 'col_totals',
        #            # aggregates
        #            'all', 'any', 'max', 'mean', 'min', 'prod', 'ptp', 'std',
        #            'sum', 'var', 'cumprod', 'cumsum',
        #            # element-wise
        #            'astype', 'clip', 'copy', 'round',
        #            # inplace
        #            'fill', 'partition', 'sort',
        #            # indirect
        #            'argmax', 'argmin', 'argpartition', 'argsort',
        #            # other
        #            'nonzero', 'reshape', 'transpose',
        #            # should go away when I implement len()
        #            '__len__'}:
        #     # excluded (and it would take some convincing to add them):
        #     # base, byteswap, ctypes, data, dump, dumps, getfield, item,
        #     # itemset, newbyteorder, put, resize, setfield, setflags, swapaxes,
        #     # take, trace, view
        #
        #     # excluded for now (I am open if anybody asks for them):
        #     # choose, compress, conj, conjugate, diagonal, dot, flags, flat,
        #     # flatten, imag, ravel, real, repeat, squeeze, strides
        #
        #     # compress (select using a boolean index for one axis) is nice but
        #     # I would like to implement a[bool_idx_wh_axis] instead in LArray
        #     return ExprAttribute(self, key)
        # else:
        #     raise AttributeError("%s (of type '%s') has no attribute '%s'"
        #                          % (self, self.__class__.__name__, key))

    def traverse(self):
        for child in self.children:
            for node in traverse_expr(child):
                yield node
        yield self

    def all_of(self, node_type):
        """
        Yields nodes corresponding to node_type.

        Parameters
        ----------
        node_type : class or tuple of class

        Yields
        ------
        all nodes of the tree (possibly including self) corresponding to
        node_type
        """
        for node in self.traverse():
            if isinstance(node, node_type):
                yield node

    def collect_variables(self):
        # TODO: it would be cleaner if we initialized _variables in __init__,
        # however it means each Expr subclass would have to call its parent
        # __init__ (which is a good thing but too much hassle at this point).
        if not hasattr(self, "_variables") or self._variables is None:
            # FIXME: this is a quick hack to make "othertable" work.
            # We should return prefixed variable instead.
            badvar = lambda v: isinstance(v, ShortLivedVariable) or \
                               (isinstance(v, GlobalVariable) and
                                v.tablename != 'periodic')
            self._variables = set(v for v in self.all_of(Variable)
                                  if not badvar(v))
        return self._variables

    # TODO: make equivalent/commutative expressions compare equal and hash to
    # the same thing.
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        def strict_equal(a, b):
            # we are intentionally very picky so that eg. True != 1
            if type(a) is not type(b):
                return False
            if isinstance(a, (tuple, list)):
                return all(strict_equal(e1, e2) for e1, e2 in zip(a, b))
            else:
                return a == b

        res = strict_equal(self.value, other.value) and \
            strict_equal(self.children, other.children)
        if res:
            if str(self) != str(other):
                print()
                print('SHOULD NOT COMPARE EQUAL!')
                print(str(self).ljust(40), '>>>', self.value, self.children)
                print(str(other).ljust(40), '>>>', other.value, other.children)
                print(type(self), type(other),
                      len(self.children), len(other.children))
                print([(x, type(x)) for x in self.children])
                print([(x, type(x)) for x in other.children])
                # raise Exception("should not compare equal")
        return res

    def __hash__(self):
        return hash((self.__class__.__name__, self.value,
                     make_hashable(self.children)))

    def __contains__(self, expr):
        for node in self.traverse():
            if expr == node:
                return True
        return False

    def get_tmp_varname(self, context):
        tmp_varname = "temp_%d" % self.num_tmp
        Expr.num_tmp += 1
        return tmp_varname

    def add_tmp_var(self, context, result):
        tmp_varname = self.get_tmp_varname(context)
        if tmp_varname in context:
            # should be consistent but nan != nan
            if isinstance(result, np.ndarray):
                assert array_nan_equal(context[tmp_varname], result)
            else:
                assert result != result or context[tmp_varname] == result
        # FIXME: we should never modify the context in-place. We should rather
        #        have a build_context method.
        context[tmp_varname] = result
        return Variable(context.entity, tmp_varname, gettype(result))


class EvaluableExpression(Expr):
    def evaluate(self, context):
        raise NotImplementedError()

    def as_simple_expr(self, context):
        return self.add_tmp_var(context, self.evaluate(context))


def non_scalar_array(a):
    return isinstance(a, np.ndarray) and a.shape


class SubscriptedExpr(EvaluableExpression):
    __children__ = ('expr', 'key')

    def __init__(self, expr, key):
        self.expr = expr
        self.key = key

    def __repr__(self):
        key = self.key
        if isinstance(key, slice):
            key_str = '%s:%s' % (key.start, key.stop)
            if key.step is not None:
                key_str += ':%s' % key.step
        else:
            key_str = str(key)
        return '%s[%s]' % (self.expr, key_str)

    def evaluate(self, context):
        expr_value = expr_eval(self.expr, context)
        key = expr_eval(self.key, context)
        filter_expr = context.filter_expr

        # When there is a contextual filter, we modify the key to avoid
        # crashes (IndexError).

        # The value returned for individuals outside the filter is
        # *undefined* ! We sometimes return missing and sometimes return the
        # value of another individual (index -1). This should not pose a
        # problem because those values should not be used anyway.
        if filter_expr is not None:
            # We need a context without filter to evaluate the filter
            # (to avoid an infinite recursion)
            sub_context = context.clone(filter_expr=None)
            # filter_value should be a bool scalar or a bool array
            filter_value = expr_eval(filter_expr, sub_context)
            assert isinstance(filter_value, (bool, np.bool_)) or \
                np.issubdtype(filter_value.dtype, np.bool_)

            if isinstance(expr_value, la.LArray):
                # ca craint, ce qui faut, c'est faire un guess axis sur la
                # première valeur valide puis utiliser le label de cet axe
                # là (ou faire un PGroup sur cet axe là), otherwise, we can
                # introduce duplicate keys
                # OR, I could replicate the first non-filtered key
                # it might be a bad key, but if so, it would break anyway
                always_good_key = expr_value.axes[0].labels[0]
                missing_value = get_default_value(expr_value)
                # print("good key", always_good_key)
                # print("type missing", missing_value, type(missing_value))
            elif isinstance(expr_value, la.core.array.LArrayPointsIndexer):
                always_good_key = expr_value.array.axes[0].labels[0]
                missing_value = get_default_value(expr_value.array)
                # print("type missing", missing_value, type(missing_value))
            elif isinstance(expr_value, np.ndarray):
                always_good_key = -1
                missing_value = get_default_value(expr_value)
            else:
                assert isinstance(expr_value, (tuple, list))
                always_good_key = -1
                missing_value = None

            def fixkey(orig_key, filter_value):
                if non_scalar_array(orig_key):
                    # print("orig_key", orig_key)
                    # not using filter_to_indices because filter_value can be an LArray
                    non_filtered_idx = filter_value.nonzero()[0]
                    if len(non_filtered_idx):
                        first_non_filtered_value = \
                            orig_key[non_filtered_idx[0]]
                    else:
                        # avoid crashing on: if(always_false,
                        # array[badindex_scalar], val)
                        return None
                    newkey = orig_key.copy()
                    newkey[~filter_value] = first_non_filtered_value
                    # print("new key", newkey)
                else:
                    # avoid crashing on: if(always_false, array[badindex], val)
                    # Note that if only some are False, we will return orig_key
                    # (ie not fix), but this is OK because it would fail anyway.
                    if np.all(~filter_value):
                        # first_non_filtered_idx = filter_value.nonzero()[0][0]
                        # first_non_filtered_value = \
                        #     orig_key[first_non_filtered_idx]
                        newkey = None #always_good_key #first_non_filtered_value
                    else:
                        newkey = orig_key
                return newkey

            # XXX: couldn't we use np.take(mode='clip') instead of all this
            # Mumbo-jumbo? (and implement it in LArray)
            if non_scalar_array(filter_value):
                if isinstance(key, tuple):
                    # nd-key
                    key = tuple(fixkey(k, filter_value) for k in key)
                    if any(k is None for k in key):
                        return missing_value
                elif isinstance(key, slice):
                    raise NotImplementedError()
                else:
                    # scalar or array key
                    key = fixkey(key, filter_value)
                    if key is None:
                        return missing_value
            else:
                if not filter_value:
                    if (non_scalar_array(key) or
                        (isinstance(key, tuple) and
                         any(non_scalar_array(k) for k in key))):
                        # scalar filter, array or tuple key
                        if isinstance(expr_value, la.core.array.LArrayPointsIndexer):
                            expr_value = expr_value.array
                        return np.full_like(expr_value, missing_value)
                    elif isinstance(key, slice):
                        raise NotImplementedError()
                    else:
                        # scalar (or tuple of scalars) key
                        return missing_value
        # print("fixed key", key)
        return expr_value[key]


class ExprAttribute(EvaluableExpression):
    __children__ = ('expr', 'key')

    def __init__(self, expr, key):
        self.expr = expr
        self.key = key

    def __repr__(self):
        return '%s.%s' % (self.expr, self.key)

    def evaluate(self, context):
        expr_value = expr_eval(self.expr, context)
        key_value = expr_eval(self.key, context)
        if hasattr(expr_value, key_value):
            return getattr(expr_value, key_value)
        else:
            raise AttributeError("%s (of type '%s') has no attribute '%s'"
                                 % (self, self.__class__.__name__, key_value))

    def __call__(self, *args, **kwargs):
        return DynamicFunctionCall(self, *args, **kwargs)

    def __getattr__(self, key):
        if key == '_variables':
            raise AttributeError("%s (of type '%s') has no attribute '%s'"
                                 % (self, self.__class__.__name__, key))
        else:
            return ExprAttribute(self, key)


# we need to inherit from ExplainTypeError, so that TypeError exceptions are
# also "explained" for functions using FillFuncNameMeta
class FillFuncNameMeta(ExplainTypeError):
    def __init__(cls, name, bases, dct):
        ExplainTypeError.__init__(cls, name, bases, dct)

        funcname = dct.get('funcname')
        if funcname is None:
            funcname = cls.__name__.lower()
            cls.funcname = funcname


# XXX: it might be a good idea to merge both
class FillArgSpecMeta(FillFuncNameMeta):
    def __init__(cls, name, bases, dct):
        FillFuncNameMeta.__init__(cls, name, bases, dct)

        compute = cls.get_compute_func()

        # make sure we are not on one of the Abstract base class
        if compute is None:
            return

        # funcname = dct.get('funcname')
        # if funcname is None:
        #     funcname = cls.__name__.lower()
        #     cls.funcname = funcname

        argspec = dct.get('argspec')
        if argspec is None:
            try:
                # >>> def a(a, b, c=1, *d, **e):
                # ...     pass
                #
                # >>> inspect.getargspec(a)
                # ArgSpec(args=['a', 'b', 'c'], varargs='d', keywords='e',
                #         defaults=(1,))
                spec = getargspec(compute)
            except TypeError:
                raise Exception('%s is not a pure-Python function so its '
                                'signature needs to be specified '
                                'explicitly. See exprmisc.Uniform for an '
                                'example' % compute.__name__)
            # On Python >= 3, method attributes appear as functions on the *class* itself
            # (they only appear as methods on the class instances)
            if isinstance(compute, types.MethodType) or not PY2:
                # for methods, strip "self" and "context" args
                args = [arg for arg in spec.args
                        if arg not in {'self', 'context'}]
                spec = (args,) + spec[1:]
            kwonly = cls.kwonlyargs
            # if we have a varkw variable but it was only needed because of
            # kwonly args
            if spec[2] is not None and kwonly and not cls.kwonlyandvarkw:
                # we set varkw to None
                spec = spec[:2] + (None,) + spec[3:]
            extra = (list(kwonly.keys()), kwonly, {})
            cls.argspec = NiceArgSpec._make(spec + extra)

    def get_compute_func(cls):
        raise NotImplementedError()


class AbstractFunction(with_metaclass(FillFuncNameMeta, Expr)):
    __children__ = ('args', 'kwargs')

    funcname = None
    argspec = None

    def __init__(self, *args, **kwargs):
        # The behavior/error messages match Python 3.4 (and probably other 3.x)
        argnames = self.argspec.args
        maxargs = len(argnames)
        defaults = self.argspec.defaults
        nreqargs = maxargs - (len(defaults) if defaults is not None else 0)
        reqargnames = argnames[:nreqargs]
        allowed_kwargs = set(argnames) | set(self.argspec.kwonlyargs)
        funcname = self.funcname
        assert funcname is not None

        nargs = len(args)
        availposargnames = set(argnames[:nargs])
        availkwargnames = set(kwargs.keys())
        dupeargnames = availposargnames & availkwargnames
        if dupeargnames:
            raise TypeError("%s() got multiple values for argument '%s'"
                            % (funcname, dupeargnames.pop()))

        # Check that we do not have invalid kwargs
        extra_kwargs = availkwargnames - allowed_kwargs
        # def f(**kwargs) => argspec.varkw = 'kwargs'
        if extra_kwargs and self.argspec.varkw is None:
            raise TypeError("%s() got an unexpected keyword argument '%s'"
                            % (funcname, extra_kwargs.pop()))

        # Check that we do not have too many args
        if self.argspec.varargs is None and nargs > maxargs:
            # f() takes 3 positional arguments but 4 were given
            # f() takes from 1 to 3 positional arguments but 4 were given
            # + 1 to be consistent with Python (to account for self) but
            # those will be modified again (-1) in ExplainTypeError
            posargs = str(nreqargs + 1) if nreqargs == maxargs \
                else "from %d to %d" % (nreqargs + 1, maxargs + 1)

            msg = "%s() takes %s positional argument%s but %d were given"
            raise TypeError(msg % (funcname, posargs,
                                   's' if maxargs > 1 else '', nargs + 1))

        # Check that we have all required args (passed either as args or kwargs)
        missing = [name for name in reqargnames
                   if name not in (availposargnames | availkwargnames)]
        if missing:
            nmissing = len(missing)
            # f() missing 1 required positional argument: 'a'
            # f() missing 2 required positional arguments: 'a' and 'b'
            # f() missing 3 required positional arguments: 'a', 'b', and 'c'
            # + 1 to be consistent with Python (to account for self) but
            # those will be modified again (-1) in ExplainTypeError
            raise TypeError("%s() missing %d positional argument%s: %s"
                            % (funcname,
                               nmissing + 1,
                               's' if nmissing > 1 else '',
                               englishenum(repr(a) for a in missing)))

        # save original arguments before we mess with them
        self._original_args = args, sorted(kwargs.items())

        # move all "non-kwonly" kwargs to args
        # def func(a, b, c, d, e=1, f=1):
        #     pass
        # nreqargs = 4, maxargs = 6
        # >>> func(1, 2, c=3, d=4, f=5)
        # nargs = 2
        # >>> func(1, 2, 3, 4, 5)
        # nargs = 5
        # 1) required arguments (without a default value) passed as kwargs
        #    pop() should not raise otherwise the "if missing" test above would
        #    have triggered an exception)
        extra_args = [kwargs.pop(name) for name in argnames[nargs:nreqargs]]

        # 2) optional args (with a default value) not passed as positional args
        if defaults is not None:
            # number of optional args passed as positional args
            nposopt = max(nargs - nreqargs, 0)
            extra_args.extend([kwargs.pop(argname, default)
                               for argname, default
                               in zip(argnames[nreqargs + nposopt:],
                                      defaults[nposopt:])])

        args = args + tuple(extra_args)
        kwargs = tuple(sorted(kwargs.items()))
        self.args = args
        self.kwargs = kwargs
        self._variables = None

    @staticmethod
    def format_args_str(args, kwargs):
        """
        :param args: list of strings
        :param kwargs: list of (k, v) where both k and v are strings
        :return: a single string
        """
        return ', '.join(list(args) + ['%s=%s' % (k, v) for k, v in kwargs])

    @staticmethod
    def format(funcname, args, kwargs):
        args = [repr(a) for a in args]
        kwargs = [(str(k), repr(v)) for k, v in kwargs]
        return '%s(%s)' % (funcname,
                           AbstractFunction.format_args_str(args, kwargs))

    def __repr__(self):
        return self.format(self.funcname, *self._original_args)


# this needs to stay in the expr module because of ExprAttribute, which uses
# DynamicFunctionCall -> GenericFunctionCall -> FunctionExpr
class FunctionExpr(with_metaclass(FillArgSpecMeta, EvaluableExpression, AbstractFunction)):
    """
    Base class for defining (python-level) functions. That is, if you want to
    make a new function available in LIAM2 models, you should inherit from this
    class. In most cases, overriding the compute and dtype methods is
    enough, but your mileage may vary.
    """
    # argspec is set automatically for pure-python functions, but needs to
    # be set manually for builtin/C functions.
    argspec = None
    kwonlyargs = {}
    kwonlyandvarkw = False
    no_eval = ()

    @classmethod
    def get_compute_func(cls):
        return cls.compute

    def _eval_args(self, context):
        """
        evaluates arguments to the function except those in no_eval
        returns args, {kwargs}

        At this point "normal" args passed as kwargs have already been
        transferred to positional args by AbstractFunction.__init__, so kwargs
        are either kwonlyargs or varkwargs
        """
        if self.no_eval:
            no_eval = self.no_eval
            assert isinstance(no_eval, tuple) and \
                all(isinstance(f, basestring) for f in no_eval), \
                "no_eval should be a tuple of strings but %r is a %s" \
                % (no_eval, type(no_eval))
            no_eval = set(no_eval)

            argspec = self.argspec
            args, kwargs = self.args, self.kwargs
            varargs = args[len(argspec.args):]

            # evaluate positional args
            args = [expr_eval(arg, context) if name not in no_eval else arg
                    for name, arg in zip(argspec.args, args)]

            # evaluate *args
            if varargs:
                assert argspec.varargs is not None
                if argspec.varargs not in no_eval:
                    varargs = [expr_eval(arg, context) for arg in varargs]
                args.extend(varargs)

            # check whether extra kwargs (from **kwargs) should be evaluated
            if argspec.varkw is not None and argspec.varkw in no_eval:
                allkwnames = set(name for name, _ in kwargs)
                # "normal" args passed as kwargs have been transferred to positional args in AbstractFunction.__init__,
                # so all remaining kwargs are either kwonlyargs or varkwargs
                varkwnames = allkwnames - set(argspec.kwonlyargs)
                no_eval |= varkwnames

            # evaluate all kwargs
            kwargs = [(name, expr_eval(arg, context))
                      if name not in no_eval else (name, arg)
                      for name, arg in kwargs]
        else:
            args, kwargs = expr_eval((self.args, self.kwargs), context)

        return args, dict(kwargs)

    # TODO: use abc.ABCMeta and abc.abstractmethod instead
    def compute(self, context, *args, **kwargs):
        raise NotImplementedError()

    def evaluate(self, context):
        args, kwargs = self._eval_args(context)
        return self.compute(context, *args, **kwargs)


class GenericFunctionCall(FunctionExpr):
    """
    GenericFunctionCall handles calling expressions where the function to run is
    passed as the first argument.
    """
    def compute(self, context, func, *args, **kwargs):
        return func(*args, **kwargs)


class DynamicFunctionCall(GenericFunctionCall):
    """
    DynamicFunctionCall handles calling expressions where the function to run is
    determined at runtime (it should be passed as the first argument).
    """
    # DynamicFunctionCall is (currently) only used for calling ndarray methods,
    # which are all builtin methods for which we do not have signatures,
    # so we cannot (at this point) check arguments nor convert kwargs to args,
    # so we deliberately do not call FunctionExpr.__init__ which does both
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = tuple(sorted(kwargs.items()))

    def compute(self, context, func, *args, **kwargs):
        return func(*args, **kwargs)

    @property
    def _original_args(self):
        return self.args[1:], self.kwargs

    @property
    def funcname(self):
        return str(self.args[0])


#############
# Operators #
#############

class UnaryOp(Expr):
    __children__ = ('expr',)

    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

    def as_simple_expr(self, context):
        return self.__class__(self.op, as_simple_expr(self.expr, context))

    def as_string(self):
        return "(%s%s)" % (self.op, as_string(self.expr))

    def dtype(self, context):
        return getdtype(self.expr, context)

    # FIXME: only add parentheses if necessary
    def __repr__(self):
        nicerop = {'~': 'not '}
        niceop = nicerop.get(self.op, self.op)
        return "(%s%r)" % (niceop, self.expr)


class BinaryOp(Expr):
    __children__ = ('expr1', 'expr2')

    def __init__(self, op, expr1, expr2):
        self.op = op
        self.expr1 = expr1
        self.expr2 = expr2

    def as_simple_expr(self, context):
        expr1 = as_simple_expr(self.expr1, context)
        expr2 = as_simple_expr(self.expr2, context)
        return self.__class__(self.op, expr1, expr2)

    # We can't simply use __str__ because of where vs if
    def as_string(self):
        expr1, expr2 = as_string(self.expr1), as_string(self.expr2)
        return "(%s %s %s)" % (expr1, self.op, expr2)

    def dtype(self, context):
        return coerce_types(context, self.expr1, self.expr2)

    # FIXME: only add parentheses if necessary
    def __repr__(self):
        nicerop = {'&': 'and', '|': 'or'}
        niceop = nicerop.get(self.op, self.op)
        return "(%r %s %r)" % (self.expr1, niceop, self.expr2)


class DivisionOp(BinaryOp):
    dtype = always(float)


class LogicalOp(BinaryOp):
    def assertbool(self, expr, context):
        dt = getdtype(expr, context)
        if dt is not bool:
            raise Exception("operands to logical operators need to be "
                            "boolean but %s is %s" % (expr, dt))

    # TODO: move the tests to a typecheck phase and use dtype = always(bool)
    def dtype(self, context):
        self.assertbool(self.expr1, context)
        self.assertbool(self.expr2, context)
        return bool


class ComparisonOp(BinaryOp):
    # TODO: move the test to a typecheck phase and use dtype = always(bool)
    def dtype(self, context):
        if coerce_types(context, self.expr1, self.expr2) is None:
            raise TypeError("operands to comparison operators need to be of "
                            "compatible types")
        return bool


#############
# Variables #
#############

class Variable(Expr):
    __children__ = ()

    def __init__(self, entity, name, dtype=None):
        # from entities import Entity
        # assert entity is None or isinstance(entity, Entity)
        self.entity = entity
        self.name = name
        self._dtype = dtype
        # self.version = 0
        # self.used = 0

    def __repr__(self):
        return "%s.%s" % (self.entity, self.name)

    def __str__(self):
        return self.name
    as_string = __str__

    def as_simple_expr(self, context):
        return self

    def dtype(self, context):
        if self._dtype is None and self.name in context:
            return gettype(context[self.name])
        else:
            return self._dtype


class ShortLivedVariable(Variable):
    pass


# class GlobalVariable(Variable):
class GlobalVariable(EvaluableExpression):
    __children__ = ()

    def __init__(self, tablename, name, dtype):
        self.tablename = tablename
        self.name = name
        self._dtype = dtype

    def __repr__(self):
        if self.name is None:
            return self.tablename
        else:
            return "%s.%s" % (self.tablename, self.name)

    def get_tmp_varname(self, context):
        period = self._eval_key(context)
        if isinstance(period, int):
            return '__%s_%s_%s' % (self.tablename, self.name, period)
        else:
            return EvaluableExpression.get_tmp_varname(self, context)

    def _eval_key(self, context):
        return context.period

    def evaluate(self, context):
        globals_data = context.global_tables
        globals_table = globals_data[self.tablename]
        if self.name is None:
            return globals_table

        if self.name not in globals_table.dtype.fields:
            raise Exception("Unknown global: %s" % self.name)

        key = self._eval_key(context)
        # TODO: this row computation should be encapsulated in the
        # globals_table object and the index column should be configurable
        colnames = globals_table.dtype.names
        if 'period' in colnames or 'PERIOD' in colnames:
            try:
                globals_periods = globals_table['PERIOD']
            except ValueError:
                globals_periods = globals_table['period']
            base_period = globals_periods[0]
            if isinstance(key, slice):
                translated_key = slice(key.start - base_period,
                                       key.stop - base_period,
                                       key.step)
            else:
                translated_key = key - base_period
        else:
            translated_key = key

        column = globals_table[self.name]
        numrows = len(column)
        missing_value = get_default_value(column)

        if isinstance(translated_key, np.ndarray) and translated_key.shape:
            return safe_take(column, translated_key, missing_value)
        elif isinstance(translated_key, slice):
            start, stop = translated_key.start, translated_key.stop
            step = translated_key.step
            if step is not None and step != 1:
                raise NotImplementedError("step != 1 (%d)" % step)
            if (isinstance(start, np.ndarray) and start.shape or
                    isinstance(stop, np.ndarray) and stop.shape):
                lengths = stop - start
                length0 = lengths[0]
                if not isinstance(start, np.ndarray) or not start.shape:
                    start = np.repeat(start, len(lengths))
                if np.all(lengths == length0):
                    # constant length => result is a 2D array:
                    # num_individuals x slice_length
                    result = np.empty((len(lengths), length0),
                                      dtype=column.dtype)
                    # we assume there are more individuals than there are
                    # "periods" (or other ticks) in the table.
                    # XXX: We might want to actually test that it is true and
                    # loop on the individuals instead if that is not the case
                    for i in range(length0):
                        result[:, i] = safe_take(column, start + i,
                                                 missing_value)
                    return result
                else:
                    # varying length => result is an array (num_individuals) of
                    # 1D arrays (slice lengths)
                    # each "item" of the result is a view, so we pay "only" for
                    # all the arrays overhead, not for the data itself.
                    result = np.empty(len(lengths), dtype=object)
                    if not isinstance(stop, np.ndarray) or not stop.shape:
                        stop = np.repeat(stop, len(lengths))
                    for i in range(len(lengths)):
                        result[i] = column[start[i]:stop[i]]
                    return IrregularNDArray(result)
            else:
                # out of bounds slices bounds are "dropped" silently (like in
                # python) -- ie the length of the slice returned can be
                # smaller than the one asked. We could return "missing_value"
                # for indices out of bounds but I do not know if it would be
                # better. Since this version is easier to implement, lets go for
                # it for now.
                return column[translated_key]
        else:
            # scalar key
            out_of_bounds = (translated_key < 0) or (translated_key >= numrows)
            return column[translated_key] if not out_of_bounds \
                else missing_value

    def __getitem__(self, key):
        return SubscriptedGlobal(self.tablename, self.name, key, self._dtype)

    def dtype(self, context):
        return self._dtype


class SubscriptedGlobal(GlobalVariable):
    __children__ = ('key',)

    def __init__(self, tablename, name, key, dtype):
        GlobalVariable.__init__(self, tablename, name, dtype)
        self.key = key

    def __repr__(self):
        return '%s[%s]' % (self.name, self.key)

    def _eval_key(self, context):
        return expr_eval(self.key, context)


def index_array_by_variables(array, context, axes):
    # TODO: either parse expressions instead of only simple Variable, or take variable value directly in
    # context instead of creating a Variable and using expr_eval
    expressions = tuple(Variable(context.entity, axis_name)
                        for axis_name in axes.names)
    columns = tuple(expr_eval(expr, context) for expr in expressions)
    axes_groups = tuple(axis[col] for axis, col in zip(axes, columns))
    return array.points[axes_groups]


# TODO: this class shouldn't be needed. GlobalArray should be handled in the
# context
class GlobalArray(Variable):
    def __init__(self, name, dtype=None, autoindex=None):
        Variable.__init__(self, None, name, dtype)
        # convert to tuple so that it is hashable
        if isinstance(autoindex, list):
            autoindex = tuple(autoindex)
        self.autoindex = autoindex

    def as_simple_expr(self, context):
        globals_data = context.global_tables
        result = globals_data[self.name]
        if self.autoindex is not None:
            result = index_array_by_variables(result, context, result.axes[self.autoindex])
        # XXX: maybe I should just use self.name?
        # FIXME: use self.add_tmp_var, because in combination with autoindex,
        # the variable could have a different value
        tmp_varname = '__%s' % self.name
        if tmp_varname in context:
            array = context[tmp_varname]
            assert isinstance(array, la.LArray)
            assert context[tmp_varname].equals(result)
        context[tmp_varname] = result
        return Variable(context.entity, tmp_varname)


class GlobalTable(object):
    def __init__(self, name, fields):
        """fields is a list of tuples (name, type)"""

        self.name = name
        self.fields = fields
        self.fields_map = dict(fields)

    def __getattr__(self, key):
        return GlobalVariable(self.name, key, self.fields_map[key])

    # noinspection PyUnusedLocal
    def traverse(self, context):
        yield self

    def __repr__(self):
        # Remember this is the expression (only used via qshow, ...), so we do
        # not want to print the data in here
        return 'Table(%s)' % ', '.join([name for name, _ in self.fields])


# XXX: can we factorise this with FunctionExpr et al.?
# for that we need argspec but we currently cannot get it when
# MethodCall.__init__ is called (the methods are potentially not created yet)
# so we would have to either: do that in two passes (first collect method
# signatures then parse method bodies) OR move AbstractFunction's function call
# arguments normalization functionality to an external function and call it
# within MethodCall.evaluate
class MethodCall(EvaluableExpression):
    __children__ = ('args', 'kwargs')

    def __init__(self, entity, name, args, kwargs):
        self.entity = entity
        self.name = name
        self.args = args
        self.kwargs = tuple(sorted(kwargs.items()))

    def evaluate(self, context):
        from liam2.process import Assignment, Function
        entity_processes = self.entity.processes
        method = entity_processes[self.name]
        # hybrid (method & variable) assignment can be called
        assert isinstance(method, (Assignment, Function))
        args = [expr_eval(arg, context) for arg in self.args]
        kwargs = dict((k, expr_eval(v, context)) for k, v in self.kwargs)
        return method.run_guarded(context, *args, **kwargs)

    def __repr__(self):
        return AbstractFunction.format(self.name, self.args, self.kwargs)


class VariableMethodHybrid(Variable):
    def __call__(self, *args, **kwargs):
        return MethodCall(self.entity, self.name, args, kwargs)


# class MethodCallToResolve(Expr):
#     def __init__(self, name, entity, args, kwargs):
#         self.name = name
#         self.entity = entity
#         self.args = args
#         self.kwargs = kwargs
#
#     def resolve(self):
#         entity_processes = self.entity.processes
#         method = entity_processes[self.name]
#         # hybrid (method & variable) assignment can be called
#         assert isinstance(method, (Assignment, Function))
#         return GenericFunctionCall(method, *self.args, **self.kwargs)


class MethodSymbol(object):
    def __init__(self, name, entity):
        self.name = name
        self.entity = entity

    def __call__(self, *args, **kwargs):
        # we cannot use self.entity.processes as they are not defined yet (we
        # are probably currently building them), so we cannot return a
        # GenericFunctionCall now like we should and instead must either
        # return an intermediary object (MethodCallToResolve) which we will
        # "resolve" later, or use DynamicFunctionCall (for which we cannot
        # have dtype yet). However that "resolve" step is currently hard to
        # do because we need ast.NodeTransformer-like machinery which we do not
        # have yet.
        # return MethodCallToResolve(self.entity, self.name, args, kwargs)
        return MethodCall(self.entity, self.name, args, kwargs)

    def __repr__(self):
        return 'MethodSymbol({}, {})'.format(self.name, self.entity)

    def __str__(self):
        return '{}.{}'.format(self.entity, self.name)


class NotHashable(Expr):
    __hash__ = None

    def __init__(self):
        pass

not_hashable = NotHashable()
