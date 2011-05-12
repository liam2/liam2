from itertools import izip, groupby, chain
from operator import itemgetter

import numpy as np

from expr import Expr, Variable, Where, functions, as_string, dtype, \
                 coerce_types, type_to_idx, idx_to_type, expr_eval, \
                 collect_variables, num_tmp, \
                 missing_values, get_missing_value, get_missing_record 
from entities import entity_registry, EntityContext, context_length
import utils


class BreakpointException(Exception):
    pass


def ispresent(values):
    dtype = values.dtype
    if np.issubdtype(dtype, float):
        return np.isfinite(values)
    elif np.issubdtype(dtype, int):
        return values != missing_values[int]
    elif np.issubdtype(dtype, bool):
        return values != missing_values[bool]


class Link(object):
    def __init__(self, name, link_type, link_field, target_entity):
        # the leading underscores are necessary to not collide with user-defined
        # fields via __getattr__.
        self._name = name
        self._link_type = link_type
        self._link_field = link_field
        self._target_entity = target_entity

    def get(self, key, missing_value=None):
        return LinkValue(self, key, missing_value)

    __getattr__ = get



class Process(object):
    def __init__(self):
        self.name = None
        self.entity = None

    def attach(self, name, entity):
        self.name = name
        self.entity = entity

    def run_guarded(self, globals):
        try:
            context = EntityContext(self.entity, globals.copy())
            self.run(context)
        except BreakpointException:
            self.entity.simulation.stepbystep = True
        
    def run(self, context):
        raise NotImplementedError()


class Assignment(Process):
    def __init__(self, expr):
        super(Assignment, self).__init__()
        self.predictor = None
        self.kind = None # period_individual, period, individual, global
        self.expr = expr

    def attach(self, name, entity, kind=None):
        super(Assignment, self).attach(name, entity)
        if self.predictor is None:
            self.predictor = name
        self.kind = kind

    def run(self, context):
        value = expr_eval(self.expr, context)
        self.store_result(value)
            
    def store_result(self, result):
        if result is None:
            return
        
        if isinstance(result, dict):
            indices = result['indices']
            result = result['values']
        else:
            indices = None

        if isinstance(result, np.ndarray):
            res_type = result.dtype.type
        else:
            res_type = type(result)

        if self.kind == 'period_individual':
            # we cannot store/cache self.entity.array[self.name] because the 
            # array object can change (eg when enlarging it due to births)
            target = self.entity.array
        elif self.kind == 'period':
            target = self.entity.per_period_array
        else:
            target = self.entity.temp_variables

        #TODO: assert type for temporary variables too
        if self.kind is not None: 
            target_type_idx = type_to_idx[target[self.predictor].dtype.type]
            res_type_idx = type_to_idx[res_type]
            if res_type_idx > target_type_idx:
                raise Exception(
                    "trying to store %s value into '%s' field which is of "
                    "type %s" % (idx_to_type[res_type_idx].__name__,
                                 self.predictor, 
                                 idx_to_type[target_type_idx].__name__))

        if indices is None:
            target[self.predictor] = result
        else:
            if self.kind is None:
                #TODO: this step should be delayed even further because it could
                # be unnecessary.
                target[self.predictor] = np.zeros(len(self.entity.array),
                                                  dtype=res_type)
            np.put(target[self.predictor], indices, result)

    def dtype(self, context):
        return dtype(self.expr, context)
    

class ProcessGroup(Process):
    def __init__(self, name, subprocesses):
        super(ProcessGroup, self).__init__()
        self.name = name
        self.subprocesses = subprocesses
    
    def run_guarded(self, globals):
        print
        for k, v in self.subprocesses:
            print "    *",
            if k is None:
#                print v,
                print v.__class__.__name__,
            else:
                print k,
            v.run_guarded(globals)
            print "done."
            v.entity.simulation.start_console(v.entity, globals['period'])


class EvaluableExpression(Expr):
    def eval(self, context):
        raise NotImplementedError

    def as_string(self, context):
        global num_tmp
        
        tmp_varname = "temp_%d" % num_tmp
        num_tmp += 1
        result = self.eval(context)
        if isinstance(result, dict):
            indices = result['indices']
            values = result['values']
        else:
            indices = None

        if indices is not None:
            if isinstance(values, np.ndarray):
                res_type = values.dtype.type
            else:
                res_type = type(values)
            result = np.zeros(context_length(context), dtype=res_type)
            np.put(result, indices, values)
        context[tmp_varname] = result
        return tmp_varname


class CompoundExpression(EvaluableExpression):
    '''expression written in terms of other expressions'''
    
    def eval(self, context): 
        expr = self.build_expr()
        context = self.build_context(context)
        return expr_eval(expr, context)

    def build_context(self, context):
        return context

    def build_expr(self):
        raise NotImplementedError

class LinkValue(EvaluableExpression):
    def __init__(self, links, target_expression, missing_value=None):
        '''
        links can be either a Link instance, a string, or a list of either
        target_expression can be any expression (it will be evaluated on the
                          target rows)
        '''
        EvaluableExpression.__init__(self)
        
        if not isinstance(links, (list, tuple)):
            links = (links,)
        self.links = links
        if isinstance(target_expression, basestring):
            target_expression = Variable(target_expression)
        self.target_expression = target_expression
        self.missing_value = missing_value
    
    def dtype(self, context):
        return self.target_entity(context).fieldtype(self.target_expression)

    def collect_variables(self, context):
        source_entity = context.entity
        fields = []
        for link in self.links:
            if isinstance(link, basestring):
                link = source_entity.links[link]
            fields.append(link._link_field)
            source_entity = entity_registry[link._target_entity]
        return set(fields)
       
    def target_entity(self, context):
        source_entity = context.entity
        for link in self.links:
            if isinstance(link, basestring):
                link = source_entity.links[link]
            # target entity becomes the next source entity
            source_entity = entity_registry[link._target_entity]
        return source_entity

    def get(self, key, missing_value=None):
        # in this case, target_expression must have been a link name
        assert isinstance(self.target_expression, Variable)
        return LinkValue(self.links + (self.target_expression.name,),
                         key,
                         missing_value)

    __getattr__ = get

    def eval(self, context):
        # build the list of link columns we will traverse
        link_columns = []
        id_to_rownums = []
        
        period = context['period']
        
        # use the context as the first entity, so that it works with subsets of
        # the entity population
        source_entity = context['__entity__']
        local_ctx = context
        for link in self.links:
            if isinstance(link, basestring):
                link = source_entity.links[link]
            
            column_name = link._link_field
            link_columns.append(expr_eval(Variable(column_name), local_ctx))
            # target entity becomes the next source entity
            source_entity = entity_registry[link._target_entity]
            local_ctx = EntityContext(source_entity, {'period': period})
            id_to_rownums.append(local_ctx.id_to_rownum)

        target_ids = link_columns.pop(0)
        id_to_rownum = id_to_rownums.pop(0)
        
        missing_int = missing_values[int]
        target_rows = id_to_rownum[target_ids]
        for column, id_to_rownum in izip(link_columns, id_to_rownums):
            valid_links = ((target_ids != missing_int) & 
                           (target_rows != missing_int))
            target_ids = np.where(valid_links,
                                  column[target_rows], missing_int)
            target_rows = id_to_rownum[target_ids]
    
        # the last/final column is a special case since it is not a simple link
        # column but can be a full-fledged expression
        target_context = EntityContext(source_entity, {'period': period})
        target_values = expr_eval(self.target_expression, target_context)
        missing_value = self.missing_value
        if missing_value is None:
            missing_value = get_missing_value(target_values)

        valid_links = (target_ids != missing_int) & (target_rows != missing_int)
        return np.where(valid_links, target_values[target_rows], missing_value)

    def __str__(self):
        names = [l._name for l in self.links] + [str(self.target_expression)]
        return '.'.join(names)
    __repr__ = __str__
        
        
class AggregateLink(EvaluableExpression):
    def __init__(self, link, target_filter=None):
        EvaluableExpression.__init__(self)
        self.link = link
        self.target_filter = target_filter

    def eval(self, context):
        assert isinstance(context, EntityContext), \
               "aggregates in groupby is currently not supported"
        source_entity = context['__entity__']
        if isinstance(self.link, basestring):
            link = source_entity.links[self.link]
        else:
            link = self.link
        assert link._link_type == 'one2many'
        
        # eg (in household entity):
        # persons: {type: one2many, target: person, field: hh_id}
        target_entity = entity_registry[link._target_entity] # eg person
        
        # this is a one2many, so the link column is on the target side
        local_ctx = EntityContext(target_entity, {'period': context['period']})
        link_column = expr_eval(Variable(link._link_field), local_ctx)
        
        missing_int = missing_values[int]
        
        if self.target_filter is not None:
            filter_context = EntityContext(target_entity, 
                                           {'period': context['period']})
            target_filter = expr_eval(self.target_filter, filter_context)
            source_ids = link_column[target_filter]
        else:
            target_filter = None
            source_ids = link_column

        id_to_rownum = context.id_to_rownum 
        if len(id_to_rownum):
            source_rows = id_to_rownum[source_ids]
            # filter out missing values: those where the value of the link
            # points to nowhere (-1)
            source_rows[source_ids == missing_int] = missing_int
        else:
            assert np.all(source_ids == missing_int)
            # we need to make a copy because eval_rows modifies the array 
            # in place
            source_rows = source_ids.copy()

        return self.eval_rows(source_entity, source_rows, target_entity,
                              target_filter, context)
    
    def eval_rows(self, source_entity, source_rows, target_entity,
                        target_filter, context):
        raise NotImplementedError



class CountLink(AggregateLink):
    func_name = 'countlink'
    
    def eval_rows(self, source_entity, source_rows, target_entity,
                  target_filter, context):
        # We can't use a negative value because that is not allowed by bincount,
        # and using a value too high will uselessly increase the size of the
        # array returned by bincount
        idx_for_missing = context_length(context)
        
        missing_int = missing_values[int]

        # filter out missing values: those where the object pointed to does not
        # exist anymore (the id corresponds to -1 in id_to_rownum)
        source_rows[source_rows == missing_int] = idx_for_missing 
   
        counts = self.count(source_rows, target_entity, target_filter, context)
        counts.resize(idx_for_missing)
        return counts
    
    def count(self, source_rows, target_entity, target_filter, context):
        if len(source_rows):
            return np.bincount(source_rows)
        else:
            return np.array([], dtype=int)

    def dtype(self, context):
        return int

    def __str__(self):
        if self.target_filter is not None:
            target_filter = ", target_filter=%s" % self.target_filter
        else:
            target_filter = ""
        return '%s(%s%s)' % (self.func_name, self.link._name, target_filter)


class SumLink(CountLink):
    func_name = 'sumlink'
    
    def __init__(self, link, target_expr, target_filter=None):
        CountLink.__init__(self, link, target_filter)
        self.target_expr = target_expr
    
    def count(self, source_rows, target_entity, target_filter, context):
        target_context = EntityContext(target_entity, 
                                       {'period': context['period']})
        value_column = expr_eval(self.target_expr, target_context)
        if target_filter is not None:
            value_column = value_column[target_filter]
        assert len(source_rows) == len(value_column)
        res = np.bincount(source_rows, value_column)

        # we need to explicitly convert to the type of the value field because
        # bincount always return floats when its weight argument is used.
        return res.astype(value_column.dtype)

    def dtype(self, context):
        #FIXME: this will probably break when target_field is an expression
        target_entity = entity_registry[self.link._target_entity] # eg person
        return target_entity.fieldtype(self.target_expr)

    def __str__(self):
        if self.target_filter is not None:
            target_filter = ", target_filter=%s" % self.target_filter
        else:
            target_filter = ""
        return '%s(%s, %s%s)' % (self.func_name, self.link._name,
                                 self.target_expr, target_filter)

class AvgLink(SumLink):
    func_name = 'avglink'
    
    def count(self, source_rows, target_entity, target_filter, context):
        sums = super(AvgLink, self).count(source_rows, target_entity, 
                                          target_filter, context)
        count = np.bincount(source_rows)
        # silence x/0 and 0/0
        np.seterr(invalid='ignore', divide='ignore')

        # this is slightly sub optimal if the value column contains integers 
        # as the data is converted from float to int then back to float
        return sums.astype(float) / count

    def dtype(self, context):
        return float


class MinLink(AggregateLink):
    func_name = 'minlink'
    aggregate_func = min

    def __init__(self, link, target_expr, target_filter=None):
        AggregateLink.__init__(self, link, target_filter)
        self.target_expr = target_expr
        
    def eval_rows(self, source_entity, source_rows, target_entity,
                  target_filter, context):
        target_context = EntityContext(target_entity, 
                                       {'period': context['period']})
        value_column = expr_eval(self.target_expr, target_context)
        if target_filter is not None:
            value_column = value_column[target_filter]
        assert len(source_rows) == len(value_column)

        result = np.empty(context_length(context), dtype=value_column.dtype)
        result.fill(get_missing_value(value_column))
        
        id_sort_indices = np.argsort(source_rows)
        sorted_rownum = source_rows[id_sort_indices]
        sorted_values = value_column[id_sort_indices]
        groups = groupby(izip(sorted_rownum, sorted_values), key=itemgetter(0))
        aggregate_func = self.aggregate_func 
        for rownum, values in groups:
            if rownum == -1:
                continue
            result[rownum] = aggregate_func(v[1] for v in values)
        return result

    def __str__(self):
        if self.target_filter is not None:
            target_filter = ", target_filter=%s" % self.target_filter
        else:
            target_filter = ""
        return '%s(%s, %s%s)' % (self.func_name, self.link._name,
                                 self.target_expr, target_filter)

        
class MaxLink(MinLink):
    func_name = 'maxlink'
    aggregate_func = max


class Min(CompoundExpression):
    def __init__(self, *args):
        CompoundExpression.__init__(self)
        assert len(args) >= 2
        self.args = args

    def build_expr(self):
        expr1, expr2 = self.args[:2]
        expr = Where(expr1 < expr2, expr1, expr2)
        for arg in self.args[2:]:
            expr = Where(expr < arg, expr, arg)
            
#        Where(Where(expr1 < expr2, expr1, expr2) < expr3, 
#              Where(expr1 < expr2, expr1, expr2),
#              expr3)
#        3 where, 3 comparisons = 6 op (or 4 if optimized)
#
#        Where(Where(Where(expr1 < expr2, expr1, expr2) < expr3, 
#                    Where(expr1 < expr2, expr1, expr2),
#                    expr3) < expr4,
#              Where(Where(expr1 < expr2, expr1, expr2) < expr3, 
#                    Where(expr1 < expr2, expr1, expr2),
#                    expr3),
#              expr4) 
#        7 where, 7 comp = 14 op (or 6 if optimized) 

        # this version scales better in theory (but in practice, it will depend
        # if numexpr factorize the common subexpression in the above version
        # or not)
#        Where(expr1 < expr2 & expr1 < expr3, 
#              expr1,
#              Where(expr2 < expr3, expr2, expr3))
#        2 where, 3 comparisons, 1 and = 6 op      
#        
#        Where(expr1 < expr2 & expr1 < expr3 & expr1 < expr4, 
#              expr1,
#              Where(expr2 < expr3 & expr2 < expr4, 
#                    expr2
#                    Where(expr3 < expr4,
#                          expr3,
#                          expr4)))
#        3 where, 6 comp, 3 and = 12 op
        return expr

    def dtype(self, context):
        return coerce_types(context, *self.args)
    
    def __str__(self):
        return 'min(%s)' % ', '.join(str(arg) for arg in self.args)


class Max(CompoundExpression):
    def __init__(self, *args):
        CompoundExpression.__init__(self)
        assert len(args) >= 2
        self.args = args

    def build_expr(self):
        expr1, expr2 = self.args[:2]
        expr = Where(expr1 > expr2, expr1, expr2)
        for arg in self.args[2:]:
            expr = Where(expr > arg, expr, arg)
        return expr
    
    def dtype(self, context):
        return coerce_types(context, *self.args)

    def __str__(self):
        return 'max(%s)' % ', '.join(str(arg) for arg in self.args)


class ZeroClip(CompoundExpression):
    def __init__(self, expr1, expr2, expr3):
        CompoundExpression.__init__(self)
        self.expr1 = expr1
        self.expr2 = expr2
        self.expr3 = expr3

    def build_expr(self):
        return Where((self.expr1 >= self.expr2) & (self.expr1 <= self.expr3), 
                     self.expr1,
                     0)

    def dtype(self, context):
        return dtype(self.expr1, context)
        

class FunctionExpression(EvaluableExpression):
    func_name = None

    #TODO: move filter to a subclass of FunctionExpression
    def __init__(self, expr, filter=None):
        self.expr = expr
        self.filter = filter

    def __str__(self):
        return '%s(%s)' % (self.func_name, self.expr)

    def collect_variables(self, context):
        return collect_variables(self.expr, context)


class ValueForPeriod(FunctionExpression):
    func_name = 'value_for_period'

    def __init__(self, expr, period, missing='auto'):
        FunctionExpression.__init__(self, expr)
        self.period = period
        self.missing = missing
        
    def eval(self, context):
        entity = context['__entity__']
        return entity.value_for_period(self.expr, self.period, context,
                                       self.missing)


class Lag(FunctionExpression):
    func_name = 'lag'

    def __init__(self, expr, num_periods=1, missing='auto'):
        FunctionExpression.__init__(self, expr)
        self.num_periods = num_periods
        self.missing = missing
        
    def eval(self, context):
        entity = context['__entity__']
        period = context['period'] - self.num_periods
        return entity.value_for_period(self.expr, period, context, self.missing)


class Duration(FunctionExpression):
    func_name = 'duration'

    def eval(self, context):
        entity = context['__entity__']
        return entity.duration(self.expr, context)


class TimeAverage(FunctionExpression):
    func_name = 'tavg'

    def eval(self, context):
        entity = context['__entity__']
        return entity.tavg(self.expr, context)


class TimeSum(FunctionExpression):
    func_name = 'tsum'

    def eval(self, context):
        entity = context['__entity__']
        return entity.tsum(self.expr, context)


class NumpyProperty(EvaluableExpression):
    func_name = None # optional (for display)
    np_func = (None,)
    # arg_names can be set automatically by using inspect.getargspec,
    # but it only works for pure Python functions, so I decided to avoid it
    # because when you add a function, it's hard to know whether it's 
    # implemented in C or not.
    arg_names = None
    skip_missing = False
        
    def __init__(self, *args, **kwargs):
        EvaluableExpression.__init__(self)
        if len(args) > len(self.arg_names):
            # + 1 to be concistent with Python (to account for self)
            raise TypeError("takes at most %d arguments (%d given)" % 
                            (len(self.arg_names) + 1, len(args) + 1))
        extra_kwargs = set(kwargs.keys()) - set(self.arg_names)
        if extra_kwargs:
            extra_kwargs = [repr(arg) for arg in extra_kwargs]
            raise TypeError("got an unexpected keyword argument %s" %
                            extra_kwargs[0])
        self.args = args
        self.kwargs = kwargs

    def eval(self, context):
        if self.skip_missing:
            def local_expr_eval(expr, context):
                values = expr_eval(expr, context)
                return values[ispresent(values)]
        else:
            local_expr_eval = expr_eval
        args = [local_expr_eval(arg, context) for arg in self.args]
        kwargs = self.kwargs.copy()
        for k, v in kwargs.items():
            kwargs[k] = expr_eval(v, context)

        if 'size' in self.arg_names and 'size' not in kwargs:
            kwargs['size'] = context_length(context)

        func = self.np_func[0]
        return func(*args, **kwargs)

    def __str__(self):
        func_name = self.func_name 
        if func_name is None:
            func_name = self.np_func[0].__name__
        kwargs = self.kwargs
        values = zip(self.arg_names, self.args)
        for name in self.arg_names[len(self.args):]:
            if name in kwargs:
                values.append((name, kwargs[name]))
        str_args = ', '.join('%s=%s' % (name, value) for name, value in values) 
        return '%s(%s)' % (func_name, str_args)
    
    def collect_variables(self, context):
        sets = [collect_variables(arg, context) for arg in self.args]
        sets.extend(collect_variables(v, context)
                    for v in self.kwargs.itervalues())
        return set.union(*sets)
    
# >>> mi = 1
# >>> ma = 10
# >>> a = np.arange(1e7)
# 
# >>> timeit np.clip(a, mi, ma)
# 10 loops, best of 3: 127 ms per loop
# >>> timeit np.clip(a, mi, ma, a)
# 10 loops, best of 3: 26.2 ms per loop
# >>> timeit ne.evaluate('where(a < mi, mi, where(a > ma, ma, a))')
# 10 loops, best of 3: 94.1 ms per loop
class Clip(NumpyProperty):
    np_func = (np.clip,)
    arg_names = ('a', 'a_min', 'a_max', 'out')


class Uniform(NumpyProperty):
    np_func = (np.random.uniform,)
    arg_names = ('low', 'high', 'size')


class Normal(NumpyProperty):
    np_func = (np.random.normal,)
    arg_names = ('loc', 'scale', 'size')


class Choice(EvaluableExpression):
    func_name = 'choice'
    
    def __init__(self, choices, weights=None):
        EvaluableExpression.__init__(self)
        self.choices = np.array(choices)
        if weights is not None:
            self.bins = np.array([0.0] + list(np.cumsum(weights)))
            # cumulative sum must be 1.0
            assert self.bins[-1] == 1.0, \
                   "the cumulative sum of choice weights must be 1"
        else:
            self.bins = None
        
    def eval(self, context):
        num = context_length(context)
        
        if num:
            if self.bins is None:
                # all values have the same probability
                choices_idx = np.random.randint(len(self.choices), size=num)
            else:
                u = np.random.uniform(size=num)
                choices_idx = np.digitize(u, self.bins) - 1
        else:
            choices_idx = []
        return self.choices[choices_idx]

    def dtype(self, context):
        return self.choices.dtype
    
    def collect_variables(self, context):
        return set()

    def __str__(self):
        if self.bins is None:
            weights = ""
        else:
            weights = ", [%s]" % ', '.join(str(v) for v in np.diff(self.bins))
        return "%s(%s%s)" % (self.func_name, list(self.choices), weights)


class RandInt(NumpyProperty):
    np_func = (np.random.randint,)
    arg_names = ('low', 'high', 'size')


class Round(NumpyProperty):
    func_name = 'round' # np.round redirects to np.round_
    np_func = (np.round,)
    arg_names = ('a', 'decimals', 'out')
    
    def dtype(self, context):
        # result dtype is the same as the input dtype
        res = dtype(self.args[0], context)
        assert res == float
        return res


class Trunc(FunctionExpression):
    func_name = 'trunc'

    def eval(self, context):
        return expr_eval(self.expr, context).astype(int)
        
    def dtype(self, context):
        assert dtype(self.args[0], context) == float
        return int


class GroupMin(NumpyProperty):
    func_name = 'grpmin'
    np_func = (np.amin,)
    arg_names = ('a', 'axis', 'out')


class GroupMax(NumpyProperty):
    func_name = 'grpmax'
    np_func = (np.amax,)
    arg_names = ('a', 'axis', 'out')


class GroupSum(NumpyProperty):
    func_name = 'grpsum'
    np_func = (np.nansum,)
    arg_names = ('a', 'axis', 'dtype', 'out')

class GroupStd(NumpyProperty):
    func_name = 'grpstd'
    np_func = (np.std,)
    arg_names = ('a', 'axis', 'dtype', 'out', 'ddof')
    skip_missing = True


class GroupCount(EvaluableExpression):
    def __init__(self, filter=None):
        self.filter = filter

    def eval(self, context):
        if self.filter is None:
            return context_length(context)
        else:
            return np.sum(expr_eval(self.filter, context))

    def dtype(self, context):
        return int

    def collect_variables(self, context):
        if self.filter is None:
            return set()
        else:
            return collect_variables(self.filter, context)

    def __str__(self):
        filter = str(self.filter) if self.filter is not None else '' 
        return "grpcount(%s)" % filter


#TODO: transform this into a CompoundExpressionn
class GroupAverage(FunctionExpression):
    func_name = 'grpavg'
    
    def eval(self, context):
        expr = self.expr
        if self.filter is not None:
            filter = expr_eval(self.filter, context)
            #FIXME: use usual temporary variable method
            context = context.copy()
            context['__tmpfilter__'] = filter
            expr = Where(Variable('__tmpfilter__'), expr, 0)
        else:
            filter = True
        values = expr_eval(expr, context)
        filter &= np.isfinite(values)
        numrows = np.sum(filter)
        if numrows:
            return np.nansum(values) / float(numrows)
        else:
            return float('nan')

    def dtype(self, context):
        return float



class NumexprFunctionProperty(FunctionExpression):
    '''For functions which are present as-is in numexpr'''
    def as_string(self, context):
        return '%s(%s)' % (self.func_name, as_string(self.expr, context))

class Abs(NumexprFunctionProperty):
    func_name = 'abs'
    
class Log(NumexprFunctionProperty):
    func_name = 'log'

    def dtype(self, context):
        return float
            
class Exp(NumexprFunctionProperty):
    func_name = 'exp'

    def dtype(self, context):
        return float


class CreateIndividual(EvaluableExpression):
    def __init__(self, entity_name=None, filter=None, number=None, **kwargs):
        self.entity_name = entity_name
        self.filter = filter
        self.kwargs = kwargs
        self.number = number
#        assert filter is not None and number is None or \
#               number is not None and filter is None

    def _initial_values(self, array, to_give_birth, num_birth):
        #TODO: use default values for fields which have one
        children = np.empty(num_birth, dtype=array.dtype)
        children.fill(get_missing_record(array))
        return children

    def eval(self, context):
        source_entity = context['__entity__']
        if self.entity_name is None:
            target_entity = source_entity
        else:
            target_entity = entity_registry[self.entity_name]
        array = target_entity.array
        ctx_filter = context.get('__filter__')

        if self.filter is not None and ctx_filter is not None:
            filter_expr = ctx_filter & self.filter
        elif self.filter is not None:
            filter_expr = self.filter
        elif ctx_filter is not None:
            filter_expr = ctx_filter
        else:
            filter_expr = None
            
        if filter_expr is not None:
            to_give_birth = expr_eval(filter_expr, context)
            num_birth = to_give_birth.sum()
        elif self.number is not None:
            to_give_birth = None
            num_birth = self.number
        else:
            raise Exception('no filter nor number in "new"')

        print "%d new %s(s)" % (num_birth, target_entity.name),
        id_to_rownum = target_entity.id_to_rownum

        num_rows = len(array)
        num_individuals = len(id_to_rownum)

        children = self._initial_values(array, to_give_birth, num_birth)
        children['id'] = np.arange(num_individuals, num_individuals + num_birth)
            
        if num_birth:
            period = context['period']
            children['period'] = period
    
            used_variables = set()
            for v in self.kwargs.itervalues():
                used_variables.update(collect_variables(v, context))
            
            # ideally we should be able to just say:    
            # child_context = context[to_give_birth]
            child_context = {'period': period, 
                             '__len__': num_birth,
                             '__entity__': source_entity}
            for varname in used_variables:
                value = context[varname]
                if isinstance(value, np.ndarray):
                    value = value[to_give_birth]
                child_context[varname] = value
            for k, v in self.kwargs.iteritems():
                children[k] = expr_eval(v, child_context)
    
            target_entity.array = np.concatenate((array, children))
            temp_variables = target_entity.temp_variables
            for name, temp_value in temp_variables.iteritems():
                if isinstance(temp_value, np.ndarray) and temp_value.shape:
                    #TODO: use default value instead (or at least missing value)
                    extra = np.zeros(num_birth, dtype=temp_value.dtype)
                    temp_variables[name] = np.concatenate((temp_value, extra))
    
            extra_variables = context.extra
            for name, temp_value in extra_variables.iteritems():
                if isinstance(temp_value, np.ndarray) and temp_value.shape:
                    #TODO: use default value instead (or at least missing value)
                    extra = np.zeros(num_birth, dtype=temp_value.dtype)
                    extra_variables[name] = np.concatenate((temp_value, extra))
    
            id_to_rownum_tail = np.arange(num_rows, num_rows + num_birth)
            target_entity.id_to_rownum = np.concatenate((id_to_rownum, 
                                                         id_to_rownum_tail))
        
        # result is the ids of the new individuals corresponding to the source
        # entity
        if to_give_birth is not None: 
            result = np.empty(context_length(context), dtype=int)
            result.fill(-1)
            if source_entity is target_entity:
                to_give_birth = np.concatenate((to_give_birth, 
                                                np.zeros(num_birth, dtype=bool)))
            np.place(result, to_give_birth, children['id'])
            return result
        else:
            return None


class Clone(CreateIndividual):
    def __init__(self, filter=None, **kwargs):
        CreateIndividual.__init__(self, None, filter, None, **kwargs)

    def _initial_values(self, array, to_give_birth, num_birth):
        return array[to_give_birth]


class Dump(EvaluableExpression):
    def __init__(self, *args, **kwargs):
        self.expressions = args
        self.filter = kwargs.pop('filter', None)
        if len(args):
            assert all(isinstance(e, Expr) for e in args), \
                   "dump arguments must be expressions, not a list of them, " \
                   "or strings" 

    def eval(self, context):
        if self.filter is not None:
            filter_value = expr_eval(self.filter, context)
        else:
            filter_value = None

        if self.expressions:
            expressions = list(self.expressions)
        else:
            expressions = [Variable(name) for name in context.keys()]
        
        str_expressions = [str(e) for e in expressions]
        if 'id' not in str_expressions:
            str_expressions.insert(0, 'id')
            expressions.insert(0, Variable('id'))
            id_pos = 0
        else:
            id_pos = str_expressions.index('id') 
        columns = []
        for expr in expressions:
            expr_value = expr_eval(expr, context)
            if filter_value is not None and isinstance(expr_value, np.ndarray):
                expr_value = expr_value[filter_value]
            columns.append(expr_value)

        numrows = len(columns[id_pos])
            
        # expand scalar columns to full columns in memory
        for idx, col in enumerate(columns):
            dtype = None
            if not isinstance(col, np.ndarray):
                dtype = type(col) 
            elif not col.shape:
                dtype = col.dtype.type
            if dtype is not None:
                newcol = np.empty(numrows, dtype=dtype)
                newcol.fill(col)
                columns[idx] = newcol
 
        return utils.PrettyTable(chain([str_expressions], izip(*columns)))


functions.update({
    # links
    'countlink': CountLink,
    'sumlink': SumLink,
    'avglink': AvgLink,
    'minlink': MinLink,
    'maxlink': MaxLink,
    # random
    'uniform': Uniform,
    'normal': Normal,
    'choice': Choice,
    'randint': RandInt,
    # past data
    'value_for_period': ValueForPeriod,
    'lag': Lag,
    'duration': Duration,
    'tavg': TimeAverage,
    'tsum': TimeSum,
    # aggregates
    'grpcount': GroupCount,
    'grpmin': GroupMin,
    'grpmax': GroupMax,
    'grpsum': GroupSum,
    'grpavg': GroupAverage,
    'grpstd': GroupStd,
    # per element
    'min': Min,
    'max': Max,
    'abs': Abs,
    'clip': Clip,
    'zeroclip': ZeroClip,
    'round': Round,
    'trunc': Trunc,
    # misc
    'new': CreateIndividual,
    'clone': Clone,
    'dump': Dump,
})
