import os
import csv

import numpy as np

from expr import functions, Expr, expr_eval
import simulation

class Action(Expr):
    def __init__(self, filter=None):
        self.filter = filter
    
    def run(self, context):
        raise NotImplementedError
    
    def eval(self, context):
        self.run(context)


class Print(Action):
    def __init__(self, *args):
        Action.__init__(self)
        self.args = args

    def run(self, context):
        values = [expr_eval(expr, context) for expr in self.args]
        print ' '.join(str(v) for v in values),


class CSV(Action):
    def __init__(self, expr, suffix=''):
        self.expr = expr
        #TODO: make base class for Dump & GroupBy
#        assert isinstance(expr, (Dump, GroupBy))
        self.suffix = suffix

    def run(self, context):
        entity = context['__entity__']
        period = context['period']
        if self.suffix:
            fname = "%s_%d_%s.csv" % (entity.name, period,
                                      self.suffix)
        else:
            fname = "%s_%d.csv" % (entity.name, period)

        print "writing to", fname, "...",          
        file_path = os.path.join(simulation.output_directory, fname)
        with open(file_path, "wb") as f:
            data = expr_eval(self.expr, context)
            dataWriter = csv.writer(f)
            dataWriter.writerows(data)


class RemoveIndividuals(Action):
    def __init__(self, filter):
        # the only difference with Action is that filter is mandatory
        Action.__init__(self, filter)

    def run(self, context):
        filter = expr_eval(self.filter, context)

        not_removed = ~filter
        if np.all(not_removed):
            return

        entity = context['__entity__']
        already_removed = entity.id_to_rownum == -1
        already_removed_indices = already_removed.nonzero()[0]
        already_removed_indices_shifted = already_removed_indices - \
                                  np.arange(len(already_removed_indices))

        # recreate id_to_rownum from scratch
        id_to_rownum = np.arange(len(entity.array))
        id_to_rownum -= filter.cumsum()
        id_to_rownum[filter] = -1
        entity.id_to_rownum = np.insert(id_to_rownum,
                                        already_removed_indices_shifted,
                                        -1)
        entity.array = entity.array[not_removed]
        temp_variables = entity.temp_variables
        for name, temp_value in temp_variables.iteritems():
            if isinstance(temp_value, np.ndarray) and temp_value.shape:
                temp_variables[name] = temp_value[not_removed]

        print "%d %s(s) removed" % (filter.sum(), entity.name),
    
functions.update({
    # can't use "print" in python 2.x because it's a keyword, not a function        
#    'print': Print,
    'csv': CSV,
    'show': Print,
    'remove': RemoveIndividuals
})