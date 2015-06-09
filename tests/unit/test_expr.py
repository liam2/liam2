import unittest

import numpy as np
from numpy import array

from context import EvaluationContext
from entities import Entity
from expr import Variable
from exprtools import parse
import links


class ArrayTestCase(unittest.TestCase):
    def assertArrayEqual(self, first, other):
        assert np.array_equal(first, other), "got: %s\nexpected: %s" % (first, 
                                                                        other)


def evaluate(s, parse_ctx, eval_ctx):
    expr = parse(s, parse_ctx)
    return expr.evaluate(eval_ctx)


class StringExprTestCase(ArrayTestCase):
    parse_ctx = None
    eval_ctx = None

    def evaluate(self, s):
        return evaluate(s, self.parse_ctx, self.eval_ctx)

    def assertEvalEqual(self, s, result):
        self.assertArrayEqual(self.evaluate(s), result)


class TestSimple(StringExprTestCase):
    def setUp(self):
        data = {'person': {'age': array([20, 10, 35, 55]),
                           'dead': array([False, True, False, True])}}
        self.eval_ctx = EvaluationContext(entity_name='person',
                                          entities_data=data)
        self.parse_ctx = {
            'person': {'age': Variable('age'), 'dead': Variable('dead')},
            '__entity__': 'person'
        }

    def test_where(self):
        self.assertEvalEqual("where(dead, 1, 2)", [2, 1, 2, 1])

    def test_min(self):        
        self.assertEvalEqual("min(age)", 10)
        self.assertEvalEqual("min(where(dead, age + 15, age))", 20)


class TestLink(StringExprTestCase):
    def setUp(self):
        entities = {}

        hh_link = links.Many2One('household', 'hh_id', 'household')
        mother_link = links.Many2One('mother', 'mother_id', 'person')
        child_link = links.One2Many('children', 'mother_id', 'person')
        persons_link = links.One2Many('persons', 'hh_id', 'person')

        dt = np.dtype([('period', int), ('id', int), ('age', int),
                       ('dead', bool),  ('mother_id', int), ('hh_id', int)])
# TODO: I can't use an EntityContext with an array containing several periods
#      of data
#        persons = array([(2000, 0, 53, False, -1, 0),
#                         (2000, 1, 23, False,  0, 1),
#                         (2000, 2, 20, False,  0, 2),
#                         (2000, 3, 43, False, -1, 3), 
#                         (2001, 0, 54,  True, -1, 0),
#                         (2001, 1, 24, False,  0, 1),
#                         (2001, 2, 21, False,  0, 2),
#                         (2001, 3, 44, False, -1, 0), # they got married 
#                         (2001, 4,  0, False,  2, 2),
        persons = array([(2002, 0, 55,  True, -1, 0),
                         (2002, 1, 25, False,  0, 1),
                         (2002, 2, 22, False,  0, 2),
                         (2002, 3, 45, False, -1, 0),
                         (2002, 4,  1, False,  2, 2)],
                        dtype=dt)
        person = Entity('person',
                        links={'household': hh_link,
                               'mother': mother_link,
                               'children': child_link},
                        array=persons)

        dt = np.dtype([('period', int), ('id', int)])
#        households = array([(2000, 0),
#                            (2000, 1),
#                            (2000, 2),
#                            (2000, 3),
#                             
#                            (2001, 0),
#                            (2001, 1),
#                            (2001, 2),
                            
        households = array([(2002, 0),
                            (2002, 1),
                            (2002, 2)],
                           dtype=dt)
        household = Entity('household',
                           links={'persons': persons_link},
                           array=households)
        entities['person'] = person
        entities['household'] = household
        self.entities = entities

        parse_ctx = {'__globals__': {}, '__entities__': entities,
                     '__entity__': 'person'}
        parse_ctx.update((entity.name, entity.all_symbols(parse_ctx))
                         for entity in entities.itervalues())
        self.parse_ctx = parse_ctx
        self.eval_ctx = EvaluationContext(entities=entities, period=2002,
                                          entity_name='person')

    def test_many2one(self):
        self.assertEvalEqual("mother.age", [-1, 55, 55, -1, 22])

    def test_one2many(self):
        self.assertEvalEqual("children.count()", [2, 0, 1, 0, 0])

    # def test_past_one2many(self):
    #     self.eval_ctx.period = 2001
    #     self.assertEvalEqual("children.count()", [2, 0, 1, 0, 0])

# short-term data provider contract:

# Q: where do I plug the CascadingContext I need?
# A: a CascadingContext per entity, but what about periods?
#      - no CascadingContext anywhere in the data provider:
#     we build an EntityContext for the last Context in the chain (the one
#     wrapping the entity "stored" data)
#     it is the duty of the entity context to transform/present a "full context"
#     (data (dataprovider) + current_entity + current_period) as a single
#     "structured array"

# Q: do I need to have access to temporary fields from other entities?
# A: I think so, at least for temporary globals. The fact that they are stored
#    or not is irrelevant.
#    => this implies that they need to be stored in the
#
# Q: what about "definitions" and other metadata stored in Entity? If I only
# have the entity name in the context, I either need to keep a global entity
# registry or add the registry (or a similar structure) in the context

# Q: what about id_to_rownum? append(), keep()
#      table.append()/table.flush()

# A: the array vs table dichotomy (or whatever we call them: working_set vs
#    history) *needs* to be somehow exposed in the object we pass around

# Q: what exactly do I really need for the history?
# A: I need lag values + fields explicitly saved

# Q: is it even possible to hide id_to_rownum in the implementation?
# A: for align_link, it seems hard to do. Can't tell if it is possible or not

# * for fill_missing_values, I need MyArray[ids] = values
#    OR do I want to keep normal indexing? In that case MyArray.set(ids, values)
#    should work
# * for M2O links, it works: I need MyThing[col][ids] where ids can contain
#   duplicates, -1 and invalid ids (ids which map to -1) (like hh_id)
# * for matching, I need MyThing.set(id, value)
# * for O2M links and tfunc, it seems more complicated, I'm not sure whether
#   it is possible to do, but I am sure that even if it is possible, it would
#   require a complete rewrite of those functions. So it might be a better idea
#   to have indeed id_to_rownum inside the array, but still have access to it:
#   array.index or array.id_to_rownum

# Q: where (in which layer) do I plug the fill_missing_values functionality
#   (which requires knowledge of the "current" period)???
# A: array or context or ???


# !! the context itself needs to support append, because append must happen for
#    local variables too !!
# the interface is tricky: the simpler would be to pass a complete array that is
# simply appended but what about the "id" column? I cannot initialize it without
# knowing what is the last id ever created
# A: it's not a huge problem: simply initialize it in the append method.
#    if an id column is passed, overwrite it, if not add one

# Q: I think I need more/something different than just cascading contexts:
#    when a function calls another function, I don't need/want the context of
#    the caller to be available to the callee. It only needs access to global
#    variables and "globals":
#    I need entity_globals/method_locals/system_or_extra
# * entity_globals can be stored on disk or not
# * context.child() needs to copy (or share) entity_globals but not
#   locals or extra
# * context needs append & keep/remove methods

# Q: General architecture
# A1: EntityContext which dispatches to 3 IndexedColumnArray (globals
#     / locals / extra)
#    > hmm, not great as they should share the same index
#    > what about disk storage? (ie array vs table)
#    > what about lag variables?
# A2: EntityContext dispatches to 3 ColumnArray and one index

#  data[entity][period][column] = vector
#  examples:
#  data['person'][2002]['period'] = [2002, 2002, 2002, 2002] 
#  data['person'][2002]['id'] = [0, 1, 3, 5] 
#  data['person'][2002]['age'] = [25, 45, 1, 37] 
#  data['person'][2002]['age'][0] = 25 

# mid-term 

#  data[entity][period_start:period_end][colname]
#      = 2d array (num_periods * num_ids)
#      OR?
#      = vector with all periods concatenated (like when I filter in vitables)
#  data[entity][period_start:period_end][rownum]
#      = struct_array (num_periods * num_columns)
#      OR?
#      = single row
#  data[entity][period_start:period_end][rownum][colname] = vector (num_periods)
#  data[entity][period][column].get(id) = value

# long-term

#  data[entity][column == value][colname] = 2d array (num_ids x num_periods)
#  data[entity][time_obj] = 1d array (vector) with structured dtype (ie several
#                           columns) # < use pandas dataframes?
#  data[entity][colname][time_obj] = vector # < use pandas dataframes?
#  data[entity][colname] = sort of 2d array # (time x individuals)
#  >>> 2darray[time_obj] = vector of individuals
#  >>> 2darray[id] = time series for individual
#  >>> 2darray.XXX == 2darray[current_time].XXX ?
#  >>> 2darray.sum() == 2darray.sum(axis=ind) == scalar

#  >>> 2darray.sum(axis=time) == 1darray (N individuals) ? >>> needs to know
#      the function before we can know what "2darray.XXX" mean
#  OR
#  >>> 2darray[:] == 2darray[time.all()] == fake 2d array
#  >>> 2darray[:].sum(axis=time) == 1darray (N individuals)

#  examples:
#  data['person'][Person.period == 2002]['id'] = [0, 1, 3, 5] 
#  data['person'][Person.period == 2002]['age'] = [25, 45, 1, 37] 
#  data['person'][Person.period == 2002]['age'][0] = 25
#  age for id == 5 is 37
#  data['person'][Person.period == 2002]['age'].get(5) = 37


if __name__ == "__main__":
    from os import path
    import sys

    # append "../" to sys.path
    test_dir = path.dirname(path.abspath(__file__))
    root_dir = path.dirname(test_dir)
    sys.path.append(root_dir)

    unittest.main()