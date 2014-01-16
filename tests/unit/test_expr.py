import unittest

import numpy as np
from numpy import array

from context import EntityContext, context_length
from entities import Entity
from expr import Variable
from exprtools import parse
import links
from registry import entity_registry


class ArrayTestCase(unittest.TestCase):
    def assertArrayEqual(self, first, other):
        assert np.array_equal(first, other), "got: %s\nexpected: %s" % (first, 
                                                                        other) 


class StringTestCase(ArrayTestCase):
    context = None
    parsing_context = None

    def assertEvalEqual(self, s, result):
        e = parse(s, self.parsing_context)
        self.assertArrayEqual(e.evaluate(self.context), result)


class Test(StringTestCase):
    def setUp(self):
        self.context = {'age': array([20, 10, 35, 55]),
                        'dead': array([False, True, False, True])}
        self.parsing_context = {
            'person': {'age': Variable('age'), 'dead': Variable('dead')},
            '__entity__': 'person'
        }

    def tearDown(self):
        pass

    def test_where(self):
        self.assertEvalEqual("where(dead, 1, 2)", [2, 1, 2, 1])

    def test_min(self):        
        self.assertEvalEqual("min(age)", 10)
        self.assertEvalEqual("min(where(dead, age + 15, age))", 20)


class FakeEntity(object):
    def __init__(self, name, links=None, data=None):
        self.name = name
        self.links = links
        self.array = data
        self.array_period = None
        self.temp_variables = {}
        self.id_to_rownum = array([0, 1, 2, 3])


class TestLink(ArrayTestCase): 
    def setUp(self):
        hh_link = links.Many2One('household', 'hh_id', 'household')
        mother_link = links.Many2One('mother', 'mother_id', 'person')
        child_link = links.One2Many('children', 'mother_id', 'person')
        persons_link = links.One2Many('persons', 'hh_id', 'person')

        dt = np.dtype([('period', int), ('id', int), ('age', int),
                       ('dead', bool),  ('mother_id', int), ('hh_id', int)])
#TODO: I can't use an EntityContext with an array containing several periods
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
#         person = FakeEntity('person',
#                             links={'household': hh_link,
#                                    'mother': mother_link,
#                                    'children': child_link},
#                             data=persons)

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
#         household = FakeEntity('household',
#                                links={'persons': persons_link},
#                                data=households)
        household = Entity('household',
                           links={'persons': persons_link},
                           array=households)
        entity_registry.add(person)
        entity_registry.add(household)

#    def test_many2one_from_dict(self):
#        person = entity_registry['person']
#        context = {'mother_id': array([-1, 0, 0, 2]),
#                   'period': 2002,
#                   '__entity__': person}
#        e = parse("mother.age", person.links, autovariables=True)
#        self.assertArrayEqual(e.evaluate(context), [-1, 55, 55, 22])
        
    def test_many2one_from_entity_context(self):
        person = entity_registry['person']
        context = EntityContext(person, {'period': 2002})
        self.assertEqual(context_length(context), 5)
        e = parse("mother.age", person.links, autovariables=True)
        self.assertArrayEqual(e.evaluate(context), [-1, 55, 55, -1, 22])

    def test_one2many_from_entity_context(self):
        person = entity_registry['person']
        context = EntityContext(person, {'period': 2002})
        e = parse("countlink(children)", person.links, autovariables=True)
        self.assertArrayEqual(e.evaluate(context), [2, 0, 1, 0, 0])

#    def test_past_one2many(self):
#        person = entity_registry['person']
#        context = EntityContext(person, {'period': 2001})
#        e = parse("countlink(children)", person.links, autovariables=True)
#        self.assertArrayEqual(e.evaluate(context), [2, 0, 1, 0])

#    def test_multi_entity_in_context_experiment(self):
#        person = entity_registry['person']
#        person.array = {'period': array([2002, 2002, 2002, 2002]),
#                        'age': array([55, 25, 22, 1])}
#        
#        data = {'person': {'id':         [ 0,  1,  2, 3,  4],
#                           'mother_id':  [-1,  0,  0, 2, -1],
#                           'age':        [55, 25, 22, 1, 45],
#                           'hh_id':      [ 0,  1,  2, 2,  0],
#                           'period':     2002,
#                           '__entity__': person},
#                'household': {'id': [0, 1, 2]}}
#        
#        annoted_expr = person.parse("household.get(persons.count())")
#        # this would be more or less equivalent to:
#        # expr = parse(expr, person.links)
#        # return AnnotedExpr(expr, self)
#        self.assertArrayEqual(annoted_expr.evaluate(context), [2, 1, 2, 2, 2])

#        OR

#        context = {'data': data, 'period': 2002, 'entity': 'person'}
#        expr = parse(expr, person.links)
#        self.assertArrayEqual(expr.evaluate(context), [2, 1, 2, 2, 2])

# pro: - I could pass the context around almost unmodified when changing from
#        an entity to another
#      - easier testing 
# con: - I can't pass the context as-is to numexpr


# short-term data provider contract:

#  data[entity][period][column] = vector
#  examples:
#  data['person'][2002]['period'] = [2002, 2002, 2002, 2002] 
#  data['person'][2002]['id'] = [0, 1, 3, 5] 
#  data['person'][2002]['age'] = [25, 45, 1, 37] 
#  data['person'][2002]['age'][0] = 25 

# mid-term 

#  data[entity][period_start:period_end][colname] = vector
#  data[entity][period_start:period_end][rownum] = row
#  data[entity][period_start:period_end][rownum][colname] = value
#  data[entity][period][column].get(id) = value

# long-term

#  data[entity][column == value][column] = vector
#  data[entity][time_obj][column] = vector # < use pandas dataframes?
#  data[entity][column][time_obj] = vector # < use pandas dataframes?
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