import unittest
import numpy as np
from numpy import array, dtype

class ArrayTestCase(unittest.TestCase):
    def assertArrayEqual(self, first, other):
        assert np.array_equal(first, other), "got: %s\nexpected: %s" % (first, 
                                                                        other) 

class Test(ArrayTestCase):
    def setUp(self):
        self.context = {'age': array([20, 10, 35, 55]),
                        'dead': array([False, True, False, True])}

    def tearDown(self):
        pass

    def test_where(self):
        e = parse("where(dead, 1, 2)", autovariables=True)
        self.assertArrayEqual(e.eval(self.context), [2, 1, 2, 1])

    def test_grpmin(self):        
        e = parse("grpmin(age)", autovariables=True)
        self.assertEqual(e.eval(self.context), 10)

        e = parse("grpmin(where(dead, age + 15, age))", autovariables=True)
        self.assertEqual(e.eval(self.context), 20)


class FakeEntity(object):
    def __init__(self, name, links=None, data=None):
        self.name = name
        self.links = links
        self.array = data
        self.temp_variables = {}
        self.id_to_rownum = array([0, 1, 2, 3])

class TestLink(ArrayTestCase): 
    def setUp(self):
        hh_link = properties.Link('household', 'many2one', 'hh_id', 'household')
        mother_link = properties.Link('mother', 'many2one', 'mother_id', 
                                      'person')
        child_link = properties.Link('children', 'one2many', 'mother_id', 
                                     'person')
        persons_link = properties.Link('persons', 'one2many', 'hh_id', 
                                       'person')

        dt = dtype([('period', int), ('id', int), ('age', int), ('dead', bool), 
                    ('mother_id', int), ('hh_id', int)])
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
        person = FakeEntity('person',
                            links={'household': hh_link,
                                   'mother': mother_link,
                                   'children': child_link},
                            data=persons)

        dt = dtype([('period', int), ('id', int)])
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
        household = FakeEntity('household',
                               links={'persons': persons_link},
                               data=households)
        entities.entity_registry.add(person)
        entities.entity_registry.add(household)

#    def test_many2one_from_dict(self):
#        person = entities.entity_registry['person']
#        context = {'mother_id': array([-1, 0, 0, 2]),
#                   'period': 2002,
#                   '__entity__': person}
#        e = parse("mother.age", person.links, autovariables=True)
#        self.assertArrayEqual(e.eval(context), [-1, 55, 55, 22])
        
    def test_many2one_from_entity_context(self):
        person = entities.entity_registry['person']
        context = entities.EntityContext(person, {'period': 2002})
        self.assertEqual(entities.context_length(context), 5)
        e = parse("mother.age", person.links, autovariables=True)
        self.assertArrayEqual(e.eval(context), [-1, 55, 55, -1, 22])

    def test_one2many_from_entity_context(self):
        person = entities.entity_registry['person']
        context = entities.EntityContext(person, {'period': 2002})
        e = parse("countlink(children)", person.links, autovariables=True)
        self.assertArrayEqual(e.eval(context), [2, 0, 1, 0, 0])

#    def test_past_one2many(self):
#        person = entities.entity_registry['person']
#        context = entities.EntityContext(person, {'period': 2001})
#        e = parse("countlink(children)", person.links, autovariables=True)
#        self.assertArrayEqual(e.eval(context), [2, 0, 1, 0])

#    def test_multi_entity_in_context_experiment(self):
#        person = entities.entity_registry['person']
#        person.array = {'period': array([2002, 2002, 2002, 2002]),
#                        'age': array([55, 25, 22, 1])}
#        
#        data = {'person': {'id':        [ 0,  1,  2, 3,  4],
#                              'mother_id': [-1,  0,  0, 2, -1],
#                              'age':       [55, 25, 22, 1, 45],
#                              'hh_id':     [ 0,  1,  2, 2,  0],
#                              'period': 2002,
#                              '__entity__': person},
#                   'household': {'id': [0, 1, 2]}}
#        
#        annoted_expr = person.parse("household.get(countlink(persons))")
#        # this would be more or less equivalent to:
#        # expr = parse(expr, person.links)
#        # return AnnotedExpr(expr, self)
#        self.assertArrayEqual(annoted_expr.eval(context), [2, 1, 2, 2, 2])

#        OR


#        context = {'data': data, 'period': 2002, 'entity': 'person'}
#        expr = parse(expr, person.links)
#        self.assertArrayEqual(expr.eval(context), [2, 1, 2, 2, 2])

#        #pro: I could pass the context around unmodified when changing from
#        #     an entity to another
#        #con: i can't pass the context as-is to numexpr


#  data provider contract:

#  data[entity][column == value][column] = vector
#  examples:
#  data['person'][period == 2002]['id'] = [0, 1, 3, 5] 
#  data['person'][period == 2002]['age'] = [25, 45, 1, 37] 
#  data['person'][period == 2002]['age'][0] = 25 
#  data['person'][period == 2002]['age'].get(5) = 37

if __name__ == "__main__":
    from os import path
    import sys
    # root_dir = "../"
    test_dir = path.dirname(path.abspath(__file__))
    root_dir = path.dirname(test_dir)
    sys.path.append(root_dir)

    import entities
        
    # needed for the different properties functions to be available
    import properties 
    from expr import Variable, parse

    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()