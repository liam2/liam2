import unittest
import numpy as np
from numpy import array, dtype



class Test(unittest.TestCase):
    def setUp(self):
        self.context = {'age': array([20, 10, 35, 55]),
                        'dead': array([False, True, False, True])}

    def tearDown(self):
        pass

    def test_where(self):
        e = parse("where(dead, 1, 2)", autovariables=True)
        assert np.array_equal(e.eval(self.context), [2, 1, 2, 1])

    def test_grpmin(self):        
        e = parse("grpmin(age)", autovariables=True)
        assert e.eval(self.context) == 10

        e = parse("grpmin(where(dead, age + 15, age))", autovariables=True)
        assert e.eval(self.context) == 20


class FakeEntity(object):
    def __init__(self, name, links=None, data=None):
        self.name = name
        self.links = links
        self.array = data
        self.temp_variables = {}
        self.id_to_rownum = array([0, 1, 2, 3])

class TestLink(unittest.TestCase): 
    def setUp(self):
        hh_link = properties.Link('household', 'many2one', 'hh_id', 'household')
        mother_link = properties.Link('mother', 'many2one', 'mother_id', 
                                      'person')
        child_link = properties.Link('children', 'one2many', 'mother_id', 
                                     'person')
        person = FakeEntity('person', links={'household': hh_link,
                                             'mother': mother_link,
                                             'children': child_link})
        household = FakeEntity('household')
        entities.entity_registry.add(person)
        entities.entity_registry.add(household)
    
    def assertArrayEqual(self, first, other):
        assert np.array_equal(first, other), "got: %s\nexpected: %s" % (first, 
                                                                        other) 
            
    def test_many2one_from_dict(self):
        person = entities.entity_registry['person']
        person.array = {'period': array([2002, 2002, 2002, 2002]),
                        'age': array([55, 25, 22, 1])}
        
        context = {'mother_id': array([-1, 0, 0, 2]),
                   'period': 2002,
                   '__entity__': person}
        
        e = parse("mother.age", person.links, autovariables=True)
        self.assertArrayEqual(e.eval(context), [-1, 55, 55, 22])
        
    def test_many2one_from_entity_context(self):
        person = entities.entity_registry['person']
        dt = dtype([('id', int), ('age', int), ('dead', bool), 
                    ('mother_id', int), ('period', int)]) 
        person.array = array([(0, 55, True, -1, 2002),
                              (1, 25, False, 0, 2002),
                              (2, 22, False, 0, 2002),
                              (3, 1, False, 2, 2002)], dtype=dt)
        context = entities.EntityContext(person, {'period': 2002})
        self.assertEqual(entities.context_length(context), 4)
        e = parse("mother.age", person.links, autovariables=True)
        self.assertArrayEqual(e.eval(context), [-1, 55, 55, 22])

    def test_one2many_from_entity_context(self):
        person = entities.entity_registry['person']
        dt = dtype([('id', int), ('age', int), ('dead', bool), 
                    ('mother_id', int), ('period', int)]) 
        person.array = array([(0, 55, True, -1, 2002),
                              (1, 25, False, 0, 2002),
                              (2, 22, False, 0, 2002),
                              (3, 1, False, 2, 2002)], dtype=dt)
        context = entities.EntityContext(person, {'period': 2002})
        e = parse("countlink(children)", person.links, autovariables=True)
        self.assertArrayEqual(e.eval(context), [2, 0, 1, 0])

#    def test_past_one2many(self):
#        person = entities.entity_registry['person']
#        dt = dtype([('id', int), ('age', int), ('dead', bool), 
#                    ('mother_id', int), ('period', int)]) 
#        person.array = array([(0, 55, True, -1, 2002),
#                              (1, 25, False, 0, 2002),
#                              (2, 22, False, 0, 2002),
#                              (3, 1, False, 2, 2002)], dtype=dt)
#        context = entities.EntityContext(person, {'period': 2001})
#        e = parse("countlink(children)", person.links, autovariables=True)
#        self.assertArrayEqual(e.eval(context), [2, 0, 1, 0])


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