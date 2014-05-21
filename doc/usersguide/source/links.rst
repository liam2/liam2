.. highlight:: yaml

.. index:: links

.. _links_label:

Links
=====

Individuals can be linked with each other or with individuals of other
entities, for example, mothers are linked to their children, partners are
linked to each other and persons belong to households. 

A typical link declaration has the following form: ::

    name: {type: <type>, target: <entity>, field: <name of link field>}
    
LIAM2 uses **integer fields** to establish the link between entities. Those
integer fields contain the id-number of the linked individual.    

For link fields, -1 is a special value meaning the link points to nothing
(eg. a person has no partner). Other negative values **should never be used**
(whatever the reason) for link fields.

LIAM2 allows two types of links: many2one and one2many.

.. index:: many2one

many2one
--------

A **many2one link** links an individual of the entity to **one** other
individual in the same (eg. a person to his/her mother) or another entity (eg. a
person to its household).

This allows the modeller to use information stored in the linked entities. ::

    entities:
        person:
            fields:
                - age: int
                - income: float
                - mother_id: int
                - father_id: int
                - partner_id: int

            links:
                mother: {type: many2one, target: person, field: mother_id}
                father: {type: many2one, target: person, field: father_id}
                partner: {type: many2one, target: person, field: partner_id}

            processes:
                age: age + 1
                mother_age: mother.age
                parents_income: mother.income + father.income
                

To access a field of a linked individual (possibly of the same entity), you
use: ::

    link_name.field_name
    
For example, the *mother_age* process uses the 'mother' link to assign the age
of the mother to the *mother_age* field. If an individual's link does not point
to anything (eg. a person has no known mother), trying to use the link would
yield the missing value (eg. for orphans, mother.age is -1 and
parents_income is *nan*).

As another example, the process below sets a variable *to_separate* to *True* if
the variable *separate* is True for either the individual or his/her partner. ::

    - to_separate: separate or partner.separate

Note that it is perfectly valid to chain links as, for example, in: ::

    grand_parents_income: mother.mother.income + mother.father.income + 
                          father.mother.income + father.father.income  
        
Another option to get values in the linked individual is to use the form: ::

    link_name.get(expr)
    
this syntax is a bit more verbose in the simple case, but is much more powerful
as it allows to evaluate (almost) any expression on the linked individual. 

For example, if you want to get the average age of both parents of the mother
of each individual, you can do it so: ::

    mother.get((mother.age + father.age) / 2)

.. index:: one2many

one2many
--------

A **one2many link** links an individual of the entity to at least one other
individual in the same (eg. a person to his/her children) or another entity (a
household to its members). ::

    entities:
        household:
            links:
                persons: {type: one2many, target: person, field: household_id}
                
        person:
            fields:
                - age: int
                - income: float
                - household_id : int

            links:
                household: {type: many2one, target: household, field: household_id}
                
- *persons* is the link from the household to its members.
- *household* is the link form a person to his/her household.

To access the information stored in the linked individuals through a one2many
link, you have to use *aggregate methods* on the link: ::

    link_name.method_name([arguments])

For example: ::

    persons.avg(age)
    
one2many links support the following methods: count(), sum(), avg(), min() and
max(). See :ref:`link_methods` for details.

*example* ::

    entities:
        household:
            fields:
                - num_children: int

            links:
                # link from a household to its members
                persons: {type: one2many, target: person, field: household_id}

            processes:
                num_children: persons.count(age <= 17)
            
        person:
            fields:
                - age: int
                - household_id: int

            links:
                # link form a person to his/her household 
                household: {type: many2one, target: household,
                            field: household_id}

            processes:
                num_kids_in_hh: household.num_children 
                
                
The num_children process, once called will compute the number of persons aged
17 or less in each household and store the result in the *num_children* field
(of the **household**).
Afterwards, that variable can be used like any other variable, for example
through a many2one link, like in the *num_kids_in_hh* process. This process
computes for each **person**, the number of children in the household of that
person. 

Note that the variable *num_kids_in_hh* could also have been
simulated by just one process, on the "person" level, by using: ::

    - num_kids_in_hh: household.get(persons.count(age <= 17))

