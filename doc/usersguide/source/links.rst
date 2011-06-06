.. highlight:: yaml

.. _links_label:

Links
=====

.. index:: links, many2one, one2many

Entities can be linked with each other or with other entities, for example, 
individuals *belong* to households, and mothers are linked to their children, 
while partners are interlinked as well.

A typical link has the following form: ::

    name: {type: <type>, target: <entity>, field: <integer field name>}
    
LIAM 2 uses field values to establish the link between entities.    

LIAM 2 allows for two types of links: 

- many2one
- one2many

many2one
--------

A **many2one** link the item of the entity to *one* other item in the same 
(eg. a person to its mother) or other entity(a person to its household).

This allows the modeller to use information stored in the linked entities. ::

    entities:
        person:
            fields:
                - age: int
                - income: float
                - mother_id: int
                - father_id: int
                - mother_age: int
                - parents_income: float

            links:
                mother: {type: many2one, target: person, field: mother_id}
                father: {type: many2one, target: person, field: father_id}

            processes:
                age: "age + 1"
                mother_age: "mother.age"
                parents_income: "mother.income + father.income"
                

The process *mother_age* uses the link mother and assigns the age of the mother to the *mother_age*  field.
If a person has no mother assigned to him (here mother_id == -1) then the mother_age will be -1 as well.
The parent_income of that individual will be *nan*.

You can use the *basic functions* (abs, log, exp, ...) in your formulas.


one2many
--------

A **one2many** links an item in an entity to at least one other item in the same 
(eg. a person to its children) or other entity (a household to its members). ::

    entities:
        household:
            fields:
                - num_children: int

            links:
                persons: {type: one2many, target: person, field: household_id}
                
        person:
            fields:
                - age: int
                - household_id : int
            links:
                household: {type: many2one, target: household, field: household_id}
                
- *persons* is the link from the household to its members.
- *household* is the link form a person to the household.

To use information stored in the linked entities you have to use *aggregate functions*

- countlink (eg. countlink(persons) gives the numbers of persons in the household)
- sumlink (eg. sumlink(persons, income) sums up all incomes from the members in a household)
- avglink (eg. avglink(persons, age) gives the average age of the members in a household)
- minlink, maxlink (eg. minlink(persons, age) gives the age of the youngest member of the household)


*example* ::

    entities:
        household:
            fields:
                - num_children_0_15: int
                - nch0_15: int

            links:
                # link from a household to its members
                persons: {type: one2many, target: person, field: household_id}
                
        person:
            fields:
                - age: int
                - age: int
                - dead: bool
                # 1: single, 2: married, 3: cohabitant, 4: divorced, 5: widowed 
                - civilstate: int
                
                - mother_id: int
                - partner_id: int
                - household_id: int
            links:
                mother: {type: many2one, target: person, field: mother_id}
                # link form a person to his/her spouse 
                partner: {type: many2one, target: person, field: partner_id}
                household: {type: many2one, target: household,
                            field: household_id}
                # link from a mother to her children
                mother_children: {type: one2many, target: person, 
                                  field: mother_id}              
                
So for example, the command below sets the variable *civilstate*. It checks 
whether the spouse is dead. If so, the variable *civilstate* is set to 5 
(widowed), otherwise nothing happens (it is set to its previous value). ::

    - civilstate: "if(partner.dead, 5, civilstate)"

As another example, the process below sets a variable *to_separate* to *True* if
the variable *separate* is True for the individual or for his or her partner. ::

    - to_separate: "separate or partner.separate"
                
As a third and last example, we can use the following two procedures on the
level of the household to count the number of children up to 16 ::

    - num_children_0_15: "countlink(persons, (age >= 0) and (age < 16))" 

Then for each individual, a variable denoting the number of children up to 16 in
his or her household can be found by ::

    - number_of_kids: "household.num_children_0_15" 

Note however that the process *num_children_0_15* is simulated on the level of
the "household", while the process *number_of_kids* pertains to the "person"
level.

Note, finally, that the variable *number_of_kids* could also have been
simulated by just one process, on the "person" level, by using: ::

    - num_kids: "household.get(countlink(persons, (age >= 0) and (age < 16)))"

