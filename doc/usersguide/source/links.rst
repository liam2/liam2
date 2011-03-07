.. highlight:: yaml

.. _links_label:

Links
=====

.. index:: links, many2one, one2many

Entities can be linked with each other or with other entities, for example, individuals *belong* to households, and mothers are
linked to their children, while partners are interlinked as well.

A typical link has the following form: ::

    name: {type: <type>, target: <entity>, field: <name link>}
    
LIAM 2 uses field values to establish the link between entities    

LIAM 2 allows for two types of links: 

- many2one
- one2many

many2one
--------

A **many2one** link the item of the entity to *one* other item in the same (eg. a person to its mother) or other entity(a person to its household).

This allows the modeler to use information stored in the linked entities. ::

    entities:
        person:
            fields:
                - age: int
                - mother_id: int
                - income : float
                - father_id: int
                - mother_age: int
                - parent_icome: float

            links:
                mother: {type: many2one, target: person, field: mother_id}
                father: {type: many2one, target: person, field: father_id}

            processes:
                age: "age + 1"
                mother_age: "mother.age"
                parent_income: "mother.income + father.income"
                

The process *mother_age* uses the link mother and assigns the age of the mother to the *mother_age*  field.
If an person has no mother assigned to him (here mother_id == -1) then the mother_age will be -1 as well.
The parent_icome of that individual will be *nan*.

You can use the *basic functions*  (abs, log, exp, ...) in your formulas.


one2many
--------

An **one2many** links an item in an entity to at least one other item in the same (eg. a person to its children) or other entity
(a household to its members). ::

    entities:
        household:
            fields:
               - nch0_15: int
            links:
                hp: {type: one2many, target: person, field: household_id}
                
        
        person:
            fields:
                - age: int
                - household_id : int 
            links:
                ph: {type: many2one, target: household, field: household_id}
                
- *hp* is the link from the household to its members.
- *ph* is the link form a person to the household.

To use information stored in the linked entities you have to use *aggregate functions*

- countlink (eg.  countlink(hp) gives the numbers of persons in the household )
- sumlink (eg. sumlink(hp, income) sums up all incomes from the members in a household)
- avglink (eg. avglink(hp, age) gives the average age of the members in a household)
- minlink, maxlink (eg. minlink(hp, age) gives the age of the youngest member of the household)


*example* ::

    entities:
        household:
            fields:
               - nch0_15: int
            links:
                hp: {type: one2many, target: person, field: household_id}
                
        
        person:
            fields:
                - age: int
                - dead: bool
                - m_id; int
                - partner_id: int
                - household_id : int 
                - civilstate: int  # 1=single, 2=married, 3=cohab, 4=divorced, 5=widowed
            links:
                pm: {type: many2one, target: person, field: m_id} # mother
                ps: {type: many2one, target: person, field: partner_id} #  partner
                ph: {type: many2one, target: household, field: household_id} # household
                mp: {type: one2many, target: person, field: m_id}  # mother children              
                
- *pm* is the link form a person to the mother
- *ps* is the link form a person to the spouse 
- *ph* is the link form a person to the household.
- *mp* is the link to the children of a mother
- *hp* is the link from the household to its members

So for example, the below command sets the variable *civilstate*. It checks whether the spouse is dead. If so, then the 
variable *civilstate* is set equal to 5 (widowed). If not, then nothing happens. ::

    - civilstate: "if(ps.dead, 5, civilstate)"

As another example, the below procedure sets a variable *to separate* to a  True if the variable *separate* is true for the
individual or for his or her partner. ::

    - to_separate: "separate or ps.separate"
                
As a third and last example, we can use the following two procedures on the level of the household to count the number of
children up to 16 ::

    - nch0_15: "countlink(hp, (age>=0) and (age <16))" 

Then for each individual, a variable denoting the number of children up to 16 in his or her household can be found by ::

    - number_of_kids:  ph.nch0_15 

Note however that the procedure *nch0_15* is simulated on the level of the household, while the procedure *number_of_kids* pertains to
the individual level.

Note, finally, that the variable *number_of_kids* could also have been simulated by just one procedure, on the individual level, being: ::

    - no_kids:  ph.get(countlink(persons, (age>=0) and (age <16)))"

