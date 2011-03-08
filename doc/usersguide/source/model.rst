.. highlight:: yaml

Model Definition
################

Microsimulation (acronym for microanalytic simulation) is a modelling technique that operates at the level of
individual *entities* such as persons, households, vehicles or firms. Within the model each entity is represented by a
record containing a unique identifier and a set of associated attributes (e.g. age, gender, work state, civil state, ...)
characteristics. A set of *processes* are then applied to these entities leading to simulated
changes in state and behaviour. These rules may be deterministic (probability = 1), such as changes in tax
liability resulting from changes in tax regulations, or stochastic (probability <=1), such as chance of dying,
marrying, giving birth or moving within a given time period. .

To define the model, we have to describe the different *entities*, the way they interact (*links*) and how they
behave (*processes*) over time. This is done in one file. We use the YAML-markup language. This format uses the level
of indentation to specify objects and sub objects.

In a LIAM 2 model file, all text following a # is considered to be comments, and therefore ignored. 

A LIAM 2 model has the following structure: ::

    globals:
        ...

    entities:
        ...

    simulation:
        ...
        
globals
=======

The *globals* are variables (aka. parameters) that do not relate to the *entities* defined in the model. They can be used in
expressions across all entities.

A global is a (periodic = time-varying) number that is global (hence the name) to all *entities*. For example, the retirement age for women in Belgium has
been gradually increasing from 61 in 1997 via 63 from 2003 onward, up to 65 in 2009. A global variable WEMRA has therefore
been included.::

    globals:
        periodic:
            - WEMRA: float

And it can then be used in the procedures, for example  ::

    workstate: "if(age >= WEMRA, 9, workstate)"

This changes the workstate of the individual to retired (9) if the age is higher than the required retirement age in that year.        

entities
========

Each entity has a unique identifier and a set of attributes (**fields**). You can use different entities in one model. You can
define the interaction between members of the same entity (eg. between partners) or among different entities (eg. a person and its
household) using the *links*. 

The **processes** describe how the entities behave. The order is not important. In the **simulation** block you define if and
when they have to be executed, this allows to simulate processes of different entities in the order you want.


LIAM 2 declares the entities as follows: ::

    entities:
        entity-name1:
            fields:  
                fields definition
            
            links:   
                links definition
                
            macros:
                macros definition
                
            processes:   
                processes definition
                
        entity-name2:
            ...
            
We use YAML as the description language. Indentation and the use of ":" are important. 

fields
------

The fields hold the information of each member in the entity. That information is global in a run of the model. Every
process defined in that entity can use and change the value. 

LIAM 2 handles three types of fields:

- bool: boolean (True or False)
- int: integer
- float: real number

There are two implicit fields that do not have to be defined:

- id: the unique identifier of the item
- period: the current period in the run of the program

*example* ::

    entities:
        person:
            fields:
                # period and id are implicit
                - age:          int
                - dead:         bool
                - gender:       bool
                - partner_id:   int
                # 1=single, 2=married, 3=cohabitant, 4=divorced, 5=widowed
                - civilstate:   int

This example defines the entity person. Each person has an age, gender, is dead or not, has a civil_state, possibly a partner. We
use the field civilstate to store the marital status as a switch of values.

The data is stored in a hdf5-data set. Not all variables defined in fields have values in the starting period. Some variables are
defined in the fields set but are calculated later by LIAM 2 (example below *agegroup_work*).

Other variables are *observed* in that their value in the starting period can be found in the data set supplied. The
observed values of the other variables in this example are not available and will therefore have to be produced by the model
(eg. below *agegroup_work*).


*example* ::

    entities:
        person:
            fields:
                # period and id are implicit
                - age:          int
                - dead:         bool
                - gender:       bool
                - partner_id:   int
                # 1=single, 2=married, 3=cohabitant, 4=divorced, 5=widowed
                - civilstate:   int
                - agegroup_work: {type: int, initialdata: false}


Note that a field name is not reserved to one entity. 


links
-----
Entities can be linked with each other or with other entities, for example, individuals ‘belong’ to households, and mothers are
linked to their children, while spouses are interlinked as well.

.. index:: links, many2one, one2many

Entities can be linked with each other or with other entities, for example, individuals *belong* to households, and mothers are
linked to their children, while partners are interlinked as well.

A typical link has the following form: ::

    name: {type: <type>, target: <entity>, field: <name link>}
    
LIAM 2 uses field values to establish the link between entities    

LIAM 2 allows for two types of links: 

- many2one
- one2many

More detail, see :ref:`links_label`.

macros
------

Macros are a way to make the code easier to read and maintain. They are defined on the entity level.
Macros are re-evaluated wherever they appear. Use *capital* letters to define macros.

*example* ::

    entities:
        person:
            fields:
                - age: int
          
            macros:
                ISCHILD: "age < 18"

            processes:
                test_macros: 
                    - ischild: "age < 18"
                    - before1: "if(ischild, 1, 2)"
                    - before2: "if(ISCHILD, 1, 2)"  # before1 == before2
                    - age: "age + 1"
                    - after1: "if(ischild, 1, 2)"
                    - after2: "if(ISCHILD, 1, 2)"   # after1 != after2 
                    
    simulation:
        processes:
            - person: [test_macros]

                    
The above example does

- ischild: creates a temporary variable *ischild* and sets it to *True* if the age of the person is under 18 and to *False* if not
- before1: creates a temporary variable *before1* and sets it to 1 if the value of the temporary variable *ischild* is *True* and to 2 if not.
- before2: creates a temporary variable *before2* and sets it to 1 if the value age < 18 is *True* and to 2 if not
- age: the age is changed
- after1: creates a temporary variable *after1* and sets it to 1 if the value of the temporary variable *ischild* is *True* and to 2 is not.
- after2: creates a temporary variable *after2* and sets it to 1 if the value age < 18 is *True* and to 2 if not.

It is clear that after1 != after2 since the age has been changed and *ischild* has not been updated since.


processes
---------

Here you define the processes you will need in the model. 

More detail, see :ref:`processes_label`.


simulation
==========

The *simulation* block includes the location of the datasets (**input**, **output**), the number of periods and
the start period. It sets what processes defined in the **entities** blook are simulated (since some can be
ommitted), and the order in which this is done.

Suppose that we have a model that starts in 2002 and has to simulate for 10 periods. Furthermore, suppose that we have two
object or entities: individuals and households. The model starts by some initial processes (grouped under the header *init*)
that precede the actual prospective simulation of the model, and that only apply to the observed dataset in 2002. These
initial simulations can pertain to the level of the individual or the household. Use the *init* block to calculate variables
for the starting period.

The prospective part of the model starts by a number of sub-processes setting the household size and composition. Next, two
processes apply on the level of the individual, changing the age and agegroup. Finally, mortality and fertility are
simulated. Seeing that this changes the numbers of individuals in households, the process establishing the household size
and composition is again used.

*example* ::

    simulation: 
        init:
            - household: [household_composition]
            - person: [agegroup]
    
        processes:  
            - household: [household_composition]
            - person: [
                   age, agegroup,
                   dead_procedure, birth
               ]
            - household: [household_composition]

        input:      
            path: "liam2"
            file: "base.h5"
        output:
            path: "liam2"
            file: "simulation.h5"
        start_period:   2002
        periods:    10



processes
---------

This block defines what processes are executed each period starting from *start_period* for *periods* times. 
Since processes change values of items in an entity, you have to specify the entity. Note that you can 
execute the same process more than once during a simulation and that you can switch between entities in the
simulation of a period. 

In the example you see that after birth and dead_procedure, the household_composition is re evaluated.

init
----

Every process specified here is executed in the *start period*. You can use it to calculate (initialise) variables derived
from observed data.

input
-----

The initial (observed) data is read from the *input* entry. 

The *path* is not compulsory. If *path* is not specified, the path is defined by the models definition path.

The hdf5-file format can be browsed with *vitables* (http://vitables.berlios.de/) or another hdf5-browser available
on the internet.

output
------

The simulation result is stored in the *output* entry. Only the variables defined at the *entity* level are stored.
Temporary (local) variables are not saved. The output file contains values for each period and each field and each item.

The *path* is not compulsory. If *path* is not specified, the path is defined by the models definition path.

start_period
------------

Defines the first period (integer) of the simulation. 

periods
-------

Defines the number of periods (integer) to be simulated.
