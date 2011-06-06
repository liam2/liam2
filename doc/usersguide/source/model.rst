.. highlight:: yaml

Model Definition
################

Microsimulation (acronym for microanalytic simulation) is a modelling technique 
that operates at the level of individual *entities* such as persons, households,
vehicles or firms. Within the model each entity is represented by a record
containing a unique identifier and a set of associated attributes (e.g. age,
gender, work state, civil state, ...) characteristics. A set of *processes* are
then applied to these entities leading to simulated changes in state and
behaviour. These rules may be deterministic (probability = 1), such as changes
in tax liability resulting from changes in tax regulations, or stochastic
(probability <= 1), such as chance of dying, marrying, giving birth or moving
within a given time period.

To define the model, we have to describe the different *entities*, the way they
interact (*links*) and how they behave (*processes*) over time. This is done in
one file. We use the YAML-markup language. This format uses the level of
indentation to specify objects and sub objects.

In a LIAM 2 model file, all text following a # is considered to be comments, and
is therefore ignored.

A LIAM 2 model has the following structure: ::

    globals:
        ...

    entities:
        ...

    simulation:
        ...
        
globals
=======

The *globals* are variables (aka. parameters) that do not relate to any 
particular *entity* defined in the model. They can be used in expressions across
all entities.

Periodic globals can have a different value for each period. For example, the
retirement age for women in Belgium has been gradually increasing from 61 in 
1997 via 63 from 2003 onward, up to 65 in 2009. A global variable WEMRA has
therefore been included.::

    globals:
        periodic:
            - WEMRA: float

Periodic globals can be used in any process. They can be used in two ways: like
a normal variable, they will evaluate to their value for the period being
simulated, for example ::

    workstate: "if(age >= WEMRA, 9, workstate)"

This changes the workstate of the individual to retired (9) if the age is higher
than the required retirement age in that year.

Another way to use them is to specify explicitly for which period you want them
to be evaluated. This is done by using GLOBALNAME[period_expr]. periodexpr can
be any expression yielding a valid period value. Here are a few artificial 
examples: ::

    workstate: "if(age >= WEMRA[2010], 9, workstate)"
    workstate: "if(age >= WEMRA[period - 1], 9, workstate)"
    workstate: "if(age >= WEMRA[year_of_birth + 60], 9, workstate)"

entities
========

Each entity has a unique identifier and a set of attributes (**fields**). You
can use different entities in one model. You can define the interaction between
members of the same entity (eg. between partners) or among different entities
(eg. a person and its household) using *links*.

The **processes** section describe how the entities behave. The order in which
they are declared is not important. In the **simulation** block you define if
and when they have to be executed, this allows to simulate processes of
different entities in the order you want.


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
            
As we use YAML as the description language, indentation and the use of ":" are
important.

fields
------

The fields hold the information of each member in the entity. That information
is global in a run of the model. Every process defined in that entity can use
and change the value. 

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
                # 1: single, 2: married, 3: cohabitant, 4: divorced, 5: widowed 
                - civilstate:   int
                - partner_id:   int
                - earnings:     float

This example defines the entity person. Each person has an age, gender, is dead
or not, has a civil state, possibly a partner. We use the field civilstate to
store the marital status as a switch of values.

By default, all declared fields are supposed to be present in the input file
(because they are *observed* or computed elsewhere and their value can be
found in the supplied data set). The value for all declared fields will also be
stored for each period in the output file. 

However, in practice, there are often some fields which are not present in the
input file. They will need to be calculated later by the model, and you need to
tell LIAM2 that the field is missing, by using "initialdata: false" in the
definition for that field (see the *agegroup* variable in the example below).

*example* ::

    entities:
        person:
            fields:
                - age:          int
                - agegroup:     {type: int, initialdata: false}

Field names must be unique per entity (i.e. several entities may have a field
with the same name). 


links
-----

Entities can be linked with each other or with other entities, for example, 
individuals belong to households, and mothers are linked to their children, 
while partners are interlinked as well.

A typical link has the following form: ::

    name: {type: <type>, target: <entity>, field: <name of link field>}
    
LIAM 2 uses integer fields to establish the link between entities. Those
integer fields contain the id-number of the linked individual.    

LIAM 2 allows two types of links: many2one and one2many.

More detail, see :ref:`links_label`.


macros
------

Macros are a way to make the code easier to read and maintain. They are defined
on the entity level. Macros are re-evaluated wherever they appear. Use *capital*
letters to define macros.

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
the start period. It sets what processes defined in the **entities** block are simulated (since some can be
omitted), and the order in which this is done.

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
        start_period: 2002
        periods: 10
        random_seed: 5235       # optional



processes
---------

This block defines which processes are executed and in what order. They will be
executed for each period starting from *start_period* for *periods* times. 
Since processes are defined on a specific entities (they change the values of 
items of that entity), you have to specify the entity before each list of 
process. Note that you can execute the same process more than once during a
simulation and that you can alternate between entities in the simulation of a
period. 

In the example you see that after dead_procedure and birth, the
household_composition procedure is re-executed.

init
----

Every process specified here is only executed in the *start period*. You can use
it to calculate (initialise) variables derived from observed data. This section
is optional (it can be entirely omitted).

input
-----

The initial (observed) data is read from the file specified in the *input* entry. 

Specifying the *path* is optional. If it is omitted, it defaults to the
directory where the simulation file is located.

The hdf5-file format can be browsed with *vitables*
(http://vitables.berlios.de/) or another hdf5-browser available on the net.

output
------

The simulation result is stored in the file specified in the *output* entry.
Only the variables defined at the *entity* level are stored. Temporary (local)
variables are not saved. The output file contains values for each period and
each field and each item.

Specifying the *path* is optional. If it is omitted, it defaults to the
directory where the simulation file is located.

start_period
------------

Defines the first period (integer) to be simulated. 

periods
-------

Defines the number of periods (integer) to be simulated.

random_seed
-----------

Defines the starting point (integer) of the pseudo-random generator. This
section is optional. This can be useful if you want to have several runs of a
simulation to use the same random numbers. 