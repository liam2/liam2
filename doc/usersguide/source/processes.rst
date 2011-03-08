.. highlight:: yaml
.. index::
    single: processes;

.. _processes_label:

Processes
#########

General setup
=============

The processes are the core of the model. For each entity-level (in this case, household or person), the block of processes starts
with the header (processes:). A process changes the variable (predictor) using a expression (expr). ::

    processes:
        process_name: 
            predictor: variable_name
            expr: "expression"

        ...
        
Or, shorter: ::         

    processes:
        variable_name: "expression"

        ...
        
The variable_name will usually be one of the variables defined in the **fields** block of the entity.

A process starts at a new line with an indentation of four spaces, and can consist of sub-
processes. These also start on a new line, again with an indentation of four spaces and a - . All definitions, be it of processes or sub-
processes, should be between double quotes. 

So the general setup is: ::

    processes:
        variable_name: "expression"
        process_name2:
            predictor: variable_name
            expr: "expression"
        process_name3:
            - subprocess_31: "expression"
            - subprocess_32: "expression"

In this example, there are three processes, of which the first two do not have sub-processes. The third process consists of two
sub-processes. If there are no sub-processes, a process obviously describes the simulation of one variable only. In this case,
the name of the process equals the name of the *endogenous variable*. 

To run the processes, they have to be specified on the simulation block of the file. This explains why the *process names* have 
to be unique for each entity.

FIXME
*example* ::


   processes:
        age: "age + 1"

This is however not the case with a process that has sub processes: it is possible for subprocess_31 to simulate one
variable of the same name, and subprocess_32 to simulate another, again of the same name as the process. In this case, the name
of the overarching process (process_name3) does not directly refer to a specific endogenous variable.

*example* ::

    processes:
        age: "age + 1"               
        agegroup:
            - agegroup5: "5 * round(age / 5)"
            - agegroup10: "10 * round(age / 10)"

The processes *agegroup5* and *agegroup10* are grouped in *agegroup*. In the simulation block you specify the
*agegroup*-process if you want to update *agegroup5* and *agegroup10*. 

By using processes and sub-processes, you can actually make *building blocks* or modules in the model. 

You can use temporary variables in a module. They only exist during the execution of that module. If you want to pass 
variables between modules you have to define them in the **fields** section.

*example* ::

    person:
        fields:
            # period and id are implicit
            - age:          int
            - dead:         bool
            - gender:       bool
            - partner_id:   int
            
            # 1=single, 2=married, 3=cohab, 4=divorced, 5=widowed
            - civilstate:          int  
            - dur_in_couple:       int
            - agegroup_work:       {type: int, initialdata: false}
            - agegroup_civilstate: {type: int, initialdata: false}
            
            # 1: in work, employee, private sector,
            # 2: in work, employee, public sector not civserv, 
            # 3: in work*, public sector civserv,
            # 4: in work, self employed,
            # 5: in education,
            # 6: unemployed including old-age unemployed,
            # 7: CELS (brugpensioen),
            # 6: disabled (including chronically ill),
            # 9: retired,
            # 10: other inactive            
            - workstate:       int     
            - inwork:          {type: bool, initialdata: false}                        
            - education_level: {type: int, initialdata: false}

        processes:
            ...
            
            divorce_procedure:
                - agediff: "if(FEMALE and MARRIED , age - ps.age, 0)"
                - inwork: "WORKING"
                # select females to divorce
                - divorce: "logit_regr(0.6713593 * ph.nch12_15 - 0.0785202 * dur_in_couple
                                + 0.1429621 * agediff - 0.0088308 * agediff**2 
                                - 0.814204 *((inwork) and (ps.inwork)) - 4.546278,
                                filter = FEMALE and MARRIED, 
                                align = 'al_p_divorce.csv')"
                # select persons to divorce
                - to_divorce: "divorce or ps.divorce"
                - partner_id: "if(to_divorce, -1, partner_id)"
                - civilstate: "if(to_divorce, 4, civilstate)"
                - dur_in_couple: "if(to_divorce, 0, dur_in_couple)"
                # move out males 
                - hh_id: "if(MALE and to_divorce, 
                    new('household', 
                        start_period=period,
                        region_id=choice([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
                    ),
                    hh_id)"

In the example *agediff*, *divorce*, *to_divorce* are temporary variables. They can only be used in the module
"divorce_procedure".

.. index::
    single: expressions;


Expressions
===========

Deterministic changes
---------------------

Let us start with a simple increment; the following process increases the value of a variable by one each simulation period. 

    age: "age + 1"

The name of the process is *age* and what it does is increasing the variable *age* of each individual by one, each period. 

.. index::
    single: simple expressions;


simple expressions
~~~~~~~~~~~~~~~~~~

- Arithmetic operators: +, -, *, /, **, %
- Comparison  operators: <, <=, ==, !=, >=, >
- Boolean operators: and, or, not
- Conditional expressions: 
    if(condition, expression_if_true, expression_if_false)

*example* ::

    agegroup_civilstate: "if(age < 50, 5 * round(age / 5), 10 * round(age / 10))"
    agegroup_work: "if(age < 70, 5 * round(age / 5), 70)"
    
    
Note that an *if*-statement has always three arguments. If you want to leave a variable unchanged if a condition is not met,
specify its value in the *expression_if_false* ::

    # retire people (set workstate = 9) when age 65 and more
    workstate: "if(age > 64, 9, workstate)"
    
You can nest if-statements. The example below retires men (gender = True) over 64. For women, this is the case when the age
equals at least the parameter WEMRA (a periodic global). ::
    
    workstate: "if(gender, 
                   if(age >= 65, 9, workstate), 
                   if(age >= WEMRA, 9, workstate))"
    
Note that you have to use parentheses when you use *Boolean operators*. ::

    inwork: "(workstate > 1) and (workstate < 5)"

.. index::
    single: mathematical functions;

mathematical functions
~~~~~~~~~~~~~~~~~~~~~~
    
- log(expr): natural logarithm (ln)
- exp(expr): exponential 
- abs(expr): absolute value
- round(expr[, n]): returns the rounded value of expr to specified n (number of digits after the decimal point). If n is not specified, 0 is used.
- clip(x, a, b): returns x if a < x < b, b if x > b, a if x < a.
- min(x, a), max(x, a): the minimum or maximum of x and a.


.. index::
    single: aggregate functions;

aggregate functions
~~~~~~~~~~~~~~~~~~~

- grpcount([filter]): count the objects in the entity
- grpsum(expr): sum the expression
- grpavg(expr): average
- grpstd(expr): standard deviation
- grpmax(expr), grpmin(expr): max or min

**grpsum** sums any variable over object types (persons, households, ...). For example *grpsum(earnings)* will
produce a sum of the earnings in the sample. The procedure *grpsum(nch0_11)* will result in the total number of
children 0 to 11 in the sample.

**grpcount** counts the number of objects (persons or households). For example, *grpcount(gender)* will produce the total number of
males in the sample. Contrary to **grpsum**, the grpcount does not need an argument: *grpcount()* will return the total number of
individuals or households in the sample.

Note that, if the variable in grpsum is a Boolean, then grpsum and grpcount will give the same results. 

*example* ::

    macros:
        WIDOW: "civilstate == 5"
    processes:
        cnt_widows: "show(grpsum(WIDOW))"

.. index:: countlink, sumlink, avglink, minlink, maxlink

link functions 
~~~~~~~~~~~~~~
(one2many links)

- countlink(link[, filter])
- sumlink(link, expr[, filter])
- avglink(link, expr[, filter])
- minlink/maxlink(link, expr[, filter])

*example* ::

    entities:
        household:
            fields:
                # period and id are implicit
                - dead:         bool
                - nb_persons:   {type: int, initialdata: false} 
                - nb_students:  {type: int, initialdata: false}
                - nch0_11:      {type: int, initialdata: false}
                - nch12_15:     {type: int, initialdata: false}
            links:
                persons: {type: one2many, target: person, field: hh_id}

            processes:            
                household_composition:
                    - nb_persons: "countlink(persons)"
                    - nb_students: "countlink(persons, workstate == 1)"
                    - nch0_11: "countlink(persons, age < 12)"
                    - nch12_15: "countlink(persons, (age > 11) and (age < 16))"

.. index:: temporal functions, lag, value_for_period, duration, tavg

temporal functions 
~~~~~~~~~~~~~~~~~~

- lag: value at previous period
- value_for_period: value at specific period
- duration: number of consecutive period the expression was True
- tavg: average of an expression since the individual was created

If an item did not exist at that period, the returned value is -1 for a int-field, nan for a float or False for a boolean.
You can overide this behaviour when you specify the *missing* parameter.

*example* ::

    lag(age, missing=0) # age of the population of last year, 0 if newborn
    grpavg(lag(age))    # average age that the current population had last year
    lag(grpavg(age))    # average age of the population of last year

    value_for_period(inwork and not male, 2002)

    duration(inwork and (earnings > 2000))
    duration(educationlevel == 4)
    
    tavg(income)

.. index:: random, uniform, normal, randint

random functions
~~~~~~~~~~~~~~~~

- uniform: random numbers with a uniform distribution
- normal: random numbers with a normal distribution
- randint: random integers between bounds

*example* ::

    normal(loc=0.0, scale=grpstd(errsal)) # a random variable with the stdev derived from errsal
    randint(0, 10)

Stochastic changes I: probabilistic simulation
----------------------------------------------

.. index:: choice

choice
~~~~~~

Monte Carlo or probabilistic simulation is a method for iteratively evaluating a deterministic model using sets of random numbers
as inputs. In microsimulation, the technique is used to simulate changes of state dependent variables. Take the simplest example:
suppose that we have an exogenous probability of an event happening, P(x=1), or not P(x=0). Then draw a random number u from an
uniform (0,1) distribution. If, for individual i, ui<p(1), then xi=1. If not, then xi=0. The expected occurrences of x after,
say, 100 runs is then P(x=1)*100 and the expected value is 1xP(1)+0xP(0)=P(1). This type of simulation hinges on the
confrontation between a random variable and an exogenous probability. In the current version of LIAM 2, it is not possible to
combine a choice with alignment.

In LIAM 2, such a probabilistic simulation is called a **choice** process. Suppose i=1..n choice options, each with a probability
prob_option_i. The choice process then has the following form: ::

    choice([option_1, option_2, ..., option_n], [prob_option_1, prob_option_2, ..., prob_option_n])

Note that both lists of options and pertaining probabilities are between []’s. Also, the variable containing the options can be
of any numeric type.

A simple example of a choice process is the simulation of the gender of newborns (51% males and 49% females), as such: ::

    gender=choice([True, False], [0.51, 0.49])

The code below illustrates a more complex example of a choice process (called *collar process*). Suppose we want to
simulate the work status (collar=1 (blue collar worker), white collar worker) for all working individuals. We however have
knowledge one’s level of education (education_level=2, 3, 4).

The process *collar_process* has collar as the key endogenous variable and has four sub-processes.

The first sub-process defines a local variable filter-bw, which will be used to separate those that the procedure should apply
to. These are all those that do not have a value for collar, and who are working, or who are in education or unemployed, which
means that they potentially could work.

The next three "collar" sub-processes simulate whether one is a white or blue collar worker, depending on the
level of education. If one meets the above filter_bw and has the lowest educational attainment level, then one has a
probability of about 84% (men) and 69% (women) of being a blue collar worker. If one has ‘education_level’ equal to 3, the
probability of being a blue collar worker is of course lower (64% for men and 31% for women), and the probability of becoming a
blue collar worker is lowest (8 and 4%, respectively) for those having the highest educational attainment level. ::

    collar_process:  # working, in education, unemployed or other inactive 
        - filter_bw: "(((workstate > 0) and (workstate < 7)) or (workstate == 10)) and (collar == 0)"
        - collar: "if(filter_bw and (education_level == 2),
                      if(gender,
                         choice([1, 2], [0.83565, 0.16435]),
                         choice([1, 2], [0.68684, 0.31316]) ),
                      collar)"
        - collar: "if(filter_bw and (education_level == 3),
                      if(gender,
                         choice([1, 2], [0.6427, 1 - 0.6427]),
                         choice([1, 2], [0.31278, 1 - 0.31278]) ),
                      collar)"
        - collar: "if(filter_bw and (education_level == 4),
                      if(gender,
                         choice([1, 2], [0.0822, 1 - 0.0822]),
                         choice([1, 2], [0.0386, 1 - 0.0386]) ),
                      collar)"

.. index:: logit, alignment

Stochastic changes II: behavioural equations
--------------------------------------------

- Logit: logit_regr(expr, filter, align)
- Alignment : 
    * align(expr, [take=take_filter,] [leave=leave_filter,] percentage) 
    * align(expr, [take=take_filter,] [leave=leave_filter,] fname='filename.csv')
- Continuous (expr + normal(0, 1) * mult + error): cont_regr(expr, filter, align, mult, error_var)
- Clipped continuous (always positive): clip_regr(expr, filter, align, mult, error_var)
- Log continuous (exponential of continuous): log_regr(expr, filter, align, mult, error_var)


*example* ::

    divorce: "logit_regr(0.6713593 * household.nch12_15 
                         - 0.0785202 * dur_in_couple
                         + 0.1429621 * agediff,
                         filter=FEMALE and (civilstate == 2), 
                         align='al_p_divorce.csv')"

    wage_earner: "if((age > 15) and (age < 65) and inwork,
                     if(MALE, 
                        align(wage_earner_score, 
                              fname='al_p_wage_earner_m.csv'),
                        align(wage_earner_score, 
                              fname='al_p_wage_earner_f.csv')),
                     False)"

.. index:: logit_regr

logit_regr                
~~~~~~~~~~

Suppose that we have a logit regression that relates the probability of some event to explanatory variables X. 
    
    p*i=logit-1(ßX + EPSi) 
    
This probability consists of a deterministic element (as before), completed by a stochastic element, EPSi, a log-normally
distributed random variable. The condition for the event occurring is p*i > 0.5.

Instead, suppose that we want the proportional occurrences of the event to be equal to an overall proportion X. In that
case, the variable p*i sets the rank of individual i according to the risk that the relevant event will happen. Then only
the first X*N individuals in the ranking will experience the event. This process is known as ‘alignment’.

In case of one logit with one alignment process -or a logit without alignment-, *logit_regr* will result in the logit
returning a Boolean whether the event is simulated. In this case, the setup becomes: ::

    - single_align: "logit_regr(<logit arguments>,
                [filter=<filter arguments>,
                            align='name.csv'])"   
                            

*example* ::

    birth:
        - to_give_birth: "logit_regr(0.0,
                                     filter=FEMALE and (age >= 15) and (age <= 50),
                                     align='al_p_birth.csv')"   

The above generic setup describes the situation where one logit pertains to one alignment process. 

.. index:: logit_score

logit_score
~~~~~~~~~~~

In many cases, however, it is convenient to use multiple logits with the same alignment process. In this case, using  a **logit_score** instead of
**logit_regr** will result in the logit returning intermediate scores that - for all conditions together- are the inputs of the
alignment process. A typical behavioural equation with alignment has the following syntax: ::

        name_process: 
            # initialise the score to -1
            - score_variable: "-1" 

            # first condition
            - score_variable: "if(condition_1,
                                  logit_score(logit_expr_1),
                                  score_variable)"
            # second condition
            - score_variable: "if(condition_2,
                                  logit_score(logit_expr_2),
                                  score_variable)"
                                  
            # ... other conditions ...
                        
            # do alignment based on the scores calculated above
            - name_endogenous_variable: 
                "if(condition,
                    if(gender, 
                       align(score_variable,
                             [take=conditions,]
                             [leave=conditions,]
                             fname='filename_m.csv'),
                       align(score_variable,  
                             [take=conditions,]
                             [leave=conditions,]
                             fname='filename_f.csv')),
                    False)"
                                
The equation needs to simulate the variable *name_endogenous_variable*. It starts however by creating a score that reflects
the event risk p*i. In a first sub-process, a variable *name_score* is set equal to -1, because this makes it highly
unlikely that the event will happen to those not included in the conditions for which the logit is applied. Next, subject to
conditions *condition_1* and *condition_2*, this score is simulated on the basis of estimated logits. The specification
*logit_score* results in the logit not returning a Boolean but instead a score.

Note that by specifying the endogenous variable *name_score* without any transformations under the ‘ELSE’ condition makes
sure that the score variable is not manipulated by a sub-process it does not pertain to.


.. index:: align, take, leave

align
~~~~~~

After this step, the score is known and this is the input for the alignment process. Suppose -as is mostly the case- that
alignment data exists for men and women separately. Then the alignment process starts by a *if* to gender. Next comes the
align command itself. This takes the form ::

    align(score_variable,
          filter=conditions,
          [take=conditions,]
          [leave=conditions,]
          fname='name.csv')
        
The file *name.csv* contains the alignment data. A standard setup is that the file starts with the prefix *al_* followed by
the name of the endogenous variable and a suffix *_m* or *_f*, depending on gender.

The optional *take* and *leave* commands forces inclusion or exclusion of objects with specified characteristics in
the selection of the event. The individuals with variables specified in the *take* command will a priori be selected for the
event. Suppose that the alignment specifies that 10 individuals should experience a certain event, and that there are 3
individuals who meet the conditions specified in the *take*. Then these 3 individuals will be selected a priori and the
alignment process will select the remaining 7 candidates from the rest of the sample. The *leave* command works the other
way around: those who match the condition in that statement, are a priori excluded from the event happening. The *take* and
*leave* are absolute conditions, which mean that the individuals meeting these conditions will always (*take*) or never
(*leave*) experience the event. 

Their *soft* counterparts can easily be included by manipulating the score of individuals.
If this score is set to a strong positive or negative number, then the individual will a priori have a high of low
probability of the event happening. These *soft take* and ‘*soft leave*’s will implement a priority order in the sample of
individuals, but will not under all circumstances conditionally include or exclude.

Note that even if the score is -1 an item can be selected by the alignment procedure. This happens when there are not enough
candidates (selected by the score) to meet the alignment needs.

The below application describes the process of being (or remaining) a wage-earner or employee. It illustrates a *soft
leave* by setting the a priori score variable *wage_earner_score* to -1. This makes sure that the a priori
selection probability for those not specified in the process is very low (but not zero, as in the case of *leave*
conditions).

Next come three sub processes setting a couple of common conditions, in the form of local (temporary) variables. These three sub-
processes are followed by six subsequent *if* conditions, separating the various behavioural equations to the sub-sample
they pertain to. The first three sub conditions pertain to women and describe the probability of being a wage-earner from in
work and employee previous year (1) from in work but not employee previous year (2), and from not in work previous year
(3). The conditions 4 to 6 describe the same transitions but for women. ::

    wage_earner_process: 
        - wage_earner_score: "-1"
        - lag_public: "lag((workstate == 2) or (workstate == 3))" 
        - inwork: "(workstate > 0) and (workstate < 5)"
        - lag_inwork: "lag((workstate > 0) and (workstate < 5))"

        # Probability of being employee from in work and employee previous year (men)
        - wage_earner_score: 
            "if(gender and (age > 15) and (age < 65) and inwork and ((lag(workstate) == 1) or (lag(workstate) == 2)),
                logit_score(0.0346714 * age + 0.9037688 * (collar == 1) - 0.2366162 * (civilstate == 3) + 2.110479),
                wage_earner_score)"
        # Probability of becoming employee from in work but not employee previous year (men)
        - wage_earner_score:
            "if(gender and (age > 15) and (age < 65) and inwork and ((lag(workstate) != 1) and (lag(workstate) != 2)),
                logit_score(-0.1846511 * age - 0.001445 * age **2 + 0.4045586 * (collar == 1) + 0.913027),
                wage_earner_score)"
        # Probability of becoming employee from not in work previous year (men)
        - wage_earner_score:
            "if(gender and(age > 15) and (age < 65) and inwork and (lag(workstate) > 4),
                logit_score(-0.0485428 * age + 1.1236 * (collar == 1) + 2.761359),
                wage_earner_score)"

        # Probability of being employee from in work and employee previous year (women)
        - wage_earner_score:
            "if(not gender and(age > 15) and (age < 65) and inwork and ((lag(workstate) == 1) or (lag(workstate) == 2)),
                logit_score(-1.179012 * age + 0.0305389 * age **2 - 0.0002454 * age **3 - 0.3585987 * (collar == 1) + 17.91888),
                wage_earner_score)"
        # Probability of becoming employee from in work but not employee previous year (women)
        - wage_earner_score:
            "if(not gender and(age > 15) and (age < 65) and inwork and ((lag(workstate) != 1) and (lag(workstate) != 2)),
                logit_score(-0.8362935*age + 0.0189809 * age **2 -0.000152 ** age **3 -0.6167602*(collar==1) + 0.6092558 * (civilstate==3) +9.152145),
                wage_earner_score)"
        # Probability of becoming employee from not in work previous year (women)
        - wage_earner_score:
            "if(not gender and (age > 15) and (age < 65) and inwork and (lag(workstate) > 4),
                logit_score(-0.6177936 * age + 0.0170716 * age **2 - 0.0001582 * age**3 + 9.388913),
                wage_earner_score)"
                                        
        - wage_earner: "if((age > 15) and (age < 65) and inwork,
                           if(gender, 
                              align(wage_earner_score, 
                                    fname='al_p_wage_earner_m.csv'),
                              align(wage_earner_score, 
                                    fname='al_p_wage_earner_f.csv')),
                           False)"

The last sub-procedure describes the alignment process. Alignment is applied to individuals between the age of 15 and 65 who
are in work. The reason for this is that those who are not working obviously cannot be working as a wage-earner. The input-
files of the alignment process are 'al_p_wage_earner_m.csv' and 'al_p_wage_earner_f.csv'. The alignment process sets the
Boolean *wage earner*, and uses as input the scores simulated previously, and the information it takes from the alignment
files. No ‘take’ or ‘leave’ conditions are specified in this case.

Note that the population to align is the population specified in the first condition, here *(age>15) and (age<65) and (inwork)* and not the
whole population.
                
.. index:: new, remove                
                
Lifecycle functions
-------------------

new
~~~

**new** creates items initiated from another item of the same entity (eg. a women gives birth) or another
entity (eg. a marriage creates a new houshold).

*generic format* ::

    new(entity, filter=expr, *set initial values of a selection of variables*)
    
The first parameter defines the entity in which the item will be created. (eg person, household)

Since an item is at origin of a creation, the fields of that origin (**__parent__**) can be used to initialise the
fields of the new item.

*example 1* ::

    birth:
        - to_give_birth: "logit_regr(0.0,
                                     filter=not gender and (age >= 15) and (age <= 50),
                                     align='al_p_birth.csv')"   
        - newbirth: "new('person', filter=to_give_birth, 
                m_id=__parent__.id
                f_id = __parent__.partner_id, 
                m_age = __parent__.age, 
                hh_id = __parent__.hh_id,
                partner_id = -1,
                civilstate = 1,
                collar = 0,
                education_level = -1,
                workstate = 5, 
                gender=choice([True, False], [0.51, 0.49]) )"  

The first sub-process (*to_give_birth*) describes the probability that a women (not gender) between 15 and 50 gives birth.
This is a process that is also aligned, but results directly in a Boolean. For this reason, the procedure *logit_regr*  is
used instead of *logit_score*. Also, note that the logit itself does not have a deterministic part (0.0), which means that
the ‘fertility rank’ of women that meet the above condition, is only determined by a logistic stochastic variable). Whether
or not a women is scheduled to give birth is the result of a stochast and the alignment process to age.

In the above case, a new person is created for each time a woman is scheduled to give birth. Secondly, a number of links are
established: the id-number and age of the parent become the *mother id* and age of the mother of the child, and the child
also receives the household number from the mother. Finally some initial variables are set for the child: the most important
of these is its gender, which is the result of a simple choice process.

**new** is not limited to items of the same entity; the below procedure *get a life* makes sure that all those that all
singles of 24 years old, leave their parents’ household for their own household. The region of this household is created
through a simple choice-process.

*example 2* ::

    get_a_life:
        - hh_id: "if(not ((civilstate == 2) or (civilstate == 3)) and (age == 24), 
                new('household', 
                    start_period=period,
                    region_id=choice([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
                ),
                hh_id)"


remove
~~~~~~

**remove** items from an entity dataset. With this command you can remove obsolete items (eg. dead persons, empty
households) thereby saving memory and improving simulation speed.


The procedure below simulates whether an individual survives or not, and what happens in the latter case. ::

    dead_procedure:  
        # decide who dies
        - dead: "if(gender, 
            logit_regr(0.0, align='al_p_dead_m.csv'), 
            logit_regr(0.0, align='al_p_dead_f.csv'))"                 
        # change the civilstate of the suriving partner
        - civilstate: "if(ps.dead, 5, civilstate)"  
        # break the link to the the suriving partner
        - partner_id: "if(ps.dead, -1, partner_id)"
        # remove the dead
        - cleanup: remove(dead)

The first sub-procedure *dead* simulates whether an individual is ‘scheduled for death’, using again only a logistic
stochastic variable and the age-gender-specific alignment process. Next some links are updated for the surviving partner.
The sub-procedure *civilstate* puts the variable of that name equal to 5 (which means that one is a widow(er) for those
individuals whose partner has been scheduled for death. Also, in that case, the partner identification code is erased. All
other procedures describing the heritage process should be included here. Finally, the command *remove* is called in the
sub-procedure *cleanup*. This command removes the *dead* from the simulation dataset.

.. index:: matching

Matching functions
------------------

**matching**: (aka Marriage market) matches individuals from set 1 with individuals from set 2 follow a particular order
(given by an expression) for each individual in set 1, computes the score of all (unmatched) individuals in set 2 and take
the best scoring one.

You have to specify the sets to match (set1filter and set2filter), the score that will be used to do the matching. 
and the criterion to decide what is a difficult match. Difficult matches are selected first. In the score the fields
of the individual and the fields of its possible partners (**other**) are used.

*generic setup* ::

    difficult_match: "abs(age - avg_age_men)"
    matching(set1filter=to_marry and not male,
             set2filter=to_marry and male,
             score='- 0.4893 * other.age 
                    + 0.0467 * (other.age - age)
                    - 0.6549 * (work and other.work)
                    - 1.3286 * (work and not other.work) 
                    - 0.9087 * (not work and other.work)',
             orderby=difficult_match)

The generic setup of the marriage market is simple; one needs to have selected those individuals who are to be coupled
(*to_couple*=true). Furthermore, one needs to have a variable (*difficult_match*) which can be used to rank individuals
according how easy they are to match. Finally, we need a function (*score*) matching potential partners.

In the first step, and for those persons that are selected to be coupled, potential partners are matched in the order set by
*difficult_match* and each woman is matched with the potential partner with the highest matching score. Once this is done,
both individuals become actual partners and the partner identification numbers are set so that the partner number of each
person equals the identification number of the partner.

*example* ::

    marriage:
        - in_couple: "MARRIED or COHAB"
        - to_couple: "if((age >= 18)  and (age <= 90) and not in_couple, 
                         if(MALE,
                            logit_regr(0.0, align='al_p_mmkt_m.csv'),
                            logit_regr(0.0, align='al_p_mmkt_f.csv')), 
                         False)"
        - difficult_match: "if(to_couple and FEMALE,
                               abs(age - grpavg(age, filter=to_couple and MALE)),
                               nan)"
        - inwork: "(workstate > 0) and (workstate <5)"                                         
        - partner_id: "if(to_couple, 
                          matching(set1filter=FEMALE, set2filter=MALE,
                                   score='- 0.4893 * other.age 
                                          + 0.0131 * other.age ** 2 
                                          - 0.0001 * other.age ** 3
                                          + 0.0467 * (other.age - age) 
                                          - 0.0189 * (other.age - age) ** 2 
                                          + 0.0003 * (other.age - age) ** 3
                                          - 0.9087 * (other.inwork and not inwork) 
                                          - 1.3286 * (not other.inwork and inwork) 
                                          - 0.6549 * (other.inwork and inwork)',
                                   orderby=difficult_match),
                          partner_id)"
        - coupled: "to_couple and (partner_id != -1)"   
        - newhousehold: "new('household', filter=coupled and FEMALE,
                             start_period=period,
                             region_id=choice([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4]) )"
        - hh_id: "if(coupled,
                     if(MALE, ps.newhousehold, newhousehold),
                     hh_id)"


The code above shows an application. First of all, individuals eligible for marriage are all those between 18 and 90 who are
not a part of a couple; the actual decision who is eligible is left to the alignment process. Next, for every women eligible
to coupling, the variable *difficult_match* is the difference between her age and the average age of men eligible for
coupling.

In a third step, a matching variable is simulated for each combination of man and woman eligible for coupling. This variable
depends on the difference in age, the work status of the potential partners, and the difference in levels of education.
Using this information, and following the order set by *difficult_match*, potential partners are coupled to become actual
partners.

In a next step, a new household is created for women who have just become a part of a couple. The household number of their
new male partners then is set equal to their new household number.



Output
======

LIAM 2 produces simulation output in three ways. First of all, by default, the simulated datasets are stored in hdf5
format. These can be accessed at the end of the run. You can use several tools to inspect the data.

You can display information during the simulation using *show* or *groupby*. You can *dump* data to csv-file for further
study.

If you run LIAM 2 in interactive mode, you can type in output functions in the console to inspect the data.

.. index::  show

show
----

*show* prints a line with information to the console. ::

    show(expr[, expr2, expr3])


*example 1* ::

    show(grpcount(age >= 18))
    show(grpcount(not dead), grpavg(age, filter=not dead))
    
The first process will print out the number of persons of age 18 and older. The second line displays the number of living
people and their average age.

*example 2* ::

    show("Count", grpcount(), "Age Average", grpavg(age), "Age Std dev", grpstd(age))
    
    gives
    
    Count 19944 Age Average 42.7496991576 Age Std dev 21.9815913417
    
.. index::  csv

csv
---

You can write the contents of a *table* to csv-file. 
The general format of the outputfile will be <entity_name>_<period>_<suffix_specifiction>.csv. 

**csv** works with any expression producing a table (eg. dump, groupby).

    csv(table_expression, suffix='suffix_specification')
    
*example*  ::

    csv(table_expr, suffix='income')
    
creates a file called "person_2002_income.csv with info for the period 2002 from the entity person
    
    
.. index::  dump

dump    
----

**dump**: produces a table with the expressions given as argument

*general format*

    dump(expr[, expr2, expr3, ..., filter=filterexpression])

*example* ::

    show(dump(age, partner.age, gender, filter=id < 10))
    
gives  ::

        id | age | partner.age | gender
         0 |  27 |          -1 |  False
         1 |  86 |          71 |  False
         2 |  16 |          -1 |   True
         3 |  19 |          -1 |  False
         4 |  27 |          21 |  False
         5 |  89 |          92 |   True
         6 |  59 |          61 |   True
         7 |  65 |          29 |  False
         8 |  38 |          35 |   True
         9 |  48 |          52 |   True
         
.. index::  groupby         

groupby
-------

**groupby** (aka *pivot table*): group individuals by their value for the given expressions, and optionally compute an
expression for each group

*general format* ::

    groupby(col_expr[, col_expr2, col_expr3, ...] [, expr=expression] [, filter=filterexpression])

*example* ::

    show(groupby(age / 10, gender))
    
gives ::    

        gender | False | True |      
    (age / 10) |       |      | total
             0 |   818 |  803 |  1621
             1 |   800 |  800 |  1600
             2 |  1199 | 1197 |  2396
             3 |  1598 | 1598 |  3196
             4 |  1697 | 1696 |  3393
             5 |  1496 | 1491 |  2987
             6 |  1191 | 1182 |  2373
             7 |   684 |  671 |  1355
             8 |   369 |  357 |   726
             9 |   150 |  147 |   297
         total | 10002 | 9942 | 19944

*example* ::

    show(groupby(inwork,gender))

gives ::            

    gender | False | True |      
    inwork |       |      | total
     False |  6170 | 5587 | 11757
      True |  3832 | 4355 |  8187
     total | 10002 | 9942 | 19944

*example* ::

     show(groupby(inwork, gender, percent=True))

gives ::    

     gender | False |  True |       
     inwork |       |       |  total
      False | 30.94 | 28.01 |  58.95
       True | 19.21 | 21.84 |  41.05
      total | 50.15 | 49.85 | 100.00