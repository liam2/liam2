#################
General overview
#################

*****
Setup
*****

Contrary to its first version, a model written in LIAM DMS consist of only one text file. So, the core of the model, the definition of the variables (previously dyvardesc), the simulation order of the routines (agespine) and relevant simulation information, including the definition of the in- and output directories, starting period and number of periods (previously dyrunset) are now in one file.

A typical model in LIAM DMS has the following setup::

    constants:
        per_period:
            - WEMRA: float
            
    entities:
        household:
            fields:
                # period is implicit
                # id is implicit
                - dead:         bool
                - start_period: {type: int, initialdata: false}
                - nb_persons:   {type: int, initialdata: false} 
                - nb_students:  {type: int, initialdata: false}            
                - nch0_11:      {type: int, initialdata: false}
                - nch12_15:     {type: int, initialdata: false}
                - equi_scale:   {type: float, initialdata: false}
                - region_id:    {type: int, initialdata: false}
    
            links:
                persons: {type: one2many, target: person, field: hh_id}
                
            processes:
                region_id: "if(period==2000, choice([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4]), region_id)"
                household_composition:
                    - nb_persons: "countlink(persons)"
                    - nb_students: "countlink(persons, workstate == 1)"
                    - nch0_11: "countlink(persons, (age <12))"
                    - nch12_15: "countlink(persons, (age>11) and (age <16))"
                    - equi_scale: "1.0 + (nb_persons - nch0_11 - nch12_15)*0.5 + (nch0_11 + nch12_15)*0.3"
    #                - sh: "show(groupby([nb_persons, nch0_11 + nch12_15]))"                
                dump_csv_h: "csv(dump())"
                show_empty: "show(grpsum(nb_persons==0))"
                clean_empty: "remove(nb_persons==0)"
                
        person:
            fields:
                # period is implicit
                # id is implicit
                - age:          int
                - dead:         bool
                - gender:       bool
                - m_id:         int
                - m_age:        {type: int, initialdata: false}
                - partner_id:   int
                - civilstate:      int  # 1=single, 2=married, 3=cohab, 4=divorced, 5=widowed
                - dur_in_couple:      int
                - agegroup_work:     {type: int, initialdata: false}
                - agegroup_civilstate:     {type: int, initialdata: false}            
                - oworkstate:    int     # 1=in education,2=in work,3=unemployed,4=old-age unemployed,5=CELS (
                - workstate:    int     # 1=in work, employee, private sector,2=in work, employee, public sect
                - work:       {type: bool, initialdata: false}  #temp work
                - inwork:         {type: bool, initialdata: false}                        
                - psinwork:       {type: bool, initialdata: false}            
                - dur_work:     int
                - education_level:       {type: int, initialdata: false}
    #            - bcw:          bool
    #            - wcw:          bool
                - collar:   int # 0=none, 1=blue collar worker, 2=white collar worker         
                - public:       bool
                - wage_earner:     {type: bool, initialdata: false} #
                - civserv:      bool
                - hh_id:        int
            
            links:
                pm: {type: many2one, target: person, field: m_id}      
                ps: {type: many2one, target: person, field: partner_id}
                ph: {type: many2one, target: household, field: hh_id}
                
            macros:
                WIDOW: "civilstate==5"
                MARRIED: "(civilstate == 2)"
                COHAB: "(civilstate == 3)"
                WORKING: "(workstate > 0) and (workstate < 5)"
                CIVSERV: "workstate == 3"
                PUBLIC: "((workstate ==2) or (workstate ==3))"
                WAGE_EARNER: "(workstate > 0) and (workstate < 4)"
                MALE: "gender"
                FEMALE: "~gender"
               
            # possible transitions and regressions. The actual list used in the
            # simulation must be defined below
            processes:
                age: "age + 1"
                cnt_widows: "show(grpsum(WIDOW))"
                init_inwork: 
                    - inwork: "WORKING"
                    - public: "PUBLIC"
                    - wage_earner: "WAGE_EARNER"
                    - civserv: "CIVSERV"
                    
                agegroup:
                    - agegroup_civilstate: "if(age<50, 5*round(age/5), 10*round(age/10))"            
                    - agegroup_work: "if(age<70, 5*round(age/5), 70)"
                dead_procedure:                
                    - dead: "if(MALE, 
                        logit_regr(0.0, align='al_p_dead_m.csv'), 
                        logit_regr(0.0, align='al_p_dead_f.csv'))"
                    # TODO implement easier way to change states                    
                    - civilstate: "if(ps.dead, 5, civilstate)"  
                    - partner_id: "if(ps.dead, -1, partner_id)"                   
                    - cleanup: remove(dead)
                                        
                birth:
                    - to_give_birth: "logit_regr(0.0,
                                filter=FEMALE and (age >= 15) and (age <= 50),
                                align='al_p_birth.csv')"   
                    - newbirth: "new('person', filter=to_give_birth, 
                         m_id = __parent__.id, 
                         m_age = __parent__.age, 
                         hh_id = __parent__.hh_id,
                         partner_id = -1,
                         civilstate = 1,
                         collar = 0,
                         education_level = -1,
                         workstate = 5, 
                         gender=choice([True, False], [0.51, 0.49]) )"  
                         
                marriage:
                    - in_couple: "MARRIED or COHAB"
                    - to_couple: "if((age >= 18)  and (age <= 90) and ~in_couple, 
                                    if(MALE,
                                        logit_regr(0.0, align='al_p_mmkt_m.csv'),
                                        logit_regr(0.0, align='al_p_mmkt_f.csv')), 
                                    False
                                  )"
    #                - difficult_match: "abs(age - grpavg(age, filter=to_couple and MALE),
    #                                        filter=to_couple and FEMALE)"
                    - difficult_match: "if(to_couple and FEMALE,
                                              abs(age - grpavg(age, filter=to_couple and MALE)),
                                              nan)"
                    - inwork: "(workstate > 0) and (workstate <5)"                                         
                    - partner_id: "if(to_couple, 
                                        matching(set1filter= FEMALE, set2filter=MALE,
                                                score='- 0.4893 * other.age 
                                                   + 0.0131 * other.age ** 2 
                                                   - 0.0001 * other.age ** 3
                                                   + 0.0467 * (other.age - age) 
                                                   - 0.0189 * (other.age - age) ** 2 
                                                   + 0.0003 * (other.age - age) ** 3
                                                   - 0.9087 * ((other.inwork) and ~(inwork)) 
                                                   - 1.3286 * (~(other.inwork) and (inwork)) 
                                                   - 0.6549 * ((other.inwork) and (inwork))
                                                   - 0.7939 * ((other.education_level == 3) and (education_lev
                                                   - 1.4128 * ((other.education_level == 2) and (education_lev
                                                   - 0.8984 * ((other.education_level == 4) and (education_lev
                                                   - 1.5530 * ((other.education_level == 4) and (education_lev
                                                   - 0.5451 * ((other.education_level == 2) and (education_lev
                                                orderby=difficult_match),
                                        partner_id)"
                    - coupled: "to_couple and (partner_id != -1)"   
                    - newhousehold: "new('household', filter=coupled and FEMALE,
                                         start_period=period,
                                         region_id=choice([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4]) )"
                        
                    - hh_id: "if(coupled,
                                    if(MALE, ps.newhousehold, newhousehold),
                                    hh_id)"
                                    
                    # Married or Cohabitant          
                    - agediff: "if(coupled, age - ps.age, -1)" 
                    - psinwork: "if(coupled, ps.inwork, False)"
                    - married: "logit_regr(
                                -0.030816 * agediff**2 + 0.0013858 * agediff **3 
                                + 1.561471 * (lag(inwork) and ~lag(psinwork))
                                + 1.576726 * (education_level - ps.education_level),
                                filter = FEMALE and coupled,
                                align = 'al_p_mar_married.csv')" 
                    - married: "if(MALE, ps.married, married)" 
                    - civilstate: "if(coupled, if(married, 2, 3), civilstate)" 
                    - dur_in_couple: "if(coupled,
                                    0, 
                                    if(in_couple, dur_in_couple+1, 0)
                                )"      
    #                - dump_csv_mmkt: "csv(dump(id, hh_id, age, MALE, married, coupled, civilstate, dur_in_cou
                                                                     
                marry_cohabitant:
                    # Cohabitant          
                    - agediff: "if(civilstate == 3, age - ps.age, -1)"
                    - inwork: "WORKING"
                    - psinwork: "if(civilstate == 3, ps.inwork, False)"
                    # Select only female-male couples to marry                                                
                    - married_cohab_f: "logit_regr(
                                -0.030816 * agediff**2 + 0.0013858 * agediff **3 
                                + 1.561471 * (lag(inwork) and ~lag(psinwork))
                                + 1.576726 * (education_level - ps.education_level),
                                filter = FEMALE and COHAB and ps.gender,
                                align = 'al_p_mar_married.csv')" 
                    - married_cohab_m: "ps.married_cohab_f" 
                    - married_cohab: "if(MALE, married_cohab_m, married_cohab_f)" 
                    - civilstate: "if(COHAB and married_cohab, 2, civilstate)" 
    #                - dump_csv_cohab: "csv(dump(id, hh_id, age, gender, dead, married_cohab_f, married_cohab_
    
                get_a_life:
                    #TODO what about the not married in 2002 and age >= 24  
                    - hh_id: "if(~(MARRIED or COHAB) and (age == 24), 
                        new('household', 
                            start_period=period,
                            region_id=choice([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
                        ),
                        hh_id)"
    
                divorce_procedure:
                    - agediff: "if(FEMALE and MARRIED , age - ps.age, 0)"
                    - inwork: "WORKING"
                    # select females to divorce
                    - divorce: "logit_regr(0.6713593 * ph.nch12_15 - 0.0785202 * dur_in_couple
                                    + 0.1429621 * agediff - 0.0088308 * agediff**2 
                                    - 0.814204 *((inwork) and (ps.inwork)) - 4.546278,
                                    filter = FEMALE and MARRIED, 
                                    align = 'al_p_divorce.csv')"
                    # change partners                             
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
    #                - dump_csv_div: "csv(dump(id, hh_id, age, gender, to_divorce, civilstate, dur_in_couple),
                                                                                                              
                separate_procedure:
                    - agediff: "if(FEMALE and COHAB , age - ps.age, 0)"
                    - inwork: "(workstate > 0) and (workstate <5)"                
                    # select females to divorce
                    - separate: "logit_regr(0.7106698 * ph.nch12_15 -0.0557708 * dur_in_couple 
                                    -0.6050533 *((inwork) and (ps.inwork)) -3.336578,
                                    filter = FEMALE and COHAB, 
                                    align = 'al_p_separate.csv')"
                    # change partners                             
                    - to_separate: "separate or ps.separate"
                    - partner_id: "if(to_separate, -1, partner_id)"
                    - civilstate: "if(to_separate, 1, civilstate)"
                    - dur_in_couple: "if(to_separate, 0, dur_in_couple)"                
                    # move out males 
                    - hh_id: "if(MALE and to_separate, 
                        new('household', 
                            start_period=period,
                            region_id=choice([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
                        ),
                        hh_id)"
    #                - dump_csv_sep: "csv(dump(id, hh_id, age, gender, to_separate, civilstate, dur_in_couple)
    #                - table_civilstate: "show(groupby([civilstate, gender]))"
    
                                                                                 
                education_level_2001:
                    - filter_2001: "education_level <0"
                    - education_level_2001_init: # initialise an education 
                         predictor: education_level
                         expr: "if(filter_2001,
                                        choice([2,3,4], [0.25, 0.39, 0.36]),
                                    education_level)"
                    - education_level_2001_1:
                         predictor: education_level
                         expr: "if(filter_2001 and (collar==1) and (workstate ==1),
                                      if(MALE, 
                                            choice([2, 3, 4], [0.45, 0.51, 0.04]),
                                            choice([2, 3, 4], [0.50, 0.46, 0.04])   ), 
                                      education_level)"                                                       
                    - education_level_2001_2:
                         predictor: education_level
                         expr: "if(filter_2001 and (collar==2) and (workstate ==1),
                                      if(MALE, 
                                            choice([2, 3, 4], [0.11, 0.31, 0.58]),
                                            choice([2, 3, 4], [0.11, 0.39, 0.50])    ), 
                                      education_level)"                                                       
                    - education_level_2001_3:
                         predictor: education_level
                         expr: "if(filter_2001 and (collar==1) and PUBLIC,
                                      if(MALE, 
                                            choice([2, 3, 4], [0.33, 0.30, 0.37]),
                                            choice([2, 3, 4], [0.15, 0.29, 0.56])   ), 
                                      education_level)"                                                       
                    - education_level_2001_4: # TODO change values
                         predictor: education_level
                         expr: "if(filter_2001 and (collar==2) and PUBLIC,
                                      if(MALE, 
                                            choice([2, 3, 4], [0.33, 0.30, 0.37]),
                                            choice([2, 3, 4], [0.15, 0.29, 0.56])   ), 
                                      education_level)"                                                       
                    - education_level_2001_5: # independent
                         predictor: education_level
                         expr: "if(filter_2001 and (workstate == 4),
                                      if(MALE, 
                                            choice([2, 3, 4], [0.20, 0.42, 0.38]),
                                            choice([2, 3, 4], [0.18, 0.42, 0.40])   ), 
                                     education_level)"
                    
                    - start_workage: "if(((workstate <5) or (workstate==6)) and (dur_work>=0), age - dur_work,
                    - education_level_2001_correction: # solve the age - education level problem
                        predictor: education_level
                        expr: "if(filter_2001 and (start_workage > 0),
                                if((start_workage < 19), 2, 
                                    if(start_workage < 24, 3, education_level)
                                    ),
                                education_level)"
                                     
    #                - sh_education_level: "show(groupby([agegroup_work, 10 * workstate + education_level], gr
                     
                show_educationlevel: "show(groupby([agegroup_work, education_level], grpsum(~dead))) "        
                                                                                                              
                education_level:
                    - education_level: "if((education_level < 0),
                                    choice([2,3,4], [0.25, 0.39, 0.36]),
                                    education_level)"
    
                table_collar: "show(groupby([education_level, collar])) " 
                collar_process:  # working, in education, unemployed or other inactive 
                    - filter_bw: "(((workstate > 0) and (workstate <7)) or (workstate == 10)) and (collar==0)"
                    - collar_process_1:
                        predictor: collar
                        expr: "if(filter_bw and (education_level == 2),
                                if(MALE,
                                    choice([1, 2], [0.83565, 0.16435]),
                                    choice([1, 2], [0.68684, 0.31316]) ),
                                    collar)"
                                    
                    - collar_process_2:
                        predictor: collar
                        expr: "if(filter_bw and (education_level == 3),
                                if(MALE,
                                    choice([1, 2], [0.6427, 1-0.6427]),
                                    choice([1, 2], [0.31278, 1-0.31278]) ),
                                    collar)"
                    - collar_process_3:
                        predictor: collar
                        expr: "if(filter_bw and (education_level == 4),
                                if(MALE,
                                    choice([1, 2], [0.0822, 1-0.0822]),
                                    choice([1, 2], [0.0386, 1-0.0386]) ),
                                    collar)"
    #                - show_collar: "show(groupby([collar, 10 * workstate + education_level] ))"
                                
                ineducation_process:
                    - show_work: "show(grpsum(workstate < 5))"            
                    - show_ineducation: "show(grpsum(workstate == 5))"
                    # decide ineducation upon age and education_level
                    - workstate: "if((workstate!=8) and
                                        ((age < 16) or 
                                         ((age < 19) and (education_level == 3)) or
                                         ((age < 24) and (education_level == 4))), 
                                         5, workstate)"                                      
                    - show_ineducation: "show(grpsum(workstate == 5))"                           
                    # unemployed if left education
                    - workstate: "if((workstate==5) and
                                        (((age == 16) and (education_level == 2)) or 
                                         ((age == 19) and (education_level == 3)) or 
                                         ((age == 24) and (education_level == 4))), 
                                        6, workstate)"
                                        
                    - show_ineducation: "show(grpsum(workstate == 5))"
                    - show_work: "show(grpsum(workstate < 5))"
    #                - show_workstate: "show(groupby([(10*workstate)+education_level, lag(workstate)]))"
                    
                table_inwork:
                    - t_inwork : "show(groupby([inwork, workstate]) )" 
                    - t_inwork_change : "show(groupby([workstate, lag(workstate)]) )"
    #                - all: "show(groupby([inwork, workstate, lag(workstate)]) )"                             
    #                - agegroup: "show(groupby([agegroup_work, workstate*10+education_level]) )"              
                
                inwork_process:
                    - before_job: "show(grpsum(inwork) )"
                    - before_no_job: "show(grpsum(workstate > 4))"
                    # retire 
                    - workstate: "if(MALE,
                                if(age >= 65, 9, workstate),
                                if(age >= WEMRA, 9, workstate)
                              )"                
                    - inwork: "WORKING"                
                    - work_score : "-1"
                    # Male 
                    - inwork_1_m:
                        predictor: work_score
                        expr: "if(MALE and (age >14) and (age<65) and (inwork),
                                    logit_score(-0.196599 * age + 0.0086552 * age **2 - 0.000988 * age **3
                                    - 1.491977 * (collar==1) + 0.1892796 * (MARRIED or COHAB) + 3.554612),
                                    work_score
                                )" 
                    - inwork_2_m:
                        predictor: work_score
                        expr: "if(MALE and (age >14) and (age<50) and (workstate > 5),
                                    logit_score(0.9780908 * age  -0.0261765 * age **2 +0.000199 * age **3
                                    +0.3113972 * (collar==1) -12.39108),
                                   work_score
                                )" 
                    - inwork_3_m:
                        predictor: work_score
                        expr: "if(MALE and (age>49) and (age<65) and (workstate == 10),
                                    logit_score(0.9780908 * age  -0.0261765 * age **2+0.000199 * age **3
                                    +0.3113972 * (collar==1) -12.39108),
                                    work_score
                                )" 
    
                    # FeMale 
                    - inwork_1_f:
                        predictor: work_score
                        expr: "if(FEMALE and (age >14) and (age<65) and (inwork),
                                    logit_score(-0.2740483 * age + 0.0109883 * age **2 -0.0001159 * age **3
                                    -1.17695 * (collar==1) + -0.0906834 * (MARRIED or COHAB) +3.648706),
                                    work_score
                                 )" 
                    - inwork_2_f:
                        predictor: work_score
                        expr: "if(FEMALE and (age >14) and (age<50) and (workstate > 5),
                                    logit_score(0.8217638 * age  -0.0219761 * age **2 +0.000166 * age **3
                                    -0.3441341 * (collar==1) -0.5590975 * (MARRIED or COHAB) -10.48043),
                                   work_score
                                )" 
                    - inwork_3_f:
                        predictor: work_score
                        expr: "if(FEMALE and (age>49) and (age<65) and (workstate == 10),
                                    logit_score(0.8217638 * age  -0.0219761 * age **2+0.000166 * age **3
                                    -0.3441341 * (collar==1) -0.5590975 * (MARRIED or COHAB) -10.48043),
                                    work_score
                                )" 
                    - job: "show(grpsum(inwork))"                
                    - work: "if((age > 15) and (age < 65),
                                if(MALE, 
                                    align(work_score, 
                                        take= (workstate == 3),
                                        leave= (age>49) and (age<65) and (workstate > 6) and (workstate < 10),
                                        fname='al_p_inwork_m.csv'),
                                    align(work_score, 
                                        take= (workstate == 3),
                                        leave=(age>49) and (age<65) and (workstate > 6) and (workstate < 10), 
                                        fname='al_p_inwork_f.csv')),
                                False
                               )"
                    # retire 
                    - workstate: "if(MALE,
                                if(age >= 65, 9, workstate),
                                if(age >= WEMRA, 9, workstate)
                              )"                
     
                    - inwork: "if(workstate==9, False, work)"
    #                - debug_inwork: "show(groupby([agegroup_work, workstate], filter=inwork and (work_score<0
                    # if not selected to work and last period in work set workstate to unemployed
                    - workstate: "if(~inwork and lag(inwork), 6, workstate)"
                    - dur_work: "if(inwork, dur_work + 1, 0)"                                                
                    - kept_job: "show(grpsum(inwork and lag(inwork)))"
                    - got_job: "show(grpsum(inwork and ~lag(inwork)))" 
                    - job: "show(grpsum(inwork))"
                    - job_problem: "show(grpsum(inwork and ~((age>14) and (age<65)) ) )"
                    - lag_workstate: "lag(workstate)"
    #                - dump_csv_job: "csv(dump((id, age, gender, work_score, work, workstate, lag_workstate, i
                    
                # Wage Earner, workstate 1,2,3
                wage_earner_process: 
                    - wage_earner_score: "-1"
                    - we_1_m: #Probability of being employee from in work and employee previous year (men)
                        predictor: wage_earner_score
                        expr: "if(MALE and(age>15) and (age<65) and (inwork) and ((lag(workstate) == 1) or (la
                                    logit_score(0.0346714*age + 0.9037688*(collar==1) -0.2366162*(civilstate==
                                    wage_earner_score
                                )"
                    - we_2_m: #Probability of becoming employee from in work but not employee previous year (m
                        predictor: wage_earner_score
                        expr: "if(MALE and(age>15) and (age<65) and (inwork) and ((lag(workstate) != 1) and (l
                                    logit_score(-0.1846511*age -0.001445 * age **2 + 0.4045586*(collar==1)+0.9
                                    wage_earner_score 
                                )"
                    - we_3_m: #Probability of becoming employee from not in work previous year (men)
                        predictor: wage_earner_score
                        expr: "if(MALE and(age>15) and (age<65) and (inwork) and (lag(workstate)>4),
                                    logit_score(-0.0485428*age + 1.1236*(collar==1)+2.761359),
                                    wage_earner_score 
                                )"
                    - we_1_f: #Probability of being employee from in work and employee previous year (women)
                        predictor: wage_earner_score
                        expr: "if(FEMALE and(age>15) and (age<65) and (inwork) and ((lag(workstate) == 1) or (
                                    logit_score(-1.179012*age + 0.0305389 * age **2 -0.0002454 * age **3 + -0.
                                    wage_earner_score 
                                )"
                    - we_2_f: #Probability of becoming employee from in work but not employee previous year (w
                        predictor: wage_earner_score
                        expr: "if(FEMALE and(age>15) and (age<65) and (inwork) and ((lag(workstate) != 1) and 
                                    logit_score(-0.8362935*age + 0.0189809 * age **2 -0.000152 ** age **3 -0.6
                                    wage_earner_score 
                                )"
                    - we_3_f: #Probability of becoming employee from not in work previous year (women)
                        predictor: wage_earner_score
                        expr: "if(FEMALE and(age>15) and (age<65) and (inwork) and (lag(workstate)>4),
                                    logit_score(-0.6177936*age + 0.0170716 * age **2 -0.0001582 *age**3 +9.388
                                    wage_earner_score 
                                )"
                    - wage_earner: "if((age>15) and (age<65) and (inwork),
                                        if(MALE, 
                                            align(wage_earner_score, 
                                                fname='al_p_wage_earner_m.csv'),
                                            align(wage_earner_score, 
                                                fname='al_p_wage_earner_f.csv')),
                                        False
                                    )"
                    - show_wage_earner: "show(grpsum(wage_earner))"
                    # Other inwork are self-employed   
                    - workstate: "if(inwork and ~wage_earner, 4, workstate)"
                    - show_selfemployed: "show(grpsum(workstate==4))"
                                         
                public_emp:
                   # Female
                   - work_age_f: "FEMALE and (age> 15) and (age < 65) and wage_earner"
                   - public_score : "-99999"
                   - public_f_1: 
                        predictor: public_score
                        expr: "if(work_age_f and lag(inwork) and ~lag(public),
                                    logit_score(
                                         0.0890169*age -0.0018488*age**2
                                        -0.4012818*(collar==1) -0.3234973*(MARRIED or COHAB)  -4.123885),
                                     public_score
                                     )" 
                   - public_f_2: 
                        predictor: public_score
                        expr: "if(work_age_f and lag(inwork) and lag(public) and (workstate !=3),
                                    logit_score(
                                         0.0266844*age 
                                        -1.209456*(collar==1) -0.7513913*(MARRIED or COHAB) +1.852208),
                                     public_score
                                    )" 
                   - public_f_3: 
                        predictor: public_score
                        expr: "if(work_age_f and ~lag(inwork),
                                    logit_score(
                                         0.3688486*age -0.0046727*age**2
                                        -0.4012818*(collar==1) -0.3234973*(MARRIED or COHAB) -6.879067),
                                     public_score 
                                    )" 
                   # Male
                   - work_age_m: "MALE and (age>15) and (age <65) and wage_earner"
                   - public_m_1: 
                        predictor: public_score
                        expr: "if(work_age_m and  lag(inwork) and ~lag(public),
                                    logit_score(
                                         -0.047131*age 
                                        -1.199581*(collar==1) -0.3863239*(MARRIED or COHAB) -2.129474),
                                     public_score
                                     )" 
                   - public_m_2: 
                        predictor: public_score
                        expr: "if(work_age_m and lag(inwork) and lag(public) and (workstate !=3),
                                    logit_score(
                                         0.216559*age -0.0020286 *age**2
                                        -0.4666532*(collar==1) -1.814848),
                                        public_score                                 
                                    )" 
                   - public_m_3: 
                        predictor: public_score
                        expr: "if(work_age_m and ~lag(inwork),
                                    logit_score(
                                        0.1964201*age -0.0020462*age**2
                                        -1.688204*(collar==1) -0.6600213*civilstate -4.0865137),
                                     public_score
                                )" 
                   - public: "if((age > 15) and (age <65 ) and wage_earner, 
                                    if(MALE, 
                                        align(public_score, 
                                            fname='al_p_public_m.csv'), 
                                        align(public_score,                                     
                                            fname='al_p_public_f.csv')),
                                    False
                               )"      
                                        
                   - show_public: "show(grpsum(public))"
                   # Other wage_earners are private sector
                   - workstate: "if(inwork and wage_earner and ~public, 1, workstate)"
                   - show_private_sector: "show(grpsum(workstate==1))"
    #               - dump_csv_public: "csv(dump(id, age, gender, public_score, inwork, workstate, public, wag
    
                civserv_process:
                    - civserv_score: "-1"            
                    # Female
                    - work_age_f: "FEMALE and (age> 15) and (age < 65) and public"
                    - civserv_f_1: #Probability of being a civil servant given that a person works as a civil 
                        predictor: civserv_score
                        expr: "if(work_age_f and lag(civserv),
                                    logit_score(2.251292), 
                                    civserv_score
                               )" 
                    - civserv_f_2: #Probability of being a civil servant given that a person works in the publ
                        predictor: civserv_score
                        expr: "if(work_age_f and not lag(civserv),
                                    logit_score(0.2278889*age -0.0034807*age**2 -0.4695609 *(collar==1) -6.266
                                    civserv_score
                               )" 
                    # MALE
                    - work_age_m: MALE and (age> 15) and (age < 65) and public
                    - civserv_m_1: #Probability of being a civil servant given that a person works as a civil 
                        predictor: civserv_score
                        expr: if(work_age_m and lag(civserv),
                                    logit_score(-3.871201 * (collar==1) + 3.060271), 
                                    civserv_score
                               ) 
                    - civserv_m_2: #Probability of being a civil servant given that a person works in the publ
                        predictor: civserv_score
                        expr: "if(work_age_m and not lag(civserv),
                                    logit_score(0.063816*age -0.001189*age**2 -0.4463475 *(collar==1)-2.886708
                                    civserv_score
                               )" 
                    - civserv: "if((age > 15) and (age <65 ) and public, 
                                    if(MALE, 
                                        align(civserv_score, 
                                            fname='al_p_civserv_m.csv'), 
                                        align(civserv_score,                                     
                                            fname='al_p_civserv_f.csv')),
                                    False
                               )" 
                   # Decide if civil servant or employee in public sector
                    - workstate: "if(public, if(civserv, 3, 2), workstate)"
                        
                    
                show_work: "show(grpsum((workstate >0) and (workstate < 5)))" 
                dump_csv_p: "csv(dump())"
                
    
    simulation:
        init:
            - household: [household_composition, clean_empty, region_id]  
            - person: [agegroup, init_inwork, education_level_2001, show_educationlevel]
                       
    
        processes:
            - household: [household_composition, clean_empty]    
            - person: [
                age, agegroup, 
                dead_procedure, 
                birth,
                cnt_widows,
                init_inwork, 
                table_inwork, table_collar,
                education_level,
                show_educationlevel, 
                ineducation_process,              
                marry_cohabitant, marriage, get_a_life, 
                ]
            - household: [household_composition]
            - person: [divorce_procedure, separate_procedure]
            - household: [household_composition]
            - person: [        
                collar_process,
                table_inwork,
                inwork_process,
    #            table_inwork,            
                wage_earner_process,
    #            table_inwork,
                public_emp,
    #            table_inwork,            
                civserv_process,
                table_inwork,
                collar_process,
    #            dump_csv_p
            ] 
    
            
    
        input:
            file: "base2001.h5"
        output:
            file: "simulation.h5"
    
        start_period: 2002   # first simulated period
        periods: 1

    
Note that all text following a "#" is considered to be comments, and therefore ignored.

A model consists of three main blocks. The first is the "constants-block". 

The second is the "entities block". This contains the body of the model, defines for every entity-level the pertaining variables, the links between these and other entities and the processes that pertain to the entity. 

Finally, the third block is the "simulation" block. This includes the location of the datasets, the number of periods and the start period, but it most importantly sets the order in which the procedures defined in the second block, are simulated.

********************
The simulation block
********************
