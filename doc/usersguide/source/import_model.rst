.. index:: importing models
.. _import_models:

Importing other models
======================

A model file can (optionally) import (an)other model file(s). An import
directive can take two forms: ::

  import: path\filename.yml

or ::

  import:
      - path1\filename1.yml
      - path2\filename2.yml

Each file in the import section will be merged with the current file in the
order it appears. Merging means that fields, links, macros and processes from
the imported file are added to the ones of the current file. If there is a
conflict (something with the same name is defined for the same entity in both
files), the "current" file (the model importing the other model) takes
priority. This means one can override entity processes defined in imported
files, or add fields to an entity defined in the imported model.

Note that both the importing model and the imported model need not be
complete/valid models (they do not need to include all required (sub)sections),
as long as the combined model is valid. See the examples below.
                             
*example* (common.yml) ::

  entities:
      person:
          fields:
              - age:      int
              - agegroup: {type: int, initialdata: false}
  
          processes:
              ageing: 
                  - age: age + 1
                  - agegroup: trunc(age / 10)
  
  simulation:
      processes:
          - person: [ageing]
  
      # we do not specify output so this model is not valid in itself
      input:
          file: simple2001.h5
  
      start_period: 2002
      periods: 2
                                  
*example* (variant1.yml) ::

  import: common.yml
  
  entities:
      person:
          processes:
              # override the ageing process
              ageing:
                  - age: age + 1
                  - agegroup: if(age < 50,
                                 5 * trunc(age / 5),
                                 10 * trunc(age / 10))
  
  simulation:
      # provide the required "output" section which is missing in common.yml
      output:
          file: variant1.h5
                               
*example* (variant2.yml) ::
    
  import: common.yml
  
  entities:
      person:
          fields:
              # adding a new field
              - dead: {type: bool, initialdata: false}
  
          processes:
              # adding a new process
              death:
                  - dead: logit_regr(0.0, align='al_p_dead.csv')
                  - show('Avg age of death', avg(age, filter=dead))
                  - remove(dead)
  
  simulation:
      # since we have a new process, we have to override the *entire* process
      # list, as LIAM2 would not know where to insert the new process otherwise.
      processes:
          - person: [ageing, death]
  
      output:
          file: variant2.h5

Imported models can themselves import other models, as for example in
variant3.yml.

*example* (variant3.yml) ::

  import: variant2.yml
  
  entities:
      person:
          processes:
              # use the "alternate" ageing procedure
              ageing:
                  - age: age + 1
                  - agegroup: if(age < 50,
                                 5 * trunc(age / 5),
                                 10 * trunc(age / 10))

This last example could also be achieved by importing both variant1.yml and
variant2.yml. Notice that the order of imports is important, since it determines
the result of conflicts between variants. For example in variant4.yml below, the
process list will be the one from variant2 and the output will go in
variant2.h5.

*example* (variant4.yml) ::

  import:
      - variant1.yml
      - variant2.yml
