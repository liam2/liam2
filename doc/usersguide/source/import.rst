.. index:: importing data
.. _import_data:

Importing data
==============

data files
----------

As of now, you can only import CSV files. LIAM2 currently supports two kinds
of data files: tables and multi-dimensional arrays. 

**table** files are used for entities data and optionally for globals. They
should have one column per field and their first row should contain the name
of the fields. These names should not contain any special character (accents,
etc.). 

For entities data, you need at least two *integer* columns: "id" and "period" 
(though they do not necessarily need to be named like that in the csv file).

**array** files are used for other external data (alignment data for example).
They are arrays of any number of dimensions of a single homogeneous type.
The first row should contain the dimension names (one dimension by cell).
The second row should contain the possible values for the last dimension.
Each subsequent row should start by the values for the first dimensions then
the actual data.

*example* ::

  gender |  work | civilstate |      |      |     
         |       |          1 |    2 |    3 |    4
   False | False |       5313 | 1912 |  695 | 1222
   False |  True |        432 |  232 |   51 |   87
    True | False |       4701 | 2185 | 1164 | 1079
    True |  True |        369 |  155 |  101 |  116

This is the same format that groupby() generates except for totals.

description file
----------------

To import CSV files, you need to create a description file. Those description 
files have the following general format: ::

    output: <path_of_hdf5_file>.csv
    
    # compression is optional. compression type can be 'zlib', 'bzip2' or 'lzo'
    # level is a digit from 1 to 9 and is optional (defaults to 5).
    # Examples of valid compression strings are: zlib, lzo-1, bzip2-9.
    # You should experiment to see which compression scheme (if any) offers the
    # best trade-off for your dataset.
    compression: <type>-<level>

    # globals are entirely optional
    globals:
        periodic:
            path: <path_of_file>.csv
            # if the csv file is transposed (each field is on a row instead of
            # a column and the field names are in the first column, instead of
            # the first row), you can use "transpose: True". You do not need to
            # specify anything if the file is not transposed.
            transposed: True

            # fields are optional (if not specified, all fields are imported)
            fields:
                # PERIOD is implicit
                - <field1_name>: <field1_type>
                - <field2_name>: <field2_type>
                - ...

        other_table:
            # same options than for periodic: path, fields, transpose, ...

        other_array:
            path: <path_of_file>.csv
            type: <field_type>

    entities:
        <entity1_name>:
            path: <path_to_entity1_data>.csv
            
            # defaults to False if not present
            transposed: True

            # if you want to manually select the fields to be used, and/or 
            # specify their types, you can do so in the following section.
            # If you want to use all the fields present in the csv file, you
            # can simply omit this section. The field types will be
            # automatically detected.
            fields:
                # period and id are implicit
                - <field1_name>: <field1_type>
                - <field2_name>: <field2_type>
                - ...

            # if you want to keep your csv files intact but use different
            # names in your simulation than in the csv files, you can specify
            # name changes here.
            oldnames:
                <fieldX_newname>: <fieldX_oldname>
                <fieldY_newname>: <fieldY_oldname>
                ...

            # another option to specify name changes (takes precedence over
            # oldnames in case of conflicts).
            newnames:
                <fieldX_oldname>: <fieldX_newname>
                <fieldY_oldname>: <fieldY_newname>
                ...

            # if you want to merge several files, use this format:
            files:
                - <path>\<to>\<file1>.<ext>:
                    # any option (renamings, ...) specified here will override
                    # the corresponding options defined at the level of the
                    # entity
                    transposed: True|False
                    newnames:
                        <fieldX_oldname>: <fieldX_newname>
                        <fieldY_oldname>: <fieldY_newname>

                # if you don't have any specific option for a file, use "{}"
                - <path>\<to>\<file2>.<ext>: {}
                - ...
                
            # OR, if all the files use the global options (the options defined
            # at the level of the entity):
            files:
                - <path>\<to>\<file1>.<ext>
                - <path>\<to>\<file2>.<ext>
                - ...
    
            # if you want to fill missing values for some fields (this only 
            # works when "files" is used).
            interpolate:
                <fieldX_name>: previous_value

            # if you want to invert the value of some boolean fields
            # (True -> False and False -> True), add them to the "invert" list
            # below.
            invert: [list, of, boolean, fields, to, invert]
                
        <entity2_name>:
            ...

Most elements of this description file are optional. The only required elements
are "output" and "entities". If an element is not specified, it uses the 
following default value:

- if *path* is omitted, it defaults to a file named after the entity in the same
  directory than the description file (ie *local_path\\name_of_the_entity.csv*).
- if the *fields* section is omitted, all columns of the csv file will be
  imported and their type will be detected automatically.
- if *compression* is omitted, the output will not be compressed.
  
Note that if an "entity section" is entirely empty, you need to use the special
code: "{}".

*simplest example* ::

    output: simplest.h5
    
    entities:
        household: {}
        person: {}

This will try to load all the fields of the household and person entities in 
"*household.csv*" and "person.csv" in the same directory than the description
file.

*simple example* ::

    output: simple.h5

    globals:
        periodic:
            path: input\globals.csv

    entities:
        household:
            path: input/household.csv

        person:
            path: input/person.csv

This will try to load all the fields of the household and person entities in 
"*household.csv*" and "person.csv" in the "input" sub-directory of the
directory where the description file is.

*example 3* ::

    output: example3.h5

    globals:
        periodic:
            path: input/globals_transposed.csv
            transposed: True

    entities:
        household:
            path: input/household.csv
    
        person:
            path: input/person.csv
            fields:
                - age:        int
                - gender:     bool
                - workstate:  int
                - civilstate: int     
                - partner_id: int
    
            oldnames:
                gender: male

This will load all the fields of the household entity in 
"*household.csv*" and load from "person.csv" only the fields listed above. 
The data will be converted (if necessary) to the type declared. In this case,
person.csv should contain at least the following columns (not necessarily in
this order): period, id, age, male, workstate, civilstate, partner_id.

If the fields of an entity are scattered in several files, you can use the
"files" key to list them, as in this fourth example : ::

    output: example4.h5

    entities:
        person:
            fields:
                - age:        int
                - gender:     bool
                - workstate:  int
                - civilstate: int     
     
            # renamings applying to all files of this entity
            newnames:
                time: period

            files:
                - param\p_age.csv:
                    # additional renamings for this file only
                    newnames:
                        value: age
                - param\p_workstate.csv:
                    newnames:
                        value: workstate
                # person.csv should have at least 4 columns:
                # period, id, age and gender
                - param/person.csv:
                    newnames:
                        # we override the "global" renaming
                        period: period
             
            interpolate:
                workstate: previous_value
                civilstate: previous_value

But this can become tedious if you have a lot of files to import and they all
have the same column names. If the name of the field can be extracted from the
name of the file, you can automate the process like this:
 
*example 5* ::

    output: example5.h5

    entities:
        person:
            fields:
                - age:  int
                - work: bool
    
            newnames:
                time: period
                # {basename} evaluates to the name of the file without
                # extension. In the examples below, that would be
                # 'p_age' and 'p_work'. We then use the "replace" method
                # on the string we got, to get rid of 'p_'.
                value: eval('{basename}'.replace('p_', ''))

            files:
                - param\p_age.csv
                - param\p_work.csv

            interpolate:
                work: previous_value
            


importing the data
------------------

Once you have your data as CSV files and created a description file, you can
import your data.

- If you are using the bundled editor, simply open the description file and
  press F5.

- If you are using the command line, use: ::

    [BUNDLEPATH]\liam2\main import <path_to_description_file>
