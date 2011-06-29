.. index:: import

.. _import_data:

Importing data
==============

data files
----------

As of now, you can only import CSV files, one file for each entity.
Their first row should contain the name of the fields. You need at least two
integer columns: "id" and "period" (though they do not necesarrily need to be
named like that in the csv file).

description file
----------------

To import CSV files, you need to create a description file. Those description 
files have the following general format: ::

    output: <path_of_hdf5_file>.csv

    globals:
        periodic:
            path: <path_of_globals_file>.csv
            # if the csv file is transposed (each field is on a row instead of a
            # column and the field names are in the first column, instead of the
            # first row), you can use "transpose: true". You do not need to
            # specify anything if the file is not transposed.
            transposed: true

    entities:
        <entity1_name>:
            path: <path_to_entity1_data>.csv
            fields:
                # period and id are implicit
                - <field1_name>: <field1_type>
                - <field2_name>: <field2_type>
                - ...

            # if you want to keep your csv files intact but you use different
            # names in your simulation that in the csv files, you can specify
            # name changes here.
            oldnames:
                <fieldX_newname>: <fieldX_oldname>
                <fieldY_newname>: <fieldY_oldname>
            
            # if you want to invert the value of some boolean fields (True -> False
            # and False -> True), add them to the "invert" list below.
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
  
Note that if an "entity section" is entirely empty, you need to use the special
code: "{}".

*example* ::

    output: normal.h5

    globals:
        periodic:
            path: input\globals_transposed.csv
            transposed: true
    
    entities:
        household:
            path: input\household.csv
    
        person:
            path: input\person.csv
            fields:
                - age:           int
                - gender:        bool
                - workstate:     int
                - civilstate:    int     
                - partner_id:    int
    
            oldnames:
                gender: male

*simpler example* ::

    output: simple.h5

    globals:
        periodic:
            path: input\globals.csv

    entities:
        household:
            path: input\household.csv
    
        person:
            path: input\person.csv

*simplest example* ::

    output: simplest.h5
    
    entities:
        household: {}
        person: {}

This will try to load all the fields of the household and person entities in 
"*household.csv*" and "person.csv" in the same directory than the description
file.

importing the data
------------------

Once you have your data as CSV files and created a description file, you can
import your data.

- If you are using the bundled editor, simply open the description file and
  press F5.

- If you are using the command line, use: ::

    liam2 import <path_to_description_file>