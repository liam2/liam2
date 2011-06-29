.. index:: import

.. _import_data:

Importing data
--------------

- Prepare your data as CSV files. The first row should contain the name of the
  fields. You need at least two integer columns: "id" and "period" (though they
  do not necesarrily need to be named like that in the csv file).

- Create an import file, as described in ... You can use 
  \\localpath\\Liam2Suite\\Synthetic\\demo_import.yml as an example.

- Press F5 to convert your CSV files to hdf5.

- Use the newly created data file with your model.