﻿.. index:: bundle, notepad++

Environment
###########

LIAM2 bundle
------------

The bundle includes:

- The executable for Windows.

- A text editor (Notepad++), pre-configured to work with LIAM2 models.

  * Notepad++ is a free (and open source) text editor that is available
    at http://sourceforge.net/projects/notepad-plus/.
    
  * We pre-configured it so that you can import csv files and run your models
    directly from within the editor by simply pressing F5 or F6 respectively.
    See below for more information.
    
- The documentation in HTML Help format (.chm). You can find .pdf and .html
  versions on our website.

- A demonstration model with a synthetic dataset.

Getting Started
---------------

- Copy the contents of the bundle in a directory on your disk (let us call it 
  *[BUNDLEPATH]*).

- Run the "Notepad++Portable.exe" from the *[BUNDLEPATH]\\editor* 
  directory.

- Open a model (eg. *[BUNDLEPATH]\\examples\\demo01.yml*)

- Press F6 to run the model. A console window will open within the editor and
  display the status of the simulation. After the simulation completes, the
  console window becomes interactive.

- Use this console to explore the results. If you already quit the
  interactive console and want to explore the results with the interactive
  console again without (re)simulating the model, you can press F7.

- Alternatively, you can explore the results "graphically" by pressing F9.
  This will open both the input dataset and the result file (if any) with an
  hdf5 viewer (ViTables).

Using your own data
-------------------

- Prepare your data as CSV files. The first row should contain the name of the
  fields. You need at least two integer columns: "id" and "period" (though they
  do not necessarily need to be named like that in the csv file).

- Create an import file, as described in the :ref:`import_data` section. You
  can use *[BUNDLEPATH]\\examples\\demo_import.yml* as an example.

- Press F5 to convert your CSV files to hdf5.

- Use the newly created data file with your model.