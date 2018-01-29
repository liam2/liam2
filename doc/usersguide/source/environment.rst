.. highlight:: yaml

.. index:: bundle, notepad++

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


.. _getting_started_bundle:

Getting Started (using the Windows bundle)
------------------------------------------

- Start the Notepad++ editor by using the shortcut you created during
  :ref:`installation <installation_bundle>`.

- Open a model (eg. ``<BUNDLEPATH>\examples\demo01.yml``)

- Press F6 to run the model. A console window will open within the editor and
  display the status of the simulation. After the simulation completes, the
  console window becomes interactive.

- Use this console to explore the results. If you already quit the
  interactive console and want to explore the results with the interactive
  console again without (re)simulating the model, you can press F7.

- Alternatively, you can explore the results "graphically" by pressing F9.
  This will open both the input dataset and the result file (if any) with an
  hdf5 viewer (ViTables).


.. _getting_started_source:

Getting Started (when running from source)
------------------------------------------

1. After you have :ref:`installed LIAM2 <installation_source>`, open a model in your favorite text editor
   (e.g. ``<SOURCEPATH>/liam2/tests/examples/demo01.yml``).

2. To run the model, open a terminal (the Mac terminal, Windows command prompt, gnome-terminal, ...)

3. Identify the path to the simulation file to be run.

    The full pathname relative to the current directory is required (e.g. ``liam2/tests/examples/demo01.yml``).

4. Type in the terminal: ::

     liam2 run -i <path/to/the/model.yml>

   For example: ::

     liam2 run -i liam2/tests/examples/demo01.yml

   After the simulation completes, the console window becomes interactive. Use this console to explore the results.
   Type 'q' or 'exit' to quit the interactive console.

5. If you already quit the interactive console and want to explore the results with the interactive
   console again without (re)simulating the model, you can use: ::

    liam2 explore liam2/tests/examples/demo01.yml

6. Alternatively, if you have ViTables correctly installed, you can explore the results "graphically" by using: ::

     liam2 view liam2/tests/examples/demo01.yml

   This will open both the input dataset and the result file (if any) with an hdf5 viewer (ViTables).

7. Other options for running LIAM2 are described by using ::

     liam2 --help


Using your own data
-------------------

- Prepare your data as CSV files. The first row should contain the name of the
  fields. You need at least two integer columns: "id" and "period" (though they
  do not necessarily need to be named like that in the csv file).

- Create an import file, as described in the :ref:`import_data` section. You
  can use ``<BUNDLEPATH>/examples/demo_import.yml`` as an example.

- Press F5 (or use `liam2 import <your_import_file.yml>` in a Terminal) to convert your CSV files to hdf5 .

- Use the newly created data file with your model.
