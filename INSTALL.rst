Introduction
============

LIAM2 is cross-platform. It has been reported to run on Windows, Mac OS X and Linux.
Installation is easier on Windows because we provide a self-contained :ref:`bundle <installation_bundle>`
containing an executable (which includes all dependencies), a text editor, documentation and a
demonstration model. If you want to install LIAM2 on other platforms (Mac OS X or Linux) or
change the framework itself, you have to :ref:`install from source <installation_source>`.


.. _installation_bundle:

Install using the Windows bundle
================================

- Extract the contents of the bundle in a directory on your disk (let us call it ``<BUNDLEPATH>``).
  The name of this directory and the path leading to it should not contain any "special"
  (non-ASCII) characters.

- Create a shortcut to ``<BUNDLEPATH>\editor\Notepad++Portable.exe``, for example by right-clicking
  on it and using "Send to" -> "Desktop (create shortcut)".

- You are ready to :ref:`run your first model <getting_started_bundle>`.


.. _installation_source:

Install from source
===================

To install LIAM2 from source, you first need to `install LIAM2 dependencies`_ then `install LIAM2 itself from source`_.

Install LIAM2 dependencies
--------------------------

LIAM2 is built on top of a number of other open-source packages. See below for the different ways to install them.

Required dependencies:

- Python 3.10+ (64b) - http://www.python.org/
- Numpy 1.8.0 or later - http://www.numpy.org/
- PyTables 3 or later - http://www.pytables.org
- Numexpr 2.0 or later - https://github.com/pydata/numexpr
- PyYAML 3.0.8 or later - http://pyyaml.org
- Cython 0.16 or later - http://www.cython.org/

Optional dependencies:

* to view hdf5 files: vitables 2.1 or later - http://vitables.org

  It requires PyQt

* to generate plots and charts: matplotlib 1.2 or later - http://matplotlib.org/

* to build the documentation to html (other formats need additional packages):

  Sphinx 1.0 or later - http://www.sphinx-doc.org/

* to import data with interpolation/data with missing data points (eg several time series with different dates for the
  same individual):

  bcolz 0.7 or later - https://github.com/Blosc/bcolz

* to create standalone executables:

  Nuitka

There are several options to install all those packages. In order of increasing difficulty:

.. * `Using Anaconda (all platforms)`_

* `Using Miniconda (all platforms)`_
* `Getting binary packages using apt-get (GNU/linux debian-based distributions)`_


..
   Using Anaconda (all platforms)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   1. Install Anaconda 2.1 or later (Python 3.X). It includes out of the box all required dependencies,
      matplotlib and sphinx. We recommend using the 64-bit version if you have a 64bit OS.

      https://www.anaconda.com/download/

   2. Install ViTables. In a command prompt, type: ::

       pip install vitables

   Note that using other Python distributions should probably work, but we have
   not tested that. As of this writing, Python(x,y) and WinPython are both missing
   the "bcolz" package, so it would need to be installed from another source, if needed.


Using Miniconda (all platforms)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install Miniconda for your platform (Python 3.X). We recommend using the 64-bit version if you have a
   64bit OS. https://conda.io/miniconda.html

2. Install required dependencies by typing in a command prompt: ::

    conda install numpy pytables numexpr pyyaml cython

3. You can also install the optional dependencies:

   - to generate plots and charts: ::

       conda install matplotlib qtpy pyqt

   - to view hdf5 files via vitables: ::

       pip install vitables

   - to build the documentation to html (other formats need additional packages): ::

       conda install sphinx

   - to import data with interpolation/data with missing data points (e.g. several time series with different dates for
     the same individual): ::

       conda install bcolz


Getting binary packages using apt-get (GNU/linux debian-based distributions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the required dependencies: ::

    sudo apt-get install python python-numpy python-tables python-numexpr python-yaml cython

You can also install the optional dependencies:

- to view hdf5 files via vitables: ::

    sudo apt-get install python-vitables

- to generate plots and charts: ::

    sudo apt-get install python-matplotlib python-dateutil python-pyparsing

- to build the documentation to html (other formats need additional packages): ::

    sudo apt-get install python-sphinx

- to import data with interpolation/data with missing data points (e.g. several time series with different dates for
  the same individual): ::

    pip install bcolz


Install LIAM2 itself from source
--------------------------------

1. Download the zip file (e.g. ``LIAM2-0.13.0-src.zip``) from LIAM2 website.

2. Unzip into a directory of your choice. Let us call that directory ``<SOURCEPATH>``. For example ::

    Programs/LIAM2-0.13.0-src/

3. Open a terminal (Mac Terminal, gnome-terminal, konsole, xterm, ...)

4. Change into the directory into which LIAM2 has been unzipped (``<SOURCEPATH>``). For example: ::

    cd Programs/LIAM2-0.13.0-src/

5. Run installation of LIAM2 ::

    python setup.py install

6. You are ready to :ref:`run your first model <getting_started_source>`.


Building the C extensions manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additionally, if you want to get maximum performance, you need to have the C extensions built and compiled.
If all goes well, this was done automatically in the "python setup.py install" step above but in case it failed for
some reason, you might need to do it manually.

For that you need to have Cython (see above) and a C compiler installed, then go to the root directory of LIAM2 and
type: ::

    python setup.py build_ext --inplace

If all goes according to plan, you should then be up and running.


Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

0. make sure both ``<PYTHONDIR>`` and ``<PYTHONDIR>/Scripts`` are in your system PATH
   where ``<PYTHONDIR>`` is the directory where you installed Python

1. Install sphinx
2. Open a command prompt
3. Go to the documentation directory. For example: ::

    cd liam2/doc/usersguide/

4. Launch the build: ::

    make html

5. Open the result in your favourite web browser. It is located in: ::

    build/html/index.html
