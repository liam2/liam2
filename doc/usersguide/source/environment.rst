.. index:: python, hdf5, yaml, notepad++

Environment
###########

Python
------

We use the Python language (http://www.python.org/) for the development of LIAM 2. 

    Python runs on Windows, Linux/Unix, Mac OS X, and has been ported to the Java and .NET virtual machines.

    Python is free to use, even for commercial products, because of its OSI-approved open source license.
    
HDF5    
----
    
We store the used data in an hdf5-format (http://www.hdfgroup.org).  

    HDF5 is a data model, library, and file format for storing and managing data. It supports an unlimited variety of
    data types, and is designed for flexible and efficient I/O and for high volume and complex data. HDF5 is portable and is
    extensible, allowing applications to evolve in their use of HDF5. The HDF5 Technology suite includes tools and
    applications for managing, manipulating, viewing, and analyzing data in the HDF5 format.
    
    HDF is open-source and the software is distributed at no cost. Potential users can evaluate HDF without any financial
    investment. Projects that adopt HDF are assured that the technology they rely on to manage their data is not dependent upon a
    proprietary format and binary-only software that a company may dramatically increase the price of, or decide to stop
    supporting altogether.
    
This allows us to handle important data sets.     

YAML
----

The definition of the data and the model is done in the YAML-language (http://www.yaml.org).

    YAML: YAML Ain't Markup Language

    What It Is: YAML is a human friendly data serialization standard for all programming languages.
    
Notepad++
---------

We bundle LIAM 2 with a portable version of the Notepad++-editor (http://sourceforge.net/projects/notepad-plus/). 
This editor allows YAML syntax highlighting.

LIAM 2
------

We package LIAM 2 into an executable. Python is bundled with the executable and does not need to be installed separately. The
package bundles all the extra modules (NumPy, NumExpr, Pytables, ...) we use.

In the future we plan a GUI-version of the program.

