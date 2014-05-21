Miscellaneous improvements
--------------------------

* if there is only one entity defined in a model (like in demo01.yml) and the
  interactive console is launched, start directly in that entity, instead of
  requiring the user to set it manually.  

* improved introduction comments in demo models.

* display whether C extensions are used or not in --versions.

* use default_entity in demos (from demo03 onward).

* do not display python version in normal execution but only in --versions.

* use cx_freeze instead of py2exe to build executables for Windows so that
  we can use the same script to build executables across platforms and tweaked
  further our build script to minimise the executable size. 
  
* compressed as many files as possible in the 32 bit Windows bundle with UPX
  to make the archive yet smaller (UPX does not support 64 bit executables
  yet).
  
* improved our build system to automate much of the release process.

Fixes
-----

* fixed the "explore" command.

* fixed integer fields on 64 bit platforms other than Windows.

* fixed demo06: WEMRA is an int now.

* fixed demo01 introduction comment (bad file name).
