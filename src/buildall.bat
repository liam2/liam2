@echo off
call build32.bat
set BDIR32=build\exe.win32-2.7
upx -9 %BDIR32%\*.exe
upx -9 %BDIR32%\*.pyd
upx -9 %BDIR32%\*.dll
upx -9 %BDIR32%\imageformats\*.dll
rem del mpl-data\sample_data?
call build64.bat
rem upx does not support 64b files
rem set BDIR64=build\exe.win-amd64-2.7