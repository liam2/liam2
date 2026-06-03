@echo off
rem This script assumes:
rem 1) an environment with all LIAM2 dependencies installed is activated
rem 2) you have upx in your PATH
python setup.py build_ext --inplace

python -m nuitka --standalone --include-package=vitables.extensions --include-package-data=vitables:*.ui --include-package-data=vitables:*.ini --include-package-data=vitables.icons --enable-plugin=pyqt6 liam2\main.py

rem Using upx makes the distribution quite a bit smaller but makes antivirus solutions (even more) prone to flag our executable as malware
rem set DIST_DIR=main.dist\
rem upx -9 %DIST_DIR%\*.exe
rem upx -9 %DIST_DIR%\*.dll
rem upx -9 %DIST_DIR%\*.pyd
rem upx -9 %DIST_DIR%\matplotlib\*.pyd
rem upx -9 %DIST_DIR%\numexpr\*.pyd
rem upx -9 %DIST_DIR%\numpy\_core\*.pyd
rem upx -9 %DIST_DIR%\numpy\fft\*.pyd
rem upx -9 %DIST_DIR%\numpy\linalg\*.pyd
rem upx -9 %DIST_DIR%\numpy\random\*.pyd
rem upx -9 %DIST_DIR%\numpy.libs\*.dll
rem upx -9 %DIST_DIR%\PIL\*.pyd
rem upx -9 %DIST_DIR%\PyQt6\*.pyd
rem upx -9 %DIST_DIR%\PyQt6\Qt6\plugins\iconengines\*.dll
rem upx -9 %DIST_DIR%\PyQt6\Qt6\plugins\imageformats\*.dll
rem upx -9 %DIST_DIR%\PyQt6\Qt6\plugins\platforms\*.dll
rem upx -9 %DIST_DIR%\PyQt6\Qt6\plugins\styles\*.dll
rem upx -9 %DIST_DIR%\PyQt6\Qt6\plugins\tls\*.dll
rem upx -9 %DIST_DIR%\tables\*
rem upx -9 %DIST_DIR%\tables.libs\*.dll
rem upx -9 %DIST_DIR%\yaml\*.pyd
