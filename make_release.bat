@echo off

set REPOSITORY=svn://filemail/liam2

if "%1"=="" (
    set SVNPATH=trunk
    rem this is a hack to retrieve the command output in a variable
    for /f "usebackq" %%a in (`python get_rev.py %REPOSITORY%`) do @set REV=%%a
    set VERSION=r%REV%
) else (
    set SVNPATH=tags/%1
    set VERSION=%1
)

set /p ANSWER=Release version "%VERSION%" (y/N)?
if "%ANSWER%"=="" goto :canceled
if /i "%ANSWER:~0,1%" NEQ "y" goto :canceled

pushd %~dp0

if /i "%VERSION:~0,1%"=="r" goto :skiptag
svn cp %REPOSITORY%/trunk %REPOSITORY%/tags/%VERSION%
if %ERRORLEVEL% GEQ 1 goto :failed
:skiptag

rmdir /s /q c:\tmp\liam2_new_release\
cd c:\tmp\
mkdir liam2_new_release
cd liam2_new_release

svn export %REPOSITORY%/%SVNPATH% build
if %ERRORLEVEL% GEQ 1 goto :failed

cd build
7z a -tzip ..\liam2-%VERSION%-src.zip *
call buildall
cd ..

mkdir html\%VERSION%

mkdir win32\documentation\html
mkdir win32\examples
mkdir win32\liam2
mkdir win32\editor

mkdir win64\documentation\html
mkdir win64\examples
mkdir win64\liam2
mkdir win64\editor

xcopy /e build\bundle\editor win32\editor
xcopy /e build\bundle\editor win64\editor

copy build\test\examples\* win32\examples
copy build\test\examples\* win64\examples

copy build\src\build\exe.win32-2.7\* win32\liam2
copy build\src\build\exe.win-amd64-2.7\* win64\liam2

copy build\doc\usersguide\build\LIAM2UserGuide.pdf win32\documentation
copy build\doc\usersguide\build\LIAM2UserGuide.pdf win64\documentation

copy build\doc\usersguide\build\LIAM2UserGuide.pdf LIAM2UserGuide-%VERSION%.pdf
copy build\doc\usersguide\build\LIAM2UserGuide.chm LIAM2UserGuide-%VERSION%.chm

xcopy /e build\doc\usersguide\build\html win32\documentation\html
xcopy /e build\doc\usersguide\build\html win64\documentation\html
xcopy /e build\doc\usersguide\build\html html\%VERSION%

cd win32
7z a -tzip ..\Liam2Suite-%VERSION%.win32.zip *
cd ..
cd win64
7z a -tzip ..\Liam2Suite-%VERSION%.win64.zip *
cd ..

rmdir /s /q win32
rmdir /s /q win64
rmdir /s /q build
popd
goto :end

:canceled
echo Aborted...
goto :end

:failed
popd
exit /b 1

:end