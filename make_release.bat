@echo off

rem USAGE: make_release [release_name] [branch]

rem Since git is a dvcs, we could make this script work locally, but it would
rem not be any more useful because making a release is usually to distribute it
rem to someone, and for that I need network access anyway.
rem Furthermore, cloning from the remote repository makes sure we do not
rem include any untracked file

rem make all environment variables we define local to this batch file and
rem reset ERRORLEVEL
setlocal

set REPOSITORY=https://github.com/liam2/liam2.git
rem git config --get remote.origin.url

if "%2"=="" (
    set BRANCH=master
) else (
    set BRANCH=%2
)

git log --format=format:%H origin/master..master | wc -l

set /p DOCONTINUE=do you want to push?es that last commit look right (ie. did
you push everything) (y/N)?

set GETREV=python get_rev.py %REPOSITORY% refs/heads/%BRANCH%
rem this is a hack to retrieve the output of a command in a variable
for /f "usebackq delims=" %%a in (`%GETREV%`) do set REV=%%a
if %ERRORLEVEL% GEQ 1 goto :failed

if "%1"=="" (
    rem take first 7 digits of commit hash
    set RELEASENAME=%REV:~0,7%
) else (
    set RELEASENAME=%1
)

set /p DORELEASE=Release version "%RELEASENAME%" (%REV%) (y/N)?
if "%DORELEASE%"=="" goto :canceled
if /i "%DORELEASE:~0,1%" NEQ "y" goto :canceled

pushd %~dp0

rmdir /s /q c:\tmp\liam2_new_release\
cd c:\tmp\
mkdir liam2_new_release
cd liam2_new_release

git clone -b %BRANCH% %REPOSITORY% build
if %ERRORLEVEL% GEQ 1 goto :failed

cd build

echo.
git log -1
echo.

set /p DOCONTINUE=does that last commit look right (ie. did you push everything) (y/N)?
if "%DOCONTINUE%"=="" goto :canceled
if /i "%DOCONTINUE:~0,1%" NEQ "y" goto :canceled

if /i "%RELEASENAME%"=="%REV:~0,7%" goto :skiptag

set TAGMSG=tag release %RELEASENAME%
git tag -a %RELEASENAME% -m "%TAGMSG%"
if %ERRORLEVEL% GEQ 1 goto :failed

:skiptag

git archive --format zip --output liam2-%RELEASENAME%-src.zip %REV%
if %ERRORLEVEL% GEQ 1 goto :failed

rem call buildall

cd ..
mkdir html\%RELEASENAME%

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

xcopy /e build\tests\examples win32\examples
xcopy /e build\tests\examples win64\examples

xcopy /e build\src\build\exe.win32-2.7 win32\liam2
xcopy /e build\src\build\exe.win-amd64-2.7 win64\liam2

copy build\doc\usersguide\build\LIAM2UserGuide.pdf win32\documentation
copy build\doc\usersguide\build\LIAM2UserGuide.pdf win64\documentation

copy build\doc\usersguide\build\LIAM2UserGuide.pdf LIAM2UserGuide-%RELEASENAME%.pdf
copy build\doc\usersguide\build\LIAM2UserGuide.chm LIAM2UserGuide-%RELEASENAME%.chm

xcopy /e build\doc\usersguide\build\html win32\documentation\html
xcopy /e build\doc\usersguide\build\html win64\documentation\html
xcopy /e build\doc\usersguide\build\html html\%RELEASENAME%

cd win32
7z a -tzip ..\Liam2Suite-%RELEASENAME%.win32.zip *
cd ..
cd win64
7z a -tzip ..\Liam2Suite-%RELEASENAME%.win64.zip *
cd ..

rmdir /s /q win32
rmdir /s /q win64

if "%TAGMSG%"=="" goto :nopush

set /p DOPUSH=is the release looking good (if so, the tag will be pushed) (y/N)?
if "%DOPUSH%"=="" goto :canceled
if /i "%DOPUSH:~0,1%" NEQ "y" goto :canceled

echo pushing tag...
rem git push
rem if %ERRORLEVEL% GEQ 1 goto :failed

:nopush
rmdir /s /q build
goto :end

:canceled
echo Aborted...
goto :end

:failed
echo Failed...
endlocal
popd
exit /b 1

:end
endlocal
popd
