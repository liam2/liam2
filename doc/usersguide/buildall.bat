@echo off
rem if "%1" == "" (
rem     echo Usage: buildall version
rem     exit /b
rem ) else (
rem     set VERSION=%1
rem )    

call make html

call make htmlhelp
pushd build\htmlhelp
hhc.exe LIAM2UserGuide.hhp > hhc.log
popd

call make latex
pushd build\latex
texify.exe --clean --pdf --tex-option=-synctex=1 --run-viewer LIAM2UserGuide.tex > texify.log
popd

rem get all the final results in build
pushd build
copy latex\LIAM2UserGuide.pdf .\LIAM2UserGuide.pdf
copy htmlhelp\LIAM2UserGuide.chm .\LIAM2UserGuide.chm
rem copy latex\LIAM2UserGuide.pdf .\LIAM2UserGuide-%VERSION%.pdf
rem copy htmlhelp\LIAM2UserGuide.chm .\LIAM2UserGuide-%VERSION%.chm
popd