@echo off
rem HTML with analytics
rem call make fpb_web
call make html

rem call make htmlhelp
rem pushd build\htmlhelp
rem hhc.exe LIAM2UserGuide.hhp > hhc.log
rem popd

rem call make latex
rem pushd build\latex
rem rem we can also use --run-viewer to open the pdf
rem texify.exe --clean --pdf --tex-option=-synctex=1 LIAM2UserGuide.tex > texify.log
rem popd
