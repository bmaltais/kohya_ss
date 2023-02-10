@echo off

REM Use this batch file with the following options:
REM -inbrowser - To launch the program in the browser
REM -server_port [port number] - To specify the server port

set inbrowserOption=
set serverPortOption=

if "%1" == "-server_port" (
  set serverPortOption=--server_port %2
  if "%3" == "-inbrowser" (
    set inbrowserOption=--inbrowser
  )
) else if "%1" == "-inbrowser" (
  set inbrowserOption=--inbrowser
  if "%2" == "-server_port" (
    set serverPortOption=--server_port %3
  )
)

call .\venv\Scripts\activate.bat
python.exe kohya_gui.py %inbrowserOption% %serverPortOption%
