import os

bat_content = r'''@echo off
REM Example of how to start the GUI with custom arguments. In this case how to auto launch the browser:
REM call gui.bat --inbrowser
REM
REM You can add many arguments on the same line
REM
call gui.bat --inbrowser
'''

ps1_content = r'''# Example of how to start the GUI with custom arguments. In this case how to auto launch the browser:
# .\gui.ps1 --inbrowser
#
# You can add many arguments on the same line
#
# & .\gui.ps1 --inbrowser --server_port 2345

& .\gui.ps1 --inbrowser
'''

bat_filename = 'gui-user.bat'
ps1_filename = 'gui-user.ps1'

if not os.path.exists(bat_filename):
    with open(bat_filename, 'w') as bat_file:
        bat_file.write(bat_content)
    print(f"File created: {bat_filename}")
else:
    print(f"File already exists: {bat_filename}")

if not os.path.exists(ps1_filename):
    with open(ps1_filename, 'w') as ps1_file:
        ps1_file.write(ps1_content)
    print(f"File created: {ps1_filename}")
else:
    print(f"File already exists: {ps1_filename}")
