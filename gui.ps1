# Example command: .\gui.ps1 -server_port 8000 -inbrowser

param([string]$username="", [string]$password="", [switch]$inbrowser, [int]$server_port)
.\venv\Scripts\activate

if ($server_port -le 0 -and $inbrowser -eq $false) {
    Write-Host "Error: You must provide either the --server_port or --inbrowser argument."
    exit 1
}

python.exe kohya_gui.py --username $username --password $password --server_port $server_port --inbrowser