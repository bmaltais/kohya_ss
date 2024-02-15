# This powershell script will create a text file for each files in the folder
#
# Useful to create base caption that will be augmented on a per image basis

$folder = "D:\test\t2\"
$file_pattern="*.*"
$text_fir_file="bigeyes style"

foreach ($file in Get-ChildItem $folder\$file_pattern -File)
{
    New-Item -ItemType file -Path $folder -Name "$($file.BaseName).txt" -Value $text_fir_file
}

foreach($directory in Get-ChildItem -path $folder -Directory)
{
    foreach ($file in Get-ChildItem $folder\$directory\$file_pattern)
    {
        New-Item -ItemType file -Path $folder\$directory -Name "$($file.BaseName).txt" -Value $text_fir_file
    }
}
