# This powershell script will create a text file for each files in the folder
#
# Useful to create base caption that will be augmented on a per image basis

$folder = "D:\some\folder\location\"
$file_pattern="*.*"
$caption_text="some caption text"

$files = Get-ChildItem $folder$file_pattern -Include *.png, *.jpg, *.webp -File
foreach ($file in $files) {
    if (-not(Test-Path -Path $folder\"$($file.BaseName).txt" -PathType Leaf)) {
        New-Item -ItemType file -Path $folder -Name "$($file.BaseName).txt" -Value $caption_text
    }
}