# This powershell script will create a text file for each files in the folder
#
# Usefull to create base caption that will be augmented on a per image basis

$folder = "D:\dreambooth\train_sylvia_ritter\raw_data\all-images\"
$file_pattern="*.*"
$text_fir_file="a digital painting of xxx, by silvery trait"

$files = Get-ChildItem $folder$file_pattern
foreach ($file in $files) {New-Item -ItemType file -Path $folder -Name "$($file.BaseName).txt" -Value $text_fir_file}