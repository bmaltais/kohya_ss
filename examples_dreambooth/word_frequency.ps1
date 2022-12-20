$txt_files_folder = "D:\dreambooth\training_twq\mad_hatter\all"
$txt_prefix_to_ignore = "asd"
$txt_postfix_ti_ignore = "asd"

# Should not need to touch anything below

# (Get-Content $txt_files_folder"\*.txt" ).Replace(",", "") -Split '\W' | Group-Object -NoElement | Sort-Object -Descending -Property Count

$combined_txt = Get-Content $txt_files_folder"\*.txt"
$combined_txt = $combined_txt.Replace(",", "")
$combined_txt = $combined_txt.Replace("$txt_prefix_to_ignore", "")
$combined_txt = $combined_txt.Replace("$txt_postfix_ti_ignore", "") -Split '\W' | Group-Object -NoElement | Sort-Object -Descending -Property Count

Write-Output "Sorted by count"
Write-Output $combined_txt