$txt_files_folder = "D:\dreambooth\train_sylvia_ritter\raw_data\all-images"
$txt_prefix_to_ignore = "a digital drawing of"
$txt_postfix_ti_ignore = "by silvery trait"

# Should not need to touch anything below

# (Get-Content $txt_files_folder"\*.txt" ).Replace(",", "") -Split '\W' | Group-Object -NoElement | Sort-Object -Descending -Property Count

$combined_txt = Get-Content $txt_files_folder"\*.txt"
$combined_txt = $combined_txt.Replace(",", "")
$combined_txt = $combined_txt.Replace("$txt_prefix_to_ignore", "")
$combined_txt = $combined_txt.Replace("$txt_postfix_ti_ignore", "") -Split '\W' | Group-Object -NoElement | Sort-Object -Descending -Property Count

Write-Output "Sorted by count"
Write-Output $combined_txt

Write-Output "Sorted by words"
$combined_txt = $combined_txt | Sort-Object -Property Name
Write-Output $combined_txt
