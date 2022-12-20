$date = Read-Host "Enter the date (yyyy-mm-dd):" -Prompt "Invalid date format. Please try again (yyyy-mm-dd):" -ValidateScript {
    # Parse the date input and return $true if it is in the correct format,
    # or $false if it is not
    $date = [DateTime]::Parse($_)
    return $date -ne $null
}