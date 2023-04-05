# Check if there are any changes that need to be committed
if (git status --short) {
    Write-Error "There are changes that need to be committed. Please stash or undo your changes before running this script."
    return
}

# Pull the latest changes from the remote repository
git pull

# Activate the virtual environment
.\venv\Scripts\activate

# Upgrade the required packages
pip install --use-pep517 --upgrade -r requirements.txt