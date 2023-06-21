import argparse
import subprocess

# Define color variables
yellow_text = "\033[1;33m"
blue_text = "\033[1;34m"
reset_text = "\033[0m"

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--no_question', action='store_true')
args = parser.parse_args()

# Run pip freeze and capture the output
output = subprocess.getoutput("pip freeze")

# Remove lines containing "WARNING"
output_lines = [line for line in output.splitlines() if "WARNING" not in line]

# Reconstruct the output string without warning lines
output = "\n".join(output_lines)

# Check if modules are found in the output
if output:
    print(f"{yellow_text}=============================================================")
    print("Modules installed outside the virtual environment were found.")
    print("This can cause issues. Please review the installed modules.\n")
    print("You can uninstall all local modules with:\n")
    print(f"{blue_text}deactivate")
    print("pip freeze > uninstall.txt")
    print("pip uninstall -y -r uninstall.txt")
    print(f"{yellow_text}============================================================={reset_text}")
    print('')
