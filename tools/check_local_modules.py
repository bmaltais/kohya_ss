import argparse
import subprocess
import sys

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

    if args.no_question:
        sys.exit(1)  # Exit with code 1 for "no" without asking the user

    # Ask the user if they want to continue
    valid_input = False
    while not valid_input:
        print('Do you want to continue?')
        print('')
        print('[1] - Yes')
        print('[2] - No')
        user_input = input("Enter your choice (1 or 2): ")
        if user_input.lower() == "2":
            valid_input = True
            sys.exit(1)  # Exit with code 1 for "no"
        elif user_input.lower() == "1":
            valid_input = True
            sys.exit(0)  # Exit with code 0 for "yes"
        else:
            print("Invalid input. Please enter '1' or '2'.")
else:
    sys.exit(0)
