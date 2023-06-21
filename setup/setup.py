#
# This is mostly used as part of the Docker build process
#

from setuptools import setup, find_packages
import subprocess
import os
import sys

# Call the create_user_files.py script
script_path = os.path.join("./", "create_user_files.py")
subprocess.run([sys.executable, script_path])

setup(name="library", version="1.0.3", packages=find_packages())
