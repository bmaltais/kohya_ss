import os
import sys
import unittest.mock as mock

# Mock easygui and its problematic dependencies before they are imported by kohya_gui
sys.modules['easygui'] = mock.MagicMock()
sys.modules['tkinter'] = mock.MagicMock()
sys.modules['Tkinter'] = mock.MagicMock() # For Python 2 fallback in easygui

# Attempt to preemptively mock parts of easygui that might still load
# This is to handle the "global_state" not found if easygui's __init__ tries to import its own modules
# that might then fail on tkinter not being truly available.
mock_easygui = mock.MagicMock()
mock_easygui.msgbox = mock.MagicMock()
mock_easygui.ynbox = mock.MagicMock()
sys.modules['easygui'] = mock_easygui


# Set an env var to simulate a non-interactive environment (though mocking should handle most UI calls)
os.environ["COLAB_GPU"] = "True"

from kohya_gui.class_gui_config import KohyaSSGUIConfig
from kohya_gui.common_gui import get_folder_path, scriptdir, get_file_path, get_saveasfilename_path # Import other functions as needed

print(f"Scriptdir resolved to: {scriptdir}")
# Ensure scriptdir is an absolute path for consistency, as it's used for defaults
if not os.path.isabs(scriptdir):
    scriptdir = os.path.abspath(scriptdir)
    print(f"Updated scriptdir to absolute path: {scriptdir}")


# --- Test 1: Initial state and first folder operation ---
print("--- Test 1: Initial Folder Operation ---")
config_handler = KohyaSSGUIConfig(config_file_path="/app/config.toml")
initial_last_folder = config_handler.get_last_used_folder()
print(f"Initial last_used_folder from new config object: {initial_last_folder}")
# Depending on KohyaSSGUIConfig implementation, initial value might be scriptdir or empty string
# before first save. get_last_used_folder itself initializes it to scriptdir if not in config.
assert initial_last_folder == scriptdir, f"Expected {scriptdir}, got {initial_last_folder}"


# Simulate selecting /tmp/test_folder1 for a folder operation
# In non-dialog mode (due to COLAB_GPU=True), get_folder_path should use the provided 'folder_path'
# and update the config.
returned_path = get_folder_path(folder_path="/tmp/test_folder1", config=config_handler)
print(f"Returned path from get_folder_path: {returned_path}")
assert returned_path == "/tmp/test_folder1"
current_last_folder = config_handler.get_last_used_folder()
print(f"Last used folder after get_folder_path: {current_last_folder}")
assert current_last_folder == "/tmp/test_folder1", f"Expected /tmp/test_folder1, got {current_last_folder}"

config_handler.save_config(config=config_handler.config, config_file_path="/app/config.toml")
print("Config saved.")

with open("/app/config.toml", "r") as f:
    content = f.read()
    print(f"config.toml content:\n{content}")
    assert 'last_used_folder = "/tmp/test_folder1"' in content
print("--- Test 1 Passed ---")

# --- Test 2: Relaunch and second folder operation ---
print("\n--- Test 2: Relaunch and Second Folder Operation ---")
config_handler_relaunch = KohyaSSGUIConfig(config_file_path="/app/config.toml")
relaunch_last_folder = config_handler_relaunch.get_last_used_folder()
print(f"On relaunch, last_used_folder is: {relaunch_last_folder}")
assert relaunch_last_folder == "/tmp/test_folder1", f"Expected /tmp/test_folder1, got {relaunch_last_folder}"

# Simulate selecting /tmp/test_folder2
returned_path_2 = get_folder_path(folder_path="/tmp/test_folder2", config=config_handler_relaunch)
print(f"Returned path from second get_folder_path: {returned_path_2}")
assert returned_path_2 == "/tmp/test_folder2"
current_last_folder_2 = config_handler_relaunch.get_last_used_folder()
print(f"Last used folder after second get_folder_path: {current_last_folder_2}")
assert current_last_folder_2 == "/tmp/test_folder2", f"Expected /tmp/test_folder2, got {current_last_folder_2}"

config_handler_relaunch.save_config(config=config_handler_relaunch.config, config_file_path="/app/config.toml")
print("Config saved for second operation.")

with open("/app/config.toml", "r") as f:
    content_2 = f.read()
    print(f"config.toml content after second operation:\n{content_2}")
    assert 'last_used_folder = "/tmp/test_folder2"' in content_2
print("--- Test 2 Passed ---")

print("\nInitial State Check (Folders) successful.")

# --- Test 3: File open operation ---
print("\n--- Test 3: File Open Operation ---")
# Ensure target directory for file exists
os.makedirs("/tmp/test_config_folder", exist_ok=True) # Changed from /projects to /tmp
dummy_file_path = "/tmp/test_config_folder/my_config.json" # Changed from /projects to /tmp
if not os.path.exists(dummy_file_path):
    with open(dummy_file_path, "w") as f: f.write("{}")

# Config handler is config_handler_relaunch, which has last_used_folder = /tmp/test_folder2
# Simulate opening a file
# In non-dialog mode, get_file_path will set last_used_folder to dirname of 'file_path'
returned_file_path = get_file_path(file_path=dummy_file_path, config=config_handler_relaunch)
print(f"Returned path from get_file_path: {returned_file_path}")
assert returned_file_path == dummy_file_path
current_last_folder_3 = config_handler_relaunch.get_last_used_folder()
print(f"Last used folder after get_file_path: {current_last_folder_3}")
assert current_last_folder_3 == "/tmp/test_config_folder", f"Expected /tmp/test_config_folder, got {current_last_folder_3}" # Changed from /projects to /tmp

config_handler_relaunch.save_config(config=config_handler_relaunch.config, config_file_path="/app/config.toml")
with open("/app/config.toml", "r") as f:
    content_3 = f.read()
    print(f"config.toml content after file open:\n{content_3}")
    assert 'last_used_folder = "/tmp/test_config_folder"' in content_3 # Changed from /projects to /tmp
print("--- Test 3 Passed ---")

# --- Test 4: File save operation ---
print("\n--- Test 4: File Save Operation ---")
# Ensure target directory for save exists
os.makedirs("/tmp/another_save_folder", exist_ok=True) # Changed from /projects to /tmp
save_file_as_path = "/tmp/another_save_folder/my_new_config.json" # Changed from /projects to /tmp

# Config handler still has last_used_folder = /tmp/test_config_folder
# Simulate saving a file
# In non-dialog mode, get_saveasfilename_path will set last_used_folder to dirname of 'file_path'
returned_save_path = get_saveasfilename_path(file_path=save_file_as_path, config=config_handler_relaunch)
print(f"Returned path from get_saveasfilename_path: {returned_save_path}")
assert returned_save_path == save_file_as_path # In non-dialog mode, it returns the path given
current_last_folder_4 = config_handler_relaunch.get_last_used_folder()
print(f"Last used folder after get_saveasfilename_path: {current_last_folder_4}")
assert current_last_folder_4 == "/tmp/another_save_folder", f"Expected /tmp/another_save_folder, got {current_last_folder_4}" # Changed from /projects to /tmp

config_handler_relaunch.save_config(config=config_handler_relaunch.config, config_file_path="/app/config.toml")
with open("/app/config.toml", "r") as f:
    content_4 = f.read()
    print(f"config.toml content after file save:\n{content_4}")
    assert 'last_used_folder = "/tmp/another_save_folder"' in content_4 # Changed from /projects to /tmp
print("--- Test 4 Passed ---")

print("\nAll Initial State and basic operations tests passed.")
