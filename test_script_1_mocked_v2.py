import os
import sys
import unittest.mock as mock

# Mock easygui and its problematic dependencies
sys.modules['easygui'] = mock.MagicMock()
sys.modules['tkinter'] = mock.MagicMock()
sys.modules['Tkinter'] = mock.MagicMock()
mock_easygui = mock.MagicMock()
mock_easygui.msgbox = mock.MagicMock()
mock_easygui.ynbox = mock.MagicMock()
sys.modules['easygui'] = mock_easygui

os.environ["COLAB_GPU"] = "True" # Simulate non-interactive environment

# Check if config.toml exists before KohyaSSGUIConfig is initialized
CONFIG_PATH = "/app/config.toml"
if os.path.exists(CONFIG_PATH):
    print(f"WARNING: {CONFIG_PATH} exists before test! Content:")
    with open(CONFIG_PATH, "r") as f:
        print(f.read())
    # Attempt to remove it again just in case
    os.remove(CONFIG_PATH)
    print(f"WARNING: Removed pre-existing {CONFIG_PATH}")
else:
    print(f"{CONFIG_PATH} does not exist before test, as expected.")

from kohya_gui.class_gui_config import KohyaSSGUIConfig
from kohya_gui.common_gui import get_folder_path, scriptdir, get_file_path, get_saveasfilename_path

print(f"Scriptdir resolved to: {scriptdir}")
# Ensure scriptdir is an absolute path for consistency
if not os.path.isabs(scriptdir): # This should not happen if common_gui.py defines it as absolute
    scriptdir = os.path.abspath(scriptdir) # Make it absolute for safety in test logic
    print(f"Updated scriptdir to absolute path: {scriptdir}")


# --- Test 1: Initial state and first folder operation ---
print("--- Test 1: Initial Folder Operation ---")
config_handler = KohyaSSGUIConfig(config_file_path=CONFIG_PATH)
initial_last_folder = config_handler.get_last_used_folder()
print(f"Initial last_used_folder from new config object: {initial_last_folder}")
assert initial_last_folder == scriptdir, f"Expected {scriptdir}, got {initial_last_folder}"

returned_path = get_folder_path(folder_path="/tmp/test_folder1", config=config_handler)
print(f"Returned path from get_folder_path: {returned_path}")
assert returned_path == "/tmp/test_folder1"
current_last_folder = config_handler.get_last_used_folder()
print(f"Last used folder after get_folder_path: {current_last_folder}")
assert current_last_folder == "/tmp/test_folder1", f"Expected /tmp/test_folder1, got {current_last_folder}"

config_handler.save_config(config=config_handler.config, config_file_path=CONFIG_PATH)
print("Config saved.")

with open(CONFIG_PATH, "r") as f:
    content = f.read()
    print(f"config.toml content:\n{content}")
    assert 'last_used_folder = "/tmp/test_folder1"' in content
print("--- Test 1 Passed ---")

# --- Test 2: Relaunch and second folder operation ---
print("\n--- Test 2: Relaunch and Second Folder Operation ---")
config_handler_relaunch = KohyaSSGUIConfig(config_file_path=CONFIG_PATH)
relaunch_last_folder = config_handler_relaunch.get_last_used_folder()
print(f"On relaunch, last_used_folder is: {relaunch_last_folder}")
assert relaunch_last_folder == "/tmp/test_folder1", f"Expected /tmp/test_folder1, got {relaunch_last_folder}"

returned_path_2 = get_folder_path(folder_path="/tmp/test_folder2", config=config_handler_relaunch)
print(f"Returned path from second get_folder_path: {returned_path_2}")
assert returned_path_2 == "/tmp/test_folder2"
current_last_folder_2 = config_handler_relaunch.get_last_used_folder()
print(f"Last used folder after second get_folder_path: {current_last_folder_2}")
assert current_last_folder_2 == "/tmp/test_folder2", f"Expected /tmp/test_folder2, got {current_last_folder_2}"

config_handler_relaunch.save_config(config=config_handler_relaunch.config, config_file_path=CONFIG_PATH)
print("Config saved for second operation.")

with open(CONFIG_PATH, "r") as f:
    content_2 = f.read()
    print(f"config.toml content after second operation:\n{content_2}")
    assert 'last_used_folder = "/tmp/test_folder2"' in content_2
print("--- Test 2 Passed ---")

print("\nInitial State Check (Folders) successful.")

# --- Test 3: File open operation ---
print("\n--- Test 3: File Open Operation ---")
os.makedirs("/tmp/test_config_folder", exist_ok=True)
dummy_file_path = "/tmp/test_config_folder/my_config.json"
if not os.path.exists(dummy_file_path):
    with open(dummy_file_path, "w") as f: f.write("{}")

returned_file_path = get_file_path(file_path=dummy_file_path, config=config_handler_relaunch)
print(f"Returned path from get_file_path: {returned_file_path}")
assert returned_file_path == dummy_file_path
current_last_folder_3 = config_handler_relaunch.get_last_used_folder()
print(f"Last used folder after get_file_path: {current_last_folder_3}")
assert current_last_folder_3 == "/tmp/test_config_folder", f"Expected /tmp/test_config_folder, got {current_last_folder_3}"

config_handler_relaunch.save_config(config=config_handler_relaunch.config, config_file_path=CONFIG_PATH)
with open(CONFIG_PATH, "r") as f:
    content_3 = f.read()
    print(f"config.toml content after file open:\n{content_3}")
    assert 'last_used_folder = "/tmp/test_config_folder"' in content_3
print("--- Test 3 Passed ---")

# --- Test 4: File save operation ---
print("\n--- Test 4: File Save Operation ---")
os.makedirs("/tmp/another_save_folder", exist_ok=True)
save_file_as_path = "/tmp/another_save_folder/my_new_config.json"

returned_save_path = get_saveasfilename_path(file_path=save_file_as_path, config=config_handler_relaunch)
print(f"Returned path from get_saveasfilename_path: {returned_save_path}")
assert returned_save_path == save_file_as_path
current_last_folder_4 = config_handler_relaunch.get_last_used_folder()
print(f"Last used folder after get_saveasfilename_path: {current_last_folder_4}")
assert current_last_folder_4 == "/tmp/another_save_folder", f"Expected /tmp/another_save_folder, got {current_last_folder_4}"

config_handler_relaunch.save_config(config=config_handler_relaunch.config, config_file_path=CONFIG_PATH)
with open(CONFIG_PATH, "r") as f:
    content_4 = f.read()
    print(f"config.toml content after file save:\n{content_4}")
    assert 'last_used_folder = "/tmp/another_save_folder"' in content_4
print("--- Test 4 Passed ---")

print("\nAll Initial State and basic operations tests passed.")
