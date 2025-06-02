import os
# Set an env var to simulate a non-interactive environment
os.environ["COLAB_GPU"] = "True"

from kohya_gui.class_gui_config import KohyaSSGUIConfig
from kohya_gui.common_gui import get_folder_path, scriptdir # scriptdir is used by get_last_used_folder as a default

# --- Test 1: Initial state and first operation ---
print("--- Test 1 ---")
# 1. Create config (simulating app launch, config file doesn't exist yet)
config_handler = KohyaSSGUIConfig(config_file_path="/app/config.toml")
# Ensure last_used_folder defaults to scriptdir (or an empty string, depending on implementation details)
# get_last_used_folder initializes it to scriptdir if not in config
print(f"Initial last_used_folder from new config object: {config_handler.get_last_used_folder()}")

# 2. Simulate get_folder_path where user "selects" /tmp/test_folder1
# In non-dialog mode, get_folder_path will set last_used_folder if folder_path is valid and config is present.
# We pass folder_path="/tmp/test_folder1" as if it was typed into a field and then an action triggered.
# Or, as if it's the path being "opened" or "selected".
returned_path = get_folder_path(folder_path="/tmp/test_folder1", config=config_handler)
print(f"Returned path from get_folder_path: {returned_path}")
print(f"Last used folder after get_folder_path: {config_handler.get_last_used_folder()}")

# 3. Save config
config_handler.save_config(config=config_handler.config, config_file_path="/app/config.toml")
print("Config saved.")

# Verify by reading the file content
with open("/app/config.toml", "r") as f:
    content = f.read()
    print(f"config.toml content:\n{content}")
    assert 'last_used_folder = "/tmp/test_folder1"' in content

print("--- Test 1 Passed ---")

# --- Test 2: Relaunch and second operation ---
print("\n--- Test 2 ---")
# 1. Simulate relaunch (load existing config)
config_handler_relaunch = KohyaSSGUIConfig(config_file_path="/app/config.toml")
print(f"On relaunch, last_used_folder is: {config_handler_relaunch.get_last_used_folder()}")
assert config_handler_relaunch.get_last_used_folder() == "/tmp/test_folder1"

# 2. Simulate opening the same dialog again. get_folder_path will determine initial_dir.
# We are interested in what initial_dir would be.
# The common_gui.get_folder_path function has this logic:
#   initial_dir_to_use = scriptdir
#   if config:
#       last_used = config.get_last_used_folder()
#       if last_used and os.path.isdir(last_used):
#           initial_dir_to_use = last_used
# So, initial_dir_to_use should become /tmp/test_folder1.
# (This part is harder to directly assert without refactoring get_folder_path to return initial_dir for testing,
# or by checking logs if we had more verbose logging for initial_dir decision)
# For now, we trust that get_last_used_folder returning /tmp/test_folder1 means the dialog would use it.

# 3. Simulate selecting another folder: /tmp/test_folder2
returned_path_2 = get_folder_path(folder_path="/tmp/test_folder2", config=config_handler_relaunch)
print(f"Returned path from second get_folder_path: {returned_path_2}")
print(f"Last used folder after second get_folder_path: {config_handler_relaunch.get_last_used_folder()}")

# 4. Save config again
config_handler_relaunch.save_config(config=config_handler_relaunch.config, config_file_path="/app/config.toml")
print("Config saved for second operation.")

# Verify by reading the file content
with open("/app/config.toml", "r") as f:
    content_2 = f.read()
    print(f"config.toml content after second operation:\n{content_2}")
    assert 'last_used_folder = "/tmp/test_folder2"' in content_2

print("--- Test 2 Passed ---")

print("\nInitial State Check successful.")
