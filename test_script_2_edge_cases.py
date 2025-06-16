import os
import sys
import unittest.mock as mock

# Mock UI elements
sys.modules['easygui'] = mock.MagicMock()
sys.modules['tkinter'] = mock.MagicMock()
sys.modules['Tkinter'] = mock.MagicMock()
mock_easygui = mock.MagicMock()
mock_easygui.msgbox = mock.MagicMock()
mock_easygui.ynbox = mock.MagicMock()
sys.modules['easygui'] = mock_easygui

os.environ["COLAB_GPU"] = "True" # Simulate non-interactive environment

from kohya_gui.class_gui_config import KohyaSSGUIConfig
from kohya_gui.common_gui import get_folder_path, scriptdir # Assuming scriptdir is /app

CONFIG_PATH = "/app/config.toml"

print(f"Scriptdir is: {scriptdir}")

# --- Test 3: Corrupted last_used_folder path ---
print("\n--- Test 3: Corrupted last_used_folder path ---")
# 1. Create config.toml with a corrupted path
with open(CONFIG_PATH, "w") as f:
    f.write('last_used_folder = "this/is/not/a/valid/path"\n')
print(f"Created corrupted {CONFIG_PATH}")

# 2. Load config
config_corrupted = KohyaSSGUIConfig(config_file_path=CONFIG_PATH)
corrupted_val = config_corrupted.get_last_used_folder()
print(f"get_last_used_folder() with corrupted path returned: {corrupted_val}")
# get_last_used_folder itself just returns the value if it's a string.
assert corrupted_val == "this/is/not/a/valid/path"

# 3. Simulate a folder operation. common_gui.get_folder_path should handle invalid path gracefully.
# It should default to scriptdir because "this/is/not/a/valid/path" is not a valid directory.
# We can't directly check initial_dir_to_use without modifying common_gui,
# but we can infer it by checking that last_used_folder is NOT updated if the dialog was "cancelled" (empty input path)
# and that it IS updated if a new path is "selected".

# Simulate calling get_folder_path, where it would try to use the corrupted path, fail, and use scriptdir.
# If user then cancels (empty string for folder_path), it should not update the corrupted value.
# However, our non-dialog mode logic in get_folder_path for COLAB_GPU is:
#   if folder_path and os.path.isdir(folder_path) and config: config.set_last_used_folder(folder_path)
# So, if folder_path is empty, it won't update.
returned_path_corr1 = get_folder_path(folder_path="", config=config_corrupted) # Simulate empty input or cancel
assert config_corrupted.get_last_used_folder() == "this/is/not/a/valid/path", "Corrupted path should not change on empty selection"
print("Graceful handling of corrupted path (no selection) passed.")

# 4. Simulate selecting a valid folder
returned_path_corr2 = get_folder_path(folder_path="/tmp/good_folder", config=config_corrupted)
assert returned_path_corr2 == "/tmp/good_folder"
assert config_corrupted.get_last_used_folder() == "/tmp/good_folder", "Path should update to /tmp/good_folder"
print("Graceful handling of corrupted path (new selection) passed.")

# 5. Save and verify config file
config_corrupted.save_config(config=config_corrupted.config, config_file_path=CONFIG_PATH)
with open(CONFIG_PATH, "r") as f:
    content = f.read()
    print(f"config.toml content after fixing corrupted path:\n{content}")
    assert 'last_used_folder = "/tmp/good_folder"' in content
print("--- Test 3 Passed ---")


# --- Test 4: last_used_folder key missing ---
print("\n--- Test 4: last_used_folder key missing ---")
# 1. Create config.toml without the key (or with other keys)
with open(CONFIG_PATH, "w") as f:
    f.write('another_key = "some_value"\n') # No last_used_folder
print(f"Created {CONFIG_PATH} with missing last_used_folder key.")

# 2. Load config. load_config() should add last_used_folder = scriptdir
config_missing = KohyaSSGUIConfig(config_file_path=CONFIG_PATH)
val_after_load = config_missing.get_last_used_folder()
print(f"get_last_used_folder() after loading config with missing key: {val_after_load}")
# The get_last_used_folder() itself ensures it returns scriptdir if key is missing or value is bad type,
# and load_config also initializes it.
assert val_after_load == scriptdir, f"Expected scriptdir, got {val_after_load}"

# 3. Simulate a folder operation
returned_path_miss = get_folder_path(folder_path="/tmp/another_good_folder", config=config_missing)
assert returned_path_miss == "/tmp/another_good_folder"
assert config_missing.get_last_used_folder() == "/tmp/another_good_folder"
print("Graceful handling of missing key (new selection) passed.")

# 4. Save and verify
config_missing.save_config(config=config_missing.config, config_file_path=CONFIG_PATH)
with open(CONFIG_PATH, "r") as f:
    content = f.read()
    print(f"config.toml content after fixing missing key:\n{content}")
    assert 'last_used_folder = "/tmp/another_good_folder"' in content
    assert 'another_key = "some_value"' in content # Ensure other keys are preserved
print("--- Test 4 Passed ---")

print("\nAll Edge Case tests passed.")
