from kohya_gui.class_gui_config import KohyaSSGUIConfig
from kohya_gui.common_gui import get_folder_path, scriptdir
import os

# Ensure the target directory exists
os.makedirs("/tmp/test_folder1", exist_ok=True)

# 1. Create config (simulating app launch)
config_handler = KohyaSSGUIConfig(config_file_path="/app/config.toml")
print(f"Initial last_used_folder from config: {config_handler.get_last_used_folder()}") # Should be scriptdir

# 2. Simulate get_folder_path
# Normally, tkinter dialog would run. We simulate its effect.
# get_folder_path internally calls config.set_last_used_folder
# Let's assume the dialog was opened with initialdir=scriptdir and user selected /tmp/test_folder1
# For the test, we'll directly set it after a simulated selection to mimic the function's behavior
# In a real scenario, get_folder_path would be called, and it would call set_last_used_folder.
# Here, we simplify by directly manipulating for test verification.

# Simulate call to get_folder_path where user selects /tmp/test_folder1
# This is a simplified mock of what would happen:
# initial_dir = config_handler.get_last_used_folder() # This would be scriptdir
# print(f"Dialog would open with initial_dir: {initial_dir}")
# selected_path_by_user = "/tmp/test_folder1" # User selects this
# if selected_path_by_user:
#     config_handler.set_last_used_folder(selected_path_by_user)
#     print(f"Set last_used_folder to: {selected_path_by_user}")
# This is what happens inside get_folder_path:
def mock_get_folder_path(current_path_in_field, cfg_obj, simulated_user_selection):
    # Logic from common_gui.get_folder_path for initial_dir_to_use
    initial_dir_to_use = scriptdir
    if cfg_obj:
        last_used = cfg_obj.get_last_used_folder()
        if last_used and os.path.isdir(last_used):
            initial_dir_to_use = last_used
        elif current_path_in_field and os.path.isdir(current_path_in_field):
            initial_dir_to_use = current_path_in_field
    elif current_path_in_field and os.path.isdir(current_path_in_field):
        initial_dir_to_use = current_path_in_field
    print(f"Dialog would open with initial_dir: {initial_dir_to_use}")

    # Simulate user selection
    if simulated_user_selection:
        if cfg_obj:
            cfg_obj.set_last_used_folder(simulated_user_selection)
            print(f"Set last_used_folder to: {simulated_user_selection}")
        return simulated_user_selection
    return current_path_in_field

# Simulate the scenario: field is empty, user selects /tmp/test_folder1
returned_path = mock_get_folder_path("", config_handler, "/tmp/test_folder1")
print(f"Returned path: {returned_path}")


# 3. Save config
config_handler.save_config(config=config_handler.config, config_file_path="/app/config.toml")
print("Config saved.")

# Verify by reloading
config_handler_verify = KohyaSSGUIConfig(config_file_path="/app/config.toml")
print(f"After save, last_used_folder from new config: {config_handler_verify.get_last_used_folder()}")
