from easygui import diropenbox, fileopenbox

def get_folder_path():
    folder_path = diropenbox("Select the directory to use")

    return folder_path

def remove_doublequote(file_path):
    if file_path != None:
        file_path = file_path.replace('"', "")

    return file_path

def get_file_path(file_path):
    file_path = fileopenbox("Select the config file to load",
                            default=file_path,
                            filetypes="*.json")

    return file_path