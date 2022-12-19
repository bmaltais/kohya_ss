from tkinter import filedialog, Tk

def get_file_path(file_path='', defaultextension='.json'):
    current_file_path = file_path
    # print(f'current file path: {current_file_path}')
    
    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes = (("Config files", "*.json"), ("All files", "*")), defaultextension=defaultextension)
    root.destroy()
    
    if file_path == '':
        file_path = current_file_path

    return file_path


def remove_doublequote(file_path):
    if file_path != None:
        file_path = file_path.replace('"', '')

    return file_path


def get_folder_path(folder_path=''):
    current_folder_path = folder_path
    
    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    folder_path = filedialog.askdirectory()
    root.destroy()
    
    if folder_path == '':
        folder_path = current_folder_path

    return folder_path

def get_saveasfile_path(file_path='', defaultextension='.json'):
    current_file_path = file_path
    # print(f'current file path: {current_file_path}')
    
    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    file_path = filedialog.asksaveasfile(filetypes = (("Config files", "*.json"), ("All files", "*")), defaultextension=defaultextension)
    root.destroy()
    
    file_path = file_path.name
    if file_path == '':
        file_path = current_file_path

    return file_path