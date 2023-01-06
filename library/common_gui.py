from tkinter import filedialog, Tk
import os
import gradio as gr
from easygui import msgbox

def get_dir_and_file(file_path):
    dir_path, file_name = os.path.split(file_path)
    return (dir_path, file_name)

def get_file_path(file_path='', defaultextension='.json', extension_name='Config files'):
    current_file_path = file_path
    # print(f'current file path: {current_file_path}')
    
    initial_dir, initial_file = get_dir_and_file(file_path)

    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=((f'{extension_name}', f'{defaultextension}'), ('All files', '*')),
        defaultextension=defaultextension, initialfile=initial_file, initialdir=initial_dir
    )
    root.destroy()

    if file_path == '':
        file_path = current_file_path

    return file_path

def get_any_file_path(file_path=''):
    current_file_path = file_path
    # print(f'current file path: {current_file_path}')
    
    initial_dir, initial_file = get_dir_and_file(file_path)

    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir=initial_dir,
        initialfile=initial_file,)
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
    
    initial_dir, initial_file = get_dir_and_file(folder_path)

    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    folder_path = filedialog.askdirectory(initialdir=initial_dir)
    root.destroy()

    if folder_path == '':
        folder_path = current_folder_path

    return folder_path


def get_saveasfile_path(file_path='', defaultextension='.json', extension_name='Config files'):
    current_file_path = file_path
    # print(f'current file path: {current_file_path}')
    
    initial_dir, initial_file = get_dir_and_file(file_path)

    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    save_file_path = filedialog.asksaveasfile(
        filetypes=((f'{extension_name}', f'{defaultextension}'), ('All files', '*')),
        defaultextension=defaultextension,
        initialdir=initial_dir,
        initialfile=initial_file,
    )
    root.destroy()

    # print(save_file_path)

    if save_file_path == None:
        file_path = current_file_path
    else:
        print(save_file_path.name)
        file_path = save_file_path.name

    # print(file_path)

    return file_path

def get_saveasfilename_path(file_path='', extensions='*', extension_name='Config files'):
    current_file_path = file_path
    # print(f'current file path: {current_file_path}')
    
    initial_dir, initial_file = get_dir_and_file(file_path)

    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    save_file_path = filedialog.asksaveasfilename(filetypes=((f'{extension_name}', f'{extensions}'), ('All files', '*')),
        defaultextension=extensions,
        initialdir=initial_dir,
        initialfile=initial_file,
    )
    root.destroy()

    if save_file_path == '':
        file_path = current_file_path
    else:
        # print(save_file_path)
        file_path = save_file_path

    return file_path


def add_pre_postfix(
    folder='', prefix='', postfix='', caption_file_ext='.caption'
):
    if prefix == '' and postfix == '':
        return

    # set caption extention to default in case it was not provided
    if caption_file_ext == '':
        caption_file_ext = '.caption'

    files = [f for f in os.listdir(folder) if f.endswith(caption_file_ext)]
    if not prefix == '':
        prefix = f'{prefix} '
    if not postfix == '':
        postfix = f' {postfix}'

    for file in files:
        with open(os.path.join(folder, file), 'r+') as f:
            content = f.read()
            content = content.rstrip()
            f.seek(0, 0)
            f.write(f'{prefix}{content}{postfix}')
    f.close()

def color_aug_changed(color_aug):
    if color_aug:
        msgbox('Disabling "Cache latent" because "Color augmentation" has been selected...')
        return gr.Checkbox.update(value=False, interactive=False)
    else:
        return gr.Checkbox.update(value=True, interactive=True)