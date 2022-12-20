from tkinter import filedialog, Tk
import os


def get_file_path(file_path='', defaultextension='.json'):
    current_file_path = file_path
    # print(f'current file path: {current_file_path}')

    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=(('Config files', '*.json'), ('All files', '*')),
        defaultextension=defaultextension,
    )
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
    save_file_path = filedialog.asksaveasfile(
        filetypes=(('Config files', '*.json'), ('All files', '*')),
        defaultextension=defaultextension,
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


def add_pre_postfix(
    folder='', prefix='', postfix='', caption_file_ext='.caption'
):
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
