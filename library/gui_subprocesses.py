import sys
import tkinter as tk
from tkinter import filedialog, messagebox


def open_file_dialog(initial_dir=None, initial_file=None, file_types="all"):
    file_type_filters = {
        "all": [("All files", "*.*")],
        "video": [("Video files", "*.mp4;*.avi;*.mkv;*.mov;*.flv;*.wmv")],
        "images": [("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff")],
        "json": [("JSON files", "*.json")],
        "lora": [("LoRa files", "*.ckpt;*.pt;*.safetensors")],
        "directory": [],
    }

    if file_types in file_type_filters:
        filters = file_type_filters[file_types]
    else:
        filters = file_type_filters["all"]

    if file_types == "directory":
        return filedialog.askdirectory(initialdir=initial_dir)
    else:
        return filedialog.askopenfilename(initialdir=initial_dir, initialfile=initial_file, filetypes=filters)


def save_file_dialog(initial_dir, initial_file, files_type="all"):
    root = tk.Tk()
    root.withdraw()

    filetypes_switch = {
        "all": [("All files", "*.*")],
        "video": [("Video files", "*.mp4;*.avi;*.mkv;*.webm;*.flv;*.mov;*.wmv")],
        "images": [("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff;*.ico")],
        "json": [("JSON files", "*.json")],
        "lora": [("LoRa files", "*.ckpt;*.pt;*.safetensors")],
    }

    filetypes = filetypes_switch.get(files_type, filetypes_switch["all"])
    save_file_path = filedialog.asksaveasfilename(initialdir=initial_dir, initialfile=initial_file, filetypes=filetypes,
                                                  defaultextension=filetypes)

    root.destroy()

    return save_file_path


def show_message_box(_message, _title="Message", _level="info"):
    root = tk.Tk()
    root.withdraw()

    message_type = {
        "warning": messagebox.showwarning,
        "error": messagebox.showerror,
        "info": messagebox.showinfo,
        "question": messagebox.askquestion,
        "okcancel": messagebox.askokcancel,
        "retrycancel": messagebox.askretrycancel,
        "yesno": messagebox.askyesno,
        "yesnocancel": messagebox.askyesnocancel
    }

    if _level in message_type:
        message_type[_level](_title, _message)
    else:
        messagebox.showinfo(_title, _message)

    root.destroy()


if __name__ == '__main__':
    mode = sys.argv[1]

    if mode == 'file_dialog':
        starting_dir = sys.argv[2] if len(sys.argv) > 2 else None
        starting_file = sys.argv[3] if len(sys.argv) > 3 else None
        file_class = sys.argv[2] if len(sys.argv) > 2 else None
        file_path = open_file_dialog(starting_dir, starting_file, file_class)
        print(file_path)

    elif mode == 'msgbox':
        message = sys.argv[2]
        title = sys.argv[3] if len(sys.argv) > 3 else ""
        show_message_box(message, title)
