import os
import pathlib
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

from library.common_gui_functions import tk_context
from library.common_utilities import CommonUtilities


class TkGui:
    def __init__(self):
        self.file_types = None

    def open_file_dialog(self, initial_dir=None, initial_file=None, file_types="all"):
        print(f"File types: {self.file_types}")
        with tk_context():
            self.file_types = file_types
            if self.file_types in CommonUtilities.file_filters:
                filters = CommonUtilities.file_filters[self.file_types]
            else:
                filters = CommonUtilities.file_filters["all"]

            if self.file_types == "directory":
                return filedialog.askdirectory(initialdir=initial_dir)
            else:
                return filedialog.askopenfilename(initialdir=initial_dir, initialfile=initial_file, filetypes=filters)

    def save_file_dialog(self, initial_dir, initial_file, file_types="all"):
        self.file_types = file_types

        # Use the tk_context function with the 'with' statement
        with tk_context():
            if self.file_types in CommonUtilities.file_filters:
                filters = CommonUtilities.file_filters[self.file_types]
            else:
                filters = CommonUtilities.file_filters["all"]

            save_file_path = filedialog.asksaveasfilename(initialdir=initial_dir, initialfile=initial_file,
                                                          filetypes=filters, defaultextension=".safetensors")

        return save_file_path

    def show_message_box(_message, _title="Message", _level="info"):
        with tk_context():
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


if __name__ == '__main__':
    try:
        mode = sys.argv[1]

        if mode == 'file_dialog':
            starting_dir = sys.argv[2] if len(sys.argv) > 2 else None
            starting_file = sys.argv[3] if len(sys.argv) > 3 else None
            file_class = sys.argv[4] if len(sys.argv) > 4 else None  # Update this to sys.argv[4]
            gui = TkGui()
            file_path = gui.open_file_dialog(starting_dir, starting_file, file_class)
            print(file_path)  # Make sure to print the result

        elif mode == 'msgbox':
            message = sys.argv[2]
            title = sys.argv[3] if len(sys.argv) > 3 else ""
            gui = TkGui()
            gui.show_message_box(message, title)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
