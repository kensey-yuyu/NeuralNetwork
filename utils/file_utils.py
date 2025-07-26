import os
import shutil
import stat


def copy_py_files(src_dir_paths: list, dest_dir_path: str) -> None:
    """
    Copy python files.

    Args:
        src_dir_paths (list): Directory path includes python files which we want to copy. 
        dest_dir_path (str): Directory path for copying.
    """

    for src_dir_path in src_dir_paths:
        for entry in os.listdir(src_dir_path):
            if not entry.endswith(".py"):
                continue
            src_file_path = os.path.join(src_dir_path, entry)
            rel_file_path = os.path.relpath(src_file_path, ".")
            dest_file_path = os.path.join(dest_dir_path, rel_file_path)

            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
            shutil.copyfile(src_file_path, dest_file_path)
    return


def remove_write(file_path: str):
    """
    Remove write permission from file.

    Args:
        file_path (str): File path.
    """

    current_mode = os.stat(file_path).st_mode
    new_mode = current_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH
    os.chmod(file_path, new_mode)
    return
