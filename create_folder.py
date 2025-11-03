import os

def mkdir(default_path, folder_name):
    path = os.path.join(default_path, folder_name)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
