import os
from os.path import join

def get_system_path(path):

    path = path
    if os.name == 'nt': # Is windows
       # path = path.replace("\\","//")
       path = path

    abs_path = os.path.abspath(os.getcwd())
    abs_path = abs_path.replace("src", "")
    abs_path = abs_path.replace("notebooks", "")

    return join(abs_path, path)