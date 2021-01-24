import os
from os.path import join

def get_system_path(path):

    path = path
    if os.name != 'nt': # Is Linux
       path = path.replace("\\","//")

    abs_path = os.path.abspath(os.getcwd())
    abs_path = abs_path.replace("src", "")
    abs_path = abs_path.replace("notebooks", "")

    return join(abs_path, path)

def get_current_path():
   return os.path.abspath(os.getcwd())

def join_paths(path, path2):
   return join(path, path2)