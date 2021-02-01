import os
from os.path import join

def apply_system_format(path):
   """
      Transform the format of the path to the correct one, depending on the operating system where the code is running
   """
   if "//" in path and os.name == 'nt': # It is Windows
      path = path.replace("//","\\")
   elif "\\" in path and os.name != 'nt': # It is Linux
      path = path.replace("\\","//")

   abs_path = os.path.abspath(os.getcwd())
   abs_path = abs_path.replace("src", "")
   abs_path = abs_path.replace("notebooks", "")

   return join(abs_path, path)

def get_current_path():
   return os.path.abspath(os.getcwd())

def join_paths(path, path2):
   return join(path, path2)