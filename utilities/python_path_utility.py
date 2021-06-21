import sys

import os

def append_to_pythonpath(paths,run_file_path):
    for path in paths:
        my_path = os.path.abspath(os.path.dirname(run_file_path))
        abs_path = os.path.join(my_path,path)
        sys.path.append(abs_path)
