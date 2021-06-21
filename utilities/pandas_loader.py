import pandas as pd

import os
import os.path as osp


def load_csv(working_dir,csv_path):

    cache_folder_name = "cache/"
    cache_folder_path = osp.join(working_dir,cache_folder_name)
    os.makedirs(cache_folder_path,exist_ok=True)

    pickle_file_name = os.path.basename(csv_path) + ".pkl"

    csv_folder = os.path.dirname(csv_path)

    csv_folder = csv_folder[1:]

    cache_pickle_folder = osp.join(cache_folder_path,csv_folder)

    os.makedirs(cache_pickle_folder,exist_ok=True)

    pickle_path = osp.join(cache_pickle_folder,pickle_file_name)



    if not os.path.isfile(pickle_path):
        print("pkl not found: {}".format(pickle_path))
        dataframe = pd.read_csv(csv_path)
        dataframe.to_pickle(pickle_path)
    else:
        print("pkl found: {}".format(pickle_path))
        dataframe = pd.read_pickle(pickle_path)

    return dataframe


if __name__ == "__main__":
    load_csv("/home/koehlp/Downloads/work_dirs","/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test/cam_0/coords_cam_0.csv" )