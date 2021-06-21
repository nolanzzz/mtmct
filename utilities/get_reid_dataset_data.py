import glob

import os.path as osp
import os
import pandas as pd
from utilities.pandas_loader import load_csv

class Get_reid_dataset_data:

    def __init__(self, dataset_test_folder
                 , dataset_train_folder
                 , reid_dataset_folder
                 , reid_data_output_path
                 , cam_ids
                 , work_dirs
                 , person_identifier="person_id"):

        self.dataset_train_folder = dataset_train_folder
        self.dataset_test_folder = dataset_test_folder
        self.reid_data_output_path = reid_data_output_path
        self.cam_ids = cam_ids
        self.work_dirs = work_dirs
        self.person_identifier = person_identifier
        self.reid_dataset_folder = reid_dataset_folder
        self.reid_folder_names = [ "distractors", "query", "test", "train" ]


    def create_csv(self):

        all_reid_frame_nos_gta = self.get_all_reid_frame_nos_gta()

        reid_dataframe = pd.DataFrame()
        def get_cam_coords_train_or_test(dataset_folder,frame_nos_gta):
            reid_train_or_test_dataframe = pd.DataFrame()
            for cam_id in self.cam_ids:
                cam_coords = load_csv(self.work_dirs, os.path.join(dataset_folder
                                                                   , "cam_{}".format(cam_id)
                                                                   , "coords_cam_{}.csv".format(cam_id)))

                coords = cam_coords[cam_coords["frame_no_gta"].isin(frame_nos_gta)]

                reid_train_or_test_dataframe = reid_train_or_test_dataframe.append(coords,ignore_index=True)

            return reid_train_or_test_dataframe

        train_df = get_cam_coords_train_or_test(dataset_folder=self.dataset_train_folder
                                                ,frame_nos_gta=all_reid_frame_nos_gta)

        reid_dataframe = reid_dataframe.append(train_df,ignore_index=True)

        test_df = get_cam_coords_train_or_test(dataset_folder=self.dataset_test_folder
                                                , frame_nos_gta=all_reid_frame_nos_gta)

        reid_dataframe = reid_dataframe.append(test_df,ignore_index=True)

        reid_dataframe.to_csv(path_or_buf=self.reid_data_output_path)



    def get_all_reid_frame_nos_gta(self):

        all_reid_frame_nos_gta = []
        for reid_folder_name in self.reid_folder_names:

            frame_nos_gta = self.get_frame_nos_gta(osp.join(self.reid_dataset_folder,reid_folder_name))


            all_reid_frame_nos_gta.extend(frame_nos_gta)

        return all_reid_frame_nos_gta



    def getAllDistractors(self):
        def getDistractors(testOrTrainSuffix):
            distractor_imgs_path = osp.join(self.reid_dataset_folder,osp.join("distractors/",testOrTrainSuffix))
            if not osp.exists(distractor_imgs_path):
                return set()
            img_paths = [ img_path for img_path in glob.glob(osp.join(distractor_imgs_path, '*.png')) ]

            return set(img_paths)

        train_img_names = getDistractors("train")
        test_img_names = getDistractors("test")
        query_img_names = getDistractors("query")

        return train_img_names.union(test_img_names,query_img_names)

    def get_frame_nos_gta(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))


        if os.path.basename(dir_path) == "distractors":
            img_paths = self.getAllDistractors()

        dataset = []
        for img_path in img_paths:


            # Image name structure framegta_2862738_camid_2_pid_2733.png
            img_name_no_ext = osp.basename(img_path).replace(".png", "")
            pid = int(img_name_no_ext.split("_")[5])
            camid = int(img_name_no_ext.split("_")[3])
            frame_no_gta = int(img_name_no_ext.split("_")[1])

            dataset.append(frame_no_gta)

        return dataset


if __name__ == "__main__":

    grdd = Get_reid_dataset_data(dataset_test_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_ext_short/test"
                                , dataset_train_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_ext_short/train"
                                 , reid_dataset_folder="/net/merkur/storage/deeplearning/users/koehl/gta/reid_camid_GTA_22.07.2019"
                                 , reid_data_output_path="/home/koehlp/Downloads/work_dirs/datasets/gta2207_reid_dataset_data/reid_data.csv"
                                 , cam_ids=range(6)
                                 , work_dirs="/home/koehlp/Downloads/work_dirs/")

    grdd = Get_reid_dataset_data(
        dataset_test_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test"
        , dataset_train_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/train"
        , reid_dataset_folder="/net/merkur/storage/deeplearning/users/koehl/gta/reid_camid_GTA_22.07.2019"
        , reid_data_output_path="/home/koehlp/Downloads/work_dirs/datasets/gta2207_reid_dataset_data/reid_data.csv"
        , cam_ids=range(6)
        , work_dirs="/home/koehlp/Downloads/work_dirs/")

    grdd.create_csv()