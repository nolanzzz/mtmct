from feature_extractors.reid_strong_extractor import Reid_strong_extractor
from tqdm import tqdm
import os.path as osp
import glob
from collections import defaultdict
import os
import cv2
import mmcv
import numpy as np
import pickle
from PIL import Image

import matplotlib.pyplot as plt

class Visualize_reid_query:

    def __init__(self,reid_dataset_folder,feature_extractor_cfg,work_dirs):

        self.reid_folder_names = ["distractors", "query", "test", "train"]

        self.reid_dataset_folder = reid_dataset_folder

        self.feature_extractor_cfg = feature_extractor_cfg

        self.work_dirs = work_dirs

    def get_reid_folder_images(self):

        folder_name_to_image_names = defaultdict(list)
        for reid_folder_name in self.reid_folder_names:

            folder_name_to_image_names[reid_folder_name] = self.get_image_names(
                osp.join(self.reid_dataset_folder,reid_folder_name))




        return folder_name_to_image_names

    def filter_distractors(self,folder_name_to_image_names):
        filtered_folder_name_to_image_names = defaultdict(list)

        distractors = set(folder_name_to_image_names["distractors"])
        for folder_name, image_names in folder_name_to_image_names.items():
            if folder_name != "distractors":
                image_names_without_distractors = set(image_names).difference(distractors)
                filtered_folder_name_to_image_names[folder_name] = list(image_names_without_distractors)

        return filtered_folder_name_to_image_names

    def get_features_pickle_path(self):

        pickle_folder =  osp.join(self.work_dirs
                         ,"visualizations"
                         ,"reid")

        os.makedirs(pickle_folder,exist_ok=True)
        pickle_path = osp.join(pickle_folder,"reid_features.pkl")
        return pickle_path

    def get_reid_image_path(self,query_image_name):
        pickle_folder = osp.join(self.work_dirs
                                 , "visualizations"
                                 , "reid")

        os.makedirs(pickle_folder, exist_ok=True)
        pickle_path = osp.join(pickle_folder, "query_visualization_{}.jpg".format(query_image_name))
        return pickle_path



    def get_gallery_distances_to_query(self,query_image,gallery_images):

        query_feature = query_image[1]
        result_image_dist = []
        for gallery_image in gallery_images:
            gallery_image_feature = gallery_image[1]
            dist = np.linalg.norm(query_feature - gallery_image_feature)
            result_image_dist.append((gallery_image[0],dist))

        result_image_dist.sort(key=lambda path_dist:path_dist[1])

        return result_image_dist



    def add_border(image, target_shape,src_image,r=0,g=0,b=0):
        r_channel = r*np.ones((target_shape[0], target_shape[1], 1), np.uint8)
        g_channel = g*np.ones((target_shape[0], target_shape[1], 1), np.uint8)
        b_channel = b*np.ones((target_shape[0], target_shape[1], 1), np.uint8)

        blank_image = np.concatenate((b_channel, g_channel,r_channel), axis=2)
        new_width = target_shape[1]
        width = src_image.shape[1]
        new_height = target_shape[0]
        height = src_image.shape[0]
        center_x = int((new_width - width) / 2)
        center_y = int((new_height - height) / 2)
        blank_image[center_y:center_y+src_image.shape[0], center_x:center_x+src_image.shape[1]] = src_image
        return blank_image

    def find_maximum_shape(self,image_list):
        max_row_size = max(image_list,key=lambda img:img.shape[0])
        max_col_size = max(image_list,key=lambda img:img.shape[1])
        return (max_row_size.shape[0],max_col_size.shape[1])

    def add_number_to_images(self,images):

        for i,img in enumerate(images):
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (0, 15)
            fontScale = 0.5
            fontColor = (255, 255, 255)
            lineType = 2

            cv2.putText(img, str(i),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)


    def add_border_to_images(self,max_rows_cols,width_border,images,images_to_draw):
        result_images = []

        query_image_pid = self.get_pid_from_image_name(images_to_draw[0][0])
        for i,img in enumerate(images):
            green = 255
            red = 255
            blue = 255
            if i > 0:
                if query_image_pid == self.get_pid_from_image_name(images_to_draw[i][0]):
                    green = 255
                    red = 0
                    blue = 0
                else:
                    green = 0
                    red = 255
                    blue = 0





            result_img = self.add_border(target_shape=(max_rows_cols[0] + width_border
                                                        ,img.shape[1] + width_border)
                                                        ,src_image=img
                                                        ,r=red
                                                        ,g=green
                                                        ,b=blue)

            result_images.append(result_img)


        return result_images

    def get_pid_from_image_name(self,image_name):

        #Image name structure framegta_2862738_camid_2_pid_2733.png
        return int(image_name.replace(".png","").split("_")[5])


    def draw_images(self,query_image,image_distances):

        query_image_path = osp.join(self.reid_dataset_folder,"query",query_image[0])
        query_img_loaded = cv2.imread(query_image_path)
        gallery_loaded_images = []

        gallery_loaded_images.append(query_img_loaded)
        images_to_draw = image_distances[:10]
        for img_dist in images_to_draw:
            gallery_image_path = osp.join(self.reid_dataset_folder,"test",img_dist[0])
            gallery_img_loaded = cv2.imread(gallery_image_path)
            #cv2.imshow("gallery result", gallery_img_loaded)
            #cv2.waitKey(0)
            gallery_loaded_images.append(gallery_img_loaded)

        images_to_draw = [query_image] + images_to_draw
        max_rows_cols = self.find_maximum_shape(gallery_loaded_images)

        #Add a border to be able to concatenate the images

        gallery_loaded_images = self.add_border_to_images(max_rows_cols=max_rows_cols
                                      ,width_border=10
                                      ,images=gallery_loaded_images
                                      ,images_to_draw=images_to_draw)

        self.add_number_to_images(images=gallery_loaded_images)


        result_image = np.concatenate(gallery_loaded_images, axis=1)

        image_path = self.get_reid_image_path(query_image_name=query_image[0])

        cv2.imwrite(filename=image_path,img=result_image)
        cv2.imshow("query result",result_image)
        #cv2.waitKey(0)



    def draw_gallery_query(self,query_image_index=888):
        folder_name_to_features = self.get_reid_features()
        query_images = folder_name_to_features["query"]
        gallery_images = folder_name_to_features["test"]

        for i in range(query_image_index):

            query_image = query_images[i]
            image_distances = self.get_gallery_distances_to_query(query_image=query_image
                                                                  ,gallery_images=gallery_images)

            self.draw_images(query_image=query_image
                             ,image_distances=image_distances)




    def get_reid_features(self):

        pickle_path = self.get_features_pickle_path()

        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as pickle_file:
                folder_name_to_image_names = pickle.load(pickle_file)
                return folder_name_to_image_names

        folder_name_to_image_names = self.get_reid_folder_images()

        folder_name_to_image_names = self.filter_distractors(folder_name_to_image_names)

        self.feature_extractor = Reid_strong_extractor(self.feature_extractor_cfg)

        folder_name_to_features = defaultdict(list)
        for folder_name, image_names in folder_name_to_image_names.items():
            folder_features = []
            print("Converting images from {} to features".format(folder_name))
            for image_name in tqdm(image_names):

                image_path = osp.join(self.reid_dataset_folder,folder_name,image_name)

                reid_img = cv2.imread(image_path)

                image_feature = self.feature_extractor.extract([reid_img])

                folder_features.append((image_name,image_feature))

            folder_name_to_features[folder_name] = folder_features



        if pickle_path is not None:
            with open(pickle_path, "wb") as pickle_file:
                pickle.dump(folder_name_to_features, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


        return folder_name_to_features


    def get_distractor_image_names(self):
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

    def get_image_names(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))


        if os.path.basename(dir_path) == "distractors":
            img_paths = self.get_distractor_image_names()

        dataset = []
        for img_path in img_paths:
            # Image name structure framegta_2862738_camid_2_pid_2733.png
            img_name = osp.basename(img_path)
            dataset.append(img_name)

        return dataset

if __name__ == "__main__":

    feature_extractor_cfg = {"feature_extractor": {
            "reid_strong_baseline_config": "/home/philipp/Dokumente/masterarbeit/JTA-MTMCT-Mod/deep_sort_mc/feature_extractors/reid-strong-baseline/configs/softmax_triplet.yml",
            "checkpoint_file": "/media/philipp/philippkoehl_ssd/work_dirs/feature_extractor/strong_reid_baseline/resnet50_model_reid_GTA_softmax_triplet.pth",
            "device": "cuda:0"
            ,"visible_device" : "0"}}

    feature_ext_cfg = mmcv.Config(feature_extractor_cfg)


    vrq = Visualize_reid_query(reid_dataset_folder="/media/philipp/philippkoehl_ssd/reid_camid_GTA_22.07.2019"
                                ,feature_extractor_cfg=feature_ext_cfg
                                ,work_dirs="/media/philipp/philippkoehl_ssd/work_dirs")

    vrq.draw_gallery_query()