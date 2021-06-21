import os.path as osp
import os
import cv2
from utilities.pandas_loader import load_csv
from shutil import copyfile
from tqdm import tqdm
import pandas as pd

def copy_renamed_mugshot_images(mugshot_base_folder,mugshot_subfolder_names, dataset_folder, output_folder,cam_ids, working_dir):

    def group_drop_except_person_id_and_appearance_id(cam_dataframe):
        cam_dataframe = cam_dataframe.groupby(by=["person_id", "appearance_id"], as_index=False).mean()
        cam_dataframe = cam_dataframe[["person_id"
            , "appearance_id"]]
        cam_dataframe = cam_dataframe.astype(int)

        return cam_dataframe



    def get_combined_cam_dataframes(dataset_folder):
        def get_combined_one_type(dataset_folder):
            combined_cam_dataframes = pd.DataFrame()
            for cam_id in cam_ids:


                coords_cam_path = os.path.join(dataset_folder,"cam_{}".format(cam_id), "coords_cam_{}.csv".format(cam_id) )
                cam_dataframe = load_csv(working_dir=working_dir,csv_path=coords_cam_path)

                cam_dataframe = group_drop_except_person_id_and_appearance_id(cam_dataframe)
                combined_cam_dataframes = combined_cam_dataframes.append(cam_dataframe,ignore_index=True)

            return combined_cam_dataframes

        combined_train_and_test = pd.DataFrame()
        for dataset_type in ["train", "test"]:
            dataset_folder_with_type = os.path.join(dataset_folder,dataset_type)
            combined_one_type = get_combined_one_type(dataset_folder=dataset_folder_with_type)
            combined_train_and_test = combined_train_and_test.append(combined_one_type,ignore_index=True)

        return combined_train_and_test

    def get_appearance_id_to_person_id(combined_cam_dataframes):
        appearance_id_to_person_id = {}
        for index, cam_row in combined_cam_dataframes.iterrows():
            appearance_id_to_person_id[cam_row["appearance_id"]] = cam_row["person_id"]

        return appearance_id_to_person_id


    def get_mugshot_folder_appearance_ids(mugshot_folder_path):
        mugshot_folder_image_names = os.listdir(mugshot_folder_path)

        appearance_ids = [ int(image_name.replace(".jpg","").split("_")[1]) for image_name in mugshot_folder_image_names ]

        return appearance_ids

    def copy_mugshot_folder_renamed(mugshot_base_folder,subfolder_name,output_folder,appearance_id_to_person_id):

        mugshot_folder_path = os.path.join(mugshot_base_folder,subfolder_name)
        mugshot_folder_image_names = os.listdir(mugshot_folder_path)

        #mugshot_folder_image_names = mugshot_folder_image_names[:10]

        print("Starting copying of {}".format(mugshot_folder_path))
        for image_name in tqdm(mugshot_folder_image_names):
            appearance_id = int(image_name.replace(".jpg","").split("_")[1])

            #This means a appearance id in the mugshot folder is not in the dataframe
            if appearance_id not in appearance_id_to_person_id:
                continue

            person_id = appearance_id_to_person_id[appearance_id]

            image_name_with_person_id = "pid_{}.jpg".format(person_id)

            src_path = os.path.join(mugshot_folder_path, image_name)


            output_folder_with_sub_folder = os.path.join(output_folder,subfolder_name)
            os.makedirs(output_folder_with_sub_folder, exist_ok=True)

            dest_path = os.path.join(output_folder_with_sub_folder,image_name_with_person_id)

            img = cv2.imread(src_path)



            img = crop_mugshot_image(img=img,subfolder_name=subfolder_name)

            #cv2.imshow("cropped img", mat=img)
            #cv2.waitKey(0)

            cv2.imwrite(filename=dest_path,img=img)



    def check_if_all_appearance_ids_exist_as_mugshots(mugshot_folder_path,appearance_id_to_person_id):
        mugshot_appearance_ids = get_mugshot_folder_appearance_ids(mugshot_folder_path)
        mugshot_appearance_ids = set(mugshot_appearance_ids)


        dataframe_appearance_ids = set(appearance_id_to_person_id.keys())

        dataframe_appearance_ids_not_in_mugshot_appearance_ids = dataframe_appearance_ids - mugshot_appearance_ids



        print("Number of appearance ids in dataframe without partner in mugshot folder: {}".format(len(dataframe_appearance_ids_not_in_mugshot_appearance_ids)))

        print(dataframe_appearance_ids_not_in_mugshot_appearance_ids)


    def copy_mugshots_for_all_subfolders(mugshot_base_folder
                                         , mugshot_subfolder_names
                                         , output_folder
                                         , appearance_id_to_person_id):

        for subfolder_name in mugshot_subfolder_names:

            copy_mugshot_folder_renamed(mugshot_base_folder=mugshot_base_folder
                                        ,subfolder_name=subfolder_name
                                        ,output_folder=output_folder
                                        ,appearance_id_to_person_id=appearance_id_to_person_id)


    def crop_mugshot_image(img,subfolder_name):

        subfolder_name_to_crop_box = {

            "front_" : {
                "x" : 629
                ,"y" : 301
                ,"w" : 376
                ,"h" : 754
            },
            "left_"  : {
                "x" : 805
                ,"y" : 252
                ,"w" : 310
                ,"h" : 809
            },
            "right_" : {
                "x" : 780
                ,"y" : 290
                ,"w" : 315
                ,"h" : 736
            },
            "back_" : {
                "x" : 755
                ,"y" : 350
                ,"w" : 325
                ,"h" : 664
            }

        }

        x = subfolder_name_to_crop_box[subfolder_name]["x"]
        y = subfolder_name_to_crop_box[subfolder_name]["y"]
        w = subfolder_name_to_crop_box[subfolder_name]["w"]
        h = subfolder_name_to_crop_box[subfolder_name]["h"]
        crop_img = img[y:y + h, x:x + w]

        return crop_img


    combined_cam_dataframes = get_combined_cam_dataframes(dataset_folder=dataset_folder)
    appearance_id_to_person_id = get_appearance_id_to_person_id(combined_cam_dataframes)

    check_if_all_appearance_ids_exist_as_mugshots(mugshot_folder_path=os.path.join(mugshot_base_folder
                                                                                   ,mugshot_subfolder_names[0]),
                                                  appearance_id_to_person_id=appearance_id_to_person_id
                                                  )

    copy_mugshots_for_all_subfolders(mugshot_base_folder=mugshot_base_folder
                                     ,mugshot_subfolder_names=mugshot_subfolder_names
                                     ,output_folder=output_folder
                                     ,appearance_id_to_person_id=appearance_id_to_person_id)



if __name__ == "__main__":
    copy_renamed_mugshot_images(mugshot_base_folder="/net/merkur/storage/deeplearning/users/koehl/gta/mugshots_recording_22.07"
                                ,mugshot_subfolder_names=["front_", "left_", "right_", "back_"]
                                , dataset_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019"
                                , output_folder="/net/merkur/storage/deeplearning/users/koehl/gta/mugshots_with_pid"
                                ,cam_ids=[0,1,2,3,4,5]
                                , working_dir="/home/koehlp/Downloads/work_dirs/")

