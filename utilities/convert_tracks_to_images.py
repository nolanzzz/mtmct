
import numpy as np
from utilities.helper import *
from utilities.pandas_loader import load_csv
import random
from shutil import copyfile


class Convert_tack_to_images:


    def __init__(self,dataset_base_folder,track_results_folder,work_dirs,cam_ids
                 ,output_folder=None
                 ,output_image_dims=(50,50)
                 ,camera_image_dims=(1920,1080)):
        self.dataset_base_folder = dataset_base_folder
        self.track_results_folder = track_results_folder
        self.work_dirs = work_dirs
        self.cam_ids = cam_ids
        self.output_folder = output_folder
        self.person_identifier = "person_id"
        self.output_image_dims = output_image_dims
        self.camera_image_dims = camera_image_dims

        if self.output_folder is not None:
            os.makedirs(self.output_folder,exist_ok=True)







    def read_ground_truth(self,person_identifier="person_id"):

        all_cam_coords = pd.DataFrame()
        for cam_id in self.cam_ids:
            dataset_base_path = os.path.join(self.dataset_base_folder
                                             ,"cam_{}".format(cam_id)
                                             ,"coords_cam_{}.csv".format(cam_id))

            cam_coords = load_csv(self.work_dirs, dataset_base_path)

            cam_coords = cam_coords.groupby(["frame_no_gta", person_identifier], as_index=False).mean()

            cam_coords = adjustCoordsTypes(cam_coords, person_identifier=person_identifier)

            cam_coords = drop_unnecessary_columns(cam_coords)

            cam_coords["cam_id"] = cam_id

            all_cam_coords = all_cam_coords.append(cam_coords,ignore_index=True)


        return all_cam_coords

    def save_images(self,images,person_id,cam_id):
        for i,image in enumerate(images):

            img_path = os.path.join(self.output_folder,"partno_{}_personid_{}_camid_{}.png".format(i,person_id,cam_id))

            cv2.imwrite(filename=img_path,img=image)



    def create_image_per_cam(self,tracks_all_cams,person_id):

        cam_ids = set(tracks_all_cams["cam_id"])

        for cam_id in cam_ids:
            track_one_cam = tracks_all_cams[tracks_all_cams["cam_id"] == cam_id]

            track_images = self.get_images_from_track(track_one_cam=track_one_cam,cam_id=cam_id)

            self.save_images(images=track_images,person_id=person_id,cam_id=cam_id)




    def normalize_in_range(self,max_old,min_old,max_new,min_new,value):

         new_value= (max_new-min_new)/(max_old-min_old)*(value-max_old)+max_new

         return int(new_value)

    def get_images_from_track(self,track_one_cam,cam_id):


        def create_img_from_image_list(image_list,output_images_dims):

            image_arr = np.array(image_list)
            image_arr = np.lib.pad(array=image_arr
                                   , pad_width=(0, image_length - len(image_arr))
                                   , mode="constant"
                                   , constant_values=(0)
                                   )

            rows = output_images_dims[0]
            cols = output_images_dims[1]

            image_arr = np.reshape(image_arr, (rows, cols))

            img = np.zeros([rows, cols, 3])

            img[:, :, 0] = image_arr
            img[:, :, 1] = image_arr
            img[:, :, 2] = image_arr

            img = img.astype(dtype=np.uint8)

            return img

        image_length = self.output_image_dims[0] * self.output_image_dims[1]
        image_list = []

        normalized_cam_id = self.normalize_in_range(max_old=max(*self.cam_ids,1)
                                , min_old=0
                                , max_new=255
                                , min_new=0
                                , value=cam_id)

        image_list.append(normalized_cam_id)
        finished_images = []

        for idx,track_row in track_one_cam.iterrows():

            bbox = get_bbox_of_row(track_row)

            bbox = constrain_bbox_to_img_dims(bbox)
            bbox_middle_pos = get_bbox_middle_pos(bbox)
            middle_x = bbox_middle_pos[0]
            middle_y = bbox_middle_pos[1]

            middle_x_normalized = self.normalize_in_range(max_old=self.camera_image_dims[0]
                                                          ,min_old=0
                                                          ,max_new=255
                                                          ,min_new=0
                                                          ,value=middle_x)

            middle_y_normalized = self.normalize_in_range(max_old=self.camera_image_dims[1]
                                                          , min_old=0
                                                          , max_new=255
                                                          , min_new=0
                                                          , value=middle_y)


            if len(image_list) + 2 <= image_length:
                image_list.append(middle_x_normalized)
                image_list.append(middle_y_normalized)
            else:
                created_img = create_img_from_image_list(image_list=image_list
                                           ,output_images_dims=self.output_image_dims)

                finished_images.append(created_img)
                image_list = []
                image_list.append(normalized_cam_id)

        if len(image_list) > 0:
            created_img = create_img_from_image_list(image_list=image_list
                                                     , output_images_dims=self.output_image_dims)

            finished_images.append(created_img)

        return finished_images




    def create_track_images(self):
        all_cam_coords = self.read_ground_truth()

        person_ids = set(all_cam_coords[self.person_identifier])

        for person_id in person_ids:
            tracks_all_cams = all_cam_coords[all_cam_coords[self.person_identifier] == person_id]
            self.create_image_per_cam(tracks_all_cams,person_id=person_id)



def create_query_folder(test_folder,new_test_folder,query_folder,query_percentage=0.2):


    def save_images_to_folder(src_folder, dest_folder,image_names):

        for image_name in image_names:

            image_src_path = os.path.join(src_folder,image_name)

            image_dst_path = os.path.join(dest_folder,image_name)

            copyfile(src=image_src_path,dst=image_dst_path)



    test_folder_images = os.listdir(test_folder)

    query_folder_size = int(len(test_folder_images)*query_percentage)

    query_images = random.sample(test_folder_images, k=query_folder_size)

    remaining_new_images = set(test_folder_images) - set(query_images)

    os.makedirs(query_folder, exist_ok=True)
    os.makedirs(new_test_folder, exist_ok=True)

    save_images_to_folder(src_folder=test_folder,dest_folder=new_test_folder,image_names=remaining_new_images)

    save_images_to_folder(src_folder=test_folder, dest_folder=query_folder, image_names=query_images)













if __name__ == "__main__":
    '''
    ctti = Convert_tack_to_images(dataset_base_folder="/media/philipp/philippkoehl_ssd/GTA_ext_short/test"
                            , track_results_folder="/media/philipp/philippkoehl_ssd/work_dirs/clustering/config_runs/multi_cam_clustering_GTA_ext_short/multicam_clustering_results/chunk_0/test"
                            , output_folder="/media/philipp/philippkoehl_ssd/work_dirs/track_images/test"
                            , work_dirs="/media/philipp/philippkoehl_ssd/work_dirs"
                            , cam_ids=[0,1,2,3,4,5])
    '''
    ctti = Convert_tack_to_images(dataset_base_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/train"
                                  , track_results_folder=""
                                  , output_folder="/home/koehlp/Downloads/work_dirs/track_images/train"
                                  , work_dirs="/home/koehlp/Downloads/work_dirs/work_dirs"
                                  , cam_ids=[0, 1, 2, 3, 4, 5])


    #ctti.create_track_images()


    create_query_folder(test_folder="/home/koehlp/Downloads/work_dirs/track_images/test"
                            ,new_test_folder="/home/koehlp/Downloads/work_dirs/track_images/test_no_query"
                            ,query_folder="/home/koehlp/Downloads/work_dirs/track_images/query")



    '''
    create_query_folder(test_folder="/media/philipp/philippkoehl_ssd/work_dirs/track_images/test"
                        ,new_test_folder="/media/philipp/philippkoehl_ssd/work_dirs/track_images/test_no_query"
                        ,query_folder="/media/philipp/philippkoehl_ssd/work_dirs/track_images/query")
    
    '''