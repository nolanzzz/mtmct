import zipfile
from utilities.helper import group_and_drop_unnecessary
from utilities.pandas_loader import *
import pandas as pd
import sys
from shutil import copyfile
from tqdm import tqdm


def get_min_frame_no(dataset_folder,dataset_type):

    cam_folder = os.path.join(dataset_folder,dataset_type,"cam_0/")

    cam_images = os.listdir(cam_folder)
    cam_images = [ image_name for image_name in cam_images if image_name.endswith(".jpg") ]
    #image_1133_0.jpg
    frame_nos = [ int(image_name.split("_")[1]) for image_name in cam_images ]

    min_frame_no = min(frame_nos)
    return min_frame_no


def convert_dataset_images_to_videos(dataset_folder,dataset_output_folder):

    for dataset_type in ["train","test"]:
        min_frame_no = get_min_frame_no(dataset_folder=dataset_folder,dataset_type=dataset_type)

        for cam_id in [0,1,2,3,4,5]:
            image_path = os.path.join(dataset_folder,dataset_type,"cam_{}".format(cam_id),"image_%d_{}.jpg".format(cam_id))
            output_video_folder = os.path.join(dataset_output_folder, dataset_type, "cam_{}".format(cam_id))
            os.makedirs(output_video_folder,exist_ok=True)
            output_video_path = os.path.join(output_video_folder,"cam_{}.mp4".format(cam_id))

            print("Starting to convert {}".format(output_video_path))
            os.system("taskset -c 0-30 ffmpeg -r 41 -start_number {} -i {} -b:v 10M {}".format(min_frame_no,image_path,output_video_path))





def zip_cam_coords(dataset_folder,dataset_output_folder):

    for dataset_type in ["train","test"]:

        for cam_id in [0,1,2,3,4,5]:
            cam_coords_path = os.path.join(dataset_folder,dataset_type,"cam_{}".format(cam_id),"coords_cam_{}.csv".format(cam_id))
            output_folder = os.path.join(dataset_output_folder, dataset_type, "cam_{}".format(cam_id))
            os.makedirs(output_folder,exist_ok=True)
            output_zip_path = os.path.join(output_folder,"coords_cam_{}.csv.zip".format(cam_id))

            zipf = zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED)
            zipf.write(cam_coords_path,arcname=os.path.basename(cam_coords_path))
            zipf.close()

            compression_ratio = 1 - os.stat(output_zip_path).st_size / os.stat(cam_coords_path).st_size
            print("compression ratio of file {} : {}".format(os.path.basename(cam_coords_path),compression_ratio))



def reduce_cam_coords(dataset_folder,dataset_output_folder,working_dir):

    for dataset_type in ["train","test"]:

        for cam_id in [0,1,2,3,4,5]:
            cam_coords_path = os.path.join(dataset_folder,dataset_type,"cam_{}".format(cam_id),"coords_cam_{}.csv".format(cam_id))
            output_folder = os.path.join(dataset_output_folder, dataset_type, "cam_{}".format(cam_id))
            os.makedirs(output_folder,exist_ok=True)
            output_path = os.path.join(output_folder, "coords_fib_cam_{}.csv".format(cam_id))

            coords_dataframe = load_csv(working_dir, cam_coords_path)

            coords_dataframe = coords_dataframe.astype(int)

            coords_dataframe = group_and_drop_unnecessary(coords_dataframe)

            coords_dataframe.to_csv(path_or_buf=output_path,index=False)

def adjust_frame_nos(dataset_folder,dataset_output_folder,dataset_types=("train","test"),coords_type=""):


    for dataset_type in dataset_types:

        #Get minimum frame no
        min_frame_no = sys.maxsize
        for cam_id in [0, 1, 2, 3, 4, 5]:
            print("Searching frame no min cam {}".format(cam_id))
            coords_name = "coords{}_cam_{}.csv".format(coords_type, cam_id)
            cam_coords_path = os.path.join(dataset_folder, dataset_type, "cam_{}".format(cam_id),coords_name)

            cam_coords = pd.read_csv(cam_coords_path,low_memory=True,nrows=10)

            min_frame_no = min(min_frame_no,*list(cam_coords["frame_no_cam"]))

        for cam_id in [0, 1, 2, 3, 4, 5]:
            print("Copying coords cam {}".format(cam_id))
            coords_name = "coords{}_cam_{}.csv".format(coords_type, cam_id)
            cam_coords_path = os.path.join(dataset_folder, dataset_type, "cam_{}".format(cam_id),coords_name)
            cam_video_name = "cam_{}.mp4".format(cam_id)
            cam_video_path = os.path.join(dataset_folder, dataset_type, "cam_{}".format(cam_id),cam_video_name)

            cam_coords_output_folder = os.path.join(dataset_output_folder, dataset_type, "cam_{}".format(cam_id))
            os.makedirs(cam_coords_output_folder, exist_ok=True)

            cam_coords_output_path = os.path.join(cam_coords_output_folder, coords_name)

            for index, cam_coords_chunk in enumerate(pd.read_csv(cam_coords_path,chunksize=10**6)):


                cam_coords_chunk["frame_no_cam"] = cam_coords_chunk["frame_no_cam"] - min_frame_no


                if index == 0:
                    write_header = True
                    mode ="w"
                else:
                    write_header = False
                    mode = "a"
                cam_coords_chunk.to_csv(cam_coords_output_path,index=False,mode=mode,header=write_header)

            cam_coords_output_folder = os.path.join(dataset_output_folder, dataset_type, "cam_{}".format(cam_id))
            cam_video_output_path = os.path.join(cam_coords_output_folder,cam_video_name)
            if os.path.exists(cam_video_path):
                copyfile(cam_video_path, cam_video_output_path)

def get_folder_images(over_folder,reid_folder_type):

    reid_image_folder = os.path.join(over_folder,reid_folder_type)
    return set(os.listdir(reid_image_folder))

def remove_distractors_from_gallery(reid_dataset_folder,output_folder):

    reid_folder_types = ["query","test","train"]

    #Get reid images
    gallery_folder_type_to_image_names = {}
    for reid_folder_type in reid_folder_types:
        gallery_folder_type_to_image_names[reid_folder_type] = get_folder_images(over_folder=reid_dataset_folder,reid_folder_type=reid_folder_type)

    distractors_folder = os.path.join(reid_dataset_folder,"distractors")
    distractors_folder_type_to_image_names = {}
    for reid_folder_type in reid_folder_types:
        distractors_folder_type_to_image_names[reid_folder_type] = get_folder_images(over_folder=distractors_folder,reid_folder_type=reid_folder_type)

    #Remove distractors from gallery
    for reid_folder_type in reid_folder_types:
        gallery_folder_type_to_image_names[reid_folder_type] = gallery_folder_type_to_image_names[reid_folder_type] - distractors_folder_type_to_image_names[reid_folder_type]

    #Create new reid gallery image folders
    for reid_folder_type in reid_folder_types:
        print("Copying images of {}".format(reid_folder_type))
        new_reid_image_folder = os.path.join(output_folder,reid_folder_type)
        reid_image_folder = os.path.join(reid_dataset_folder, reid_folder_type)
        os.makedirs(new_reid_image_folder,exist_ok=True)

        for image_name in tqdm(gallery_folder_type_to_image_names[reid_folder_type]):
            old_image_path = os.path.join(reid_image_folder,image_name)
            new_image_path = os.path.join(new_reid_image_folder,image_name)
            copyfile(old_image_path,new_image_path)


    print("Finished!")





if __name__ == "__main__":
    remove_distractors_from_gallery(reid_dataset_folder="/media/philipp/philippkoehl_ssd/reid_clean_GTA_22.07.2019"
                                    ,output_folder="/media/philipp/philippkoehl_ssd/MTA_reid")