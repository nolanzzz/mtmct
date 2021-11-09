
import os.path as osp
from shutil import copyfile

import mmcv
import numpy as np
import os.path as osp
import pandas as pd

import os


from tqdm import tqdm
from utilities.pandas_loader import load_csv

import multiprocessing as mp

from utilities.helper import constrain_bbox_to_img_dims


current_annotation_id = 0

def get_frame_annotation(cam_coords_frame: pd.DataFrame, image_id: int, image_size: tuple) -> dict:
    global current_annotation_id

    annotations = []

    for ped_row in cam_coords_frame.iterrows():



        bbox = [
            int(ped_row[1].loc["x_top_left_BB"]),
            int(ped_row[1].loc["y_top_left_BB"]),
            int(ped_row[1].loc["x_bottom_right_BB"]),
            int(ped_row[1].loc["y_bottom_right_BB"])
        ]
        bbox = constrain_bbox_to_img_dims(bbox)

        #bbox is [x,y,width,height]
        bbox = xyxy2xywh(np.asarray(bbox))

        width = bbox[2]
        height = bbox[3]

        area = float(width * height)



        annotation = {
            "id": current_annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "iscrowd": 0,
            "area": area,
            "bbox": bbox,
            "width": image_size[0],
            "height": image_size[1],
        }

        annotations.append(annotation)

        current_annotation_id += 1



    return annotations



def convert_annotations(gta_dataset_path,coco_gta_dataset_path,work_dirs,data_type,cam_number=6,img_dims=(1920,1080),person_id_name="person_id",samplingRate=1):
    coco_dict = {
        'info': {
            'description': 'GTA_MTMCT',
            'url': 'gtav.',
            'version': '1.0',
            'year': 2019,
            'contributor': 'Philipp Koehl',
            'date_created': '2019/07/16',
        },
        'licences': [{
            'url': 'http://creativecommons.org/licenses/by-nc/2.0',
            'id': 2,
            'name': 'Attribution-NonCommercial License'
        }],
        'images': [],
        'annotations': [],
        'categories': [{
                           'supercategory':'person',
                           'id':1,
                           'name':'person'
                        },
                        {
                            'supercategory': 'background',
                            'id': 2,
                            'name': 'background'
                        }
                        ]
    }

    coco_gta_dataset_images_path = osp.join(coco_gta_dataset_path, "images")
    os.makedirs(coco_gta_dataset_images_path,exist_ok=True)
    os.makedirs(coco_gta_dataset_path,exist_ok=True)





    for cam_id in range(cam_number):
        print("processing cam_{}".format(cam_id))

        cam_path = os.path.join(gta_dataset_path,"cam_{}".format(cam_id))
        csv_path = osp.join(cam_path, "coords_fib_cam_{}.csv".format(cam_id))

        cam_coords = load_csv(work_dirs, csv_path)
        cam_coords = cam_coords.groupby(["frame_no_cam", person_id_name], as_index=False).mean()
        cam_frames = cam_coords.groupby("frame_no_cam", as_index=False).mean()

        cam_frames = cam_frames.iloc[::samplingRate]




        pbar = tqdm(total=len(cam_frames))

        def updateTqdm(*a):
            pbar.update()

        pool = mp.Pool(processes=10)  # use all available cores, otherwise specify the number you want as an argument
        for cam_frame in cam_frames.iterrows():

            frame_no_cam = int(cam_frame[1]["frame_no_cam"])
            # train - 1; test - 2
            if data_type == "train":
                code = "1"
            else:
                code = "2"
            image_id = int(code + str(cam_id) + '{:06d}'.format(frame_no_cam))

            cam_coords_frame = cam_coords[cam_coords["frame_no_cam"] == frame_no_cam ]


            frame_annotations = get_frame_annotation(cam_coords_frame,image_id, img_dims)
            image_name = "image_{}_{}.png".format(frame_no_cam,cam_id)
            # image_path_gta = osp.join(cam_path, "img1", image_name)
            # image_path_gta_coco = osp.join(coco_gta_dataset_images_path,image_name)


            coco_dict['images'].append({
                'license': 4,
                'file_name': image_name,
                'height': img_dims[1],
                'width': img_dims[0],
                'date_captured': '2019-07-28 00:00:00',
                'id': image_id
            })

            coco_dict['annotations'].extend(frame_annotations)
            updateTqdm()
            # pool.apply_async(copyfile,args=(image_path_gta,image_path_gta_coco),callback=updateTqdm)
        pool.close()
        pool.join()



    mmcv.dump(coco_dict, osp.join(coco_gta_dataset_path,"coords_{}.json".format(data_type)))
    return coco_dict


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]

def main():

    gta_dataset_path = "/Users/nolanzhang/Projects/mtmct/data/MTA_short_new/train"
    coco_dataset_path = "/Users/nolanzhang/Projects/mtmct/data/MTA_short_new/train"
    work_dirs = "/Users/nolanzhang/Projects/mtmct/work_dirs/"
    convert_annotations(gta_dataset_path,coco_dataset_path,work_dirs,"train")

    gta_dataset_path = "/Users/nolanzhang/Projects/mtmct/data/MTA_short_new/test"
    coco_dataset_path = "/Users/nolanzhang/Projects/mtmct/data/MTA_short_new/test"
    work_dirs = "/Users/nolanzhang/Projects/mtmct/work_dirs/"
    convert_annotations(gta_dataset_path, coco_dataset_path, work_dirs,"test")



if __name__ == '__main__':
    main()


