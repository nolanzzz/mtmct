import argparse
import os.path as osp


import mmcv
import numpy as np
import os.path as osp
import pandas as pd
import re
import os
from PIL import Image
import PIL


def get_frame_annotation(cam_coords_frame: pd.DataFrame, img_path: str, image_size: tuple) -> dict:

    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for ped_row in cam_coords_frame.iterrows():
        labels.append(0)
        bbox = [
            int(ped_row[1].loc["x_top_left_BB"]),
            int(ped_row[1].loc["y_top_left_BB"]),
            int(ped_row[1].loc["x_bottom_right_BB"]),
            int(ped_row[1].loc["y_bottom_right_BB"])
        ]
        bboxes.append(bbox)


    annotation = {
        'filename': img_path,
        'width': image_size[0],
        'height': image_size[1],
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def readFilesInDirectory(folder):
    allFiles = []
    for f in os.listdir(folder):
        filePath = osp.join(folder, f)
        if osp.isfile(filePath) and osp.splitext(filePath)[1] in [".jpg", ".png"]:
            allFiles.append(f)



    allFiles.sort(key=natural_keys)

    return allFiles

def getAllNumbersFromString(text):

    return re.findall(r'\d+', text)

def getFrameListFromImageNames(images):

    frame_numbers = []
    for img_name in images:
        frame_number = getAllNumbersFromString(img_name)[0]
        frame_numbers.append(int(frame_number))

    return frame_numbers

def convert_annotations(cam_path, input_file, output_file):

    cam_coords = pd.read_csv(input_file)

    image_names = readFilesInDirectory(cam_path)



    image_size = PIL.Image.open(osp.join(cam_path, image_names[0])).size

    frame_ids = getFrameListFromImageNames(image_names)

    annotations = []

    for i,frame_id in enumerate(frame_ids):
        cam_coords_frame = cam_coords[cam_coords["frame_no_gta"] == frame_id]

        cam_coords_frame = cam_coords_frame.groupby("ped_id").mean()

        frame_annotation = get_frame_annotation(cam_coords_frame,image_names[i],image_size)
        annotations.append(frame_annotation)






    mmcv.dump(annotations, output_file)
    return annotations


def main():
    cam_path = '/home/philipp/Downloads/Recording_12.07.2019/cam_1'
    convert_annotations(cam_path, osp.join(cam_path,"coords_cam_1.csv"),osp.join(cam_path,"coords_cam_1.json"))



if __name__ == '__main__':
    main()
