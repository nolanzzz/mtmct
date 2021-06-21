

import os
import pandas as pd

from tqdm import tqdm

from utilities.helper import xtylwh_to_xyxy

'''
        A row should have this shape:
        { "frame_no_cam" : frame_no_cam , "cam_id" : cam_id , "person_id" : person_id , "xtl" : xtl , "ytl" : ytl , "xbr" : xbr , "ybr" : ybr }
        
        
        but has field_names = ['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'wx', 'wy', 'wz']
'''


def convert(input_file_path,output_file_path):

    cam_id = int(os.path.basename(input_file_path).replace(".csv","").split("_")[1])
    input_frame = pd.read_csv(input_file_path)
    output_frame = pd.DataFrame({ "frame_no_cam" : [] , "cam_id" : [] , "person_id" : [] , "xtl" : [] , "ytl" : [] , "xbr" : [] , "ybr" : [] })
    for index,input_row in tqdm(input_frame.iterrows(),total=len(input_frame)):


        xyxy_bbox = xtylwh_to_xyxy((input_row[2],input_row[3],input_row[4],input_row[5]))
        output_frame = output_frame.append({ "frame_no_cam" : input_row[0]
                                , "cam_id" : cam_id
                                , "person_id" : input_row[1]
                                , "xtl" : xyxy_bbox[0]
                                , "ytl" : xyxy_bbox[1]
                                , "xbr" : xyxy_bbox[2]
                                , "ybr" : xyxy_bbox[3]  },ignore_index=True)

    output_frame = output_frame.astype({"frame_no_cam" : int , "cam_id": int, "person_id": int, "xtl" : int, "ytl" : int , "xbr" : int, "ybr" : int})
    output_frame.to_csv(output_file_path,index=False)




if __name__ == "__main__":

    convert("/home/koehlp/Downloads/work_dirs/tracker/iou_tracker/Recording_12.07.2019_17/cam_3.csv"
            ,"/home/koehlp/Downloads/work_dirs/tracker/iou_tracker/Recording_12.07.2019_17/cam_3_motmetrics_evaluation_format.csv")
