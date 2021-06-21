import re
import numpy as np
import logging
import time
import io
import sys
import cv2

import pandas as pd
from math import ceil
from datetime import timedelta
import os
from utilities.pandas_loader import load_csv

import cv2

def atoi(text):
    return int(text) if text.isdigit() else text

def get_elapsed_time_and_msg(start_time,message):
    time_difference_in_seconds = time.time() - start_time
    return str(timedelta(seconds=time_difference_in_seconds)) + " : " + message

def put_text_with_background(img,text,org,font_scale=1.5):
    font = cv2.FONT_HERSHEY_PLAIN

    # set the rectangle background to white
    rectangle_bgr = (0, 0, 0)

    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    # set the text start position
    text_offset_x = org[0]
    text_offset_y = org[1]
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(255, 255, 255), thickness=3)


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def get_string_sizes_of_all_variables():
    result_string = ""
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                             key= lambda x: -x[1])[:10]:
        result_string += "{:>30}: {:>8} \n".format(name, sizeof_fmt(size))

    return result_string


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]

def many_xyxy2xywh(bboxes):

    return np.array([ xyxy2xywh(bbox) for bbox in bboxes ])

def get_bbox_middle_pos(bbox):

    xtl, ytl, xbr, ybr = bbox
    x = xtl + ((xbr - xtl) / 2.)
    y = ytl + ((ybr - ytl) / 2.)

    return (x,y)

def get_bbox_of_row(row):
    return (row["x_top_left_BB"] ,row["y_top_left_BB"], row["x_bottom_right_BB"], row["y_bottom_right_BB"])

def get_bbox_of_row_track_results(row):
    return (row["xtl"],row["ytl"],row["xbr"],row["ybr"])

def get_all_frame_numbers(dataset_folder,cam_ids):
    '''
    Because the coord csv files only contain frame nubmers with people visible, the images have to be used to
    find all frame numbers.
    :param dataset_folder:
    :return:
    '''

    #Any cam can be used
    dataset_cam_folder = os.path.join(dataset_folder, "cam_{}/".format(cam_ids[0]),)
    cam_folder_filenames = os.listdir(dataset_cam_folder)
    #image_{frame_no_cam}_{cam_id}.jpg
    cam_folder_images = [ filename for filename in  cam_folder_filenames if filename.endswith(".jpg")]
    frame_nos_cam = [ int(image_name.replace(".jpg","").split('_')[1]) for image_name in cam_folder_images ]

    frame_nos_cam = sorted(frame_nos_cam)

    return frame_nos_cam

def get_all_frame_numbers_via_videos(dataset_folder,cam_ids):
    '''
    Because the coord csv files only contain frame nubmers with people visible, the images have to be used to
    find all frame numbers.
    :param dataset_folder:
    :return:
    '''

    #Any cam can be used
    cam_video_path = os.path.join(dataset_folder, "cam_{}/".format(cam_ids[0]), "cam_{}.mp4".format(cam_ids[0]))

    video_capture = cv2.VideoCapture(cam_video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_nos_cam = list(range(frame_count))

    return frame_nos_cam

def get_track_results_and_dataset(track_results_folder, dataset_folder, working_dir, cam_count, until_frame_no):

    take_frame_numbers = None
    if until_frame_no is not None:
        all_frame_numbers = get_all_frame_numbers(dataset_folder,list(range(cam_count)))

        take_frame_numbers = all_frame_numbers[:until_frame_no]


    track_results_dataframes = load_track_result_dataframes(track_results_folder=track_results_folder
                                 ,cam_count=cam_count
                                 ,take_frame_nos=take_frame_numbers)

    dataset_dataframes = load_dataset_dataframes(dataset_folder=dataset_folder
                                                 , working_dir=working_dir
                                                 , cam_count=cam_count
                                                 , take_frame_nos=take_frame_numbers)

    return track_results_dataframes,dataset_dataframes


def group_and_drop_unnecessary(cam_dataframe):
    cam_dataframe = cam_dataframe.groupby(by=["frame_no_cam", "person_id"],as_index=False).mean()
    cam_dataframe = cam_dataframe[["frame_no_cam"
        , "person_id"
        , "x_top_left_BB"
        , "y_top_left_BB"
        , "x_bottom_right_BB"
        , "y_bottom_right_BB"]]
    cam_dataframe = cam_dataframe.astype(int)
    return cam_dataframe

def load_dataset_dataframes(dataset_folder, working_dir, cam_count , take_frame_nos):
    dataset_cam_dataframes = []
    for cam_id in range(cam_count):
        coords_path = os.path.join(dataset_folder,"cam_{}".format(cam_id), "coords_cam_{}.csv".format(cam_id))

        coords_dataframe = load_csv(working_dir,coords_path)

        if take_frame_nos is not None:
            coords_dataframe = coords_dataframe[coords_dataframe["frame_no_cam"].isin(take_frame_nos)]

        coords_dataframe = coords_dataframe.astype(int)

        coords_dataframe = group_and_drop_unnecessary(coords_dataframe)

        dataset_cam_dataframes.append(coords_dataframe)

    return dataset_cam_dataframes


def load_track_result_dataframes(track_results_folder,cam_count, take_frame_nos):
    track_dataframes = []
    for cam_id in range(cam_count):
        track_result_path = os.path.join(track_results_folder, "track_results_{}.txt".format(cam_id))

        track_result_dataframe = pd.read_csv(track_result_path)


        if take_frame_nos is not None:

            track_result_dataframe = track_result_dataframe[track_result_dataframe["frame_no_cam"].isin(take_frame_nos)]

        track_result_dataframe = track_result_dataframe.astype(int)


        track_dataframes.append(track_result_dataframe)

    return track_dataframes

def get_bbox_width(bbox):
    xtl, ytl, xbr, ybr = bbox
    return xbr - xtl

def get_bbox_height(bbox):
    xtl, ytl, xbr, ybr = bbox
    return ybr - ytl

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



def rename_short_motmetrics(evaluation_results):

    rename_rules =  {
        'idf1' : 'IDF1',
        'idp' : 'IDP',
        'idr' : 'IDR',
        'recall' : 'Rcll',
        'precision' : 'Prcn',
        'num_unique_objects' : 'GT',
        'mostly_tracked' : 'MT',
        'partially_tracked' : 'PT',
        'mostly_lost': 'ML',
        'num_false_positives' : 'FP',
        'num_misses' : 'FN',
        'num_switches' : 'IDs',
        'num_fragmentations' : 'FM',
        'mota' : 'MOTA',
        'motp' : 'MOTP'
    }
    return evaluation_results.rename(columns=rename_rules)




def bboxes_round_int(bboxes):
    result = []
    for bbox in bboxes:

        bbox = np.rint(bbox)
        bbox = np.array(bbox,dtype=np.int32)
        result.append(bbox)
    return np.array(result)


def xcycwh_to_xtylwh(bbox_xywh):
    bbox_xywh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
    bbox_xywh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
    return bbox_xywh

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def xcycwh_to_xyxy(bbox_xywh,img_dims):
    '''

    :param bbox_xywh:
    :param img_dims: (width,height)
    :return:
    '''
    x,y,w,h = bbox_xywh
    x1 = max(int(x-w/2),0)
    x2 = min(int(x+w/2),img_dims[0]-1)
    y1 = max(int(y-h/2),0)
    y2 = min(int(y+h/2),img_dims[1]-1)
    return x1,y1,x2,y2

def xtylwh_to_xyxy(bbox_tlwh,img_dims=(1920,1080)):
    '''

    :param bbox_tlwh:
    :param img_dims: (width,height)
    :return:
    '''
    x,y,w,h = bbox_tlwh
    x1 = max(int(x),0)
    x2 = min(int(x+w),img_dims[0]-1)
    y1 = max(int(y),0)
    y2 = min(int(y+h),img_dims[1]-1)
    return x1,y1,x2,y2


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)


def constrain_bbox_to_img_dims(xyxy_bbox,img_dims=(1920,1080)):
    img_width, img_height = img_dims
    xtl_res, ytl_res, xbr_res, ybr_res = xyxy_bbox

    if xtl_res < 0: xtl_res = 0

    if xtl_res >= img_width: xtl_res = img_width  # should not occur

    if ytl_res < 0: ytl_res = 0

    if ytl_res >= img_height: ytl_res = img_height

    if xbr_res < 0: xbr_res = 0

    if xbr_res >= img_width: xbr_res = img_width

    if ybr_res < 0: ybr_res = 0

    if ybr_res >= img_height: ybr_res = img_height



    return [xtl_res, ytl_res,xbr_res,ybr_res]


def timing(f,output=sys.stdout):
    def wrap(*args):
        time1 = time.time()
        print('Starting execution of: {:s}'.format(f.__name__), file=output)
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0),file=output)

        return ret
    return wrap

def takespread(sequence, num):

    result = []
    if num >= len(sequence):
        return sequence

    length = float(len(sequence))
    for i in range(num):
        result.append(sequence[int(ceil(i * length / num))])

    return result

def drawBoundingBoxes(image, xyxy_bboxes):

    for bbox in xyxy_bboxes:
        drawBoundingBox(image,bbox)



def drawBoundingBox(image, xyxy_bbox, color=(0,255,0),thickness=1):
    topLeft = (xyxy_bbox[0],xyxy_bbox[1])
    topRight = (xyxy_bbox[2],xyxy_bbox[1])
    bottomLeft = (xyxy_bbox[0],xyxy_bbox[3])
    bottomRight = (xyxy_bbox[2],xyxy_bbox[3])


    cv2.line(image, topLeft , bottomLeft, color=color, thickness=thickness)
    cv2.line(image, topLeft, topRight, color=color, thickness=thickness)
    cv2.line(image, topRight, bottomRight, color=color, thickness=thickness)
    cv2.line(image, bottomLeft, bottomRight, color=color, thickness=thickness)



def constraint_point_to_img_dims(point,img_dims=(1920,1080)):
    img_width, img_height = img_dims
    x_res, y_res = point

    if x_res < 0: x_res = 0

    if x_res >= img_width: x_res = img_width  # should not occur

    if y_res < 0: y_res = 0

    if y_res >= img_height: y_res = img_height

    return [x_res, y_res]


def drop_unnecessary_columns(coords_file:pd.DataFrame):
    return coords_file.drop(columns=[   "appearance_id"
                                        , "joint_type"
                                        , "x_2D_joint"
                                        , "y_2D_joint"
                                        , "x_3D_joint"
                                        , "y_3D_joint"
                                        , "z_3D_joint"
                                        , "joint_occluded"
                                        , "joint_self_occluded"
                                        , "x_3D_cam"
                                        , "y_3D_cam"
                                        , "z_3D_cam"
                                        , "x_rot_cam"
                                        , "y_rot_cam"
                                        , "z_rot_cam"
                                        , "fov"
                                        , "x_3D_person"
                                        , "y_3D_person"
                                        , "z_3D_person"
                                        , "x_2D_person"
                                        , "y_2D_person"
                                        , "ped_type"
                                        , "wears_glasses"
                                        , "yaw_person"
                                        , "hours_gta"
                                        , "minutes_gta"
                                        , "seconds_gta"

                                      ])

def adjustCoordsTypes(coords,person_identifier="ped_id"):

    return coords.astype(int)

if __name__ == "__main__":
    #arr = np.array([[1.2,2.3,4.,5.],[1.6,2.1,4.,5.]])

    #print(bboxes_round_int(arr))

    #print(list(takespread([12,4,5,6,7],2)))

    find_weights_start_time = time.time() - 1000000

    print(get_elapsed_time_and_msg(find_weights_start_time,"Test message"))