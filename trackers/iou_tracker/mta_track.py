
from dataclasses import dataclass
from iou_tracker import track_iou
from util_iou import load_mot, save_to_csv
from time import time
import os

from multiprocessing.pool import ThreadPool
import pandas as pd

@dataclass
class Config_mta_tracker:
    detections_folder: str
    track_output_folder : str
    cam_ids: list
    sigma_l = 0.0
    sigma_h = 0.9
    sigma_iou = 0.3
    t_min = 3



def run_tracker_task(config_mta_tracker,cam_id):
    print("Running tracker on cam no. {}".format(cam_id))

    detections_name = "detections_cam_{}.csv".format(cam_id)
    detections_path = os.path.join(config_mta_tracker.detections_folder, detections_name)

    loaded_detections = load_mot(detections=detections_path)
    start = time()
    tracks = track_iou(loaded_detections
                       , config_mta_tracker.sigma_l
                       , config_mta_tracker.sigma_h
                       , config_mta_tracker.sigma_iou
                       , config_mta_tracker.t_min)
    end = time()

    num_frames = len(loaded_detections)
    print("finished with " + str(int(num_frames / (end - start))) + " fps!")

    save_tracks(tracks, cam_id=cam_id, output_folder=config_mta_tracker.track_output_folder)


def run_iou_tracker(config_mta_tracker):

    pool = ThreadPool(processes=10)

    for cam_id in config_mta_tracker.cam_ids:
        pool.apply_async(func=run_tracker_task,args=(config_mta_tracker,cam_id))
        #run_tracker_task(config_mta_tracker,cam_id)

    pool.close()
    pool.join()



def save_tracks(tracks,cam_id,output_folder):
    #frame_no_cam,cam_id,person_id,detection_idx,xtl,ytl,xbr,ybr

    tracks_list = []
    for person_id, track in enumerate(tracks):

        for frame_no_cam, bbox in enumerate(track["bboxes"],start=track["start_frame"]):
            tracks_list.append({ "frame_no_cam" : frame_no_cam
                                                        , "cam_id" : cam_id
                                                        , "person_id" : person_id
                                                        , "detection_idx" : person_id
                                                        , "xtl" : bbox[0]
                                                        , "ytl" : bbox[1]
                                                        , "xbr" : bbox[2]
                                                        , "ybr" : bbox[3]
                                                         })

    tracks_dataframe = pd.DataFrame(tracks_list)
    tracks_dataframe = tracks_dataframe.astype(int)
    tracks_dataframe = tracks_dataframe[["frame_no_cam","cam_id","person_id","detection_idx","xtl","ytl","xbr","ybr"]]
    tracks_dataframe = tracks_dataframe.sort_values(['frame_no_cam', 'person_id'],ignore_index=True)

    os.makedirs(output_folder,exist_ok=True)
    tracks_name = "track_results_{}.txt".format(cam_id)
    tracks_path = os.path.join(output_folder, tracks_name)

    tracks_dataframe.to_csv(path_or_buf=tracks_path,index=False)


if __name__ == "__main__":

    '''
    cmt = Config_mta_tracker(
        detections_folder="/media/philipp/philippkoehl_ssd/work_dirs/detector/detections/frcnn50_abdnet"
        , track_output_folder="/media/philipp/philippkoehl_ssd/work_dirs/tracker/frcnn50_abdnet_iou_try"
        , cam_ids=[0, 1, 2, 3, 4, 5])
        
    '''

    '''
    cmt = Config_mta_tracker(
        detections_folder="/media/philipp/philippkoehl_ssd/work_dirs/detector/detections/frcnn50_abdnet"
        , track_output_folder="/media/philipp/philippkoehl_ssd/work_dirs/tracker/frcnn50_abdnet_iou_try"
        , cam_ids=[0, 1, 2, 3, 4, 5])
        
    '''
    '''
    cmt = Config_mta_tracker(
        detections_folder="/home/koehlp/Downloads/work_dirs/detector/detections/cascade_abdnet_mta_test_cpt_iosb"
        , track_output_folder="/home/koehlp/Downloads/work_dirs/tracker/cascade_abdnet_mta_test_iou_iosb"
        , cam_ids=[0, 1, 2, 3, 4, 5])
        
    '''

    cmt = Config_mta_tracker(
        detections_folder="/home/koehlp/Downloads/work_dirs/detector/detections/faster_rcnn_r50_gta_trained_strong_reid_Gta2207_iosb"
        , track_output_folder="/home/koehlp/Downloads/work_dirs/tracker/faster_rcnn_r50_gta_trained_strong_reid_Gta2207_iou_fast_iosb"
        , cam_ids=[0, 1, 2, 3, 4, 5])





    run_iou_tracker(config_mta_tracker=cmt)