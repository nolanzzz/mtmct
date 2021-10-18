from utilities.python_path_utility import append_to_pythonpath

append_to_pythonpath(['feature_extractors/reid_strong_baseline'
                      ,'feature_extractors/ABD_Net'
                        ,'detectors/mmdetection'
                       ,'evaluation/py_motmetrics'], __file__)

import argparse
import mmcv
from trackers.deep_sort import DeepSort
from util import draw_bboxes
from tqdm import tqdm
from utilities.helper import xtylwh_to_xyxy
import warnings
import cv2
import logging
import json
from utilities.track_result_statistics import count_tracks


from utilities.python_path_utility import *

import pandas as pd

from feature_extractors.reid_strong_baseline.utils.logger import setup_logger

from datasets.mta_dataset_cam_iterator import get_cam_iterators
from detectors.mmdetection_detector import Mmdetection_detector
# from trackers.fair.src.lib.tracker import multitracker
# from trackers.fair.src.lib.models import *
# from trackers.fair.src.lib.utils import *
from trackers.fair.src.lib import *


class Run_tracker:
    def __init__(self,args):


        self.cfg = mmcv.Config.fromfile(args.config).root

        self.cfg.general.config_basename = os.path.basename(args.config).replace(".py","")

        self.use_original_wda = self.cfg.general.config_basename[:4] != "fair"

        self.cfg.general.repository_root = os.path.abspath(os.path.dirname(__file__))

        self.set_tracker_config_run_path()

        #mmdetection does not put everything to the device that is being set in its function calls
        #E.g. With torch.cuda.set_device(4) it will run without errors. But still using GPU 0
        #With os.environ['CUDA_VISIBLE_DEVICES'] = '4' the visibility will be restricted to only the named GPUS starting internatlly from zero
        os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg.general.cuda_visible_devices


        #Initializes the detector class by calling the constructor and creating the object
        if self.use_original_wda:
            self.detector = Mmdetection_detector(self.cfg)


        self.deep_sort = DeepSort(self.cfg, use_original_wda=self.use_original_wda)


        #Set up the logger
        logger = setup_logger("mtmct", self.cfg.general.config_run_path, 0)

        logger.info(args)
        logger.info(json.dumps(self.cfg,sort_keys=True, indent=4))

    def get_detections_path(self,cam_id):
        detections_path_folder = os.path.join(self.cfg.general.config_run_path
                                              , "detections")
        os.makedirs(detections_path_folder, exist_ok=True)

        detections_path = os.path.join(detections_path_folder, "detections_cam_{}.csv".format(cam_id))
        return detections_path

    def get_detections_frame_nos_path(self, cam_id):
        detections_path_folder = os.path.join(self.cfg.general.config_run_path
                                              , "detections")
        os.makedirs(detections_path_folder, exist_ok=True)

        detections_path = os.path.join(detections_path_folder, "detections_frame_nos_cam_{}.csv".format(cam_id))
        return detections_path

    def load_detections(self,cam_iterator):

        self.detections_to_store = []
        self.detections_frame_nos = []
        detections_frame_nos_loaded = pd.DataFrame({ "frame_no_cam" : []  })

        self.detections_loaded = pd.DataFrame()
        self.detections_path = self.get_detections_path(cam_id=cam_iterator.cam_id)
        self.detections_frame_nos_path = self.get_detections_frame_nos_path(cam_id=cam_iterator.cam_id)

        if os.path.exists(self.detections_frame_nos_path):
            detections_frame_nos_loaded = pd.read_csv(self.detections_frame_nos_path)


        #This will assure that if the selection of frame nos is changed new detections have to be generated
        cam_iterator_frame_nos = cam_iterator.get_all_frame_nos()
        frame_nos_union_length = len(set(detections_frame_nos_loaded["frame_no_cam"]).intersection(set(cam_iterator_frame_nos)))
        cam_iterator_frame_nos_length = len(cam_iterator_frame_nos)

        if frame_nos_union_length == cam_iterator_frame_nos_length and os.path.exists(self.detections_path):
            self.detections_loaded = pd.read_csv(self.detections_path)




    def set_tracker_config_run_path(self):

        # Build the path where results and logging files etc. should be stored

        self.cfg.general.config_run_path = os.path.join(self.cfg.general.repository_root
                                                        ,"work_dirs"
                                                        , "tracker"
                                                        , "config_runs"
                                                        , self.cfg.general.config_basename)
        os.makedirs(self.cfg.general.config_run_path, exist_ok=True)


    def save_detections(self):
        if len(self.detections_to_store) > 0:
            detections_to_store_df = pd.DataFrame(self.detections_to_store)

            detections_to_store_df = detections_to_store_df.astype({"frame_no_cam": int
                                                                           , "id": int
                                                                           , "x": int
                                                                           , "y": int
                                                                           , "w": int
                                                                           , "h": int
                                                                           , "score": float})
            detections_to_store_df.to_csv(self.detections_path, index=False)

            detections_frame_nos = pd.DataFrame(self.detections_frame_nos)
            detections_frame_nos.to_csv(self.detections_frame_nos_path, index=False)


    def store_detections_one_frame(self, frame_no_cam, xywh_bboxes, scores):
        '''
        "frame_no_cam,id,x,y,w,h,score"
        :param xywh_bboxes:
        :return:
        '''



        for index, xywh_bbox_score in enumerate(zip(xywh_bboxes,scores)):
            xywh_bbox, score = xywh_bbox_score

            self.detections_to_store.append({ "frame_no_cam" : frame_no_cam
                                        , "id" : index
                                        , "x" : xywh_bbox[0]
                                        , "y" : xywh_bbox[1]
                                        , "w" : xywh_bbox[2]
                                        , "h" : xywh_bbox[3]
                                        , "score" : score })


    def img_callback_original_wda(self, dataset_img):


        if len(self.detections_loaded) > 0:
            detections_current_frame =  self.detections_loaded[self.detections_loaded["frame_no_cam"] == dataset_img.frame_no_cam]
            scores = detections_current_frame["score"].tolist()
            bboxes_xtlytlwh = list(zip(detections_current_frame["x"], detections_current_frame["y"],detections_current_frame["w"],detections_current_frame["h"]))
        else:
            bboxes_xtlytlwh, scores = self.detector.detect(dataset_img.img)
            self.store_detections_one_frame(dataset_img.frame_no_cam, bboxes_xtlytlwh, scores)
            self.detections_frame_nos.append({ "frame_no_cam" : dataset_img.frame_no_cam })


        draw_img = dataset_img.img
        if bboxes_xtlytlwh is not None:

            outputs = self.deep_sort.update(bboxes_xtlytlwh, scores, dataset_img)

            if len(outputs) > 0:

                bboxes_xtylwh = outputs[:, :4]
                bboxes_xyxy = [ xtylwh_to_xyxy(bbox_xtylwh,dataset_img.img_dims) for bbox_xtylwh in bboxes_xtylwh ]
                identities = outputs[:, -2]
                detection_idxs = outputs[:, -1]
                draw_img = draw_bboxes(dataset_img.img, bboxes_xyxy, identities)


                for detection_idx, person_id, bbox in zip(detection_idxs,identities,bboxes_xyxy):
                    print('%d,%d,%d,%d,%d,%d,%d,%d' % (
                            dataset_img.frame_no_cam, dataset_img.cam_id, person_id, detection_idx,int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), file=self.track_results_file)

        if self.cfg.general.display_viewer:
            cv2.imshow("Annotation Viewer", draw_img)
            cv2.waitKey(1)


    def img_callback(self, dataset_img):

        draw_img = dataset_img.img

        outputs = self.deep_sort.update_with_fair_detections(dataset_img)

        if len(outputs) > 0:

            bboxes_xtylwh = outputs[:, :4]
            bboxes_xyxy = [ xtylwh_to_xyxy(bbox_xtylwh,dataset_img.img_dims) for bbox_xtylwh in bboxes_xtylwh ]
            identities = outputs[:, -2]
            detection_idxs = outputs[:, -1]
            draw_img = draw_bboxes(dataset_img.img, bboxes_xyxy, identities)


            for detection_idx, person_id, bbox in zip(detection_idxs,identities,bboxes_xyxy):
                print('%d,%d,%d,%d,%d,%d,%d,%d' % (
                        dataset_img.frame_no_cam, dataset_img.cam_id, person_id, detection_idx,int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), file=self.track_results_file)

        if self.cfg.general.display_viewer:
            cv2.imshow("Annotation Viewer", draw_img)
            cv2.waitKey(1)


    def run_on_cam_images(self,cam_iterator):
        for image in cam_iterator:
            self.pbar_tracker.update()
            if self.use_original_wda:
                self.img_callback_original_wda(image)
            else:
                self.img_callback(image)

    @staticmethod
    def get_cam_iterator_len_sum(cam_image_iterators):

        overall_len = 0
        for cam_iterator in cam_image_iterators:
            overall_len += len(cam_iterator)
        return overall_len

    def get_track_results_path(self,cam_id):

        tracker_results_folder =  os.path.join(self.cfg.general.config_run_path,"tracker_results")
        os.makedirs(tracker_results_folder,exist_ok=True)

        return os.path.join(tracker_results_folder,"track_results_{}.txt".format(cam_id))


    def run_on_dataset(self):
        logger = logging.getLogger("mtmct")
        logger.info("Starting tracking on dataset.")

        cam_iterators = get_cam_iterators(self.cfg, self.cfg.data.source.cam_ids)
        frame_count_all_cams = self.get_cam_iterator_len_sum(cam_iterators)
        self.pbar_tracker = tqdm(total=frame_count_all_cams)



        for cam_iterator in cam_iterators:
            logger.info("Processing cam {}".format(cam_iterator.cam_id))

            self.track_results_path = self.get_track_results_path(cam_iterator.cam_id)
            logger.info(self.track_results_path)
            self.track_results_file = open(self.track_results_path, 'w')
            print("frame_no_cam,cam_id,person_id,detection_idx,xtl,ytl,xbr,ybr", file=self.track_results_file)
            if self.use_original_wda:
                self.load_detections(cam_iterator)

            self.run_on_cam_images(cam_iterator)

            self.track_results_file.close()

            if self.use_original_wda:
                self.save_detections()

            track_count_string = count_tracks(self.track_results_path)
            logger.info(track_count_string)



    def run(self):

        self.run_on_dataset()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)

    return parser.parse_args()


if __name__=="__main__":
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    args = parse_args()

    run_tracker = Run_tracker(args)

    run_tracker.run()