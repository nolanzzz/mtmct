import numpy as np

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker
import importlib
import os
import pickle

from feature_extractors.reid_strong_extractor import Reid_strong_extractor
from feature_extractors.abd_net_extractor import Abd_net_extractor

__all__ = ['DeepSort']


class DeepSort(object):

    def __init__(self, cfg, max_dist=0.2, use_original_wda=False):

        self.cfg = cfg

        if use_original_wda:
            #Create absolute paths for reid
            cfg.feature_extractor.reid_strong_extractor.reid_strong_baseline_config = os.path.join(cfg.general.repository_root
                                                                                                   ,cfg.feature_extractor.reid_strong_extractor.reid_strong_baseline_config)
            cfg.feature_extractor.reid_strong_extractor.checkpoint_file = os.path.join(cfg.general.repository_root
                                                                                       ,cfg.feature_extractor.reid_strong_extractor.checkpoint_file)

            cfg.feature_extractor.abd_net_extractor.load_weights = os.path.join(cfg.general.repository_root
                                                                                       ,cfg.feature_extractor.abd_net_extractor.load_weights)



            if cfg.feature_extractor.feature_extractor_name == "abd_net_extractor":
                self.feature_extractor = Abd_net_extractor(cfg.feature_extractor.abd_net_extractor)
            elif cfg.feature_extractor.feature_extractor_name == "fair":
                self.feature_extractor = Abd_net_extractor(cfg.feature_extractor.fair)
            else:
                self.feature_extractor = Reid_strong_extractor(cfg.feature_extractor)

        max_cosine_distance = max_dist

        nn_budget = cfg.tracker.nn_budget
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

        self.feature_pickle_folder = self.get_features_pickle_folder()
        self.track_feature_pickle_folder = cfg.general.track_features_folder



    def get_features_pickle_folder(self):
        feature_pickle_folder = os.path.join(self.cfg.general.config_run_path, "features")
        os.makedirs(feature_pickle_folder, exist_ok=True)

        return feature_pickle_folder


    # def get_track_features_pickle_folder(self):
    #     track_feature_pickle_folder = os.path.join(self.cfg.general.config_run_path, "track_features")
    #     os.makedirs(track_feature_pickle_folder, exist_ok=True)
    #
    #     return track_feature_pickle_folder


    def update(self, bboxes_xtylwh, confidences, dataset_img):
        '''
        Attention the input bounding boxes have to be in the x-top y-left and width height format!
         Not the x-center y-center width
        :param bboxes_xtylwh: input bounding boxes have to be in the x-top y-left and width height format
        :param confidences:
        :param ori_img:
        :return:
        '''
        self.height, self.width = dataset_img.img.shape[:2]
        # generate detections

        feature_pkl_path = os.path.join(self.feature_pickle_folder, "frame_no_cam_{}_cam_id_{}.pkl".format(dataset_img.frame_no_cam,dataset_img.cam_id))


        if os.path.exists(feature_pkl_path):
            with open(feature_pkl_path, 'rb') as handle:
                detections = pickle.load(handle)

        else:

            features = self._get_features(bboxes_xtylwh, dataset_img.img)
            detections = [Detection(bbox_xtylwh, conf, feature) for bbox_xtylwh,conf, feature in zip(bboxes_xtylwh, confidences, features)]
            # print("Detections: ", detections)
            with open(feature_pkl_path, 'wb') as handle:
                pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)



        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x,y,w,h = track.to_tlwh()
            track_id = track.track_id
            outputs.append(np.array([x,y,w,h,track_id,track.last_detection_idx()], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs


    def update_with_wda_features(self, bboxes_xtylwh, confidences, dataset_img):
        '''
        Attention the input bounding boxes have to be in the x-top y-left and width height format!
         Not the x-center y-center width
        :param bboxes_xtylwh: input bounding boxes have to be in the x-top y-left and width height format
        :param confidences:
        :param ori_img:
        :return:
        '''
        self.height, self.width = dataset_img.img.shape[:2]
        # generate detections

        feature_pkl_path = os.path.join(self.feature_pickle_folder, "frame_no_cam_{}_cam_id_{}.pkl".format(dataset_img.frame_no_cam,dataset_img.cam_id))
        ''' reformat features for multi-cam clustering '''
        track_feature_pkl_path = os.path.join(self.track_feature_pickle_folder,"frame_no_cam_{}_cam_id_{}.pkl".format(dataset_img.frame_no_cam,dataset_img.cam_id))
        person_id_to_feature = {}

        if os.path.exists(feature_pkl_path):
            with open(feature_pkl_path, 'rb') as handle:
                detections = pickle.load(handle)

        else:

            features = self._get_features(bboxes_xtylwh, dataset_img.img)
            detections = [Detection(bbox_xtylwh, conf, feature) for bbox_xtylwh,conf, feature in zip(bboxes_xtylwh, confidences, features)]
            # print("Detections: ", detections)
            with open(feature_pkl_path, 'wb') as handle:
                pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)



        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, keep_features=True)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x,y,w,h = track.to_tlwh()
            track_id = track.track_id
            outputs.append(np.array([x,y,w,h,track_id,track.last_detection_idx()], dtype=np.int))
            ''' store track features so that multi_clustering does not need to do it again '''
            if len(track.features) > 0:
                person_id_to_feature[track.track_id] = track.features[-1]
                with open(track_feature_pkl_path, 'wb') as handle:
                    pickle.dump(person_id_to_feature, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs


    def update_with_fair_detections(self, dataset_img):
        '''
        Attention the input bounding boxes have to be in the x-top y-left and width height format!
         Not the x-center y-center width
        :return:
        '''
        self.height, self.width = dataset_img.img.shape[:2]
        feature_pkl_path = os.path.join(self.feature_pickle_folder, "frame_no_cam_{}_cam_id_{}.pkl".format(dataset_img.frame_no_cam,dataset_img.cam_id))
        ''' reformat features for multi-cam clustering '''
        track_feature_pkl_path = os.path.join(self.track_feature_pickle_folder, "frame_no_cam_{}_cam_id_{}.pkl".format(dataset_img.frame_no_cam,dataset_img.cam_id))
        person_id_to_feature = {}

        outputs = []

        if os.path.exists(feature_pkl_path):
            with open(feature_pkl_path, 'rb') as handle:
                detections = pickle.load(handle)

            # update tracker
            self.tracker.predict()
            self.tracker.update(detections, keep_features=True)

            # output bbox identities
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                x,y,w,h = track.to_tlwh()
                track_id = track.track_id
                outputs.append(np.array([x,y,w,h,track_id,track.last_detection_idx()], dtype=np.int))
                if len(track.features) > 0:
                    person_id_to_feature[track.track_id] = track.features[-1]
                    with open(track_feature_pkl_path, 'wb') as handle:
                        pickle.dump(person_id_to_feature, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if len(outputs) > 0:
                outputs = np.stack(outputs,axis=0)
        return outputs

    
    def _get_features(self, bboxes_xtylwh, ori_img):
        im_crops = []
        for bbox in bboxes_xtylwh:
            x,y,w,h = bbox
            im = ori_img[y:y+h, x:x+w]
            im_crops.append(im)
        if im_crops:
            features = self.feature_extractor.extract(im_crops)
        else:
            features = np.array([])
        return features


