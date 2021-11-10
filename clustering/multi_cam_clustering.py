import os.path as osp
import gc

import pandas as pd
import mmcv
from clustering.feature_extraction_track_results import Feature_extraction
import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from utilities.dataset_statistics import *
from clustering.clustering_utils import *
from scipy import spatial
import copy
from clustering import heapq
from shapely.geometry import Point
from utilities.non_daemonic_pool import NonDeamonicPool
from utilities.helper import *
from evaluation.multicam_evaluation import calculate_multi_cam_mean_and_std
from clustering.velocity_calculation import Velocity_calculation
from utilities.helper import rename_short_motmetrics
from evaluation.multicam_evaluation import Multicam_evaluation, Motmetrics_distance
import pickle
from evaluation.motmetrics_evaluation import eval_single_cam_multiple_cams
import multiprocessing
import traceback


from evaluation.motmetrics_evaluation import calculate_single_cam_mean_and_std
from evaluation.multicam_evaluation import split_data

import trackers.fair.src.lib.tracker

def get_feature_pickle_folder(work_dirs, config_basename, dataset_type):
    feature_pickle_folder = os.path.join(work_dirs
                                         , "clustering"
                                         , "config_runs"
                                         , config_basename
                                         , "pickled_appearance_features"
                                         , dataset_type
                                          )

    return feature_pickle_folder


def get_feature_pickle_path(work_dirs, frame_no_cam, cam_id, config_basename, dataset_type):
    feature_pickle_folder = get_feature_pickle_folder(work_dirs, config_basename, dataset_type)

    os.makedirs(feature_pickle_folder, exist_ok=True)
    # feature_pickle_filename = os.path.join(feature_pickle_folder, "frameno_{}_camid_{}.pkl".format(frame_no_cam,cam_id))
    feature_pickle_filename = os.path.join(feature_pickle_folder, "frame_no_cam_{}_cam_id_{}.pkl".format(frame_no_cam,cam_id))

    return feature_pickle_filename

def get_frame_nos_track_results(track_results):
    frame_numbers = track_results.groupby("frame_no_cam", as_index=False).mean()
    frame_numbers = frame_numbers["frame_no_cam"].tolist()
    frame_numbers = list(map(int, frame_numbers))
    return frame_numbers


def load_tracks_all_cams(cam_count, track_results_folder):
    track_result_dataframes = []


    if isinstance(track_results_folder, list):

        for cam_id, track_result_dataframe in enumerate(track_results_folder):

            track_result_dataframe = track_result_dataframe.astype(int)

            track_result_dataframes.append({"cam_id": cam_id, "track_results": track_result_dataframe})

        return track_result_dataframes


    # load track results for each cam
    for cam_id in range(cam_count):
        track_result_path = osp.join(track_results_folder, "track_results_{}.txt".format(cam_id))

        track_result_dataframe = pd.read_csv(track_result_path)

        track_result_dataframe = track_result_dataframe.astype({"frame_no_cam": int
                                                                   , "cam_id": int
                                                                   , "person_id": int
                                                                   , "xtl": int
                                                                   , "ytl": int
                                                                   , "xbr": int,
                                                                "ybr": int})

        track_result_dataframes.append({"cam_id": cam_id, "track_results": track_result_dataframe})

    return track_result_dataframes

def pickle_all_reid_features(work_dirs
                             ,mc_cfg
                             ,track_results_folder
                             ,dataset_folder
                             ,config_basename
                             ,dataset_type
                             ,cam_count
                             ,feature_extraction):


    track_result_dataframes = load_tracks_all_cams(cam_count=cam_count,
                                                   track_results_folder=track_results_folder)
    print("Creating features of dataset path: {}".format(dataset_folder))
    print("Creating features of dataset type: {}".format(dataset_type))

    if isinstance(track_results_folder,str):
        print("Track results folder: {}".format(track_results_folder))

    for track_results_one_cam in track_result_dataframes:

        frame_nos = get_frame_nos_track_results(track_results_one_cam["track_results"])
        track_results_df = track_results_one_cam["track_results"]
        cam_id = track_results_one_cam["cam_id"]

        cam_video_path = os.path.join(dataset_folder,"cam_{}".format(cam_id),"cam_{}.mp4".format(cam_id))
        print("cam_video_path:", cam_video_path)
        video_capture = cv2.VideoCapture(cam_video_path)



        print("Creating features of the track results for cam {}".format(cam_id))
        for frame_no_cam in tqdm(frame_nos):

            one_frame = track_results_df[track_results_df["frame_no_cam"] == frame_no_cam]
            xyxy_bboxes = zip(one_frame["xtl"], one_frame["ytl"], one_frame["xbr"], one_frame["ybr"])

            feature_pickle_filename = get_feature_pickle_path(work_dirs=work_dirs
                                                              , frame_no_cam=frame_no_cam
                                                              , cam_id=cam_id
                                                              , config_basename=config_basename
                                                              , dataset_type=dataset_type)
            # # load Fair detections & features here
            # if os.path.exists(feature_pickle_filename):
            #     with open(feature_pickle_filename, 'rb') as handle:
            #         detections = pickle.load(handle)
            #         print("detections:", detections)


            # check if need to use wda extractor
            if mc_cfg.config_basename[:4] != "fair" and not os.path.exists(feature_pickle_filename):
                if len(feature_extraction) == 0:
                    feature_extraction.append(Feature_extraction(mc_cfg))

                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no_cam)

                ret, frame = video_capture.read()
                if frame_no_cam == 4920:
                    print(ret)
                    print(frame)

                if not ret and frame_no_cam < 4919:
                    print(frame_no_cam)
                    print(ret)
                    print(frame)
                    raise Exception("Unable to read video frame.")


                # print("\nframe_no_cam: ",frame_no_cam)
                # print("frame:", frame.shape)

                features_frame = feature_extraction[0].get_features(xyxy_bboxes, frame)
                person_id_to_feature = {}
                for person_id, feature in zip(one_frame["person_id"], features_frame):
                    person_id_to_feature[person_id] = feature
                with open(feature_pickle_filename, 'wb') as handle:
                    pickle.dump(person_id_to_feature, handle, protocol=pickle.HIGHEST_PROTOCOL)



class Multi_cam_clustering:

    def __init__(self,test_track_results_folder
                 ,train_track_results_folder
                 ,test_dataset_folder
                 ,train_dataset_folder
                 ,work_dirs
                 ,cam_count
                 ,person_identifier
                 ,config_basename
                 ,chunk_id):

        self.logger = logging.getLogger("clustering_logger")
        self.person_identifier = person_identifier
        self.train_track_results_folder = train_track_results_folder
        self.test_track_results_folder = test_track_results_folder
        self.test_dataset_folder = test_dataset_folder
        self.train_dataset_folder = train_dataset_folder
        self.work_dirs = work_dirs
        self.cam_count = cam_count
        self.track_result_dataframes = []
        self.person_id_tracks = []
        self.track_no_to_track_frame_no_cam_to_track_pos = {} #to cache the results of the function which calculates this
        self.query_pos_cam_id_to_nearest_node = {} #To cache the searches for the nearest nodes in one cam
        self.config_basename = config_basename
        self.chunk_id = chunk_id
        self.should_save_track_results = True

    def get_person_id_tracks_pickle_path(self,config_basename, dataset_type):


        person_id_tracks_pickle_folder = os.path.join(self.work_dirs
                                             , "clustering"
                                             , "config_runs"
                                             , config_basename
                                             , "person_id_tracks"
                                             , "chunk_{}".format(self.chunk_id)
                                             , dataset_type)

        os.makedirs(person_id_tracks_pickle_folder,exist_ok=True)

        person_id_tracks_pickle_path = osp.join(person_id_tracks_pickle_folder,"person_id_tracks.pkl")

        return person_id_tracks_pickle_path


    def load_person_id_tracks(self,track_results_folder):
        if len(self.person_id_tracks) > 0:
            return

        self.track_result_dataframes = load_tracks_all_cams(cam_count=self.cam_count,
                                                            track_results_folder=track_results_folder)

        for track_results in self.track_result_dataframes:
            track_results_df = track_results["track_results"]
            cam_id = track_results["cam_id"]
            person_id_to_track = get_person_id_to_track(track_results_df)

            for person_id, track in person_id_to_track.items():
                self.person_id_tracks.append({ "person_id" : person_id , "cam_id" : cam_id , "track" : track})


    def get_all_tracks_with_feature_mean(self,track_results_folder,dataset_type):

        get_person_id_tracks_pickle_path = self.get_person_id_tracks_pickle_path(self.config_basename,dataset_type)

        if os.path.exists(get_person_id_tracks_pickle_path):
            print("Found pickled person_id_tracks")
            print(get_person_id_tracks_pickle_path)
            with open(get_person_id_tracks_pickle_path, 'rb') as handle:
                self.person_id_tracks = pickle.load(handle)

            return

        self.load_person_id_tracks(track_results_folder)
        print("Did not find pickled person_id_tracks. Calculating them now.")

        for track_dict in tqdm(self.person_id_tracks):


            feature_mean = self.calculate_track_feature_mean(track=track_dict
                                                             ,dataset_type=dataset_type)
            # print("feature_mean:", feature_mean)
            track_dict["feature_mean"] = feature_mean



        with open(get_person_id_tracks_pickle_path, 'wb') as handle:
            print("Writing person_id_tracks")
            print(get_person_id_tracks_pickle_path)
            pickle.dump(self.person_id_tracks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def calculate_dist_clustered_tracks(self, track1_indices, track2_indices):
        min_dist = sys.maxsize
        for track1_idx in track1_indices:
            for track2_idx in track2_indices:
                pair_dist = self.track_pair_to_dist[frozenset([track1_idx, track2_idx])]

                min_dist = min(min_dist, pair_dist)
                # if there is a hard constraint it should not be clustered
                if pair_dist == np.Inf:
                    return np.Inf

        return min_dist


    def calculate_track_feature_mean(self,track,dataset_type):
        track_list = track["track"]
        person_id = track["person_id"]
        cam_id = track["cam_id"]
        track_features = []
        for track_entry in track_list:

            frame_no_cam = track_entry["frame_no_cam"]

            feature_pickle_name = get_feature_pickle_path(work_dirs=self.work_dirs
                                                          ,frame_no_cam=frame_no_cam
                                                          ,cam_id=cam_id
                                                          ,config_basename=self.config_basename
                                                          ,dataset_type=dataset_type)
            if os.path.exists(feature_pickle_name):
                with open(feature_pickle_name, 'rb') as handle:
                    feature_dict = pickle.load(handle)
                    if person_id in feature_dict:
                        track_features.append(feature_dict[person_id])

        track_features = np.array(track_features)
        track_mean = np.mean(track_features,axis=0)
        return track_mean





    def are_tracks_cam_id_frame_disjunct(self, track1, track2):
        '''
        Check if none of the track positions has the same framenumber in one camera.
        Combining such tracks should be avoided because one person can only be at one position in a cam image.
        :param track1:
        :param track2:
        :return:
        '''
        track1 = get_cluster_tracks_as_list(track1)
        track2 = get_cluster_tracks_as_list(track2)

        track1_cam_id_set = set([(track_pos["frame_no_cam"],track_pos["cam_id"]) for track_pos in track1])
        track2_cam_id_set = set([(track_pos["frame_no_cam"],track_pos["cam_id"]) for track_pos in track2])

        return track1_cam_id_set.isdisjoint(track2_cam_id_set)



    def tracks_to_frame_no_to_track_pos(self, track,track_no):
        cache_key = (track_no,len(track))
        if cache_key in self.track_no_to_track_frame_no_cam_to_track_pos:
            return self.track_no_to_track_frame_no_cam_to_track_pos[cache_key]

        track_frame_no_cam_to_track_pos = defaultdict(list)
        for track_pos in track:
            track_frame_no_cam_to_track_pos[track_pos["frame_no_cam"]].append(track_pos)

        self.track_no_to_track_frame_no_cam_to_track_pos[cache_key] = track_frame_no_cam_to_track_pos
        return track_frame_no_cam_to_track_pos

    def are_tracks_frame_overlap_disjunct(self, track1, track2):


        def can_exist_in_cams_at_same_time(track1_positions,track2_positions):

            for track1_pos in track1_positions:
                for track2_pos in track2_positions:
                    cam_id_track1 = track1_pos["cam_id"]
                    cam_id_track2 = track2_pos["cam_id"]
                    point_track1 = get_bbox_middle_pos(track1_pos["bbox"])
                    point_track2 = get_bbox_middle_pos(track2_pos["bbox"])


                    track1_point_in_polygon = self.cam_id_to_cam_id_to_polygon[cam_id_track1][cam_id_track2].contains(Point(point_track1))
                    track2_point_in_polygon = self.cam_id_to_cam_id_to_polygon[cam_id_track2][cam_id_track1].contains(Point(point_track2))

                    if not track1_point_in_polygon or not track2_point_in_polygon:
                        return False
            return True

        #As we have this check of single camera time restrictions as a separate distance
        if track1[0]["cam_id"] == track2[0]["cam_id"]:
            return True

        track1_no = track1[0]["track_unclustered_no"]
        track2_no = track2[0]["track_unclustered_no"]

        track1 = get_cluster_tracks_as_list(track1)
        track2 = get_cluster_tracks_as_list(track2)


        track1_frame_no_cam_to_track_pos = self.tracks_to_frame_no_to_track_pos(track1,track1_no)
        track2_frame_no_cam_to_track_pos = self.tracks_to_frame_no_to_track_pos(track2,track2_no)

        for track1_frame_no_cam, track1_positions in track1_frame_no_cam_to_track_pos.items():
            if track1_frame_no_cam in track2_frame_no_cam_to_track_pos:
                track2_positions = track2_frame_no_cam_to_track_pos[track1_frame_no_cam]

                if not can_exist_in_cams_at_same_time(track1_positions,track2_positions):
                    return False




        return True

    def initialize_maximum_link_predict_distance(self,track_results_folder,dataset_type,maximum_link_frames=500):
        def get_velocity_stats_pickle_path(dataset_type):

            folder = os.path.join(self.work_dirs
                             ,"clustering"
                             ,"config_runs"
                             ,self.config_basename
                             ,"velocity_stats"
                             , "chunk_{}".format(self.chunk_id)
                             , dataset_type)


            os.makedirs(folder, exist_ok=True)

            path = os.path.join(folder, "velocity_stats.pkl")

            return path

        pickle_path = get_velocity_stats_pickle_path(dataset_type=dataset_type)
        vc = Velocity_calculation(track_results_folder=track_results_folder
                                  ,cam_ids=list(range(self.cam_count))
                                  ,pickle_path=pickle_path)


        velocity_stats = vc.get_velocity_stats(n_tail=40,step_width=10)
        velocity_mean = velocity_stats["velocity_mean"]

        self.maximum_link_pred_distance = velocity_mean * maximum_link_frames


    def get_camera_transition_constraint(self, track1, track2, transitions_threshold=1):
        track1, track2 = self.make_track1_older_then_track2(track1, track2)

        track1_cam_id, track2_cam_id = track1[-1]['cam_id'], track2[0]['cam_id']

        if track1_cam_id == track2_cam_id:
            return 0

        if self.camera_transition_matrix[track1_cam_id,track2_cam_id] >= transitions_threshold:
            return 0
        else:
            return np.Inf


    def get_track_pred_pos_start_distance(self, track1, track2):

        track1, track2 = self.make_track1_older_then_track2(track1,track2)

        if track1[-1]['cam_id'] != track2[0]['cam_id']:
            return 0

        pred_position = self.get_predicted_position(track1, track2)

        track1 = track1[-1]["track"]
        track2 = track2[0]["track"]

        track2_start = get_bbox_middle_pos(track2[0]["bbox"])
        bbox_height = get_bbox_height(track1[-1]["bbox"])
        pred_pos_start_distance = np.linalg.norm(np.array(pred_position)-np.array(track2_start)) / bbox_height

        if pred_pos_start_distance > self.maximum_link_pred_distance:
            return 0
        else:
            #Because this distance can't be calculated for every track pair a negative distance will be returned
            return -(1.0 - pred_pos_start_distance / self.maximum_link_pred_distance)

    def get_predicted_position(self,track1,track2,n_tail=40):

        track1, track2 = self.make_track1_older_then_track2(track1, track2)

        track1 = track1[-1]["track"]
        track2 = track2[0]["track"]

        last_n_positons = track1[-n_tail:]


        first_pos_track1 = get_bbox_middle_pos(last_n_positons[0]["bbox"])
        second_pos_track1 = get_bbox_middle_pos(last_n_positons[-1]["bbox"])
        first_pos_frame_no_track1 = last_n_positons[0]["frame_no_cam"]
        second_pos_frame_no_track1 = last_n_positons[-1]["frame_no_cam"]

        if len(track1) == 1:
            return np.array(first_pos_track1)

        velocity = (np.array(second_pos_track1) - np.array(first_pos_track1)) / (second_pos_frame_no_track1 - first_pos_frame_no_track1)


        start_pos_frame_no_track2 = track2[0]["frame_no_cam"]

        passed_time = start_pos_frame_no_track2 - second_pos_frame_no_track1

        predicted_position = np.array(second_pos_track1) + (velocity * passed_time)


        return predicted_position


    def get_feature_mean_distance(self,track1,track2):

        track1, track2 = self.make_track1_older_then_track2(track1,track2)

        #return np.linalg.norm(np.array(track1[-1]["feature_mean"])-np.array(track2[0]["feature_mean"]))

        #divided by to to put this distance which is between 0 and 2 to a range between 0 and 1
        return spatial.distance.cosine(np.array(track1[-1]["feature_mean"]),np.array(track2[0]["feature_mean"])) / 2

    def make_track1_older_then_track2(self,track1,track2):
        # We need track1[-1] < track2[0]
        #the [-1]["track"][-1] is needed because one track is stored as fragments that are in one list
        if not (track1[-1]["track"][-1]["frame_no_cam"] < track2[0]["track"][0]["frame_no_cam"]):
            track_tmp = track1
            track1 = track2
            track2 = track_tmp

        return track1,track2

    def combine_tracks(self,track1,track2):
        track1,track2 = self.make_track1_older_then_track2(track1,track2)

        return track1 + track2

    def strip_outer_dict_from_tracks(self,tracks):

        return [track["track"] for track in tracks]

    def assign_track_unclustered_no(self,track_clusters):
        '''

        :param track_clusters:
        :return:
        '''
        for track_cluster_no, track_cluster in enumerate(track_clusters):
            track_cluster["track_unclustered_no"] = track_cluster_no

    def get_tracks_per_cam(self,track_clusters):
        cam_id_to_tracks = defaultdict(list)

        for track_cluster in track_clusters:
            for track_dict in track_cluster:

                cam_id = track_dict["cam_id"]
                cam_id_to_tracks[cam_id].append(track_dict)

        return cam_id_to_tracks



    def save_track_all_cams(self,track_clusters,config_basename,dataset_type):
        def fetch_async_results(async_tracking_results):
            tracking_results = []
            for async_result in async_tracking_results:
                tracking_result = async_result.get()
                tracking_results.append(tracking_result)

            tracking_results.sort(key=lambda x: x["cam_id"])

            tracking_results = list(map(lambda x: x["tracks"],tracking_results))
            return tracking_results

        save_results_folder = os.path.join(self.work_dirs
                                           , "clustering"
                                           , "config_runs"
                                           , config_basename
                                           , "multicam_clustering_results"
                                           , "chunk_{}".format(self.chunk_id)
                                           , dataset_type
                                           )
        os.makedirs(save_results_folder, exist_ok=True)

        cam_id_to_tracks = self.get_tracks_per_cam(track_clusters)
        pool = ThreadPool(processes=len(cam_id_to_tracks))
        async_results = []
        for cam_id, tracks in cam_id_to_tracks.items():
            async_result = pool.apply_async(self.save_tracks_one_cam,args=(tracks,cam_id, save_results_folder))

            async_results.append(async_result)
        pool.close()
        pool.join()

        tracking_results = fetch_async_results(async_results)

        return save_results_folder,tracking_results


    def save_tracks_one_cam(self,tracks, cam_id, track_results_folder):
        try:

            print("Starting save procedure of combined tracks cam {}".format(cam_id))
            combined_tracks = []
            for track_dict in tracks:
                track_no = track_dict["track_no"]
                track = track_dict["track"]
                track_unclustered_no = track_dict["track_unclustered_no"]
                for track_pos in track:
                    combined_tracks.append({"frame_no_cam": track_pos["frame_no_cam"]
                                                                 , "cam_id": cam_id
                                                                 , "person_id": track_no
                                                                 , "track_unclustered_no" : track_unclustered_no
                                                                 , "xtl": track_pos["bbox"][0]
                                                                 , "ytl": track_pos["bbox"][1]
                                                                 , "xbr": track_pos["bbox"][2]
                                                                 , "ybr": track_pos["bbox"][3]})

            combined_tracks = pd.DataFrame(combined_tracks)
            combined_tracks = combined_tracks.astype({"frame_no_cam": int
                                                         , "cam_id": int
                                                         , "person_id": int
                                                         , "track_unclustered_no" : int
                                                         , "xtl": float
                                                         , "ytl": float
                                                         , "xbr": float
                                                         , "ybr": float})
            combined_tracks = combined_tracks.sort_values(by=["frame_no_cam"]) #Removed sort by person id to increase the speed

            if self.should_save_track_results:
                track_results_path = os.path.join(track_results_folder, "track_results_{}.txt".format(cam_id))
                combined_tracks.to_csv(track_results_path, index=False)
                print("Saved combined track results to: {}".format(track_results_path))

            print("Returned tracks of cam: {}".format(cam_id))
            return { "cam_id" : cam_id , "tracks" : combined_tracks }

        except Exception as e:
            print(e)
            raise


    def initialize_overlapping_area_tester(self):



        overlapping_area_hulls_path = get_overlapping_area_hulls_path(work_dirs=self.work_dirs
                                                                      ,config_basename=self.config_basename
                                                                      ,dataset_type="test")

        cam_id_to_cam_id_to_hull = get_overlapping_areas(dataset_path=self.train_dataset_folder
                                                         ,working_dirs=self.work_dirs
                                                         ,cam_count=self.cam_count
                                                         ,person_identifier=self.person_identifier
                                                         ,pickle_path=overlapping_area_hulls_path)



        self.cam_id_to_cam_id_to_polygon = hull_to_polygon(cam_id_to_cam_id_to_hull)


    def initialize_cam_transition_matrix(self):

        def get_cam_transition_matrix_path(work_dirs, config_basename, dataset_type):
            cam_transitions_folder = os.path.join(work_dirs
                                                         , "clustering"
                                                         , "config_runs"
                                                         , config_basename
                                                         , "cam_transitions"
                                                         , dataset_type)

            os.makedirs(cam_transitions_folder, exist_ok=True)

            cam_homographies_path = os.path.join(cam_transitions_folder, "cam_transitions_matrix.txt")

            return cam_homographies_path

        transition_matrix_path = get_cam_transition_matrix_path(work_dirs=self.work_dirs
                                                                ,config_basename=self.config_basename
                                                                ,dataset_type="test")

        if os.path.exists(transition_matrix_path):
            print("Found stored camera transition matrix.")
            print("Loading cam transition matrix from {}".format(transition_matrix_path))
            self.camera_transition_matrix = np.loadtxt(fname=transition_matrix_path)
            return

        print("Found no camera transition matrix. Calculating cam transition matrix now.")
        self.camera_transition_matrix = get_camera_transition_matrix(dataset_path=self.train_dataset_folder
                                                                         ,cam_count=self.cam_count
                                                                         ,output_path=transition_matrix_path
                                                                         ,working_dirs=self.work_dirs)








    def initialize_cam_homographies(self):

        cam_homographies_path = get_cam_homographies_path(work_dirs=self.work_dirs
                                                          ,config_basename=self.config_basename
                                                          , dataset_type="test")

        self.cam_homographies = get_cam_homographies(self.train_dataset_folder
                             , self.work_dirs
                             , self.cam_count
                             , person_identifier=self.person_identifier
                             , pickle_path=cam_homographies_path)



    def overlapping_match_score(self,track1,track2,match_distance_threshold=10):

        def count_homography_matches(track1_time_matches,track2_time_matches,match_count,lq_track_length):
            '''
            Counts the matches in one frame. It may happen that there is more than one match because of
            overlappings of more than two cams.
            :param track1_time_matches:
            :param track2_time_matches:
            :return:
            '''
            for track1_pos in track1_time_matches:
                for track2_pos in track2_time_matches:
                    track1_pos_bbox_height = get_bbox_height(track1_pos["bbox"])
                    track2_pos_bbox_height = get_bbox_height(track2_pos["bbox"])

                    #This will be done to get a sense which track might have a better quality
                    #Assuming that a greater bounding box means better quality
                    if track1_pos_bbox_height > track2_pos_bbox_height:
                        hq_track_pos = track1_pos
                        lq_track_pos = track2_pos
                    else:
                        hq_track_pos = track2_pos
                        lq_track_pos = track1_pos

                    hq_track_pos_middle = get_bbox_middle_pos(hq_track_pos["bbox"])
                    lq_track_pos_middle = get_bbox_middle_pos(lq_track_pos["bbox"])

                    hq_pos_in_overlapping_area = self.cam_id_to_cam_id_to_polygon[hq_track_pos["cam_id"]][lq_track_pos["cam_id"]].contains(Point(hq_track_pos_middle))
                    lq_pos_in_overlapping_area = self.cam_id_to_cam_id_to_polygon[lq_track_pos["cam_id"]][hq_track_pos["cam_id"]].contains(Point(lq_track_pos_middle))

                    if not hq_pos_in_overlapping_area or not lq_pos_in_overlapping_area:
                        return

                    homography = self.cam_homographies[hq_track_pos["cam_id"]][lq_track_pos["cam_id"]]

                    homogen_point = np.append(np.array(hq_track_pos_middle), [1])
                    transformed_point = np.matmul(homography, homogen_point)

                    # Homogeneous coords back to cartesian
                    transformed_point = np.true_divide(transformed_point[:2], transformed_point[-1])

                    transformed_hq_to_lq_distance = np.linalg.norm(np.array(lq_track_pos_middle)-transformed_point)

                    lq_track_length[0] += 1

                    if transformed_hq_to_lq_distance < match_distance_threshold:
                        match_count[0] += 1

        track1_no = track1[0]["track_unclustered_no"]
        track2_no = track2[0]["track_unclustered_no"]


        track1 = get_cluster_tracks_as_list(track1)
        track2 = get_cluster_tracks_as_list(track2)

        #Previously this distance had to be able to work with track clusters
        #it is able to handle multiple track positions in one track with the same frame number
        track1_frame_no_cam_to_track_pos = self.tracks_to_frame_no_to_track_pos(track1,track1_no)
        track2_frame_no_cam_to_track_pos = self.tracks_to_frame_no_to_track_pos(track2,track2_no)


        tracks_frames_intersection = set(track1_frame_no_cam_to_track_pos.keys()).intersection(track2_frame_no_cam_to_track_pos.keys())
        match_count = [0]
        lq_track_length = [0]
        for frame_no in tracks_frames_intersection:
            track1_time_matches = track1_frame_no_cam_to_track_pos[frame_no]
            track2_time_matches = track2_frame_no_cam_to_track_pos[frame_no]

            count_homography_matches(track1_time_matches,track2_time_matches,match_count,lq_track_length)

        if lq_track_length[0] == 0:
            return 0
        match_score = match_count[0] / lq_track_length[0]


        #Returning a negative match score as the weight is positve so something will be subtracted from the distance
        return -match_score




    def valid_heap_node(self, heap_node, old_clusters):
        pair_dist = heap_node[0]
        pair_data = heap_node[1]
        for old_cluster in old_clusters:
            if old_cluster in pair_data:
                return False
        return True


    def load_pairwise_distances(self,dataset,dist_name_to_distance_weights,dataset_type,distance_cache_active):

        def get_distances_and_indices_path():

            folder = os.path.join(self.work_dirs
                                                         , "clustering"
                                                         , "config_runs"
                                                         , self.config_basename
                                                         , "multicam_distances_and_indices"
                                                         , "chunk_{}".format(self.chunk_id)
                                                         , dataset_type )


            os.makedirs(folder, exist_ok=True)

            cam_homographies_path = osp.join(folder, "multicam_distances_and_indices.pkl")

            return cam_homographies_path


        pickle_path = get_distances_and_indices_path()


        if distance_cache_active and os.path.exists(pickle_path):
            print("Found distances_and_indices.")
            print(pickle_path)
            with open(pickle_path, "rb") as pickle_file:
                distances_and_indices = pickle.load(pickle_file)
        else:

            distances_and_indices = get_distances_and_indices(dataset,self.calculate_track_distances)

            if distance_cache_active:
                print("Writing pickled distances_and_indices.")
                print(pickle_path)
                os.makedirs(os.path.dirname(os.path.abspath(pickle_path)), exist_ok=True)
                with open(pickle_path, "wb") as pickle_file:
                    pickle.dump(distances_and_indices, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        return compute_pairwise_distance_normalized(distances_and_indices
                                                    ,dist_name_to_distance_weights=dist_name_to_distance_weights)






    def cluster_tracks_via_hierarchical(self,dataset_type
                                            ,dist_name_to_distance_weights
                                            ,distance_cache_active
                                            ,person_count=350
                                            ,threshold=4000.0):

        track_results_folder = None
        if dataset_type == "train":
            track_results_folder = self.train_track_results_folder
        elif dataset_type == "test":
            track_results_folder = self.test_track_results_folder

        self.get_all_tracks_with_feature_mean(track_results_folder,dataset_type)
        self.initialize_overlapping_area_tester()
        self.initialize_cam_homographies()
        self.initialize_maximum_link_predict_distance(track_results_folder=track_results_folder
                                                      ,dataset_type=dataset_type)



        tracks_all_persons = self.person_id_tracks

        #tracks_all_persons = tracks_all_persons[-100:] #For faster debugging

        self.assign_track_unclustered_no(tracks_all_persons)

        current_clusters = set([(track_idx,) for track_idx,_ in enumerate(tracks_all_persons)])
        old_clusters = []


        data_pairwise_dist = self.load_pairwise_distances(dataset=tracks_all_persons
                                                          ,dist_name_to_distance_weights=dist_name_to_distance_weights
                                                          ,dataset_type=dataset_type
                                                          ,distance_cache_active=distance_cache_active)

        self.track_pair_to_dist = get_track_pair_to_dist(data_pairwise_dist)

        heap = build_priority_queue(data_pairwise_dist)

        pbar = tqdm(total=(len(data_pairwise_dist)))

        print("Clustering tracks")
        while len(current_clusters) > person_count:
            pbar.update()

            dist, min_item = heapq.heappop(heap)
            if dist > threshold:
                break

            pair_data = min_item[1]

            if not valid_heap_node(min_item, old_clusters):
                continue

            new_cluster = sum(pair_data, [])


            for pair_item in pair_data:
                old_clusters.append(pair_item)
                current_clusters.remove(tuple(pair_item))

            self.add_heap_entry(heap, new_cluster, current_clusters)

            current_clusters.add(tuple(new_cluster))

        print("Number of tracks: {}".format(len(current_clusters)))
        print_sorted_distances(heap)
        cluster_tracks = map_clusters_track_indices_to_tracks(current_clusters,tracks_all_persons)

        #Uses a different save_track_all_cams function because the current in single cam clustering uses
        #enumerate to create the person ids but this is not possible here because the tracks will be separated to cams
        assign_track_nos_to_track_dict(cluster_tracks)

        save_results_folder,tracking_results = self.save_track_all_cams(track_clusters=cluster_tracks
                                                                 ,config_basename=self.config_basename
                                                                 ,dataset_type=dataset_type)

        return { "track_results_path" : save_results_folder
                , "track_count" : len(current_clusters)
                , "tracking_results" : tracking_results}

    def add_heap_entry(self,heap, new_cluster, current_clusters):
        for ex_cluster in current_clusters:
            new_heap_entry = []
            dist = self.calculate_dist_clustered_tracks(ex_cluster,new_cluster)
            new_heap_entry.append(dist)
            new_heap_entry.append([list(ex_cluster), list(new_cluster)])
            heapq.heappush(heap, (dist, new_heap_entry))



    def calculate_track_distances(self,candidate_track_idx,partner_track_idx,dataset):

        result_distances = {}
        candidate_track = map_idx_to_tracks(candidate_track_idx, dataset)

        partner_track = map_idx_to_tracks(partner_track_idx, dataset)

        result_distances["are_tracks_cam_id_frame_disjunct"] = 0
        result_distances["are_tracks_frame_overlap_disjunct"] = 0
        result_distances["overlapping_match_score"] = 0
        result_distances["feature_mean_distance"] = 0
        result_distances["track_pred_pos_start_distance"] = 0


        if not self.are_tracks_cam_id_frame_disjunct(candidate_track, partner_track):
            result_distances["are_tracks_cam_id_frame_disjunct"] = np.Inf

        if not self.are_tracks_frame_overlap_disjunct(candidate_track, partner_track):
            result_distances["are_tracks_frame_overlap_disjunct"] = np.Inf

        result_distances["track_pred_pos_start_distance"] = self.get_track_pred_pos_start_distance(candidate_track, partner_track)
        result_distances["overlapping_match_score"] = self.overlapping_match_score(candidate_track, partner_track)
        result_distances["feature_mean_distance"] = self.get_feature_mean_distance(candidate_track, partner_track)

        return result_distances


    def get_best_cam_settings_path(self):

        folder_path = os.path.join(self.work_dirs , "clustering"
                                                     , "best_multi_cam_settings"
                                                     , os.path.basename(self.train_track_results_folder))


        os.makedirs(folder_path, exist_ok=True)

        csv_path = os.path.join(folder_path, "best_single_cam_settings.csv")

        return csv_path

    def get_cam_tresh_results_path(self):

        folder_path = os.path.join(self.work_dirs , "clustering"
                                                     , "multi_cam_multiple_tresh_results"
                                                     , os.path.basename(self.train_track_results_folder))


        os.makedirs(folder_path, exist_ok=True)

        csv_path = os.path.join(folder_path, "single_cam_multiple_tresh_results.csv")

        return csv_path



    def get_best_weights_path(self):

        best_weights_folder = os.path.join(self.work_dirs
                                           , "clustering"
                                           , "config_runs"
                                           , self.config_basename
                                           , "best_weights" )

        os.makedirs(best_weights_folder,exist_ok=True)

        best_weights_path = os.path.join(best_weights_folder,"best_weights.csv")

        return best_weights_path

    def find_weights(self,weight_search_configs,dist_name_to_distance_weights):

        def get_best_weight(evaluation_results,metric="MOTA"):

            max_row = evaluation_results.loc[evaluation_results[metric].idxmax()]

            return max_row["weight"]

        def evaluate_one_config(weight_search_config,dist_name_to_distance_weights):
            dist_name = weight_search_config["dist_name"]
            weight_start = weight_search_config["start_value"]
            weight_stop = weight_search_config["stop_value"]
            steps = weight_search_config["steps"]
            search_weights = np.linspace(start=weight_start,stop=weight_stop,num=steps)

            evaluation_results = pd.DataFrame()

            for weight_no,weight in enumerate(search_weights):
                dist_name_to_distance_weights[dist_name] = weight

                self.logger.info(get_elapsed_time_and_msg(start_time=find_weights_start_time
                                                          , message="Starting clustering for dist_name: {} weight: {} "
                                                                    "weight_no of weights: {} of {}".format(dist_name,weight,weight_no+1,len(search_weights))))

                clustering_results = self.cluster_tracks_via_hierarchical(dataset_type="test"
                                              ,dist_name_to_distance_weights=dist_name_to_distance_weights
                                              ,person_count=1
                                                ,distance_cache_active=True
                                              ,threshold=1)

                self.logger.info(get_elapsed_time_and_msg(start_time=find_weights_start_time
                                                          ,
                                                          message="Starting evaluation for dist_name: {} weight: {} ".format(
                                                              dist_name, weight)))

                result = Multicam_evaluation(dataset_folder=self.test_dataset_folder
                    ,track_results_folder=clustering_results["tracking_results"]
                    ,cam_ids=list(range(self.cam_count))
                    ,working_dir=self.work_dirs
                    ,motmetrics_distance=Motmetrics_distance.iou_matrix).evaluate()


                evaluation_summary = result["summary"]

                self.logger.info(get_elapsed_time_and_msg(start_time=find_weights_start_time
                                                          ,
                                                          message="Finished evaluation for dist_name: {} weight: {} \n result: {}".format(
                                                              dist_name, weight, result["strsummary"])))

                evaluation_summary["weight"] = weight
                evaluation_summary["dist_name"] = dist_name
                evaluation_summary["track_count"] = clustering_results["track_count"]
                evaluation_results = evaluation_results.append(evaluation_summary, ignore_index=True)

            return evaluation_results

        def save_weight_evaluation_results(evaluation_results):

            weight_evaluation_output_folder = os.path.join(self.work_dirs
                                                           ,"clustering"
                                                           ,"config_runs"
                                                           , self.config_basename
                                                           , "weight_evaluations")


            os.makedirs(weight_evaluation_output_folder,exist_ok=True)

            output_path = os.path.join(weight_evaluation_output_folder,"weight_evaluation.csv")

            print("Saving weight evaluation results to: {}".format(output_path))

            evaluation_results.to_csv(output_path)

        def save_best_weights(dist_name_to_distance_weights):

            best_weights_df = pd.DataFrame({ "dist_name" : [] , "dist_weight" : [] })


            for dist_name, distance_weight in dist_name_to_distance_weights.items():
                best_weights_df = best_weights_df.append({ "dist_name" : dist_name
                                                           ,"dist_weight" : distance_weight
                                                           },ignore_index=True)

            best_weights_path = self.get_best_weights_path()

            print("Saving best weights to: {}".format(best_weights_path))
            best_weights_df.to_csv(best_weights_path)

        find_weights_start_time = time.time()

        all_evaluation_results = pd.DataFrame()


        for weight_search_config in weight_search_configs:
            self.logger.info(get_elapsed_time_and_msg(start_time=find_weights_start_time
                                                      , message="Starting evaluation of weights for dist_name: {}".format(
                    weight_search_config["dist_name"])))

            evaluation_results = evaluate_one_config(weight_search_config,dist_name_to_distance_weights)

            best_weight = get_best_weight(evaluation_results)
            dist_name = evaluation_results.iloc[0]["dist_name"]

            dist_name_to_distance_weights[dist_name] = best_weight

            all_evaluation_results = all_evaluation_results.append(evaluation_results)

            save_weight_evaluation_results(all_evaluation_results)
        save_best_weights(dist_name_to_distance_weights)



    def cluster_from_weights(self,best_weights_path,default_weights,dataset_type):

        if dataset_type == "test":
            dataset_folder = self.test_dataset_folder
        elif dataset_type == "train":
            dataset_folder = self.train_dataset_folder
        else:
            raise Exception("Wrong dataset type provided: {}".format(dataset_type))


        def overwrite_best_weights_in_default_weights(best_weights,default_weights):

            overwritten_weights = copy.deepcopy(default_weights)
            for _,best_weight_row in best_weights.iterrows():
                overwritten_weights[best_weight_row["dist_name"]] = best_weight_row["dist_weight"]

            return overwritten_weights


        start_time = time.time()

        if os.path.exists(best_weights_path):
            print("best weights path: {}".format(best_weights_path))
            best_weights = pd.read_csv(best_weights_path)
            best_weights = overwrite_best_weights_in_default_weights(best_weights, default_weights)
        else:
            print("no best weights path found using default weights")
            best_weights = default_weights

        self.logger.info(get_elapsed_time_and_msg(start_time,str(best_weights)))
        self.logger.info("Starting multi cam clustering chunk_{}.".format(self.chunk_id))
        clustering_results = self.cluster_tracks_via_hierarchical(dist_name_to_distance_weights=best_weights
                                                            ,threshold=1
                                                            ,distance_cache_active=True
                                                            ,person_count=1
                                                            ,dataset_type=dataset_type)

        self.logger.info("Starting single cam evaluation chunk_{}.".format(self.chunk_id))


        single_cam_eval_summary  = eval_single_cam_multiple_cams(dataset_base=dataset_folder
                                                                ,track_results_folder=clustering_results["tracking_results"]
                                                                ,cam_ids=list(range(self.cam_count))
                                                                ,working_dir=self.work_dirs)

        self.logger.info("Finished single cam evaluation chunk_{}.".format(self.chunk_id))



        self.logger.info("Starting multi cam evaluation chunk_{}.".format(self.chunk_id))
        multi_cam_eval_result = Multicam_evaluation(
                                    dataset_folder=dataset_folder
                                    , track_results_folder=clustering_results["tracking_results"]
                                    , working_dir=self.work_dirs
                                    , cam_ids=list(range(self.cam_count))
                                    ,motmetrics_distance=Motmetrics_distance.iou_matrix).evaluate()

        self.logger.info(get_elapsed_time_and_msg(start_time, multi_cam_eval_result["strsummary"])
                         + " chunk_id: {}".format(self.chunk_id))
        evaluation_summary = multi_cam_eval_result["summary"]



        return {"multi_cam_eval_summary" : evaluation_summary
                ,"single_cam_eval_summary" : single_cam_eval_summary }



def find_clustering_weights(test_track_results_folder
                             , train_track_results_folder
                             , work_dirs
                             , test_dataset_folder
                             , train_dataset_folder
                             , mc_cfg
                             , cam_count
                             , dist_name_to_distance_weights
                             , weight_search_configs
                             , config_basename
                             , person_identifier
                             , take_frames_per_cam
                             ):



    feature_extraction = []

    train_track_results_dataframes,train_dataset_dataframes = get_track_results_and_dataset(
                                                                track_results_folder=test_track_results_folder
                                                               , dataset_folder=test_dataset_folder
                                                               , working_dir=work_dirs
                                                               , cam_count=cam_count
                                                               , until_frame_no=take_frames_per_cam)



    pickle_all_reid_features(work_dirs=work_dirs
                             , mc_cfg=mc_cfg
                             , track_results_folder=train_track_results_dataframes
                             , dataset_folder=train_dataset_folder
                             , config_basename=config_basename
                             , dataset_type="test"
                             , cam_count=cam_count
                             , feature_extraction=feature_extraction)



    multi_cam_clustering = Multi_cam_clustering(
        test_track_results_folder=test_track_results_folder,
        train_track_results_folder=train_track_results_dataframes,
        work_dirs=work_dirs,
        test_dataset_folder=test_dataset_folder,
        train_dataset_folder=train_dataset_dataframes,
        person_identifier=person_identifier,
        config_basename=config_basename,
        cam_count=cam_count,
        chunk_id=0
    )

    multi_cam_clustering.find_weights(weight_search_configs=weight_search_configs
                                        ,dist_name_to_distance_weights=dist_name_to_distance_weights)





def splitted_clustering_from_weights(test_track_results_folder
                                     , train_track_results_folder
                                     , work_dirs
                                     , test_dataset_folder
                                     , train_dataset_folder
                                     , mc_cfg
                                     , cam_count
                                     , best_weights_path
                                     , default_weights
                                     , config_basename
                                     , person_identifier
                                     , n_split_parts
                                     ):

    def get_multi_cam_chunks_evaluation_path(config_basename):
        folder = os.path.join(work_dirs
                              , "clustering"
                              , "config_runs"
                              , config_basename
                              , "multi_cam_evaluation")

        os.makedirs(folder, exist_ok=True)

        evaluations_path = os.path.join(folder,"chunks_evaluations.csv")

        return evaluations_path

    def get_single_cam_chunks_evaluation_path(config_basename):
        folder = os.path.join(work_dirs
                              , "clustering"
                              , "config_runs"
                              , config_basename
                              , "single_cam_evaluation")

        os.makedirs(folder, exist_ok=True)

        evaluations_path = os.path.join(folder,"chunks_evaluations.csv")

        return evaluations_path

    def get_mean_std_multi_cam_evaluation_path(config_basename):
        folder = os.path.join(work_dirs
                              , "clustering"
                              , "config_runs"
                              , config_basename
                              , "multi_cam_evaluation"
                              )

        os.makedirs(folder, exist_ok=True)

        path = os.path.join(folder, "chunks_mean_std_evaluation.csv")

        return path

    def get_mean_std_single_cam_evaluation_path(config_basename):
        folder = os.path.join(work_dirs
                              , "clustering"
                              , "config_runs"
                              , config_basename
                              , "single_cam_evaluation"
                              )

        os.makedirs(folder, exist_ok=True)

        path = os.path.join(folder, "chunks_mean_std_evaluation.csv")

        return path

    def get_async_tracking_results(eval_results):

        tracking_results = []
        for result in eval_results:

            if isinstance(result, multiprocessing.pool.AsyncResult):
                try:
                    result = result.get()
                except Exception:
                    print("Exception in worker")
                    traceback.print_exc()
                    raise

            tracking_results.append(result)

        return tracking_results

    def combine_tracking_results(eval_results):

        tracking_results = pd.DataFrame()
        for result in eval_results:

            if isinstance(result,multiprocessing.pool.AsyncResult):
                result_dataframe = result.get()
            else:
                result_dataframe = result

            tracking_results = tracking_results.append(result_dataframe,ignore_index=True)

        tracking_results.sort_values('chunk_id',inplace=True)

        return tracking_results


    def evaluate_after_clustering(eval_results):
        single_cam_eval_summarys = list(map(lambda x: x["single_cam_eval_summary"], eval_results))
        multi_cam_eval_summarys = list(map(lambda x: x["multi_cam_eval_summary"], eval_results))
        single_cam_eval_dataframe = combine_tracking_results(single_cam_eval_summarys)
        multi_cam_eval_dataframe = combine_tracking_results(multi_cam_eval_summarys)

        multi_cam_chunks_evaluation_path = get_multi_cam_chunks_evaluation_path(config_basename)

        multi_cam_eval_dataframe.to_csv(multi_cam_chunks_evaluation_path, index=False)

        multi_cam_mean_std_chunks = calculate_multi_cam_mean_and_std(multi_cam_eval_dataframe)

        multi_cam_mean_std_evaluation_path = get_mean_std_multi_cam_evaluation_path(config_basename)

        multi_cam_mean_std_chunks.to_csv(multi_cam_mean_std_evaluation_path, index=False)

        single_cam_chunks_evaluation_path = get_single_cam_chunks_evaluation_path(config_basename)

        single_cam_mean_std_evaluation_path = get_mean_std_single_cam_evaluation_path(config_basename)

        single_cam_mean_and_std = calculate_single_cam_mean_and_std(single_cam_eval_dataframe)

        single_cam_mean_and_std.to_csv(single_cam_mean_std_evaluation_path, index=False)

        single_cam_eval_dataframe.to_csv(single_cam_chunks_evaluation_path, index=False)



    chunk_id_to_gt_chunks, chunk_id_to_tr_chunks = split_data(dataset_folder=test_dataset_folder
                                                              , track_results_folder=test_track_results_folder
                                                              , cam_ids=list(range(cam_count))
                                                              , working_dir=work_dirs
                                                              , n_split_parts=n_split_parts)

    # As pickle_reid_features will be called in a loop.
    # In this list the initialized extractor will be stored
    feature_extraction = []

    pool = NonDeamonicPool(processes=11)
    eval_results = []
    for chunk_id, test_chunk_gt in chunk_id_to_gt_chunks.items():
        test_chunk_tr = chunk_id_to_tr_chunks[chunk_id]

        cluster_from_weights_task_args = (
            test_chunk_tr,
            train_track_results_folder,
            work_dirs,
            test_chunk_gt,
            train_dataset_folder,
            person_identifier,
            best_weights_path,
            default_weights,
            "test",
            config_basename,
            chunk_id,
            cam_count
        )

        pickle_all_reid_features(work_dirs=work_dirs
                                 , mc_cfg=mc_cfg
                                 , track_results_folder=test_chunk_tr
                                 , dataset_folder=test_dataset_folder
                                 , config_basename=config_basename
                                 , dataset_type="test"
                                 , cam_count=cam_count
                                 , feature_extraction=feature_extraction)


        #evaluation_result = cluster_from_weights_task(*cluster_from_weights_task_args)
        #Will return pool.AsyncResult
        evaluation_result = pool.apply_async(cluster_from_weights_task,cluster_from_weights_task_args)
        eval_results.append(evaluation_result)



    pool.close()
    pool.join()
    eval_results = get_async_tracking_results(eval_results)
    evaluate_after_clustering(eval_results)




def cluster_from_weights_task(
            test_track_results_folder,
            train_track_results_folder,
            work_dirs,
            test_dataset_folder,
            train_dataset_folder,
            person_identifier,
            best_weights_path,
            default_weights,
            dataset_type,
            config_basename,
            chunk_id,
            cam_count):
    tracking_results = None
    try:
        print("Started clustering Task")
        track_clustering = Multi_cam_clustering(
            test_track_results_folder=test_track_results_folder,
            train_track_results_folder=train_track_results_folder,
            work_dirs=work_dirs,
            test_dataset_folder=test_dataset_folder,
            train_dataset_folder=train_dataset_folder,
            person_identifier=person_identifier,
            config_basename=config_basename,
            cam_count=cam_count,
            chunk_id=chunk_id
        )

        tracking_results = track_clustering.cluster_from_weights(best_weights_path=best_weights_path
                                                                 , default_weights=default_weights
                                                                 , dataset_type=dataset_type
                                                                 )

        tracking_results["multi_cam_eval_summary"]["chunk_id"] = chunk_id
        tracking_results["single_cam_eval_summary"]["chunk_id"] = chunk_id

        variable_sizes_string = get_string_sizes_of_all_variables()

        logger = logging.getLogger("clustering_logger")

        logger.info("Sizes of all variables after clustering"
                    " and evaluation of chunk {} : {}".format(chunk_id
                                                                ,variable_sizes_string))


        gc.collect()

    except Exception as e:
        print(e)
        raise

    return tracking_results


if __name__ == "__main__":

    pass
