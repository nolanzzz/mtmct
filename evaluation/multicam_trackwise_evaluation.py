import gc
import motmetrics as mm
import numpy as np
from tqdm import tqdm
import pandas as pd
from utilities.helper import adjustCoordsTypes
import multiprocessing as mp
import sys
import time
from collections import defaultdict
import os
from utilities.non_daemonic_pool import NonDeamonicPool, NoDaemonProcess
from utilities.helper import constraint_point_to_img_dims, drop_unnecessary_columns
from utilities.helper import rename_short_motmetrics
import os.path as osp
from utilities.helper import get_all_frame_numbers

try:
    from utilities import pandas_loader
    from utilities import pandas_loader

    from utilities.helper import many_xyxy2xywh
    from utilities.helper import constrain_bbox_to_img_dims


except ImportError as error:

    print(str(error))
    print("Maybe it will be imported later. When sys.path was extended")


class Motmetrics_distance:
    norm2squared_matrix = 1
    iou_matrix = 2


class Multicam_trackwise_evaluation:

    def __init__(self, dataset_folder
                 , track_results_folder
                 , evaluation_results_path
                 , working_dir: str
                 , cam_ids: list
                 , motmetrics_distance=Motmetrics_distance.norm2squared_matrix
                 , img_dims=(1920, 1080)
                 , print_output=sys.stdout):
        '''
        Loading track_results via csv file.
        :param track_results_path:
        :param ground_truth_path:
        '''
        self.motmetrics_distance = motmetrics_distance
        self.print_output = print_output
        self.cam_ids = sorted(cam_ids)
        self.track_results_folder = track_results_folder
        self.dataset_folder = dataset_folder
        self.evaluation_results_path = evaluation_results_path
        self.img_dims = img_dims
        self.working_dir = working_dir

    def load_evaluation_files(self):

        if isinstance(self.track_results_folder, str):
            self.track_result_dataframes = load_track_results(cam_ids=self.cam_ids
                                                              , track_results_folder=self.track_results_folder)
        else:
            self.track_result_dataframes = self.track_results_folder

        if isinstance(self.dataset_folder, str):
            self.ground_truth_dataframes = load_ground_truth_dataframes(cam_ids=self.cam_ids
                                                                        , dataset_folder=self.dataset_folder
                                                                        , working_dir=self.working_dir)
        else:
            self.ground_truth_dataframes = self.dataset_folder

    def initialize_base(self):
        self.load_evaluation_files()
        # Create an accumulator that will be updated during each frame
        self.acc = mm.MOTAccumulator(auto_id=True)
        self.overall_frame_count = 0

        self.gt_cams_df_frame_numbers = []
        for gt_cam_dataframe in self.ground_truth_dataframes:
            gt_cam_df_grouped = gt_cam_dataframe.groupby("frame_no_cam", as_index=False).mean()
            gt_cam_frame_numbers = gt_cam_df_grouped["frame_no_cam"].tolist()
            gt_cam_frame_numbers = list(map(int, gt_cam_frame_numbers))
            self.overall_frame_count += len(gt_cam_frame_numbers)
            self.gt_cams_df_frame_numbers.extend(gt_cam_frame_numbers)

        self.max_all_frame_numbers = max(self.gt_cams_df_frame_numbers)
        self.min_all_frame_numbers = min(self.gt_cams_df_frame_numbers)

    def calculate_distances(self, motmetrics_distance, tr_rows, gt_rows):

        gt_tr_distances = np.array([])
        if motmetrics_distance == Motmetrics_distance.norm2squared_matrix:

            tr_points_this_frame = list(zip(tr_rows["xtl"], tr_rows["ytl"]))
            tr_points_this_frame = list(
                map(list, tr_points_this_frame))  # Because zip delivers tuples but we need lists.

            gt_points_this_frame = list(zip(gt_rows["x_top_left_BB"], gt_rows["y_top_left_BB"]))
            gt_points_this_frame = list(map(list, gt_points_this_frame))
            gt_points_this_frame = list(map(constraint_point_to_img_dims, gt_points_this_frame))

            gt_tr_distances = self.pool.apply_async(mm.distances.norm2squared_matrix
                                                    , args=(gt_points_this_frame, tr_points_this_frame, 200.)
                                                    , callback=lambda *a: self.pbar.update()).get()

        elif motmetrics_distance == Motmetrics_distance.iou_matrix:

            tr_bboxes_this_frame = list(zip(tr_rows["xtl"]
                                            , tr_rows["ytl"]
                                            , tr_rows["xbr"]
                                            , tr_rows["ybr"]))

            tr_bboxes_this_frame = list(map(np.array, tr_bboxes_this_frame))
            tr_bboxes_this_frame = many_xyxy2xywh(tr_bboxes_this_frame)
            gt_bboxes_this_frame = list(zip(gt_rows["x_top_left_BB"]
                                            , gt_rows["y_top_left_BB"]
                                            , gt_rows["x_bottom_right_BB"]
                                            , gt_rows["y_bottom_right_BB"]))

            gt_bboxes_this_frame = list(map(constrain_bbox_to_img_dims, gt_bboxes_this_frame))
            gt_bboxes_this_frame = list(map(np.array, gt_bboxes_this_frame))
            gt_bboxes_this_frame = many_xyxy2xywh(gt_bboxes_this_frame)
            gt_tr_distances = self.pool.apply_async(mm.distances.iou_matrix,
                                                    args=(gt_bboxes_this_frame, tr_bboxes_this_frame, .5),
                                                    callback=lambda *a: self.pbar.update()).get()

        return gt_tr_distances

    def evaluate(self):
        # Call update once for per frame. For now, assume distances between
        # frame objects / hypotheses are given.
        self.initialize_base()
        self.pool = mp.Pool(processes=2)

        self.pbar = tqdm(total=self.overall_frame_count)
        gt_tr_objects_distances = []
        print("Calculating distances of objects")

        for gt_df, tr_df in zip(self.ground_truth_dataframes, self.track_result_dataframes):

            for frame_no_cam in range(self.min_all_frame_numbers, self.max_all_frame_numbers + 1):

                gt_rows_this_frame = gt_df[gt_df["frame_no_cam"] == frame_no_cam]
                tr_rows_this_frame = tr_df[tr_df["frame_no_cam"] == frame_no_cam]

                if (len(gt_rows_this_frame) == 0) and (len(tr_rows_this_frame) == 0):
                    continue

                tr_objects_this_frame = tr_rows_this_frame["person_id"].tolist()

                # In old csv files it was called ped_id than another id was used: person_id
                if "ped_id" in gt_rows_this_frame.columns:
                    gt_objects_this_frame = gt_rows_this_frame["ped_id"].tolist()
                else:
                    gt_objects_this_frame = gt_rows_this_frame["person_id"].tolist()

                gt_tr_distances = self.calculate_distances(self.motmetrics_distance
                                                           , tr_rows_this_frame
                                                           , gt_rows_this_frame)

                gt_tr_objects_distances.append((gt_objects_this_frame, tr_objects_this_frame, gt_tr_distances))

        self.pool.close()
        self.pool.join()

        print("Calling update on metrics accumulator.")
        for gt_objects_this_frame, tr_objects_this_frame, gt_tr_distances in tqdm(gt_tr_objects_distances):
            self.acc.update(
                gt_objects_this_frame,  # Ground truth objects in this frame
                tr_objects_this_frame,  # Detector hypotheses in this frame
                gt_tr_distances
            )

        print("Computing Metrics")


        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=["idfn_track_matching"
            , "idfp_track_matching"
            , "idtp_track_matching"
            , "id_global_assignment"
            , "obj_frequencies"
            , "pred_frequencies"], return_dataframe=False, name='acc')

        track_eval_res_df = self.get_track_eval_res_df(summary)

        track_eval_res_df = track_eval_res_df.astype(int)

        track_eval_res_df.to_csv(self.evaluation_results_path)



    def get_track_eval_res_df(self, summary):

        df = pd.DataFrame({ "rid" : [] , "oid" : [] , "cid" : [] , "hid" : [] , "idfn" : [] , "idfp" : [] , "idtp" : [] , "idfp_idfn_sum" : [] })

        idtp_track_matching = summary["idtp_track_matching"]

        idfp_track_matching = summary["idfp_track_matching"]

        idfn_track_matching = summary["idfn_track_matching"]

        id_global_assignment = summary["id_global_assignment"]

        pred_frequencies = summary["pred_frequencies"]

        obj_frequencies = summary["obj_frequencies"]

        idx_hids = id_global_assignment["idx_hids"]
        idx_oids = id_global_assignment["idx_oids"]

        for rid_cid, idfn in idfn_track_matching.items():

            row_dict = {"rid": rid_cid[0], "oid": -1, "cid": -1, "hid": -1, "idfn": -1, "idfp": -1, "idtp": -1,
                        "idfp_idfn_sum": -1, "obj_frequency": -1, "hid_frequency": -1}

            if rid_cid[0] in idx_oids:
                row_dict["oid"] = idx_oids[rid_cid[0]]

            row_dict["cid"] = rid_cid[1]

            if rid_cid[1] in idx_hids:
                row_dict["hid"] = idx_hids[rid_cid[1]]

            row_dict["idfn"] = idfn

            if rid_cid in idfp_track_matching:
                row_dict["idfp"] = idfp_track_matching[rid_cid]

            row_dict["idfp_idfn_sum"] = row_dict["idfp"] + row_dict["idfn"]


            if rid_cid in idtp_track_matching:
                row_dict["idtp"] = idtp_track_matching[rid_cid]

            if row_dict["oid"] in obj_frequencies:
                row_dict["obj_frequency"] = obj_frequencies[row_dict["oid"]]

            if row_dict["hid"] in pred_frequencies:
                row_dict["hid_frequency"] = pred_frequencies[row_dict["hid"]]


            df = df.append(row_dict,ignore_index=True)

        return df



def load_track_results(track_results_folder, cam_ids):
    track_result_dataframes = []
    # load track results for each cam
    for cam_id in cam_ids:
        track_result_path = osp.join(track_results_folder, "track_results_{}.txt".format(cam_id))

        track_result_dataframe = pd.read_csv(track_result_path)

        track_result_dataframes.append(track_result_dataframe)

    return track_result_dataframes


def load_ground_truth_dataframes(dataset_folder, working_dir, cam_ids):
    cam_ids.sort()
    # load ground truth data for each cam
    ground_truth_dataframes = []
    for cam_id in cam_ids:
        cam_coords_path = osp.join(dataset_folder, "cam_{}/".format(cam_id), "coords_cam_{}.csv".format(cam_id))

        cam_coords = pandas_loader.load_csv(working_dir, cam_coords_path)

        # In old csv files it was called ped_id than another id was used: person_id
        if "ped_id" in cam_coords.columns:
            person_identifier = "ped_id"
        else:
            person_identifier = "person_id"

        cam_coords = cam_coords.groupby(["frame_no_gta", person_identifier], as_index=False).mean()

        cam_coords = adjustCoordsTypes(cam_coords, person_identifier=person_identifier)

        cam_coords = drop_unnecessary_columns(cam_coords)

        ground_truth_dataframes.append(cam_coords)

    return ground_truth_dataframes



if __name__ == "__main__":
    '''
    result = Multicam_evaluation(dataset_folder="/media/philipp/philippkoehl_ssd/GTA_ext_short/test"
                        ,track_results_folder="/media/philipp/philippkoehl_ssd/work_dirs/clustering/multi_camera/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_test"
                         ,cam_ids=[0,1,2,3,4,5]
                        ,working_dir="/media/philipp/philippkoehl_ssd/work_dirs"
                        ,motmetrics_distance=Motmetrics_distance.iou_matrix).evaluate()

    print(result["strsummary"])

    '''


    result = Multicam_trackwise_evaluation(dataset_folder="/media/philipp/philippkoehl_ssd/GTA_ext_short/test"
                                             , track_results_folder="/media/philipp/philippkoehl_ssd/work_dirs/clustering/config_runs/multi_cam_clustering_GTA_ext_short/multicam_clustering_results/chunk_0/test"
                                             , cam_ids=[0,1,2,3,4,5]
                                             , working_dir="/media/philipp/philippkoehl_ssd/work_dirs"
                                             , evaluation_results_path="/media/philipp/philippkoehl_ssd/work_dirs/evaluation/multi_cam_trackwise_evaluation/eval_results.csv"
                                             , motmetrics_distance=Motmetrics_distance.iou_matrix).evaluate()





