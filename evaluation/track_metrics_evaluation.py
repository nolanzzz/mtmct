

import motmetrics as mm
import numpy as np
from tqdm import tqdm
import pandas as pd

import multiprocessing as mp
import sys
import time

from utilities.helper import constraint_point_to_img_dims

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

class Motmetrics_evaluation:

    def __init__(self, ground_truth_path: str
                 , track_results_path:str = None
                 , track_results: pd.DataFrame = None
                 , working_dir = None
                 , evaluation_results_path = None
                 ,img_dims = (1920,1080)
                 ,print_output=sys.stdout):
        '''
        Loading track_results via csv file.
        :param track_results_path:
        :param ground_truth_path:
        '''

        self.evaluation_results_path = evaluation_results_path

        self.print_output = print_output

        self.track_results = pd.DataFrame({ "frame_no_cam" : []
                                              , "cam_id" : []
                                              , "person_id" : []
                                              , "xtl" : []
                                              , "ytl" : []
                                              , "xbr" : []
                                              , "ybr" : [] })

        if track_results is not None:
            self.track_results = track_results

        if track_results_path is not None:
            self.track_results = pd.read_csv(track_results_path)

        self.ground_truth_path = ground_truth_path
        self.img_dims = img_dims
        self.working_dir = working_dir


    def append_to_track_results(self,row):
        '''
        A row should have this shape:
        { "frame_no_cam" : frame_no_cam , "cam_id" : cam_id , "person_id" : person_id , "xtl" : xtl , "ytl" : ytl , "xbr" : xbr , "ybr" : ybr }
        :param row:
        :return:
        '''

        self.track_results = self.track_results.append(row , ignore_index=True)

    def initialize_base(self,ground_truth_path,working_dir_path):
        # Create an accumulator that will be updated during each frame
        self.acc = mm.MOTAccumulator(auto_id=True)

        self.ground_truth = pandas_loader.load_csv(working_dir_path,ground_truth_path)

        # In old csv files it was called ped_id than another id was used: person_id
        if "ped_id" in self.ground_truth.columns:
            self.ground_truth = self.ground_truth.groupby(["frame_no_gta", "ped_id"], as_index=False).mean()
        else:
            self.ground_truth = self.ground_truth.groupby(["frame_no_gta", "person_id"], as_index=False).mean()

        self.gt_frame_numbers_cam = self.ground_truth.groupby("frame_no_cam", as_index=False).mean()
        self.gt_frame_numbers_cam = self.gt_frame_numbers_cam["frame_no_cam"].tolist()

        self.gt_frame_numbers_cam = list(map(int, self.gt_frame_numbers_cam))






    def evaluate(self,motmetrics_distance=Motmetrics_distance.norm2squared_matrix):
        # Call update once for per frame. For now, assume distances between
        # frame objects / hypotheses are given.
        if len(self.track_results) == 0:
            return "Empty track results file."

        self.initialize_base(self.ground_truth_path,self.working_dir)

        pool = mp.Pool(processes=10)
        pbar = tqdm(total=len(self.gt_frame_numbers_cam))
        gt_tr_objects_distances = []
        for frame_no_cam in self.gt_frame_numbers_cam: #Will go through all frame_number of one cam

            gt_rows_this_frame = self.ground_truth[self.ground_truth["frame_no_cam"] == frame_no_cam]
            tr_rows_this_frame = self.track_results[self.track_results["frame_no_cam"] == frame_no_cam]

            tr_objects_this_frame = tr_rows_this_frame["person_id"].tolist()



            #In old csv files it was called ped_id than another id was used: person_id
            if "ped_id" in gt_rows_this_frame.columns:
                gt_objects_this_frame = gt_rows_this_frame["ped_id"].tolist()
            else:
                gt_objects_this_frame = gt_rows_this_frame["person_id"].tolist()



            if motmetrics_distance == Motmetrics_distance.norm2squared_matrix:

                tr_points_this_frame = list(zip(tr_rows_this_frame["xtl"], tr_rows_this_frame["ytl"]))
                tr_points_this_frame = list(map(list, tr_points_this_frame))  # Because zip delivers tuples but we need lists.

                gt_points_this_frame = list(zip(gt_rows_this_frame["x_top_left_BB"], gt_rows_this_frame["y_top_left_BB"]))
                gt_points_this_frame = list(map(list, gt_points_this_frame))
                gt_points_this_frame = list(map(constraint_point_to_img_dims, gt_points_this_frame))

                gt_tr_distances = pool.apply_async(mm.distances.norm2squared_matrix
                                                   , args=(gt_points_this_frame, tr_points_this_frame, 200.)
                                                   , callback=lambda *a: pbar.update()).get()

            elif motmetrics_distance == Motmetrics_distance.iou_matrix:



                tr_bboxes_this_frame = list(zip(tr_rows_this_frame["xtl"]
                                                ,tr_rows_this_frame["ytl"]
                                                ,tr_rows_this_frame["xbr"]
                                                ,tr_rows_this_frame["ybr"]))



                tr_bboxes_this_frame = list(map(np.array, tr_bboxes_this_frame))

                tr_bboxes_this_frame = many_xyxy2xywh(tr_bboxes_this_frame)

                gt_bboxes_this_frame = list(zip(gt_rows_this_frame["x_top_left_BB"]
                                                ,gt_rows_this_frame["y_top_left_BB"]
                                                ,gt_rows_this_frame["x_bottom_right_BB"]
                                                ,gt_rows_this_frame["y_bottom_right_BB"]))




                gt_bboxes_this_frame = list(map(constrain_bbox_to_img_dims, gt_bboxes_this_frame))

                gt_bboxes_this_frame = list(map(np.array, gt_bboxes_this_frame))

                gt_bboxes_this_frame = many_xyxy2xywh(gt_bboxes_this_frame)

                gt_tr_distances = pool.apply_async(mm.distances.iou_matrix,
                                                   args=(gt_bboxes_this_frame, tr_bboxes_this_frame, .5),
                                                   callback=lambda *a: pbar.update()).get()


            gt_tr_objects_distances.append((gt_objects_this_frame,tr_objects_this_frame,gt_tr_distances))

        pool.close()
        pool.join()

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
                                                ,"idfp_track_matching"
                                                ,"idtp_track_matching"
                                                ,"id_global_assignment"
                                                ,"obj_frequencies"
                                                ,"pred_frequencies"],return_dataframe=False, name='acc')

        track_eval_res_df = self.get_track_eval_res_df(summary)

        track_eval_res_df = track_eval_res_df.astype(int)

        track_eval_res_df.to_csv(self.evaluation_results_path)

        return track_eval_res_df

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




if __name__ == "__main__":

    '''
    result = Motmetrics_evaluation(ground_truth_path="/home/philipp/Downloads/Recording_12.07.2019/cam_2/coords_cam_2.csv"
                        ,track_results_path="/home/philipp/work_dirs/clustering/single_camera_refinement/track_results_2.txt"
                        , evaluation_results_path="/home/philipp/work_dirs/clustering/evaluation_per_track_results.csv"
                        ,working_dir="/home/philipp/work_dirs").evaluate(motmetrics_distance=Motmetrics_distance.iou_matrix)

    '''

    result = Motmetrics_evaluation(
        ground_truth_path="/net/merkur/storage/deeplearning/users/koehl/gta/Recording_12.07.2019_17/cam_2/coords_cam_2.csv"
        , track_results_path="/home/koehlp/Downloads/work_dirs/clustering/single_camera_refinement_blade/track_results_2.txt"
        , evaluation_results_path="/home/koehlp/Downloads/work_dirs/clustering/evaluation_per_track_results.csv"
        , working_dir="/home/koehlp/Downloads/work_dirs/").evaluate(motmetrics_distance=Motmetrics_distance.iou_matrix)



    print(result)



