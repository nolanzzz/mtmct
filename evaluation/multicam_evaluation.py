
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
from utilities.non_daemonic_pool import NonDeamonicPool,NoDaemonProcess
from utilities.helper import constraint_point_to_img_dims, drop_unnecessary_columns
from utilities.helper import rename_short_motmetrics
import os.path as osp
from utilities.helper import get_all_frame_numbers_via_videos



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



class Multicam_evaluation:

    def __init__(self, dataset_folder
                 , track_results_folder
                 , working_dir:str
                 , cam_ids:list
                 ,motmetrics_distance=Motmetrics_distance.norm2squared_matrix
                 ,img_dims = (1920,1080)
                 ,print_output=sys.stdout):
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
        self.img_dims = img_dims
        self.working_dir = working_dir




    def load_evaluation_files(self):


        if isinstance(self.track_results_folder,str):
            self.track_result_dataframes = load_track_results(cam_ids=self.cam_ids
                                                          ,track_results_folder=self.track_results_folder)
        else:
            self.track_result_dataframes = self.track_results_folder


        if isinstance(self.dataset_folder, str):
            self.ground_truth_dataframes = load_ground_truth_dataframes(cam_ids=self.cam_ids
                                                                        ,dataset_folder=self.dataset_folder
                                                                        ,working_dir=self.working_dir)
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


    def calculate_distances(self,motmetrics_distance,tr_rows,gt_rows):


        gt_tr_distances = np.array([])
        if motmetrics_distance == Motmetrics_distance.norm2squared_matrix:

            tr_points_this_frame = list(zip(tr_rows["xtl"], tr_rows["ytl"]))
            tr_points_this_frame = list(map(list, tr_points_this_frame))  # Because zip delivers tuples but we need lists.

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


        for gt_df,tr_df in zip(self.ground_truth_dataframes,self.track_result_dataframes):

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
                                                           ,tr_rows_this_frame
                                                           ,gt_rows_this_frame)

                gt_tr_objects_distances.append((gt_objects_this_frame,tr_objects_this_frame,gt_tr_distances))

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
        summary = mh.compute(self.acc, metrics=mm.metrics.motchallenge_metrics)


        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )

        summary = rename_short_motmetrics(summary)

        return {"summary" : summary , "strsummary" : strsummary }



def load_track_results(track_results_folder,cam_ids):
    track_result_dataframes = []
    # load track results for each cam
    for cam_id in cam_ids:
        track_result_path = osp.join(track_results_folder, "track_results_{}.txt".format(cam_id))

        track_result_dataframe = pd.read_csv(track_result_path)

        track_result_dataframes.append(track_result_dataframe)

    return track_result_dataframes

def load_ground_truth_dataframes(dataset_folder,working_dir,cam_ids):
    cam_ids.sort()
    # load ground truth data for each cam
    ground_truth_dataframes = []
    for cam_id in cam_ids:
        cam_coords_path = osp.join(dataset_folder, "cam_{}/".format(cam_id), "coords_fib_cam_{}.csv".format(cam_id))

        cam_coords = pandas_loader.load_csv(working_dir, cam_coords_path)

        cam_coords = adjustCoordsTypes(cam_coords, person_identifier="person_id")

        ground_truth_dataframes.append(cam_coords)

    return ground_truth_dataframes

def split_data(dataset_folder,track_results_folder,working_dir, n_split_parts, cam_ids):

    def get_splitted_dataframes(n_split_parts,cam_dataframes):
        chunk_id_to_chunks_per_cam = defaultdict(list)
        frame_nos_cam = get_all_frame_numbers_via_videos(dataset_folder,cam_ids)

        frame_no_cams_splitted = np.array_split(frame_nos_cam,n_split_parts)

        frame_no_cams_splitted = list(map(list,frame_no_cams_splitted))

        for cam_id, cam_dataframe in enumerate(cam_dataframes):
            for chunk_id,frame_nos_chunk in enumerate(frame_no_cams_splitted):

                dataframe_chunk = cam_dataframe[cam_dataframe["frame_no_cam"].isin(frame_nos_chunk)]
                chunk_id_to_chunks_per_cam[chunk_id].append(dataframe_chunk)

        return chunk_id_to_chunks_per_cam




    ground_truth_dataframes = load_ground_truth_dataframes(dataset_folder=dataset_folder
                                                           ,working_dir=working_dir
                                                           ,cam_ids=cam_ids)

    track_results_dataframes = load_track_results(track_results_folder=track_results_folder
                                                  ,cam_ids=cam_ids)
    chunk_id_to_gt_chunks = get_splitted_dataframes(n_split_parts,ground_truth_dataframes)

    chunk_id_to_tr_chunks = get_splitted_dataframes(n_split_parts,track_results_dataframes)

    return (chunk_id_to_gt_chunks,chunk_id_to_tr_chunks)


def calculate_multi_cam_mean_and_std(results):
    results_mean = results.mean()
    results_mean["type"] = "mean"
    results_var = results.std()
    results_var["type"] = "std"
    eval_results = pd.DataFrame()

    eval_results = eval_results.append(results_mean, ignore_index=True)

    eval_results = eval_results.append(results_var, ignore_index=True)

    return eval_results


def evaluate_chunk(dataset_folder,track_results_folder,cam_ids,working_dir,chunk_id):
    result = None
    try:
        result = Multicam_evaluation(dataset_folder=dataset_folder
                                     ,track_results_folder=track_results_folder
                                     , cam_ids=cam_ids
                                     , working_dir=working_dir
                                     , motmetrics_distance=Motmetrics_distance.iou_matrix).evaluate()
        gc.collect()
        result["summary"]["chunk_id"] = chunk_id
    except Exception as e:
        print(e)
        raise

    return result


def splitted_multi_cam_evaluation(dataset_folder, track_results_folder, working_dir, results_output_folder, cam_ids, n_parts):



    chunk_id_to_gt_chunks,chunk_id_to_tr_chunks = split_data(dataset_folder=dataset_folder
                                                           ,track_results_folder=track_results_folder
                                                           ,working_dir=working_dir
                                                           ,cam_ids=cam_ids
                                                           ,n_split_parts=n_parts)
    pool = NonDeamonicPool(processes=5)

    chunk_evaluation_results = pd.DataFrame()
    async_results = []
    for chunk_id, gt_chunks in chunk_id_to_gt_chunks.items():
        tr_chunks = chunk_id_to_tr_chunks[chunk_id]

        args = (gt_chunks
                , tr_chunks
                , cam_ids
                , working_dir
                , chunk_id)
        async_result = pool.apply_async(func=evaluate_chunk,args=args)

        #async_result = evaluate_chunk(*args)
        async_results.append(async_result)

    pool.close()
    pool.join()

    for async_result in async_results:

        if isinstance(async_result,pd.DataFrame):
            result = async_result
        else:
            result  = async_result.get()
        chunk_evaluation_results = chunk_evaluation_results.append(result["summary"],ignore_index=True)



    mean_and_std_result = calculate_multi_cam_mean_and_std(chunk_evaluation_results)


    os.makedirs(results_output_folder,exist_ok=True)

    print(mean_and_std_result.to_string())

    mean_and_std_result.to_csv(os.path.join(results_output_folder, "multi_cam_eval_mean_std.csv")
                               ,index=False)

    chunk_evaluation_results.to_csv(os.path.join(results_output_folder, "multi_cam_eval_chunks.csv")
                                    ,index=False)


if __name__ == "__main__":

    '''
    result = Multicam_evaluation(dataset_folder="/media/philipp/philippkoehl_ssd/GTA_ext_short/test"
                        ,track_results_folder="/media/philipp/philippkoehl_ssd/work_dirs/clustering/multi_camera/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_test"
                         ,cam_ids=[0,1,2,3,4,5]
                        ,working_dir="/media/philipp/philippkoehl_ssd/work_dirs"
                        ,motmetrics_distance=Motmetrics_distance.iou_matrix).evaluate()

    print(result["strsummary"])
    
    '''

    splitted_multi_cam_evaluation(dataset_folder="/media/philipp/philippkoehl_ssd/GTA_ext_short/test"
                                  , track_results_folder="/media/philipp/philippkoehl_ssd/work_dirs/clustering/config_runs/multi_cam_clustering_GTA_ext_short_non_clean/multicam_clustering_results/chunk_0/test"
                                  , results_output_path="/media/philipp/philippkoehl_ssd/work_dirs/clustering/multi_camera_clustering_results/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_test.csv"
                                  , cam_ids=[0,1,2,3,4,5]
                                  , working_dir="/media/philipp/philippkoehl_ssd/work_dirs"
                                  , n_parts=1
                                  )

    '''
    splitted_multi_cam_evaluation(dataset_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test"
                        ,
                        track_results_folder="/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_Gta2207_iosb"
                        ,
                        results_output_path="/home/koehlp/Downloads/work_dirs/clustering/multi_camera_clustering_results/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_test_iosb.csv"
                        , cam_ids=[0, 1, 2, 3, 4, 5]
                        , working_dir="/home/koehlp/Downloads/work_dirs"
                        , n_parts=10
                        )
    '''




