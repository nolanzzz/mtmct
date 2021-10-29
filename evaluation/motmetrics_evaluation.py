

import motmetrics as mm
import numpy as np
from tqdm import tqdm
import pandas as pd
import multiprocessing
from multiprocessing.pool import ThreadPool
import sys
import time
import os
from utilities.non_daemonic_pool import NonDeamonicPool
from utilities.helper import rename_short_motmetrics

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

    def __init__(self, ground_truth
                 , track_results
                 , working_dir = None
                 ,img_dims = (1920,1080)
                 ,print_output=sys.stdout):
        '''
        Loading track_results via csv file.
        :param track_results_path:
        :param ground_truth_path:
        '''

        self.print_output = print_output

        self.track_results = pd.DataFrame({ "frame_no_cam" : []
                                              , "cam_id" : []
                                              , "person_id" : []
                                              , "xtl" : []
                                              , "ytl" : []
                                              , "xbr" : []
                                              , "ybr" : [] })


        if isinstance(track_results,str):
            self.track_results = pd.read_csv(track_results)
        else:
            self.track_results = track_results


        self.ground_truth = ground_truth
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

    def initialize_base(self,ground_truth,working_dir_path):
        # Create an accumulator that will be updated during each frame
        self.acc = mm.MOTAccumulator(auto_id=True)
        self.ground_truth = ground_truth

        if isinstance(ground_truth,str):
            self.ground_truth = pandas_loader.load_csv(working_dir_path,ground_truth)


        self.gt_frame_numbers_cam = set(self.ground_truth["frame_no_cam"].tolist())

        self.gt_frame_numbers_cam = list(map(int, self.gt_frame_numbers_cam))




    def evaluate(self,motmetrics_distance=Motmetrics_distance.norm2squared_matrix):
        # Call update once for per frame. For now, assume distances between
        # frame objects / hypotheses are given.
        if len(self.track_results) == 0:
            return "Empty track results file."

        self.initialize_base(self.ground_truth,self.working_dir)

        pool = ThreadPool(processes=10)
        pbar = tqdm(total=len(self.gt_frame_numbers_cam))
        gt_tr_objects_distances = []
        for frame_no_cam in self.gt_frame_numbers_cam: #Will go through all frame_number of one cam

            gt_rows_this_frame = self.ground_truth[self.ground_truth["frame_no_cam"] == frame_no_cam]
            tr_rows_this_frame = self.track_results[self.track_results["frame_no_cam"] == frame_no_cam]

            tr_objects_this_frame = tr_rows_this_frame["person_id"].tolist()

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


        metrics = mm.metrics.motchallenge_metrics

        summary = mh.compute(self.acc, metrics=metrics)


        metric_names = mm.io.motchallenge_metric_names

        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=metric_names
        )


        summary = rename_short_motmetrics(summary)
        return {"strsummary" : strsummary, "summary" : summary }


def evaluate_one_cam_task(ground_truth,track_results,working_dir,cam_id):
    eval_result = None
    try:

        if isinstance(ground_truth, str):
            ground_truth = os.path.join(ground_truth, "cam_{}".format(cam_id), "coords_fib_cam_{}.csv".format(cam_id))

        if isinstance(track_results,str):
            track_results = os.path.join(track_results, "track_results_{}.txt".format(cam_id))


        eval_result = Motmetrics_evaluation(
            ground_truth=ground_truth
            , track_results=track_results
            , working_dir=working_dir).evaluate(motmetrics_distance=Motmetrics_distance.iou_matrix)


        print("results for cam_{}: \n {}".format(cam_id,eval_result["strsummary"]))

        result_summary = eval_result["summary"]

        result_summary["cam_id"] = cam_id



    except Exception as e:
        print(e)
        raise

    return result_summary

def eval_single_cam_multiple_cams(dataset_base, track_results_folder, working_dir, cam_ids, results_csv_ouput_path=None):

    def combine_results_to_dataframe(results):
        results_dataframe = pd.DataFrame()

        for result in results:

            if isinstance(result,multiprocessing.pool.AsyncResult):
                result = result.get()


            results_dataframe = results_dataframe.append(result,ignore_index=True)

        return results_dataframe


    pool = NonDeamonicPool(processes=10)
    eval_results = []
    for cam_id in tqdm(cam_ids):

        if isinstance(track_results_folder,list):
            track_results = track_results_folder[cam_id]
        else:
            track_results = track_results_folder


        if isinstance(dataset_base, list):
            ground_truth = dataset_base[cam_id]
        else:
            ground_truth = dataset_base

        evaluate_one_cam_task_args = (ground_truth,track_results,working_dir,cam_id)
        result = evaluate_one_cam_task(*evaluate_one_cam_task_args)

        # result = pool.apply_async(evaluate_one_cam_task,evaluate_one_cam_task_args)
        eval_results.append(result)


    pool.close()
    pool.join()

    results_dataframe = combine_results_to_dataframe(eval_results)

    if results_csv_ouput_path is not None:
        results_dataframe.to_csv(results_csv_ouput_path,index=False)

    return results_dataframe


def calculate_single_cam_mean_and_std(results:pd.DataFrame):
    results_mean = results.groupby(by="cam_id",as_index=False).mean()
    results_mean["type"] = "mean"
    results_std = results.groupby(by="cam_id",as_index=False).std()
    results_std["type"] = "std"
    eval_results = pd.DataFrame()
    eval_results = eval_results.append(results_mean,ignore_index=True)
    eval_results = eval_results.append(results_std,ignore_index=True)

    return eval_results



def evaluate_chunk(dataset_folder,track_results_folder,cam_ids,working_dir,chunk_id,results_output_folder):
    results_dataframe = None
    try:
        results_dataframe = eval_single_cam_multiple_cams(dataset_base=dataset_folder
                                      ,track_results_folder=track_results_folder
                                      ,cam_ids=cam_ids
                                      ,working_dir=working_dir)

        results_dataframe["chunk_id"] = chunk_id

        results_dataframe.to_csv(os.path.join(results_output_folder, "chunk_{}_result.csv".format(chunk_id))
         , index=False)


    except Exception as e:
        print(e)
        raise

    return results_dataframe


def splitted_single_cam_evaluation(dataset_folder, track_results_folder, working_dir, results_output_folder, cam_ids, n_parts):


    from evaluation.multicam_evaluation import split_data
    chunk_id_to_gt_chunks,chunk_id_to_tr_chunks = split_data(dataset_folder=dataset_folder
                                                           ,track_results_folder=track_results_folder
                                                           ,working_dir=working_dir
                                                           ,cam_ids=cam_ids
                                                           ,n_split_parts=n_parts)
    pool = NonDeamonicPool(processes=13)

    chunk_evaluation_results = pd.DataFrame()
    async_results = []

    os.makedirs(results_output_folder, exist_ok=True)
    for chunk_id, gt_chunks in chunk_id_to_gt_chunks.items():
        tr_chunks = chunk_id_to_tr_chunks[chunk_id]


        async_result = pool.apply_async(func=evaluate_chunk,args=(gt_chunks
                                                                  ,tr_chunks
                                                                  ,cam_ids
                                                                  ,working_dir
                                                                  ,chunk_id
                                                                  ,results_output_folder))


        #async_result = evaluate_chunk(gt_chunks,tr_chunks,cam_ids,working_dir,chunk_id,results_output_folder)



        async_results.append(async_result)

    pool.close()
    pool.join()


    for async_result in async_results:

        if not isinstance(async_result,pd.DataFrame):
            result  = async_result.get()
        else:
            result = async_result
        chunk_evaluation_results = chunk_evaluation_results.append(result,ignore_index=True)




    mean_and_std_result = calculate_single_cam_mean_and_std(chunk_evaluation_results)

    print(mean_and_std_result.to_string())

    mean_and_std_result.to_csv(os.path.join(results_output_folder,"single_cam_evaluation_mean_std.csv")
                               ,index=False)

    chunk_evaluation_results.to_csv(os.path.join(results_output_folder,"single_cam_evaluation_chunks.csv")
                                    ,index=False)


if __name__ == "__main__":


    '''
    evaluate_multiple_files(dataset_base="/home/philipp/Downloads/Recording_12.07.2019"
                            ,track_results_folder="/media/philipp/philippkoehl_ssd/work_dirs/clustering/single_camera_refinement"
                            ,working_dir="/media/philipp/philippkoehl_ssd/work_dirs"
                            ,cam_ids=[5])

    '''


    eval_single_cam_multiple_cams(dataset_base="/Users/nolanzhang/Projects/mtmct/data/MTA_ext_short/test"
                                  ,
                                  track_results_folder="/Users/nolanzhang/Projects/mtmct/work_dirs/tracker/config_runs/20e_full_testing_cam_1/tracker_results"
                                  # track_results_folder="/Users/nolanzhang/Projects/mtmct/work_dirs/clustering/config_runs/mta_es_abd_non_clean_server/multicam_clustering_results/chunk_0/test"
                                  , working_dir="/Users/nolanzhang/Projects/mtmct/work_dirs"
                                  , cam_ids=[1]
                                  , results_csv_ouput_path="/Users/nolanzhang/Projects/mtmct/work_dirs/tracker/config_runs/20e_full_testing_cam_1/single_camera_clustering_results.csv")
                                  # , results_csv_ouput_path="/Users/nolanzhang/Projects/mtmct/work_dirs/tracker/config_runs/frcnn50_new_abd_test/multi_camera_clustering_results.csv")




    '''
    evaluate_multiple_files(dataset_base="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test"
                            , track_results_folder="/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_Gta2207_iosb"
                            , working_dir="/home/koehlp/Downloads/work_dirs"
                            , cam_ids=list(range(6))
                            ,results_csv_ouput_path="/home/koehlp/Downloads/work_dirs/clustering/single_camera_clustering_results/faster_rcnn_r50_gta_trained_strong_reid_Gta2207_iosb/results.csv")
                            
    '''



'''

if __name__ == "__main__":


    sys.path.append("/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc")
    from utilities import pandas_loader
    from utilities import pandas_loader

    from utilities.helper import many_xyxy2xywh
    from utilities.helper import constrain_bbox_to_img_dims

    start_time = time.time()
    result  = Motmetrics_evaluation(ground_truth_path="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test/cam_2/coords_cam_2.csv",
                          track_results_path="/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_fpn_1x_strong_reid_gta2207_iosb/track_results.txt"
                        ,working_dir="/home/koehlp/Downloads/work_dirs/").evaluate(motmetrics_distance=Motmetrics_distance.iou_matrix)
    end_time = time.time()
    with open('/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_fpn_1x_strong_reid_gta2207_iosb/evaluation_result.txt', 'w') as f:
        print(result,file=f)
        print('Evaluation took {:.3f} ms'.format((end_time - start_time) * 1000.0),file=f)



'''



