
import os
from tqdm import tqdm
import pandas as pd
from utilities.helper import get_bbox_middle_pos,drawBoundingBox
from clustering.clustering_utils import get_person_id_to_track,get_groundtruth_person_id_to_track
import cv2
from utilities.helper import *
from utilities.pandas_loader import load_csv

class Track_Visualization:

    def __init__(self,dataset_base_folder,track_results_path,track_evaluation_results_path,cam_id, work_dirs,output_folder=None):
        self.work_dirs = work_dirs
        self.dataset_base_folder = dataset_base_folder
        self.track_results_path = track_results_path
        self.cam_id = cam_id
        self.track_evaluation_results_path = track_evaluation_results_path

        if output_folder is None:
            self.output_folder = os.path.join(self.work_dirs, "clustering", "drawn_track_videos")
        else:
            self.output_folder = output_folder


        self.track_colors = [(0,0,255),(0,255,0)]
        self.track_circle_radi = [2,1]



    def read_track_results(self,track_results_path):
        track_results = pd.read_csv(track_results_path)

        person_id_to_tracks = get_person_id_to_track(track_results)

        return person_id_to_tracks


    def read_ground_truth(self,person_identifier="ped_id"):

        dataset_base_path = os.path.join(self.dataset_base_folder
                                         ,"cam_{}".format(self.cam_id)
                                         ,"coords_cam_{}.csv".format(self.cam_id))

        ground_truth = load_csv(self.work_dirs, dataset_base_path)

        ground_truth = ground_truth.groupby(["frame_no_gta", person_identifier], as_index=False).mean()

        ground_truth = adjustCoordsTypes(ground_truth, person_identifier=person_identifier)

        ground_truth = drop_unnecessary_columns(ground_truth)

        person_id_to_track = get_groundtruth_person_id_to_track(ground_truth)

        return person_id_to_track

    def read_track_evaluation_results(self):
        track_evaluation_results = pd.read_csv(self.track_evaluation_results_path)


        return track_evaluation_results


    def get_union_frame_nos(self,track1,track2):

        def track_to_frame_nos(track):
            result = []
            for track_pos in track:
                result.append(track_pos["frame_no_cam"])

            return result

        track1_frame_nos = track_to_frame_nos(track1)
        track2_frame_nos = track_to_frame_nos(track2)

        frame_no_union = set(track1_frame_nos).union(track2_frame_nos)

        frame_no_union = list(frame_no_union)

        frame_no_union.sort()

        return frame_no_union




    def draw_one_frame(self,img,until_frame_no,track,color,radius):

        for track_pos in track:

            bbox = track_pos["bbox"]
            bbox = tuple(map(int, bbox))
            person_pos = get_bbox_middle_pos(bbox)

            person_pos = tuple(map(int, person_pos))
            cv2.circle(img, person_pos, radius=radius, color=color, thickness=-1)

            if until_frame_no == track_pos["frame_no_cam"]:
                drawBoundingBox(img, bbox, color=color)

            if until_frame_no <= track_pos["frame_no_cam"]:
                break




    def draw_all_frames(self,union_frames,tracks,hid,oid):
        current_frame = union_frames[-1]
        for current_frame in union_frames:

            img_path = os.path.join(self.dataset_base_folder
                                    , "cam_{}".format(self.cam_id)
                                    , "image_{}_{}.jpg".format(current_frame, self.cam_id))
            img = cv2.imread(img_path)

            for track_idx,track in enumerate(tracks):
                track_color = self.track_colors[track_idx % len(self.track_colors)]
                circle_radius = self.track_circle_radi[track_idx % len(self.track_circle_radi)]
                self.draw_one_frame(img,current_frame,track,track_color,circle_radius)

            track_output_folder = os.path.join(self.output_folder,"hid_{}_oid_{}".format(hid,oid))
            os.makedirs(track_output_folder,exist_ok=True)

            track_output_image_path = os.path.join(track_output_folder,"image_{}_{}.jpg".format(current_frame, self.cam_id))

            cv2.imwrite(track_output_image_path,img)




    def run_visualization(self):

        track_evaluation_results = self.read_track_evaluation_results()

        gt_person_id_to_track = self.read_ground_truth()

        tr_person_id_to_track = self.read_track_results(self.track_results_path)

        for idx, eval_res_row in tqdm(track_evaluation_results.iterrows(),total=len(track_evaluation_results)):
            hid = eval_res_row["hid"]

            oid = eval_res_row["oid"]


            if oid not in gt_person_id_to_track or hid not in tr_person_id_to_track:
                break

            gt_track = gt_person_id_to_track[oid]

            tr_track = tr_person_id_to_track[hid]

            union_frames = self.get_union_frame_nos(gt_track,tr_track)

            self.draw_all_frames(union_frames,[gt_track,tr_track],hid,oid)













if __name__ == "__main__":
    trv = Track_Visualization(dataset_base_folder="/home/philipp/Downloads/Recording_12.07.2019"
                        ,track_results_path="/home/philipp/work_dirs/clustering/single_camera_refinement/track_results_2.txt"
                        ,track_evaluation_results_path="/home/philipp/work_dirs/clustering/evaluation_per_track_results.csv"
                        ,work_dirs="/home/philipp/work_dirs"
                        ,cam_id=2)

    trv = Track_Visualization(dataset_base_folder="/net/merkur/storage/deeplearning/users/koehl/gta/Recording_12.07.2019_17"
                              ,
                              track_results_path="/home/koehlp/Downloads/work_dirs/clustering/single_camera_refinement/track_results_2.txt"
                              ,
                              track_evaluation_results_path="/home/koehlp/Downloads/work_dirs/clustering/evaluation_per_track_results.csv"
                              , work_dirs="/home/koehlp/Downloads/work_dirs"
                              , cam_id=2
                              ,  output_folder="/net/merkur/storage/deeplearning/users/koehl/gta/drawn_tracks_matched")

    trv.run_visualization()

