
import cv2

import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import pickle
import os
import random

class Draw_track_start_end:

    def __init__(self,work_dirs,track_results_path, image_path):
        self.track_results_path = track_results_path

        self.work_dirs = work_dirs
        self.image_path = image_path


    def load_track_results(self):
        self.track_results = pd.read_csv(self.track_results_path)



    def get_bbox_middle_pos(self,bbox):

        xtl, ytl, xbr, ybr = bbox
        x = xtl + ((xbr - xtl) / 2.)
        y = ytl + ((ybr - ytl) / 2.)

        return (x,y)


    def load_track_positions(self):

        track_results_folder, track_results_filename = os.path.split(self.track_results_path)
        track_results_folder = os.path.normpath(track_results_folder)
        config_run_name = track_results_folder.split(os.sep)[0]

        draw_tracks_config_run_folder = os.path.join(self.work_dirs,"tracker","draw_tracks",config_run_name)

        os.makedirs(draw_tracks_config_run_folder,exist_ok=True)

        draw_tracks_pickle_path = os.path.join(draw_tracks_config_run_folder,"draw_tracks.pkl")

        if os.path.exists(draw_tracks_pickle_path):
            with open(draw_tracks_pickle_path, "rb") as pickle_file:

                draw_tracks_dict = pickle.load(pickle_file)
                self.start_positions = draw_tracks_dict["start_positions"]

                self.end_positions = draw_tracks_dict["end_positions"]

                self.person_id_to_tracks = draw_tracks_dict["person_id_to_tracks"]


        else:
            self.calculate_draw_tracks()
            with open(draw_tracks_pickle_path, "wb") as pickle_file:


                draw_tracks_dict = { "start_positions" : self.start_positions
                                    , "end_positions" : self.end_positions
                                    , "person_id_to_tracks" : self.person_id_to_tracks }

                print("Saved draw_tracks_dict")
                pickle.dump(draw_tracks_dict, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


    def get_bbox_from_row(self,row):

        return (row["xtl"],row["ytl"],row["xbr"],row["ybr"])

    def calculate_draw_tracks(self):
        self.load_track_results()

        seen_person_id_to_positions_current_frame = defaultdict(list)
        seen_person_id_to_positions_last_frame = defaultdict(list)
        last_frame_id = -1

        self.start_positions = []
        self.end_positions = []
        self.person_id_to_tracks = defaultdict(list)



        for index, track_result_row in tqdm(self.track_results.iterrows(),total=len(self.track_results)):
            current_frame_id = track_result_row["frame_no_cam"]

            if current_frame_id != last_frame_id:

                last_frame_id = current_frame_id

                vanished_person_ids = set(seen_person_id_to_positions_last_frame.keys()) - set(seen_person_id_to_positions_current_frame.keys())

                for person_id in vanished_person_ids:
                    persons_positions = seen_person_id_to_positions_last_frame[person_id]
                    self.person_id_to_tracks[person_id].append(persons_positions)
                    self.end_positions.append(persons_positions[-1])

                seen_person_id_to_positions_last_frame = seen_person_id_to_positions_current_frame.copy()
                seen_person_id_to_positions_current_frame.clear()


            current_person_id = track_result_row["person_id"]

            current_position = self.get_bbox_middle_pos(self.get_bbox_from_row(track_result_row))


            if current_person_id not in seen_person_id_to_positions_last_frame:

                self.start_positions.append(current_position)

            seen_person_id_to_positions_current_frame[current_person_id].append(current_position)


    def draw_end_positions(self):
        img = cv2.imread(self.image_path)

        for end_position in self.end_positions:

            end_position = tuple(map(int,end_position))

            cv2.circle(img, end_position, radius=1, color=(0, 0, 255), thickness=-1)



        cv2.imshow("show", img)
        cv2.waitKey(0)

    def draw_start_end_distances(self):
        img = cv2.imread(self.image_path)
        overall_person_id_to_tracks = list(self.person_id_to_tracks.items())
        print("overall_person_id_to_tracks len: {}".format(len(overall_person_id_to_tracks)))

        sampled_person_id_to_tracks = random.choices(overall_person_id_to_tracks, k=1000)

        for person_id, tracks in sampled_person_id_to_tracks:

            for track_idx in range(len(tracks)-1):
                track1, track2 = tracks[track_idx],tracks[track_idx+1]

                track1_end, track2_start = track1[-1],track2[0]

                track1_end = tuple(map(int,track1_end))

                track2_start = tuple(map(int, track2_start))

                cv2.circle(img, track1_end, radius=2, color=(255, 0, 0), thickness=-1)
                cv2.circle(img, track2_start, radius=2, color=(0, 255, 0), thickness=-1)

                cv2.line(img, track1_end, track2_start, color=(0, 0, 255), thickness=1)



        cv2.imshow("show", img)
        cv2.waitKey(0)

    def draw_start_positions(self):
        img = cv2.imread(self.image_path)

        for start_position in self.start_positions:

            start_position = tuple(map(int,start_position))

            cv2.circle(img, start_position, radius=1, color=(0, 0, 255), thickness=-1)



        cv2.imshow("show", img)
        cv2.waitKey(0)




if __name__ == "__main__":
    draw_track_start_end = Draw_track_start_end("/home/koehlp/Downloads/work_dirs"
                                                ,"/home/koehlp/Downloads/work_dirs/config_runs/faster_rcnn_r50_fpn_1x_strong_reid_Gta2207_iosb/track_results_cam_2.txt"
                                                ,"/net/merkur/storage/deeplearning/users/koehl/gta/Recording_12.07.2019_17/cam_2/image_0_2.jpg")
    draw_track_start_end.load_track_positions()

    draw_track_start_end.draw_end_positions()