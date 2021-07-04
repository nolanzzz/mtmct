
import os
from tqdm import tqdm
import pandas as pd
from helper import get_bbox_middle_pos,drawBoundingBox
# from clustering.clustering_utils import get_person_id_to_track,get_groundtruth_person_id_to_track
import cv2
from utilities.helper import *
from utilities.pandas_loader import load_csv

from utilities.glasbey import Glasbey


class Draw_multi_camera_tracks:

    def __init__(self, dataset_base_folder, track_results_folder, track_evaluation_results_path, cam_ids, work_dirs, output_folder=None):
        self.work_dirs = work_dirs
        self.dataset_base_folder = dataset_base_folder
        self.track_results_folder = track_results_folder
        self.cam_ids = cam_ids
        self.track_evaluation_results_path = track_evaluation_results_path

        if output_folder is None:
            self.output_folder = os.path.join(self.work_dirs, "visualization", "drawn_multicam_tracks")
        else:
            self.output_folder = output_folder

        os.makedirs(self.output_folder,exist_ok=True)
        self.track_colors = [(0,0,255),(0,255,0)]
        self.track_circle_radi = [2,1]



    def read_track_results(self):

        all_cam_track_results = pd.DataFrame()
        for cam_id in self.cam_ids:


            track_results_path = os.path.join(self.track_results_folder,"track_results_{}.txt".format(cam_id))
            track_results = pd.read_csv(track_results_path)

            track_results["cam_id"] = cam_id

            all_cam_track_results = all_cam_track_results.append(track_results,ignore_index=True)

        return all_cam_track_results


    def read_ground_truth(self,person_identifier="person_id"):

        all_cam_coords = pd.DataFrame()
        for cam_id in self.cam_ids:
            dataset_base_path = os.path.join(self.dataset_base_folder
                                             ,"cam_{}".format(cam_id)
                                             ,"coords_cam_{}.csv".format(cam_id))

            cam_coords = load_csv(self.work_dirs, dataset_base_path)

            cam_coords = cam_coords.groupby(["frame_no_gta", person_identifier], as_index=False).mean()

            cam_coords = adjustCoordsTypes(cam_coords, person_identifier=person_identifier)

            cam_coords = drop_unnecessary_columns(cam_coords)

            cam_coords["cam_id"] = cam_id

            all_cam_coords = all_cam_coords.append(cam_coords,ignore_index=True)


        return all_cam_coords

    def read_track_evaluation_results(self):
        track_evaluation_results = pd.read_csv(self.track_evaluation_results_path)


        return track_evaluation_results


    def get_union_frame_nos(self,track1,track2):


        frame_no_union = set(track1["frame_no_cam"]).union(track2["frame_no_cam"])

        frame_no_union = list(frame_no_union)

        frame_no_union.sort()

        return frame_no_union


    def write_frame_no(self,img,track,color,gt_min_frame_no=None):
        i = 0
        for idx, track_pos in track.iterrows():

            if "x_top_left_BB" in track_pos:
                bbox = get_bbox_of_row(track_pos)
            else:
                bbox = get_bbox_of_row_track_results(track_pos)

            bbox = tuple(map(int, bbox))


            person_pos = get_bbox_middle_pos(bbox)

            person_pos = tuple(map(int, person_pos))

            if i % 500 == 100:
                draw_frame_no = int(track_pos["frame_no_cam"] - gt_min_frame_no)
                cv2.putText(img=img, text=str(draw_frame_no), org=person_pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.0, color=color, thickness=2)

            i += 1

        return img



    def get_distinct_colors(self,N):

        gb = Glasbey(base_palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255)])  # base_palette can also be rbg-list
        palette = gb.generate_palette(size=N)
        palette = gb.convert_palette_to_rgb(palette)

        return palette

    def get_max_single_cam_tracks_per_clustered_track(self,calculated_tracks_df):

        person_ids = set(calculated_tracks_df["person_id"])
        max_single_cam_track_count = -sys.maxsize
        for person_id in person_ids:
            clustered_track = calculated_tracks_df[calculated_tracks_df["person_id"] == person_id]

            single_cam_track_count = len(set(clustered_track["track_unclustered_no"]))

            max_single_cam_track_count = max(max_single_cam_track_count,single_cam_track_count)

        return max_single_cam_track_count

    def create_unclustered_no_to_color_mapping(self,track):




        track_unclustered_nos = set(track["track_unclustered_no"])


        assert len(track_unclustered_nos) <= len(self.distinct_colors)

        unclustered_no_to_color = {}

        for i,track_no in enumerate(track_unclustered_nos):
            unclustered_no_to_color[track_no] = self.distinct_colors[i]

        return unclustered_no_to_color



    def draw_track(self,img,track,color,radius,gt_draw_bbox_frame_no=None,gt_min_frame_no=None,):

        unclustered_no_to_color_mapping = None
        if "track_unclustered_no" in track:
            unclustered_no_to_color_mapping = self.create_unclustered_no_to_color_mapping(track=track)



        for idx,track_pos in track.iterrows():

            if "x_top_left_BB" in track_pos:
                bbox = get_bbox_of_row(track_pos)
            else:
                bbox = get_bbox_of_row_track_results(track_pos)
                track_unclustered_no = track_pos["track_unclustered_no"]
                color  = unclustered_no_to_color_mapping[track_unclustered_no]

            bbox = tuple(map(int, bbox))

            if gt_draw_bbox_frame_no is not None and track_pos["frame_no_cam"] == gt_draw_bbox_frame_no:
                drawBoundingBox(img, bbox, color=color)

            person_pos = get_bbox_middle_pos(bbox)

            person_pos = tuple(map(int, person_pos))

            cv2.circle(img, person_pos, radius=radius, color=color, thickness=-1)


        return img



    def concat_images(self,images,shape=(3,2)):

        if len(images) == 1:
            return images[0]

        if len(images) % 2 != 0:
            image_shape = images[0].shape
            white_image = np.zeros([image_shape[0], image_shape[1], 3], dtype=np.uint8)
            white_image.fill(255)

            images.append(white_image)



        rows = []
        image_index = 0
        for i in range(shape[0]):
            row = []
            for j in range(shape[1]):

                if image_index >= len(images):
                    break

                row.append(images[image_index])
                image_index += 1



            if len(row) > 0: rows.append(np.concatenate(row,axis=1))
        return np.concatenate(rows,axis=0)


    def draw_track_cams(self,gt_track,cl_track,hid,oid):

        if len(gt_track["frame_no_cam"]) == 0:
            return

        gt_min_frame_no = min(gt_track["frame_no_cam"])

        cam_images = []
        for cam_id in self.cam_ids:
            gt_cam_track = gt_track[gt_track["cam_id"] == cam_id]
            cl_cam_track = cl_track[cl_track["cam_id"] == cam_id]

            union_frames = self.get_union_frame_nos(track1=gt_cam_track,track2=cl_cam_track)
            gt_cam_track_frame_nos = sorted(list(gt_cam_track["frame_no_cam"]))

            if len(union_frames) == 0:
                continue

            if len(gt_cam_track_frame_nos) > 0:
                gt_draw_bbox_frame_no = int(max(gt_cam_track_frame_nos[:100]))
            else:
                continue


            img_path = os.path.join(self.dataset_base_folder
                                    , "cam_{}".format(cam_id)
                                    , "image_{}_{}.jpg".format(gt_draw_bbox_frame_no, cam_id))

            img = cv2.imread(img_path)

            #write cam id
            cv2.putText(img=img, text=str("cam_id: {}".format(cam_id)), org=(800,1000), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2, color=(255, 255, 255), thickness=3)

            img = self.draw_track(img=img,track=gt_cam_track,color=(0,255,0),radius=4,gt_draw_bbox_frame_no=gt_draw_bbox_frame_no,gt_min_frame_no=gt_min_frame_no)

            img = self.draw_track(img=img, track=cl_cam_track,color=(0,0,255),radius=2)

            img = self.write_frame_no(img=img,track=gt_cam_track,color=(255,255,255),gt_min_frame_no=gt_min_frame_no)


            #img = self.write_frame_no(img=img,track=cl_cam_track,color=(0,0,255),gt_min_frame_no=gt_min_frame_no)


            cam_images.append(img)




        track_output_image_path = os.path.join(self.output_folder,"img_hid_{}_oid_{}.jpg".format(hid,oid))

        concatenated_cam_images = self.concat_images(images=cam_images)
        cv2.imwrite(track_output_image_path,concatenated_cam_images, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print("Image written to {}".format(track_output_image_path))




    def run_visualization(self):

        track_evaluation_results = self.read_track_evaluation_results()

        ground_truth_df = self.read_ground_truth()

        calculated_tracks_df = self.read_track_results()

        max_single_cam_tracks = self.get_max_single_cam_tracks_per_clustered_track(calculated_tracks_df)

        #self.distinct_colors = self.get_distinct_colors(max_single_cam_tracks+1)

        #self.distinct_colors = [ color for color in self.distinct_colors if color != (0,255,0)]

        #print(self.distinct_colors)

        self.distinct_colors = [(255, 0, 0), (0, 0, 255), (212, 203, 255), (47, 80, 0), (96, 0, 67), (239, 0, 216), (255, 192, 103), (0, 171, 150), (0, 2, 2), (148, 112, 108), (145, 103, 254), (184, 255, 225), (130, 153, 0), (0, 92, 104), (140, 36, 0), (4, 172, 255), (255, 150, 179), (42, 0, 95), (97, 76, 122), (59, 31, 0), (255, 255, 3), (206, 115, 0), (174, 179, 168), (195, 0, 92), (111, 130, 152), (17, 231, 255), (255, 235, 230), (125, 88, 2), (91, 123, 87), (179, 136, 188), (156, 1, 156), (171, 212, 114), (48, 50, 56), (30, 0, 10), (39, 113, 218), (108, 0, 230), (0, 34, 0), (104, 63, 65), (0, 190, 91), (216, 98, 100), (0, 129, 0), (255, 131, 255), (1, 56, 109), (248, 255, 176), (84, 0, 7), (191, 4, 253), (0, 245, 154), (211, 151, 121), (149, 186, 211), (95, 0, 127), (155, 163, 255), (172, 87, 138), (91, 94, 96), (255, 47, 145), (154, 134, 73), (14, 70, 51), (202, 161, 3), (200, 236, 255), (130, 206, 189), (0, 0, 43), (194, 66, 1), (73, 69, 41), (137, 0, 50), (0, 133, 141), (160, 71, 73), (255, 199, 236), (125, 164, 115), (145, 95, 174), (56, 25, 59), (255, 123, 71), (61, 89, 138)]

        #track_evaluation_results = track_evaluation_results[:10]

        for idx, eval_res_row in tqdm(track_evaluation_results.iterrows(),total=len(track_evaluation_results)):



            hid = int(eval_res_row["hid"])

            oid = int(eval_res_row["oid"])

            # img_hid_17_oid_1966.jpg

            if hid != 60 and oid != 2041:
                pass

            cl_track = calculated_tracks_df[calculated_tracks_df["person_id"] == hid]
            gt_track = ground_truth_df[ground_truth_df["person_id"] == oid]




            self.draw_track_cams(cl_track=cl_track,gt_track=gt_track,hid=hid,oid=oid)















if __name__ == "__main__":
    trv = Draw_multi_camera_tracks(dataset_base_folder="/u40/zhanr110/MTA_ext_short/test"
                              , track_results_folder="/u40/zhanr110/wda_tracker/work_dirs/clustering/config_runs/mta_es_abd_non_clean/multicam_clustering_results/chunk_0/test"
                              , track_evaluation_results_path="/u40/zhanr110/wda_tracker/work_dirs/evaluation/multi_cam_trackwise_evaluation/eval_results.csv"
                              , work_dirs="/u40/zhanr110/wda_tracker/work_dirs"
                              , cam_ids=[0,1,2,3,4,5])



    trv.run_visualization()

