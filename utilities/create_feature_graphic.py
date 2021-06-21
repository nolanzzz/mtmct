

from utilities.joint import Joint
from utilities.pose import Pose
import os
import colorsys
from utilities.glasbey import Glasbey
from utilities.helper import *



from utilities.pandas_loader import load_csv

def draw_all_bboxes_and_joints(img,person_dataframe,person_ids,distinct_colors):

    for i, person_id in enumerate(person_ids):
        color = distinct_colors[i]
        one_person_joints = person_dataframe[person_dataframe["person_id"] == person_id]

        if len(one_person_joints) == 0:
            continue

        pose = Pose(one_person_joints)
        pose.drawBoundingBox(image=img,color=color)
        #pose.drawPose(image=img,color=color)






def draw_tracks_visible_in_frame(dataset_path,working_dir,cam_ids,frame_no_to_draw,shape,output_path):

    def get_frame_no_cam(image_name):
        frame_no_cam = int(image_name.split("_")[1])
        return frame_no_cam

    def get_cam_id(image_name):
        cam_id = int(image_name.split("_")[2])
        return cam_id

    def get_visible_person_ids(frame_no_cam, cam_dataframe):

        person_ids = set(cam_dataframe[cam_dataframe["frame_no_cam"] == frame_no_cam]["person_id"])

        person_ids = list(person_ids)
        person_ids = sorted(person_ids)

        return person_ids

    def float_tuple_to_int(tup):
        return tuple(map(lambda x:int(x),tup))

    def draw_one_person_id_track(img,one_person_track_dataframe,color):

        for index, track_entry in one_person_track_dataframe.iterrows():
            bbox = get_bbox_of_row(track_entry)
            bbox = constrain_bbox_to_img_dims(bbox)
            track_point = get_bbox_middle_pos(bbox)
            track_point = float_tuple_to_int(track_point)
            cv2.circle(img=img,center=track_point,radius=2,color=color,thickness=-1)



    def get_last_frame(cam_dataframe,frame_no_cam):

        last_frame = cam_dataframe[cam_dataframe["frame_no_cam"] == frame_no_cam]

        return last_frame

    def get_distinct_colors(N):

        gb = Glasbey(base_palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255)])  # base_palette can also be rbg-list
        palette = gb.generate_palette(size=N)
        palette = gb.convert_palette_to_rgb(palette)

        return palette

    def draw_all_person_tracks(person_ids, cam_dataframe, frame_no_cam, img, distinct_colors):



        for i, person_id in enumerate(person_ids):
            #Draw track of person id until frame_no_cam
            one_person_track_dataframe = cam_dataframe[(cam_dataframe["person_id"] == person_id)
                                       & (cam_dataframe["frame_no_cam"]  <= frame_no_cam)]

            current_color = distinct_colors[i]

            one_person_track_dataframe = one_person_track_dataframe.tail(500)
            draw_one_person_id_track(img=img
                                     ,one_person_track_dataframe=one_person_track_dataframe
                                     ,color=current_color)







    def get_last_frames_and_dfs_all_cams(cam_ids):

        cam_id_to_last_frame = {}
        cam_id_to_cam_dataframe = {}

        for cam_id in cam_ids:
            cam_df_path = os.path.join(dataset_path
                                       ,"cam_{}".format(cam_id)
                                       ,"coords_cam_{}.csv".format(cam_id))

            cam_dataframe = load_csv(working_dir=working_dir, csv_path=cam_df_path)
            last_frame_df = get_last_frame(cam_dataframe=cam_dataframe, frame_no_cam=frame_no_to_draw)
            cam_id_to_last_frame[cam_id] = last_frame_df

            cam_dataframe = group_and_drop_unnecessary(cam_dataframe)
            cam_id_to_cam_dataframe[cam_id] = cam_dataframe

        return cam_id_to_last_frame,cam_id_to_cam_dataframe

    def get_all_cams_visible_person_ids(cam_id_to_cam_dataframe):
        visible_person_ids = []
        for cam_id, cam_dataframe in cam_id_to_cam_dataframe.items():
            cam_visible_person_ids = get_visible_person_ids(frame_no_cam=frame_no_to_draw, cam_dataframe=cam_dataframe)
            visible_person_ids.extend(cam_visible_person_ids)
        visible_person_ids = set(visible_person_ids)
        visible_person_ids = sorted(visible_person_ids)

        return visible_person_ids


    def stitch_images_together(images,shape):
        rows = []
        row = []
        image_no = 0
        for row_no in range(shape[0]):
            for column_no in range(shape[1]):
                image= images[image_no]
                image_no += 1

                row.append(image)

            row_concatenated = np.concatenate(row, axis=1)
            row = []
            rows.append(row_concatenated)

        vis = np.concatenate(rows, axis=0)
        resized_image = cv2.resize(vis, (0, 0), fx=0.7, fy=0.7)

        return resized_image


    def draw_all_cams(cam_id_to_last_frame,cam_id_to_cam_dataframe,visible_person_ids,distinct_colors):
        images = []
        for cam_id in cam_ids:


            image_path = os.path.join(dataset_path
                                      , "cam_{}".format(cam_id)
                                      , "image_{}_{}.jpg".format(frame_no_to_draw, cam_id))

            image_to_draw = cv2.imread(image_path)

            cam_dataframe = cam_id_to_cam_dataframe[cam_id]
            draw_all_person_tracks(person_ids=visible_person_ids
                                   , cam_dataframe=cam_dataframe
                                   , frame_no_cam=frame_no_to_draw
                                   , img=image_to_draw
                                   , distinct_colors=distinct_colors)


            last_frame_df = cam_id_to_last_frame[cam_id]
            draw_all_bboxes_and_joints(img=image_to_draw
                                       , person_dataframe=last_frame_df
                                       , person_ids=visible_person_ids
                                       , distinct_colors=distinct_colors)

            images.append(image_to_draw)

        return images


    def show_images(images):
        if isinstance(images,list):

            for image in images:

                cv2.imshow("image",image)
                cv2.waitKey(0)
        else:
            cv2.imshow("image", images)
            cv2.waitKey(0)

    cam_id_to_last_frame,cam_id_to_cam_dataframe = get_last_frames_and_dfs_all_cams(cam_ids=cam_ids)

    visible_person_ids = get_all_cams_visible_person_ids(cam_id_to_cam_dataframe)



    distinct_colors = get_distinct_colors(N=len(visible_person_ids))

    drawn_cam_images = draw_all_cams(cam_id_to_last_frame=cam_id_to_last_frame
                                      ,cam_id_to_cam_dataframe=cam_id_to_cam_dataframe
                                      ,visible_person_ids=visible_person_ids
                                      ,distinct_colors=distinct_colors)

    image_grid = stitch_images_together(images=drawn_cam_images,shape=shape)

    show_images(image_grid)

    cv2.imwrite(filename=output_path,img=image_grid)

if __name__ == "__main__":
    draw_tracks_visible_in_frame(dataset_path="/media/philipp/philippkoehl_ssd/GTA_ext_short/test/"
                                 ,working_dir="/media/philipp/philippkoehl_ssd/work_dirs"
                                 ,output_path="/media/philipp/philippkoehl_ssd/work_dirs/visualizations/feature_graphics_cam_0_1_no_joints.jpg"
                                 ,cam_ids=[0,1]
                                 ,shape=[1,2]
                                 ,frame_no_to_draw=188600)