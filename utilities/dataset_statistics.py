

from utilities.pandas_loader import load_csv
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import pickle
from scipy.spatial import ConvexHull
from tqdm import tqdm
from utilities.helper import get_bbox_middle_pos,get_bbox_of_row,constrain_bbox_to_img_dims,put_text_with_background


from shapely.geometry.polygon import Polygon

import cv2

def count_tracks(dataset_path,working_dirs,cam_count,person_identifier="ped_id"):
    print("Dataset path: {}".format(dataset_path) )
    for cam_id in range(cam_count):
        cam_path = os.path.join(dataset_path,"cam_{}/".format(cam_id),"coords_cam_{}.csv".format(cam_id))

        cam_coords = load_csv(working_dirs,cam_path)

        cam_coords = cam_coords.groupby(person_identifier, as_index=False).mean()

        person_id_count = len(cam_coords)

        print("Cam {} number of person ids (tracks): {}".format(cam_id,person_id_count))


def defaultdict_list():
    return defaultdict(list)

def defaultdict_defaultdict_list():
    return defaultdict(defaultdict_list)

def get_cam_id_to_cam_id_to_points(dataset_path,working_dirs,cam_count,person_identifier="ped_id",pickle_path=None):
    '''
    Returns the points of persons that occur at the same time in at least two cams.
    :param dataset_path:
    :param working_dirs:
    :param cam_count:
    :param person_identifier:
    :return:
    '''
    def get_bbox_of_row(row):
        return (row["x_top_left_BB"] ,row["y_top_left_BB"], row["x_bottom_right_BB"], row["y_bottom_right_BB"])

    def add_overlap_points_frame(cam_id_to_cam_id_to_points,same_frame_and_no_person_id):
        if len(same_frame_and_no_person_id) <= 1:
            return

        for row in range(len(same_frame_and_no_person_id)):
            for col in range(row):
                row_outer = same_frame_and_no_person_id.iloc[row,:]
                row_inner = same_frame_and_no_person_id.iloc[col,:]

                cam_id_outer = row_outer["cam_id"]
                cam_id_inner = row_inner["cam_id"]
                bbox_outer = constrain_bbox_to_img_dims(get_bbox_of_row(row_outer))
                bbox_inner = constrain_bbox_to_img_dims(get_bbox_of_row(row_inner))
                cam_id_to_cam_id_to_points[cam_id_inner][cam_id_outer].append(get_bbox_middle_pos(bbox_inner))
                cam_id_to_cam_id_to_points[cam_id_outer][cam_id_inner].append(get_bbox_middle_pos(bbox_outer))


    if pickle_path is not None and os.path.exists(pickle_path):
        with open(pickle_path, "rb") as pickle_file:
            return pickle.load(pickle_file)

    combined_df = get_combined_dataframe(dataset_path=dataset_path
                                         ,working_dirs=working_dirs
                                         ,cam_ids=range(cam_count)
                                         ,frame_count_per_cam=20000
                                         , person_identifier=person_identifier)

    group_counts = combined_df.groupby(["frame_no_cam", person_identifier]).size().to_frame('count').reset_index()
    group_counts = group_counts[group_counts["count"] >= 2]

    # This will map two cam_ids to a dict which maps cam_id to the points in this cam that overlap
    cam_id_to_cam_id_to_points = defaultdict(defaultdict_list)
    for index, coord_row in tqdm(group_counts.iterrows(), total=len(group_counts)):
        same_frame_no_person_id = combined_df[(combined_df["frame_no_cam"] == coord_row["frame_no_cam"])
                                              & (combined_df[person_identifier] == coord_row[person_identifier])]

        add_overlap_points_frame(cam_id_to_cam_id_to_points, same_frame_no_person_id)

    if pickle_path is not None:
        with open(pickle_path,"wb") as pickle_file:
            pickle.dump(cam_id_to_cam_id_to_points,pickle_file,protocol=pickle.HIGHEST_PROTOCOL)

    return cam_id_to_cam_id_to_points


def hull_to_polygon(cam_id_to_cam_id_to_hull):
    # Initializing with a polygon that will return that every point in the image dimensions is not contained
    cam_id_to_cam_id_to_polygon = defaultdict(
        lambda: defaultdict(lambda: Polygon([(-100, -100), (-200, -200), (-300, -300)])))

    for cam_id_outer, cam_id_to_hull in cam_id_to_cam_id_to_hull.items():
        for cam_id_inner, hull in cam_id_to_hull.items():
            cam_id_to_cam_id_to_polygon[cam_id_outer][cam_id_inner] = Polygon(list(map(tuple, hull)))

    return cam_id_to_cam_id_to_polygon


def get_overlapping_area_hulls_path(work_dirs,config_basename,dataset_type):

    overlapping_area_hulls_folder = os.path.join(work_dirs
                                                 , "clustering"
                                                 , "config_runs"
                                                 , config_basename
                                                 , "overlapping_area_hulls"
                                                 , dataset_type)

    os.makedirs(overlapping_area_hulls_folder, exist_ok=True)

    overlapping_area_hulls_path = os.path.join(overlapping_area_hulls_folder, "cams_hulls.pkl")

    return overlapping_area_hulls_path


def get_cam_homographies_path(work_dirs,config_basename, dataset_type):

    overlapping_area_hulls_folder = os.path.join(work_dirs
                                                 , "clustering"
                                                 , "config_runs"
                                                 , config_basename
                                                 , "cam_homographies"
                                                 , dataset_type)

    os.makedirs(overlapping_area_hulls_folder, exist_ok=True)

    cam_homographies_path = os.path.join(overlapping_area_hulls_folder, "cams_homographies.pkl")

    return cam_homographies_path

def get_overlapping_areas(dataset_path,working_dirs,cam_count,person_identifier="ped_id",pickle_path=None):


    def cam_points_to_convex_hull(cam_id_to_cam_id_to_points):

        result_cam_ids_to_cam_id_to_hull = defaultdict(dict)


        for cam_id_outer, cam_id_to_points in cam_id_to_cam_id_to_points.items():
            for cam_id_inner, points in cam_id_to_points.items():

                #At least 3 points are needed to create a convex hull
                if len(points) <= 3:
                    continue
                points = np.array(points).reshape(-1,2)

                hull = ConvexHull(points)

                result_cam_ids_to_cam_id_to_hull[cam_id_outer][cam_id_inner] = points[hull.vertices,:]

        return result_cam_ids_to_cam_id_to_hull

    if os.path.exists(pickle_path):
        print("Found cam overlapping area hulls.")
        print(pickle_path)
        with open(pickle_path, "rb") as pickle_file:
            return pickle.load(pickle_file)
    print("Calculating cam overlapping area hulls.")
    cam_id_to_cam_id_to_points = get_cam_id_to_cam_id_to_points(dataset_path,working_dirs,cam_count,person_identifier)

    cam_ids_to_cam_id_to_hull = cam_points_to_convex_hull(cam_id_to_cam_id_to_points)


    with open(pickle_path,"wb") as pickle_file:
        print("Writing cam_ids_to_cam_id_to_hull")
        print(pickle_path)
        pickle.dump(cam_ids_to_cam_id_to_hull,pickle_file,protocol=pickle.HIGHEST_PROTOCOL)

    return cam_ids_to_cam_id_to_hull



def draw_overlapping_areas(dataset_path,working_dirs,cam_count,person_identifier="ped_id",pickle_path=None,draw_folder=None):

    def draw_poly_line(img,points,color):

        for point_n_0 in range(len(points)-1):
            point_n_1 = point_n_0 + 1


            point_n_0 = tuple(points[point_n_0])
            point_n_1 = tuple(points[point_n_1])

            cv2.line(img=img,pt1=point_n_0,pt2=point_n_1,color=color,thickness=7)

        point_n_0 = tuple(points[0])
        point_n_1 = tuple(points[-1])

        cv2.line(img=img, pt1=point_n_0, pt2=point_n_1, color=color, thickness=7)

    def put_images_into_grid(images,output_path,shape=(2,3)):
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
        resized_image = cv2.resize(vis, (0, 0), fx=0.6, fy=0.6)

        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        cv2.imwrite(filename=output_path
                    , img=resized_image
                    , params=[int(cv2.IMWRITE_JPEG_QUALITY), 90])

        cv2.imshow("imshow window", resized_image)
        cv2.waitKey(0)


    def cam_img_path(dataset_path,cam_id):
        return  os.path.join(dataset_path, "cam_{}".format(cam_id),
                                                          "image_12300_{}.jpg".format(cam_id))

    cam_colors = sorted([(255,0,0),(255,125,0),(0,255,0),(0,0,255),(0,125,255),(155,0,155)])


    os.makedirs(os.path.split(pickle_path)[0],exist_ok=True)

    cam_id_to_cam_id_to_hull = get_overlapping_areas(dataset_path,working_dirs,cam_count,person_identifier,pickle_path)


    convex_hull_images = []
    cam_0_img_path = cam_img_path(dataset_path=dataset_path,cam_id=0)
    cam_from_img = cv2.imread(cam_0_img_path)
    convex_hull_images.append((0,cam_from_img))
    for cam_id_from, cam_id_to_hull in cam_id_to_cam_id_to_hull.items():

        cam_from_img_path = cam_img_path(dataset_path=dataset_path,cam_id=cam_id_from)
        cam_from_img = cv2.imread(cam_from_img_path)


        for cam_id_overlap, hull in cam_id_to_hull.items():

            hull = np.array(hull,dtype=np.int32)

            overlay = cam_from_img.copy()
            cv2.fillConvexPoly(overlay, hull, color=(22,22,22),lineType=1)
            draw_poly_line(img=overlay,points=hull,color=cam_colors[cam_id_overlap])


            alpha = 0.5  # Transparency factor.

            # Following line overlays transparent rectangle over the image
            cam_from_img = cv2.addWeighted(overlay, alpha, cam_from_img, 1 - alpha, 0)

            put_text_with_background(img=cam_from_img
                                     ,text="area cam_id: {}".format(cam_id_overlap)
                                     ,org=(hull[0,0]+5, hull[0,1]+30)
                                     ,font_scale=4)

        put_text_with_background(img=cam_from_img
                                 , text="image cam_id: {}".format(cam_id_from)
                                 , org=(20, 1020)
                                 ,font_scale=4.0)

        convex_hull_images.append((cam_id_from,cam_from_img))
        #cv2.imshow("hull image",cam_from_img)
        cv2.imwrite(os.path.join(draw_folder,"overlaps_cam_{}.jpg".format(cam_id_from)),cam_from_img)
        cv2.waitKey(0)
    convex_hull_images.sort(key=lambda x: x[0])
    convex_hull_images = [ cam_id_and_image[1] for cam_id_and_image in convex_hull_images]
    put_images_into_grid(images=convex_hull_images,output_path=os.path.join(draw_folder,"overlapping_cams_grid.jpg"))




def get_combined_dataframe(dataset_path,working_dirs,cam_ids,frame_count_per_cam=None,person_identifier="ped_id"):
    print("Dataset path: {}".format(dataset_path))
    combined_dataset = pd.DataFrame()

    for cam_id in cam_ids:

        if isinstance(dataset_path,str):

            cam_path = os.path.join(dataset_path, "cam_{}/".format(cam_id), "coords_fib_cam_{}.csv".format(cam_id))

            cam_coords = load_csv(working_dirs, cam_path)




        else:
            cam_coords = dataset_path[cam_id]

        cam_coords = cam_coords[["frame_no_cam"
            , person_identifier
            , "x_top_left_BB"
            , "y_top_left_BB"
            , "x_bottom_right_BB"
            , "y_bottom_right_BB"]]

        cam_coords = cam_coords.astype({"frame_no_cam": int
                                           , person_identifier: int
                                           , "x_top_left_BB": int
                                           , "y_top_left_BB": int
                                           , "x_bottom_right_BB": int
                                           , "y_bottom_right_BB": int})

        cam_coords["cam_id"] = cam_id

        #Will just take a limited number of frames. That means the first frame_count_per_cam frame frames.
        if frame_count_per_cam is not None:
            all_frame_nos = list(cam_coords["frame_no_cam"])

            all_frame_nos = set(all_frame_nos)

            all_frame_nos = list(all_frame_nos)

            all_frame_nos = sorted(all_frame_nos)

            get_frame_nos = all_frame_nos[:frame_count_per_cam]

            get_frame_nos = set(get_frame_nos)

            cam_coords = cam_coords[cam_coords["frame_no_cam"].isin(get_frame_nos)]





        combined_dataset = combined_dataset.append(cam_coords)
    return combined_dataset



def get_person_id_to_track_multicam(dataset_path,working_dirs,cam_count,person_identifier="ped_id"):
    print("Dataset path: {}".format(dataset_path))
    person_id_to_frame_no_cam_to_cam_id = {}

    for cam_id in range(cam_count):
        cam_path = os.path.join(dataset_path, "cam_{}/".format(cam_id), "coords_cam_{}.csv".format(cam_id))

        cam_coords = load_csv(working_dirs, cam_path)

        cam_coords = cam_coords.groupby(["frame_no_gta", person_identifier], as_index=False).mean()

        for index,coord_row in tqdm(cam_coords.iterrows(),total=len(cam_coords)):

            person_id = int(coord_row[person_identifier])
            frame_no_cam = int(coord_row["frame_no_cam"])

            if person_id not in person_id_to_frame_no_cam_to_cam_id:
                person_id_to_frame_no_cam_to_cam_id[person_id] = {}

            if frame_no_cam not in person_id_to_frame_no_cam_to_cam_id[person_id]:
                person_id_to_frame_no_cam_to_cam_id[person_id][frame_no_cam] = set()

            person_id_to_frame_no_cam_to_cam_id[person_id][frame_no_cam].add(cam_id)


    return person_id_to_frame_no_cam_to_cam_id




def get_camera_transition_matrix(dataset_path
                                 ,working_dirs
                                 ,cam_count
                                 ,output_path
                                 ,inside_cam_missing_frames_transition=5*41
                                 ,person_identifier="person_id"
                                 ):


    def add_new_cam_transitions(cam_ids_i,cam_ids_i_1,transition_matrix):
        '''
           Example cam distribution for every frame:
           frame_no_cam: 0 | 1 | 2 | 3 | 4
           cam_id:       2   2   2   2   2
           cam_id:           0   0   0
           cam_id:       1

           Now taking always subsequent cam_id sets and subtracting them: n and n+1 -> for frame 0 and 1 this
           means set([2,0]) - set([2,1]) so 0 is new. That means 2 - 0 and 1 - 0 are new transitions.

           '''

        new_cam_ids = cam_ids_i_1 - cam_ids_i

        for cam_id_i in cam_ids_i:
            for new_cam_id in new_cam_ids:
                if cam_id_i != new_cam_id:
                    transition_matrix[cam_id_i,new_cam_id] += 1

    def add_all_transitions(cam_ids_i,cam_ids_i_1,transition_matrix):

        for cam_id_i in cam_ids_i:
            for cam_id_i_1 in cam_ids_i_1:
                transition_matrix[cam_id_i,cam_id_i_1] += 1



    person_id_to_frame_no_cam_to_cam_id = get_person_id_to_track_multicam(dataset_path
                                                                       ,working_dirs
                                                                       ,cam_count
                                                                       ,person_identifier=person_identifier)


    cam_transition_matrix = np.zeros((cam_count,cam_count))


    print("Calculating camera transition matrix")
    for person_id, frame_no_cam_to_cam_id in person_id_to_frame_no_cam_to_cam_id.items():
        frame_no_cam_to_cam_id = list(frame_no_cam_to_cam_id.items())

        frame_no_cam_to_cam_id = sorted(frame_no_cam_to_cam_id,key=lambda x:x[0])


        for i in range(len(frame_no_cam_to_cam_id)-1):
            cam_ids_i = frame_no_cam_to_cam_id[i][1]
            cam_ids_i_1 = frame_no_cam_to_cam_id[i+1][1]

            frame_no_cam_i = frame_no_cam_to_cam_id[i][0]
            frame_no_cam_i_1 = frame_no_cam_to_cam_id[i+1][0]

            if abs(frame_no_cam_i-frame_no_cam_i_1) > inside_cam_missing_frames_transition:
                #If the frame difference is greater than one also within cam transitions will be added
                add_all_transitions(cam_ids_i,cam_ids_i_1,cam_transition_matrix)
            else:
                add_new_cam_transitions(cam_ids_i,cam_ids_i_1,cam_transition_matrix)

    np.set_printoptions(suppress=True)
    print(cam_transition_matrix)

    os.makedirs(os.path.split(output_path)[0],exist_ok=True)
    print("Saved cam transition matrix to: {}".format(output_path))

    cam_transition_matrix = cam_transition_matrix.astype(int)

    np.savetxt(output_path, cam_transition_matrix, fmt = '%i')

    return cam_transition_matrix


def get_cam_homographies(dataset_path,working_dirs,cam_count,person_identifier="ped_id",pickle_path=None):

    if os.path.exists(pickle_path):
        print("Found cam homographies.")
        print(pickle_path)
        with open(pickle_path, "rb") as pickle_file:
            return pickle.load(pickle_file)
    print("Calculating cam homographies.")
    cam_id_to_cam_id_to_points = get_cam_id_to_cam_id_to_points(dataset_path
                                                                , working_dirs
                                                                , cam_count
                                                                , person_identifier
                                                                )

    cam_id_to_cam_id_homography = defaultdict(dict)

    for outer_cam_id,cam_id_to_points in cam_id_to_cam_id_to_points.items():
        for inner_cam_id,inner_cam_id_points in cam_id_to_points.items():


            row_cam_id_points = np.array(cam_id_to_cam_id_to_points[outer_cam_id][inner_cam_id])
            column_cam_id_points = np.array(cam_id_to_cam_id_to_points[inner_cam_id][outer_cam_id])



            homography_row_cam_to_column_cam, mask = cv2.findHomography(row_cam_id_points
                                                                        ,column_cam_id_points,method=cv2.RANSAC,
                                                                        ransacReprojThreshold=5.0)



            cam_id_to_cam_id_homography[outer_cam_id][inner_cam_id] = homography_row_cam_to_column_cam

    os.makedirs(os.path.dirname(os.path.abspath(pickle_path)),exist_ok=True)
    with open(pickle_path,"wb") as pickle_file:
        print("Writing homographies.")
        print(pickle_path)
        pickle.dump(cam_id_to_cam_id_homography,pickle_file,protocol=pickle.HIGHEST_PROTOCOL)

    return cam_id_to_cam_id_homography


def test_homographies(cam_id_to_cam_id_homography):
    cam_id_to_cam_id_to_points = get_cam_id_to_cam_id_to_points("/home/philipp/Downloads/Recording_12.07.2019/"
                                                                 ,"/home/philipp/work_dirs/"
                                                                     ,6
                                                                     ,person_identifier="ped_id"
                                                                    , pickle_path="/home/philipp/work_dirs/clustering/overlapping_areas/cam_ids_to_points.pkl")

    tranformed_cam_id_to_cam_id_points = defaultdict(lambda: defaultdict(list))

    for outer_cam_id, cam_id_to_points in cam_id_to_cam_id_to_points.items():
        for inner_cam_id, points in cam_id_to_points.items():
            homography = cam_id_to_cam_id_homography[outer_cam_id][inner_cam_id]
            transformed_points = pd.DataFrame()

            for point in points:
                homogen_point = np.append(np.array(point),[1])
                transformed_point = np.matmul(homography,homogen_point)

                # Homogeneous coords back to cartesian
                transformed_point = np.true_divide(transformed_point[:2],transformed_point[-1])
                tranformed_cam_id_to_cam_id_points[outer_cam_id][inner_cam_id].append(tuple(transformed_point))

            print(list(zip(tranformed_cam_id_to_cam_id_points[outer_cam_id][inner_cam_id]
                           ,cam_id_to_cam_id_to_points[inner_cam_id][outer_cam_id])))




if __name__ == "__main__":
    '''
    count_tracks("/net/merkur/storage/deeplearning/users/koehl/gta/Recording_12.07.2019_17/"
                 ,"/home/koehlp/Downloads/work_dirs"
                     ,6
                     ,person_identifier="ped_id")
    '''

    
                 
    

    draw_overlapping_areas("/media/philipp/philippkoehl_ssd/GTA_ext_short/train"
                                 , "/media/philipp/philippkoehl_ssd/work_dirs"
                                 , 6
                                 , person_identifier="person_id"
                                ,pickle_path="/media/philipp/philippkoehl_ssd/work_dirs/clustering/overlapping_areas/cams_hulls.pkl"
                                ,draw_folder="/media/philipp/philippkoehl_ssd/work_dirs/clustering/overlapping_areas/")


    '''
    cam_id_to_cam_id_homography = get_cam_homographies("/home/philipp/Downloads/Recording_12.07.2019/"
                                                         ,"/home/philipp/work_dirs/"
                                                             ,6
                                                             ,person_identifier="ped_id"
                                                            ,pickle_path="/home/philipp/work_dirs/clustering/overlapping_areas/cams_homographies.pkl")

    test_homographies(cam_id_to_cam_id_homography)
    
    /net/merkur/storage/deeplearning/users/koehl/gta/GTA_ext_short/test
    
    '''

    '''
    get_camera_transition_matrix(dataset_path="/media/philipp/philippkoehl_ssd/GTA_ext_short/test"
                                 ,working_dirs="/media/philipp/philippkoehl_ssd/work_dirs"
                                 ,cam_count=6
                                 ,output_path="/media/philipp/philippkoehl_ssd/camera_transition_matrix.csv"
                                 ,inside_cam_missing_frames_transition=5*41)
    '''

    '''
    get_camera_transition_matrix(dataset_path="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/train"
                                 , working_dirs="/home/koehlp/Downloads/work_dirs"
                                 , cam_count=6
                                 , output_path="/home/koehlp/Downloads/work_dirs/statistics/camera_transition_matrix/camera_transition_matrix_gta2207_train.csv"
                                 , inside_cam_missing_frames_transition=5 * 41)
                                 
    '''