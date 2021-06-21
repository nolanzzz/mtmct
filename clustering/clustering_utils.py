import pandas as pd
from collections import defaultdict
import os
from tqdm import tqdm
from clustering import heapq
import sys
import numpy as np
from utilities.helper import constrain_bbox_to_img_dims


def valid_heap_node(heap_node, old_clusters):
    pair_data = heap_node[1]
    for old_cluster in old_clusters:
        if old_cluster in pair_data:
            return False
    return True


def assign_track_nos_to_track_dict(tracks):
    for track_cluster_no, track_cluster in enumerate(tracks):
        for track_dict in track_cluster:
            track_dict["track_no"] = track_cluster_no


def print_sorted_distances(heap):
    print("Printing sorted heap distances: ")
    sorted_heap_distances = sorted([heap_entry[0] for heap_entry in heap])
    sorted_heap_distances = sorted_heap_distances[:10]
    for dist in sorted_heap_distances:
        print(dist)


def map_clusters_track_indices_to_tracks(clusters,all_tracks):
    result = []
    for cluster in clusters:
        result.append(map_idx_to_tracks(cluster,all_tracks))

    return result



def map_idx_to_tracks(track_indices, all_tracks):
    result = []
    for track_idx in track_indices:
        result.append(all_tracks[track_idx])
    return result

def flatten_list(l):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)

    return flat_list

def add_up_lists(l):
    result = []
    for sublist in l:
        combined_lists = []
        for item in sublist:
            combined_lists += item
        result.append(combined_lists)
    return result

def get_distances_and_indices(dataset,calculate_track_distances):
    print("Calculating pairwise distances")
    result = []
    dataset_size = len(dataset)
    for i in tqdm(range(dataset_size - 1)):  # ignore last i
        for j in range(i + 1, dataset_size):  # ignore duplication

            distances = calculate_track_distances([i], [j], dataset)

            # duplicate dist, need to be remove, and there is no difference to use tuple only
            # leave second dist here is to take up a position for tie selection

            result.append([distances, [i, j]])

    return result



def compute_pairwise_distance_normalized(distances_and_indices, dist_name_to_distance_weights):

    def get_dist_name_to_dist_list(distances_and_indices):
        dist_name_to_dist_list = defaultdict(list)
        for distances_pair in distances_and_indices:
            distances = distances_pair[0]
            for dist_name, dist in distances.items():
                dist_name_to_dist_list[dist_name].append(dist)
        return dist_name_to_dist_list


    def get_dist_name_to_median_dist(dist_name_to_dist_list):
        dist_name_to_median_dist = {}
        for dist_name, dist_list in dist_name_to_dist_list.items():
            dist_list_no_inf_zero = [dist for dist in dist_list if dist != np.Inf and dist > 0]

            if len(dist_list_no_inf_zero) == 0:
                dist_name_to_median_dist[dist_name] = 1
            else:
                dist_name_to_median_dist[dist_name] = np.median(dist_list_no_inf_zero)

        return dist_name_to_median_dist



    def get_dist_name_to_max_dist(dist_name_to_dist_list):
        dist_name_to_max_dist = {}
        for dist_name, dist_list in dist_name_to_dist_list.items():
            dist_list_no_inf_zero = [dist for dist in dist_list if dist != np.Inf and dist > 0]

            if len(dist_list_no_inf_zero) == 0:
                dist_name_to_max_dist[dist_name] = 1
            else:
                dist_name_to_max_dist[dist_name] = max(dist_list_no_inf_zero)

        return dist_name_to_max_dist

    def get_dist_name_to_dist_list_div_val(dist_name_to_dist_list, dist_name_to_val):
        dist_name_to_dist_list_div_max = {}
        for dist_name, dist_list in dist_name_to_dist_list.items():
            val = dist_name_to_val[dist_name]
            if val == 0:
                dist_name_to_dist_list_div_max[dist_name] = np.array(dist_list)
            else:
                dist_name_to_dist_list_div_max[dist_name] = np.divide(dist_list, val)

        return dist_name_to_dist_list_div_max

    def get_dist_name_to_weighted_dist_list(dist_name_to_dist_list, dist_name_to_distance_weights):
        dist_name_to_weighted_dist_list = {}
        for dist_name, distance_weight in dist_name_to_distance_weights.items():

            dist_list = dist_name_to_dist_list[dist_name]

            #Workaround for the problem that np.Inf*0 results in nan
            if distance_weight == 0:
                dist_name_to_weighted_dist_list[dist_name] = np.full(len(dist_list),0)
                continue

            weighted_dist_list = np.multiply(dist_list, distance_weight)

            dist_name_to_weighted_dist_list[dist_name] = weighted_dist_list

        return dist_name_to_weighted_dist_list

    def get_sum_of_distances(dist_name_to_dist_list):
        dist_lists = []
        for dist_name, dist_list in dist_name_to_dist_list.items():
            dist_lists.append(dist_list)

        dist_arr = np.array(dist_lists)
        dist_sums = np.sum(dist_arr, axis=0)

        return dist_sums

    def get_distances_with_track_indices(dist_sums, distances_and_indices):
        distances_with_indices = []
        for dist, dist_and_indices in zip(dist_sums, distances_and_indices):
            indices = dist_and_indices[1]
            i = indices[0]
            j = indices[1]
            distances_with_indices.append((dist, [dist, [[i], [j]]]))

        return distances_with_indices

    dist_name_to_dist_list = get_dist_name_to_dist_list(distances_and_indices)

    dist_name_to_weighted_dist_list = get_dist_name_to_weighted_dist_list(dist_name_to_dist_list
                                                                          , dist_name_to_distance_weights)

    dist_sums = get_sum_of_distances(dist_name_to_weighted_dist_list)

    distances_with_track_indices = get_distances_with_track_indices(dist_sums, distances_and_indices)

    return distances_with_track_indices


def get_cluster_tracks_as_list(track_cluster):
    '''
    This will append all track position elements to one list.

    :param track_cluster:
    :return:
    '''

    def add_cam_id_to_track_elements(cam_id, track):
        for track_pos in track:
            track_pos["cam_id"] = cam_id
        return track

    result = []
    for track_dict in track_cluster:
        result += add_cam_id_to_track_elements(track_dict["cam_id"], track_dict["track"])
    return result

def compute_pairwise_distance(dataset,calculate_track_distance):
    print("Calculating pairwise distances")
    result = []
    dataset_size = len(dataset)
    for i in tqdm(range(dataset_size-1)):    # ignore last i
        for j in range(i+1, dataset_size):     # ignore duplication


            dist = calculate_track_distance([i], [j],dataset)

            # duplicate dist, need to be remove, and there is no difference to use tuple only
            # leave second dist here is to take up a position for tie selection

            result.append( (dist, [dist, [[i], [j]]]) )

    return result


def build_priority_queue(distance_list):
    heapq.heapify(distance_list)
    return distance_list

def get_track_pair_to_dist(pairwise_distances):
    pair_to_dist = {}
    for pairwise_distance in pairwise_distances:
        #pairwise_distance = (dist, [dist, [[i], [j]]])
        idx1 = pairwise_distance[1][1][0][0]
        idx2 = pairwise_distance[1][1][1][0]
        track_idx_pair = frozenset([idx1,idx2])
        pair_to_dist[track_idx_pair] = pairwise_distance[0]

    return pair_to_dist

def get_person_id_to_track(track_results):

    if isinstance(track_results,str):
        print("Loading track results.")
        print(track_results)
        track_results = pd.read_csv(track_results)

    person_id_to_track = defaultdict(list)
    frame_numbers = track_results.groupby("frame_no_cam",as_index=False).mean()
    frame_numbers = frame_numbers["frame_no_cam"].tolist()
    frame_numbers = list(map(int,frame_numbers))

    frame_numbers = sorted(frame_numbers)
    for frame_no in frame_numbers:
        one_frame = track_results[track_results["frame_no_cam"] == frame_no]

        for index,track_row in one_frame.iterrows():
            person_id = int(track_row["person_id"])
            bbox = (track_row["xtl"],track_row["ytl"],track_row["xbr"],track_row["ybr"])
            person_id_to_track[person_id].append({"frame_no_cam" : int(track_row["frame_no_cam"])
                                                               ,"bbox" : bbox})



    return person_id_to_track

def get_groundtruth_person_id_to_track(track_results,person_identifier="ped_id"):

    if isinstance(track_results,str):
        track_results = pd.read_csv(track_results)

    person_id_to_track = defaultdict(list)
    frame_numbers = track_results.groupby("frame_no_cam",as_index=False).mean()
    frame_numbers = frame_numbers["frame_no_cam"].tolist()
    frame_numbers = list(map(int,frame_numbers))

    frame_numbers = sorted(frame_numbers)
    for frame_no in frame_numbers:
        one_frame = track_results[track_results["frame_no_cam"] == frame_no]

        for index,track_row in one_frame.iterrows():
            person_id = int(track_row[person_identifier])


            bbox = (track_row["x_top_left_BB"],track_row["y_top_left_BB"],track_row["x_bottom_right_BB"],track_row["y_bottom_right_BB"])

            bbox = constrain_bbox_to_img_dims(bbox)

            person_id_to_track[person_id].append({"frame_no_cam" : int(track_row["frame_no_cam"])
                                                               ,"bbox" : bbox})



    return person_id_to_track


def save_combined_tracks(tracks,cam_id,output_result_path):
    print("Starting save procedure of combined tracks.")
    combined_tracks:pd.DataFrame = pd.DataFrame({ "frame_no_cam" : []
                                       ,"cam_id" : []
                                       ,"person_id" : []
                                       ,"xtl" : []
                                       ,"ytl" : []
                                       ,"xbr" : []
                                       ,"ybr" : [] })
    for track_no, track in enumerate(tracks):
        for track_pos in track:
            combined_tracks = combined_tracks.append({ "frame_no_cam" : track_pos["frame_no_cam"]
                                           ,"cam_id" : cam_id
                                           ,"person_id" : track_no
                                           ,"xtl" : track_pos["bbox"][0]
                                           ,"ytl" : track_pos["bbox"][1]
                                           ,"xbr" : track_pos["bbox"][2]
                                           ,"ybr" : track_pos["bbox"][3] },ignore_index=True)

    combined_tracks = combined_tracks.astype({ "frame_no_cam" : int
                                       ,"cam_id" : int
                                       ,"person_id" : int
                                       ,"xtl" : float
                                       ,"ytl" : float
                                       ,"xbr" : float
                                       ,"ybr" : float })
    combined_tracks = combined_tracks.sort_values(by=["frame_no_cam"]) # removed sorting by frame_no_cam and person_id to increase speed

    combined_tracks.to_csv(output_result_path,index=False)
    print("Saved combined track results to: {}".format(output_result_path))

    return output_result_path