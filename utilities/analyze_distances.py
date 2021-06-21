
from collections import defaultdict,Counter
import pickle
from tqdm import tqdm


def get_dist_name_to_dist_list(distances_and_indices):
    distance_name_to_distance_list = defaultdict(list)
    for distances_and_ind in tqdm(distances_and_indices):

        distance_name_to_dist = distances_and_ind[0]
        for distance_name, dist in distance_name_to_dist.items():
            distance_name_to_distance_list[distance_name].append(dist)
    return distance_name_to_distance_list

def analyze_distances_pickle(distances_path):
    def create_histogram(occurrence_list,distances_path,save_fig_name,yscale=None):
        import matplotlib.pyplot as plt
        import numpy as np
        if yscale == "log":
            plt.yscale('log')

        from matplotlib import rcParams
        rcParams.update({'figure.autolayout': True})
        plt.tight_layout(pad=2.5)
        plt.hist(occurrence_list, bins=30)
        plt.ylabel('frequency')
        plt.xlabel('distance value')
        import os
        histogram_path = os.path.join(os.path.split(distances_path)[0],"distances_histogram_" + save_fig_name)
        plt.savefig(histogram_path)
        plt.show()

    def print_most_frequent_distance_values(distance_name_to_distance,print_count):
        distance_name_to_counts = {}
        for distance_name, distances in distance_name_to_distance.items():
            distance_counter = Counter(distances)

            distance_name_to_counts[distance_name] = distance_counter
            print(distance_name)

            distance_counter_items_sorted = sorted(list(distance_counter.items()),key=lambda x: x[1],reverse=True)
            for i, key_value in enumerate(distance_counter_items_sorted):
                key, value = key_value

                print(key, value)

                if i == print_count-1:
                    break

    distances_and_indices = None


    with open(distances_path, "rb") as pickle_file:
        distances_and_indices = pickle.load(pickle_file)

    distance_name_to_distance_list = get_dist_name_to_dist_list(distances_and_indices)

    print_most_frequent_distance_values(distance_name_to_distance=distance_name_to_distance_list,print_count=2)

    create_histogram(distance_name_to_distance_list["overlapping_match_score"]
                     ,distances_path
                     ,"overlapping_match_score.pdf"
                     ,yscale="log")

    create_histogram(distance_name_to_distance_list["feature_mean_distance"]
                     ,distances_path
                     ,"feature_mean_distance.pdf"
                     )
    create_histogram(distance_name_to_distance_list["track_pred_pos_start_distance"]
                     ,distances_path
                     ,"track_pred_pos_start_distance.pdf"
                     ,yscale="log")


def analyze_feature_mean(multicam_distances_and_indices_path,person_id_tracks_path):
    '''
    To find out why there are some feature mean distance zero.
    :return: 
    '''

    def get_indices_with_feature_mean_zero(distances_and_indices):
        indices = []
        for distances_and_ind in tqdm(distances_and_indices):
            distance_name_to_dist = distances_and_ind[0]

            if distance_name_to_dist["feature_mean_distance"] == 0:
                indices.append(distances_and_ind[1])

        return indices

    def get_feature_means_of_indices(indices,all_tracks):
        feature_means_list = []
        for index_pair in indices:
            index_0 = index_pair[0]
            index_1 = index_pair[1]

            feature_track_0 = all_tracks[index_0]
            feature_track_1 =  all_tracks[index_1]

            feature_means_list.append((feature_track_0,feature_track_1))
        return feature_means_list


    with open(person_id_tracks_path, "rb") as pickle_file:
        person_id_tracks = pickle.load(pickle_file)

    with open(multicam_distances_and_indices_path, "rb") as pickle_file:
        distances_and_indices = pickle.load(pickle_file)

    indices_with_feature_mean_zero = get_indices_with_feature_mean_zero(distances_and_indices)

    feature_means = get_feature_means_of_indices(indices_with_feature_mean_zero,person_id_tracks)

    print(feature_means)

if __name__ == "__main__":
    analyze_distances_pickle("/media/philipp/philippkoehl_ssd/multi_cam_distances_good_and_bad/multicam_distances_and_indices.pkl")

    #analyze_feature_mean(multicam_distances_and_indices_path="/media/philipp/philippkoehl_ssd/multi_cam_distances_good_and_bad/multicam_distances_and_indices.pkl"
     #                    ,person_id_tracks_path="/media/philipp/philippkoehl_ssd/multi_cam_distances_good_and_bad/person_id_tracks.pkl")


