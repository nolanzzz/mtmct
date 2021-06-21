


import pandas as pd
import sys

def count_tracks(track_results_path):

    if isinstance(track_results_path,str):
        track_results = pd.read_csv(track_results_path)


    if len(track_results) > 0:
        track_results = track_results.groupby("person_id",as_index=False).mean()

    number_of_tracks = len(track_results)

    result = "Track results from: {} \n" \
             "Number of tracks: {}".format(track_results_path,number_of_tracks)

    return result


def print_track_lengths(track_results_path):

    if isinstance(track_results_path,str):
        track_results = pd.read_csv(track_results_path)



    track_results = track_results.groupby("person_id",as_index=False).count()

    for index, row in track_results.iterrows():

        print("person_id: {} track_length: {}".format(row[0],row[1]))







if __name__ == "__main__":
    result = print_track_lengths("/home/philipp/work_dirs/clustering/single_camera_refinement/track_results_2.txt")

    print(result)