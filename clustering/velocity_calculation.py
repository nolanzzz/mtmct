import os

from clustering.clustering_utils import *
from utilities.helper import *
import pickle


class Velocity_calculation:


    def __init__(self,track_results_folder,cam_ids,pickle_path):

        self.track_results_folder = track_results_folder
        self.cam_ids = cam_ids
        self.pickle_path = pickle_path


    def get_cam_id_to_tracks(self):

        cam_id_to_tracks = {}
        for cam_id in self.cam_ids:

            if isinstance(self.track_results_folder,str):
                track_results = os.path.join(self.track_results_folder,"track_results_{}.txt".format(cam_id))
            else:
                track_results = self.track_results_folder[cam_id]




            cam_id_to_tracks[cam_id] = get_person_id_to_track(track_results)



        return cam_id_to_tracks



    def print_velocities_multiple_cams(self,cam_id_to_person_id_to_mean_velocity):


        for cam_id, person_id_to_mean_velocity in cam_id_to_person_id_to_mean_velocity.items():
            person_id_to_mean_velocity.keys()



    def get_velocity_stats(self,n_tail=40, step_width=10):


        if os.path.exists(self.pickle_path):
            print("Found velocity stats pickle path.")
            print(self.pickle_path)
            with open(self.pickle_path, "rb") as pickle_file:
                velocity_stats = pickle.load(pickle_file)
                print(velocity_stats)
        else:

            velocity_stats = self.calculate_velocity_stats(n_tail=n_tail,step_width=step_width)
            print(velocity_stats)
            with open(self.pickle_path, "wb") as pickle_file:
                print("Writing velocity stats pickle path.")
                print(self.pickle_path)
                pickle.dump(velocity_stats, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        return velocity_stats


    def calculate_velocity_stats(self,n_tail=40, step_width=10):

        print("Starting calculation of velocity stats")


        cam_id_to_tracks = self.get_cam_id_to_tracks()


        velocities = []

        for cam_id, person_id_to_track in cam_id_to_tracks.items():
            for person_id, track in person_id_to_track.items():
                person_velocities = []
                person_velocity = None
                for track_idx in range(0,len(track)-n_tail,step_width):

                    #if the track is too short, a velocity won't be calculated
                    if len(track) < n_tail:
                        continue


                    start_idx = track_idx
                    end_idx = track_idx+n_tail

                    start_pos = track[start_idx]
                    end_pos = track[end_idx]


                    start_bbox = start_pos["bbox"]
                    start_frame_no = start_pos["frame_no_cam"]
                    start_point = get_bbox_middle_pos(start_bbox)

                    end_bbox = end_pos["bbox"]
                    end_frame_no = end_pos["frame_no_cam"]
                    end_point = get_bbox_middle_pos(end_bbox)

                    start_bbox_height = get_bbox_height(start_bbox)

                    frame_difference = (end_frame_no-start_frame_no)

                    #If the frames are missing the velocity should not be calculated
                    if frame_difference > n_tail:
                        continue

                    velocity = (np.array(end_point) - np.array(start_point)) / frame_difference

                    person_velocities.append(np.linalg.norm(velocity) / start_bbox_height)


                velocities.extend(person_velocities)


        velocity_mean = np.mean(velocities)
        velocity_std = np.std(velocities)

        return { "velocity_mean" : velocity_mean
                , "velocity_std" : velocity_std}


if __name__ == '__main__':
    vc = Velocity_calculation(track_results_folder="/media/philipp/philippkoehl_ssd/work_dirs/config_runs/faster_rcnn_r50_gta_trained_strong_reid_GtaExtShort_test"
                              ,cam_ids=[0,1,2,3,4,5])


    print(vc.calculate_velocity_stats())
