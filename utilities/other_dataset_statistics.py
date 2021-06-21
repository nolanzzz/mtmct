from utilities.helper import *
from utilities.pandas_loader import load_csv
import os
import numpy as np
import matplotlib.pyplot as plt
from utilities.dataset_statistics import get_combined_dataframe
from tqdm import tqdm
import pickle
import time

def return_zero():
    return 0


class Other_dataset_statistics:
    def __init__(self,dataset_test_folder
                 ,dataset_train_folder
                 ,cam_ids
                 ,work_dirs
                 ,person_identifier="person_id"
                 ,load_test = True):
        self.dataset_train_folder = dataset_train_folder
        self.dataset_test_folder = dataset_test_folder
        self.cam_ids = cam_ids
        self.work_dirs = work_dirs
        self.person_identifier = person_identifier
        self.load_test = load_test

    def print_dataset_length(self):

        def get_folder_contained_image_count(dataset_folder):
            filenames_in_folder = os.listdir(os.path.join(dataset_folder,"cam_0"))
            images_in_folder = [ filename for filename in filenames_in_folder if filename.endswith(".jpg") ]
            return len(images_in_folder)

        def print_dataset_folder_stats(dataset_folder,frames_per_second=41):
            number_of_frames = get_folder_contained_image_count(dataset_folder)
            print("dataset folder: {}".format(dataset_folder))
            print("number of frames: {}".format(number_of_frames))
            print("frames per second: {}".format(frames_per_second))


            dataset_length_seconds = number_of_frames / frames_per_second

            dataset_length_hours = time.strftime('%H:%M:%S', time.gmtime(dataset_length_seconds))

            print("dataset length in hours: {}".format(dataset_length_hours))

        print_dataset_folder_stats(self.dataset_train_folder)
        print_dataset_folder_stats(self.dataset_test_folder)


    def get_bbox_distribution_pickle_path(self):

        path = os.path.normpath(self.dataset_train_folder)
        dataset_path_splitted = path.split(os.sep)

        bbox_distribution_folder = os.path.join(self.work_dirs
                                                ,"statistics"
                                                ,"bbox_distribution"
                                                ,dataset_path_splitted[-2]
                                                )

        os.makedirs(bbox_distribution_folder,exist_ok=True)

        pickle_path = os.path.join(bbox_distribution_folder,"bbox_distribution.pkl")

        return pickle_path

    def get_track_lenghts_pickle_path(self):

        path = os.path.normpath(self.dataset_train_folder)
        dataset_path_splitted = path.split(os.sep)

        bbox_distribution_folder = os.path.join(self.work_dirs
                                                ,"statistics"
                                                ,"track_lengths"
                                                ,dataset_path_splitted[-2]
                                                )

        os.makedirs(bbox_distribution_folder,exist_ok=True)

        pickle_path = os.path.join(bbox_distribution_folder,"track_lenghts.pkl")

        return pickle_path


    def get_person_count_over_time_plot_path(self,dataset_folder):

        path = os.path.normpath(dataset_folder)
        dataset_path_splitted = path.split(os.sep)

        bbox_distribution_folder = os.path.join(self.work_dirs
                                                ,"statistics"
                                                ,"person_count_over_time"
                                                ,dataset_path_splitted[-2]
                                                )

        os.makedirs(bbox_distribution_folder,exist_ok=True)

        pickle_path = os.path.join(bbox_distribution_folder,"person_count_over_time_{}.pdf".format(dataset_path_splitted[-1]))

        return pickle_path

    def divide_chunks(self,l, n):

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]


    def get_combined_tracks_all_cams(self):

        train_combined_coords = get_combined_dataframe(dataset_path=self.dataset_train_folder
                                                 , working_dirs=self.work_dirs
                                                 , cam_ids=self.cam_ids
                                                 , person_identifier="person_id")

        test_combined_coords = get_combined_dataframe(dataset_path=self.dataset_test_folder
                                                 , working_dirs=self.work_dirs
                                                 , cam_ids=self.cam_ids
                                                 , person_identifier="person_id")

        combined_train_test = train_combined_coords.append(test_combined_coords)

        return combined_train_test


    def calculate_track_lengths(self):

        def get_person_id_to_frame_nos_cam():

            combined_tracks = self.get_combined_tracks_all_cams()

            person_id_to_frame_nos_cam = defaultdict(list)
            for idx, cam_coord_row in tqdm(combined_tracks.iterrows(),total=len(combined_tracks)):
                person_id = cam_coord_row['person_id']
                frame_no_cam = cam_coord_row['frame_no_cam']

                person_id_to_frame_nos_cam[person_id].append(frame_no_cam)

            return person_id_to_frame_nos_cam


        def get_track_lengths(person_id_to_frame_nos_cam):
            track_lengths = []
            for frame_nos_cam_one_track in person_id_to_frame_nos_cam.values():
                min_frame_no = min(frame_nos_cam_one_track)
                max_frame_no = max(frame_nos_cam_one_track)

                frame_no_span = max_frame_no - min_frame_no

                track_lengths.append(frame_no_span)


            return track_lengths

        pickle_path = self.get_track_lenghts_pickle_path()

        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as pickle_file:
                track_lengths = pickle.load(pickle_file)
                return track_lengths



        person_id_to_frame_nos_cam = get_person_id_to_frame_nos_cam()

        track_lengths = get_track_lengths(person_id_to_frame_nos_cam)


        if pickle_path is not None:
            with open(pickle_path, "wb") as pickle_file:
                pickle.dump(track_lengths, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


        return track_lengths


    def plot_tracks_per_frame_per_cam(self,frames_per_point=41):

        def calculate_chunk_mean(frame_no_chunks,person_id_count_chunks,min_frame_no):
            new_time_points = []
            new_person_id_counts = []
            for frame_no_chunk,person_id_count_chunk in zip(frame_no_chunks,person_id_count_chunks):

                new_time_points.append(max(frame_no_chunk) - min_frame_no)
                new_person_id_counts.append(np.mean(person_id_count_chunk))

            return new_time_points, new_person_id_counts


        def plot_one_cam(cam_id,person_id_per_frame_count,ax,title):
            import matplotlib.pyplot as plt

            # x axis values
            frame_nos = list(person_id_per_frame_count["frame_no_cam"])

            min_frame = min(frame_nos)

            # corresponding y axis values
            person_id_count = list(person_id_per_frame_count["person_id"])

            frame_no_chunks = list(self.divide_chunks(frame_nos, frames_per_point))

            person_id_count_chunks = list(self.divide_chunks(person_id_count, frames_per_point))

            new_time_points, new_person_id_counts = calculate_chunk_mean(frame_no_chunks, person_id_count_chunks,
                                                                         min_frame)

            new_time_points = list(map(int,new_time_points))
            new_person_id_counts = list(map(int, new_person_id_counts))

            # plotting the points
            ax.plot(new_time_points, new_person_id_counts)

            ax.set(ylabel='#persons cam {}'.format(cam_id))


            if cam_id == self.cam_ids[-1]:
                ax.set_xlabel("#frame", rotation=0, size='large')

            if cam_id == self.cam_ids[0]:
                ax.set_title(title)



            # function to show the plot


        def plot_multiple_cams(dataset_folder,train_test_cam_id_to_frame_no_cam_to_track_count):
            fig, axs = plt.subplots(nrows=len(self.cam_ids),ncols=2, sharex='col', sharey='row')
            fig.suptitle('Number of persons for all cameras over time')

            train_cam_id_to_frame_no_cam_to_track_count, test_cam_id_to_frame_no_cam_to_track_count \
                = train_test_cam_id_to_frame_no_cam_to_track_count


            for ax, cam_id in zip(axs[:,0], self.cam_ids):

                frame_no_cam_to_track_count = train_cam_id_to_frame_no_cam_to_track_count[cam_id]
                plot_one_cam(cam_id, frame_no_cam_to_track_count, ax, title="train")


            for ax, cam_id in zip(axs[:,1], self.cam_ids):

                frame_no_cam_to_track_count = test_cam_id_to_frame_no_cam_to_track_count[cam_id]
                plot_one_cam(cam_id, frame_no_cam_to_track_count, ax, title="test")



            figure = plt.gcf()  # get current figure
            figure.set_size_inches(12, 10)


            plt.draw()
            graphics_path = self.get_person_count_over_time_plot_path(dataset_folder)

            plt.savefig(graphics_path)
            plt.show()


        cam_id_to_frame_no_cam_to_track_count_test = self.get_number_of_tracks_per_frame(self.dataset_test_folder)

        cam_id_to_frame_no_cam_to_track_count_train = self.get_number_of_tracks_per_frame(self.dataset_train_folder)




        plot_multiple_cams(self.dataset_train_folder,(cam_id_to_frame_no_cam_to_track_count_train,cam_id_to_frame_no_cam_to_track_count_test))









    def get_number_of_tracks_per_frame(self,dataset_folder):


        cam_id_to_frame_no_cam_to_track_count = {}
        for cam_id in self.cam_ids:

            frame_no_cam_to_track_count = defaultdict(return_zero)
            cam_coords = load_csv(self.work_dirs, os.path.join(dataset_folder
                                                                     , "cam_{}".format(cam_id)
                                                                     , "coords_cam_{}.csv".format(cam_id)))

            cam_coords = self.get_reduced_cam_coords(cam_coords)

            person_id_per_frame_count = cam_coords.groupby(['frame_no_cam'],as_index=False).count()



            cam_id_to_frame_no_cam_to_track_count[cam_id] = person_id_per_frame_count

        return cam_id_to_frame_no_cam_to_track_count





    def get_reduced_cam_coords(self,cam_coords):

        cam_coords = cam_coords.groupby(["frame_no_gta", self.person_identifier], as_index=False).mean()

        cam_coords = cam_coords[["frame_no_cam"
            , self.person_identifier
            , "x_top_left_BB"
            , "y_top_left_BB"
            , "x_bottom_right_BB"
            , "y_bottom_right_BB"]]

        cam_coords = cam_coords.astype({"frame_no_cam": int
                                           , self.person_identifier: int
                                           , "x_top_left_BB": int
                                           , "y_top_left_BB": int
                                           , "x_bottom_right_BB": int
                                           , "y_bottom_right_BB": int})

        return cam_coords


    def get_combined_test_train_cam(self,cam_id):
        #It is possible to combine the train and test set without modification of frame numbers because previously
        # they were separated
        combined = pd.DataFrame()

        if self.load_test:
            cam_coords_test = load_csv(self.work_dirs,  os.path.join(self.dataset_test_folder
                                                                     , "cam_{}".format(cam_id)
                                                                    , "coords_cam_{}.csv".format(cam_id)))

            cam_coords_test = self.get_reduced_cam_coords(cam_coords_test)

            combined = combined.append(cam_coords_test,ignore_index=True)

        cam_coords_train = load_csv(self.work_dirs, os.path.join(self.dataset_train_folder
                                                                 , "cam_{}".format(cam_id)
                                                                , "coords_cam_{}.csv".format(cam_id)))

        cam_coords_train = self.get_reduced_cam_coords(cam_coords_train)
        combined = combined.append(cam_coords_train,ignore_index=True)

        return combined

    def calculate_bbox_distribution(self,img_width_and_height=(1920,1080)):

        pickle_path = self.get_bbox_distribution_pickle_path()

        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as pickle_file:
                bbox_distribution =  pickle.load(pickle_file)
                return bbox_distribution

        cam_id_to_bbox_freqs = {}

        for cam_id in self.cam_ids:

            bbox_freqs = np.zeros((img_width_and_height[0],img_width_and_height[1]), dtype=int)

            cam_coords = self.get_combined_test_train_cam(cam_id)

            for idx, coord_row in cam_coords.iterrows():

                bbox = get_bbox_of_row(coord_row)

                bbox = constrain_bbox_to_img_dims(bbox)

                bbox_height = get_bbox_height(bbox)
                bbox_width = get_bbox_width(bbox)

                bbox_freqs[bbox_height,bbox_width] += 1

            cam_id_to_bbox_freqs[cam_id] = bbox_freqs

        if pickle_path is not None:
            with open(pickle_path, "wb") as pickle_file:
                pickle.dump(cam_id_to_bbox_freqs, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        return cam_id_to_bbox_freqs


    def sum_up_bbox_distribution_all_cams(self):
        bbox_distributions = self.calculate_bbox_distribution()
        bbox_distribution_list = []
        for cam_id, bbox_distribution in bbox_distributions.items():
            bbox_distribution_list.append(bbox_distribution)


        bbox_distribution_sum = np.sum(bbox_distribution_list,axis=0)

        return bbox_distribution_sum



    def plot_track_lengths_histogram(self,x_axis_frames_per_point=41):

        track_lengths = self.calculate_track_lengths()
        track_lengths = list(map(lambda x: x / x_axis_frames_per_point, track_lengths))

        print("Number of tracks: {}".format(len(track_lengths)))
        plt.hist(track_lengths, bins=100)

        plt.xlabel('track duration in seconds', fontsize=12)
        #plt.yscale('log', nonposy='clip')
        plt.ylabel('frequency', fontsize=12)

        plt.tight_layout(pad=1.5)

        distribution_pickle_path = self.get_track_lenghts_pickle_path()
        plt.savefig(distribution_pickle_path + ".pdf")
        plt.show()



    def plot_bbox_height_distribution(self,):

        bbox_distribution = self.calculate_bbox_bbox_height_distribution()

        print("number of bboxes: {}".format(len(bbox_distribution)))

        plt.hist(bbox_distribution, bins=100, range=(0, 360))

        plt.xlabel('bbox height in pixel', fontsize=12)
        plt.yscale('log', nonposy='clip')
        plt.ylabel('bbox frequency', fontsize=12)

        plt.tight_layout(pad=1.5)

        distribution_pickle_path = self.get_bbox_distribution_pickle_path()
        plt.savefig(distribution_pickle_path + ".pdf")
        plt.show()


    def plot_bbox_distribution(self):

        bbox_distribution = self.sum_up_bbox_distribution_all_cams()


        bbox_distribution = bbox_distribution[:150,:150]



        plt.xlabel('bbox width in pixel', fontsize=12)
        plt.ylabel('bbox height in pixel', fontsize=12)
        plt.imshow(bbox_distribution)
        plt.colorbar().set_label('bbox frequency', rotation=270)



        distribution_pickle_path = self.get_bbox_distribution_pickle_path()
        plt.savefig(distribution_pickle_path + ".pdf")
        plt.show()


    def calculate_bbox_bbox_height_distribution(self):
        bbox_distribution = self.sum_up_bbox_distribution_all_cams()

        # area_to_count = defaultdict(lambda:0)
        heights = []
        height, width = bbox_distribution.shape
        for column in range(width):
            for row in range(height):
                heights.extend([column] * bbox_distribution[row, column])

        return heights

    def calculate_bbox_area_distribution(self):
        bbox_distribution = self.sum_up_bbox_distribution_all_cams()

        #area_to_count = defaultdict(lambda:0)
        areas = []
        height,width = bbox_distribution.shape
        for column in range(width):
            for row in range(height):
                area = row * column
                areas.extend([area]*bbox_distribution[row,column])


        return areas


    def plot_area_histogram(self):

        bbox_area_distribution = self.calculate_bbox_area_distribution()

        plt.xlabel('bbox area in pixel^2', fontsize=12)
        plt.ylabel('frequency', fontsize=12)
        #plt.yscale('log', nonposy='clip')
        plt.hist(bbox_area_distribution, bins=100,range=(0,4000))

        distribution_pickle_path = self.get_bbox_distribution_pickle_path()
        plt.tight_layout(pad=1.5)
        plt.savefig(distribution_pickle_path + ".area.pdf")
        plt.show()
        plt.draw()


if __name__ == "__main__":


    '''
    
    
        ods = Other_dataset_statistics(
            dataset_test_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_ext_short/test"
            , dataset_train_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_ext_short/train"
            , cam_ids=list(range(6))
            , work_dirs="/home/koehlp/Downloads/work_dirs/"
        )
    
    
        ods = Other_dataset_statistics(dataset_test_folder="/media/philipp/philippkoehl_ssd/GTA_ext_short/test"
                                       , dataset_train_folder="/media/philipp/philippkoehl_ssd/GTA_ext_short/train"
                                       , cam_ids=list(range(6))
                                       , work_dirs="/media/philipp/philippkoehl_ssd/work_dirs"
                                       )
    
    '''

    ods = Other_dataset_statistics(
        dataset_test_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test"
        , dataset_train_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/train"
        , cam_ids=list(range(6))
        , work_dirs="/home/koehlp/Downloads/work_dirs/"
    )

    ods.plot_bbox_height_distribution()


    ods.plot_track_lengths_histogram()

    #ods.plot_area_histogram()

    #ods.plot_tracks_per_frame_per_cam()