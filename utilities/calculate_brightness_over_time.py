
import os
os.environ["OMP_NUM_THREADS"] = "3" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "3" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "3" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "3" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "3" # export NUMEXPR_NUM_THREADS=6


import cv2
import numpy as np
from utilities.pandas_loader import load_csv
import os
from tqdm import tqdm

import pandas as pd


import matplotlib.pyplot as plt

def calculate_image_brightness(image_path):
    img = cv2.imread(image_path)

    img_float = np.float32(img)
    hsv = cv2.cvtColor(img_float, cv2.COLOR_BGR2HSV_FULL)
    brightness_values = hsv[:,:,2]
    brightness_values = np.reshape(brightness_values,-1)
    brightness_mean = np.mean(brightness_values)
    return brightness_mean



def calculate_brightness_images_in_folder(image_folder,take_every_nth=41,output_csv_path=None):


    if output_csv_path is not None and os.path.exists(output_csv_path):
        return pd.read_csv(output_csv_path)



    brightness_dataframe = pd.DataFrame()
    image_names = os.listdir(image_folder)

    image_names = [ image_name for image_name in image_names if image_name.endswith(".jpg") ]

    #sort by frame no cam as image_{frame_no_cam}_{cam_id}
    image_names.sort(key=lambda image_name: int(image_name.split("_")[1]))

    image_names = image_names[0::take_every_nth]

    for image_name in tqdm(image_names):
        frame_no_cam = int(image_name.split("_")[1])

        image_path = os.path.join(image_folder,image_name)

        brightness_mean = calculate_image_brightness(image_path)

        brightness_dataframe = brightness_dataframe.append({ "frame_no_cam" : frame_no_cam, "brightness_mean" : brightness_mean }
                                                           ,ignore_index=True)

        brightness_dataframe = brightness_dataframe.astype({ "frame_no_cam" : int })

        brightness_dataframe = brightness_dataframe[["frame_no_cam","brightness_mean"]]


    if output_csv_path is not None:

        os.makedirs(os.path.split(output_csv_path)[0],exist_ok=True)
        brightness_dataframe.to_csv(path_or_buf=output_csv_path,index=False)


    return brightness_dataframe


def plot_brightness_plot(train_image_folder,test_image_folder,output_graphic_path,take_every_nth=41, output_csv_path=None):

    def get_adjusted_frame_nos(brightness_df):
        frame_nos = list(brightness_df["frame_no_cam"])

        min_frame_no = min(frame_nos)

        print("min frame no: {}".format(min_frame_no))

        # let the frame nos begin by 0
        frame_nos = [frame_no - min_frame_no for frame_no in frame_nos]

        return frame_nos


    def normalize_min_max(values):
        values = np.array(values)
        normalized_values = (values - min(values)) / (
                                max(values) - min(values))

        return list(normalized_values)

    train_brightness_df = calculate_brightness_images_in_folder(image_folder=train_image_folder
                                          ,take_every_nth=take_every_nth
                                          ,output_csv_path=output_csv_path + "train.csv")

    test_brightness_df = calculate_brightness_images_in_folder(image_folder=test_image_folder
                                                          , take_every_nth=take_every_nth
                                                          , output_csv_path=output_csv_path + "test.csv")

    train_frame_nos = get_adjusted_frame_nos(train_brightness_df)

    train_brightness_mean = list(train_brightness_df["brightness_mean"])

    train_brightness_mean = normalize_min_max(train_brightness_mean)

    test_frame_nos = get_adjusted_frame_nos(test_brightness_df)

    test_brightness_mean = list(test_brightness_df["brightness_mean"])

    test_brightness_mean = normalize_min_max(test_brightness_mean)


    os.makedirs(os.path.split(output_graphic_path)[0],exist_ok=True)



    color = 'tab:red'
    plt.xlabel('#frame')
    plt.ylabel('brightness')
    train_set_plot = plt.plot(train_frame_nos, train_brightness_mean,color=color, label='train set')


    color = 'tab:blue'
    test_set_plot = plt.plot(test_frame_nos, test_brightness_mean, color=color, label='test set')

    plt.legend()


    plt.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.draw()
    plt.savefig(output_graphic_path)

    plt.show()




def plot_idf1_and_brightness_together(image_folder
                                      , take_every_nth
                                      , output_csv_path
                                      , tracking_results_chunks_path
                                      , frames_per_chunk=41*60*5):

    chunk_eval_results = pd.read_csv(tracking_results_chunks_path)

    idf1_chunks = list(chunk_eval_results["IDF1"])

    chunk_nos = [ frames_per_chunk / 2 + frames_per_chunk*i for i, score in enumerate(idf1_chunks)]

    brightness_df = calculate_brightness_images_in_folder(image_folder=image_folder
                                                          , take_every_nth=take_every_nth
                                                          , output_csv_path=output_csv_path)

    frame_nos = list(brightness_df["frame_no_cam"])

    min_frame_no = min(frame_nos)

    # let the frame nos begin by 0
    frame_nos = [frame_no - min_frame_no for frame_no in frame_nos]

    brightness_mean = np.array((brightness_df["brightness_mean"]))

    normalized_brightness_mean = (brightness_mean - min(brightness_mean)) / (max(brightness_mean) - min(brightness_mean))


    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('#frame')
    ax1.set_ylabel('idf1', color=color)
    ax1.plot(chunk_nos, idf1_chunks, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('brightness', color=color)  # we already handled the x-label with ax1
    ax2.plot(frame_nos, normalized_brightness_mean, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


if __name__ == "__main__":
    dark_img_brightness = calculate_image_brightness("/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test/cam_1/image_173234_1.jpg")
    bright_img_brightness = calculate_image_brightness("/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test/cam_1/image_301101_1.jpg")
    print("dark image brightness: {}".format(dark_img_brightness))
    print("bright image brightness: {}".format(bright_img_brightness))

    brightness_df = plot_brightness_plot(test_image_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test/cam_2"
                                         ,train_image_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/train/cam_2"
                                        ,output_csv_path="/home/koehlp/Downloads/work_dirs/statistics/cam_frames_brightness/GTA_2207/brightness.csv"
                                         ,output_graphic_path="/home/koehlp/Downloads/work_dirs/visualizations/brightness_plot_gta2207.pdf")

    '''
    plot_idf1_and_brightness_together(image_folder="/net/merkur/storage/deeplearning/users/koehl/gta/GTA_Dataset_22.07.2019/test/cam_2"
                                                          ,output_csv_path="/home/koehlp/Downloads/work_dirs/statistics/cam_frames_brightness/GTA_2207/test/brightness.csv"
                                                        ,tracking_results_chunks_path="/home/koehlp/Downloads/work_dirs/evaluation/config_runs/evaluate_frcnn_gst_reid_Gta2207_test_iosb/multi_cam_evaluation/multi_cam_eval_chunks.csv"
                                                        ,take_every_nth=41)
    print(brightness_df)
    
    '''

