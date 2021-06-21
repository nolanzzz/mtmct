import matplotlib.pyplot as plt
import scipy.io as sio

import os

class Reid_dataset_statistics:
    def __init__(self,dataset_path):
        self.dataset_path = dataset_path

    def draw_bbox_height_histogram(self):

        test_mat = "0791_c6s3_092792_00_good.mat"

        test_mat_path = os.path.join(self.dataset_path,test_mat)

        print(sio.loadmat(test_mat_path))
        pass


if __name__ == "__main__":

    rds = Reid_dataset_statistics(dataset_path="/net/merkur/storage/deeplearning/datasets/reid/market-1501/gt_query")

    rds.draw_bbox_height_histogram()
    pass