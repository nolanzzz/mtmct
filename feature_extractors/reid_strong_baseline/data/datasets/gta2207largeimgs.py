# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import cv2
import numpy as np

import os.path as osp

from .bases import BaseImageDataset


class Gta2207largeimgs(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'gta2207'

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(Gta2207largeimgs, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Gta2207 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _getAllDistractors(self):
        def getDistractors(testOrTrainSuffix):
            distractor_imgs_path = osp.join(self.dataset_dir,osp.join("distractors/",testOrTrainSuffix))
            if not osp.exists(distractor_imgs_path):
                return set()
            img_names = [ osp.basename(img_path) for img_path in glob.glob(osp.join(distractor_imgs_path, '*.png')) ]

            return set(img_names)

        train_img_names = getDistractors("train")
        test_img_names = getDistractors("test")
        query_img_names = getDistractors("query")

        return train_img_names.union(test_img_names,query_img_names)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))

        all_distractors = self._getAllDistractors()
        pid_container = set()
        for img_path in img_paths:

            # Ignore distractors
            if osp.basename(img_path) in all_distractors: continue

            #Image name structure framegta_2862738_camid_2_pid_2733.png
            img_name_no_ext = osp.basename(img_path).replace(".png","")
            pid = int(img_name_no_ext.split("_")[5])

            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:

            if osp.basename(img_path) in all_distractors: continue # junk images are just ignored

            img = cv2.imread(img_path)
            height, width, channels = img.shape

            if height < 50: continue

            # Image name structure framegta_2862738_camid_2_pid_2733.png
            img_name_no_ext = osp.basename(img_path).replace(".png", "")
            pid = int(img_name_no_ext.split("_")[5])
            camid = int(img_name_no_ext.split("_")[3])

            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
