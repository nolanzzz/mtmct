from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts


def load_tracks_per_cam(cam_id, track_results_folder):
    track_result_path = osp.join(track_results_folder, "cam_{}.txt".format(cam_id))
    track_result_df = pd.read_csv(track_result_path)
    track_result_df = track_result_df.astype({"frame": int,
                                              "id": int,
                                              "x1": int,
                                              "y1": int,
                                              "w": int,
                                              "h": int})
    track_result_df = track_result_df.set_index(['frame'])
    return track_result_df


def load_gt_per_cam(cam_id, track_results_folder):
    track_result_path = osp.join(track_results_folder, "cam_{}".format(cam_id), "gt", "gt_test.csv")
    track_result_df = pd.read_csv(track_result_path)
    track_result_df = track_result_df.astype({"frame": int,
                                              "id": int,
                                              "x1": int,
                                              "y1": int,
                                              "w": int,
                                              "h": int})
    track_result_df = track_result_df.set_index(['frame'])
    return track_result_df


def vis_seq(opt, dataloader, tracks_df, save_dir=None, show_image=True, frame_rate=41):
    if save_dir:
        mkdir_if_missing(save_dir)
    print("save_dir: ", save_dir)
    timer = Timer()
    frame_id = 1
    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        timer.tic()

        online_tlwhs = []
        online_ids = []

        if (tracks_df.index == frame_id).any():
            if isinstance(tracks_df.loc[frame_id], pd.DataFrame):
                for index, t in tracks_df.loc[frame_id].iterrows():
                    tlwh = [t.x1, t.y1, t.w, t.h]
                    tid = t.id

                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            else:
                tlwh = [tracks_df.loc[frame_id].x1,
                        tracks_df.loc[frame_id].y1,
                        tracks_df.loc[frame_id].w,
                        tracks_df.loc[frame_id].h]
                tid = tracks_df.loc[frame_id].id

                online_tlwhs.append(tlwh)
                online_ids.append(tid)
            timer.toc()
            if show_image or save_dir is not None:
                online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                              fps=1. / timer.average_time)
            # if show_image:
            #     cv2.imshow('online_im', online_im)
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    return frame_id, timer.average_time, timer.calls


def main(opt, seqs, data_root, exp_name='demo',
         save_images=True, save_videos=False):
    # logger.setLevel(logging.INFO)
    # result_root = '/u40/zhanr110/mtmct/work_dirs/clustering/config_runs/mta_es_abd_non_clean/multicam_clustering_results/chunk_0/test'
    result_root = '/Users/nolanzhang/Projects/mtmct/work_dirs/tracker/config_runs/' + exp_name + '/fair_results'
    gt_root = '/Users/nolanzhang/Projects/mtmct/trackers/fair/data/MTA_short/mta_data/images/test'
    # output_root = '/u40/zhanr110/mtmct/work_dirs/clustering/config_runs/mta_es_abd_non_clean/'
    output_root = '/Users/nolanzhang/Projects/mtmct/work_dirs/tracker/config_runs/'
    print("\n", result_root, "\n")

    # run visualizing
    n_frame = 0
    timer_avgs, timer_calls = [], []

    for seq in seqs:

        output_dir = os.path.join(output_root, exp_name, 'outputs_fair', seq) if save_images or save_videos else None

        # logger.info('start seq: {}'.format(seq))

        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)

        seq_track_result_df = load_tracks_per_cam(seq[-1], result_root)
        # seq_track_result_df = load_gt_per_cam(seq[-1], gt_root)

        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        # set use_cuda to False if run with cpu
        nf, ta, tc = vis_seq(opt, dataloader, seq_track_result_df, save_dir=output_dir, show_image=True, frame_rate=frame_rate)
        n_frame += nf

        timer_avgs.append(ta)
        timer_calls.append(tc)

        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    print(opt)

    if opt.test_mta:
        seqs_str = '''cam_1'''
        data_root = os.path.join(opt.data_dir, 'mta_data/images/test')
    if opt.train_mta:
        seqs_str = '''cam_2'''
        data_root = os.path.join(opt.data_dir, 'mta_data/images/train')
    if opt.test_mta_partial:
        seqs_str = '''cam_0_t
                      cam_1_t
                      cam_2_t
                      cam_3_t
                      cam_4_t
                      cam_5_t'''
        data_root = os.path.join(opt.data_dir, 'mta_data/images/test')

    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         seqs=seqs,
         data_root=data_root,
         exp_name=opt.exp_id,
         save_images=True,
         save_videos=False)
