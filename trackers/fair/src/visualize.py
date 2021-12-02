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
    track_result_path = osp.join(track_results_folder, "track_results_{}.txt".format(cam_id))
    track_result_df = pd.read_csv(track_result_path)
    # track_result_df = track_result_df.astype({"frame_no_cam": int,
    #                                           "cam_id": int,
    #                                           "person_id": int,
    #                                           "detection_idx": int,
    #                                           "xtl": int,
    #                                           "ytl": int,
    #                                           "xbr": int,
    #                                           "ybr": int})
    track_result_df = track_result_df.set_index(['frame_no_cam'])
    return track_result_df


def load_gt_per_cam(cam_id, track_results_folder):
    track_result_path = osp.join(track_results_folder, "cam_{}".format(cam_id), "coords_fib_cam_{}.csv".format(cam_id))
    track_result_df = pd.read_csv(track_result_path)
    track_result_df = track_result_df.astype({"frame_no_cam": int,
                                              "person_id": int,
                                              "x_top_left_BB": int,
                                              "y_top_left_BB": int,
                                              "x_bottom_right_BB": int,
                                              "y_bottom_right_BB": int})
    track_result_df = track_result_df.set_index(['frame_no_cam'])
    return track_result_df


def vis_seq(opt, dataloader, tracks_df, save_dir=None, show_image=True, frame_rate=41):
    if save_dir:
        mkdir_if_missing(save_dir)
    print("save_dir: ", save_dir)
    timer = Timer()
    frame_id = 1

    all_tlwhs = []
    all_ids = []

    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        #if i % 8 != 0:
            #continue
        # if frame_id % 20 == 0:
        #     logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()

        online_tlwhs = []
        online_ids = []

        # track_tlwhs = []
        # track_ids = []
        # if frame_id > 677:
        if (tracks_df.index == frame_id).any():
            if isinstance(tracks_df.loc[frame_id], pd.DataFrame):
                for index, t in tracks_df.loc[frame_id].iterrows():
                    tlwh = [t.xtl, t.ytl, t.xbr - t.xtl, t.ybr - t.ytl]
                    tid = t.person_id
                    # cam 0
                    # if frame_id <= 367 and (tid == 97 or (frame_id > 112 and tid == 158)):
                    # cam 1
                    # if tid == 97 and 678 <= frame_id <= 784:
                    # cam 2
                    # if 1914 <= frame_id <= 3926 and tid == 151:
                    # if frame_id <= 246 and tid == 11:
                    # cam 3
                    # if 2255 <= frame_id and tid == 151:
                    # if tid == 11:
                    # track_tlwhs.append(tlwh)
                    # track_ids.append(tid)

                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

            else:
                tlwh = [tracks_df.loc[frame_id].xtl,
                        tracks_df.loc[frame_id].ytl,
                        tracks_df.loc[frame_id].xbr - tracks_df.loc[frame_id].xtl,
                        tracks_df.loc[frame_id].ybr - tracks_df.loc[frame_id].ytl]

                tid = tracks_df.loc[frame_id].person_id

                # cam 0
                # if frame_id <= 367 and (tid == 97 or (frame_id > 112 and tid == 158)):
                # cam 1
                # if tid == 97 and 678 <= frame_id <= 784:
                # cam 2
                # if 1914 <= frame_id <= 3926 and tid == 151:
                # if frame_id <= 246 and tid == 11:
                # cam 3
                # if 2255 <= frame_id and tid == 151:
                # if tid == 11:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)

                # track_tlwhs.append(tlwh)
                # track_ids.append(tid)
            timer.toc()

            # all_tlwhs.append(track_tlwhs)
            # all_ids.append(track_ids)

            if show_image or save_dir is not None:
                online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                              fps=1. / timer.average_time)
                # track_im = plot_trajectory(online_im, all_tlwhs, all_ids)
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '{:06d}.png'.format(frame_id)), online_im)
                # cv2.imwrite(os.path.join(save_dir, '{:06d}.png'.format(frame_id)), track_im)
        else:
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '{:06d}.png'.format(frame_id)), img0)

        frame_id += 1
    return frame_id, timer.average_time, timer.calls


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_trajectory(image, tlwhs, track_ids):
    image = image.copy()
    for one_tlwhs, track_id in zip(tlwhs, track_ids):
        for tlwh, id in zip(one_tlwhs, track_id):
            color = get_color(int(id))
            x1, y1, w, h = tuple(map(int, tlwh))
            cv2.circle(image, (int(x1 + 0.5 * w), int(y1 + h)), 2, color, thickness=3)

    return image


def main(opt, seqs, data_root, exp_name='demo',
         save_images=True, save_videos=False):
    # logger.setLevel(logging.INFO)
    result_root = '/Users/nolanzhang/Projects/mtmct/work_dirs/clustering/config_runs/' + exp_name + '/multicam_clustering_results/chunk_0/test'
    # result_root = '/Users/nolanzhang/Projects/mtmct/work_dirs/clustering/config_runs/' + exp_name + '/tracker_results'
    gt_root = '/Users/nolanzhang/Projects/mtmct/data/MTA_ext_short/test'
    output_root = '/Users/nolanzhang/Projects/mtmct/work_dirs/clustering/config_runs'
    # output_root = '/u40/zhanr110/mtmct/work_dirs/clustering/config_runs/fair_high_20e/vis'
    print("\n", result_root, "\n")

    # run visualizing
    n_frame = 0
    timer_avgs, timer_calls = [], []

    for seq in seqs:

        output_dir = os.path.join(output_root, exp_name, 'outputs_wda', seq) if save_images or save_videos else None

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
            cmd_str = 'ffmpeg -f image2 -i {}/%06d.png -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    print(opt)

    if opt.test_mta:
        seqs_str = '''cam_5'''
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
