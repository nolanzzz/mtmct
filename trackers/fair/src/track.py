from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def write_results(filename, results, data_type, cam_id=None):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'wda':
        save_format = '{frame_no_cam},{cam_id},{person_id},{detection_idx},{xtl},{ytl},{xbr},{ybr}\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        if data_type == 'wda':
            f.write("frame_no_cam,cam_id,person_id,detection_idx,xtl,ytl,xbr,ybr\n")
        for frame_id, tlwhs, track_ids, det_idxs in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, det_idx in zip(tlwhs, track_ids, det_idxs):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                if data_type == 'wda':
                    line = save_format.format(frame_no_cam=frame_id, cam_id=cam_id, person_id=track_id,
                                              detection_idx=det_idx, xtl=x1, ytl=y1, xbr=x2, ybr=y2, w=w, h=h)
                else:
                    line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)

    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, seq_id, root_path, exp_name, frame_rate=41, use_cuda=True):

    tracker = JDETracker(opt, frame_rate=frame_rate, use_cuda=use_cuda)
    timer = Timer()
    frame_id = 0
    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        tracker.update_store_det_features_only(blob, img0, seq_id[-1], frame_id, root_path, exp_name)
        timer.toc()
        frame_id += 1
    return frame_id, timer.average_time, timer.calls


def original_eval_seq(opt, dataloader, data_type, result_filename, result_filename_wda, seq_id, save_dir=None,
                      show_image=True, frame_rate=41, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate, use_cuda=use_cuda)
    timer = Timer()
    results = []
    frame_id = 0
    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        #if i % 8 != 0:
            #continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        online_det_idxs = [i for i in range(len(online_targets))]
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids, online_det_idxs))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename_wda, results, 'wda', seq_id[-1])
    write_results(result_filename, results, data_type, seq_id[-1])
    # write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)

    mtmct_root = os.path.abspath(os.path.join(data_root, '../../../../../../..'))

    if opt.use_original:
        result_root = os.path.join(data_root, '..', 'results', exp_name)
        result_root_wda = os.path.join('../../../work_dirs/tracker/config_runs', exp_name, 'fair_tracker_results')
        mkdir_if_missing(result_root)
        mkdir_if_missing(result_root_wda)
    accs = []

    data_type = 'mot'

    # run tracking
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))

        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)

        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        # set use_cuda to False if run with cpu
        if opt.use_original:
            result_filename = os.path.join(result_root, '{}.txt'.format(seq))
            result_filename_wda = os.path.join(result_root_wda, 'track_results_{}.txt'.format(seq[-1]))
            nf, ta, tc = original_eval_seq(opt, dataloader, data_type, result_filename, result_filename_wda, seq,
                                           save_dir=output_dir, show_image=show_image, frame_rate=frame_rate,
                                           use_cuda=True)
            # eval
            logger.info('Evaluate seq: {}'.format(seq))
            evaluator = Evaluator(data_root, seq, data_type)
            accs.append(evaluator.eval_file(result_filename))
            if save_videos:
                output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
                cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
                os.system(cmd_str)
        else:
            nf, ta, tc = eval_seq(opt, dataloader, seq, mtmct_root, exp_name, frame_rate=frame_rate, use_cuda=True)

        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    if opt.use_original:
        # get summary
        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        summary = Evaluator.get_summary(accs, seqs, metrics)
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        print(strsummary)
        Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    opt = opts().init()
    print(opt)

    if opt.val_mot16:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mta:
        seqs_str = '''cam_0
                      cam_1
                      cam_2
                      cam_3
                      cam_4
                      cam_5'''
        data_root = os.path.join(opt.data_dir, 'mta_data/images/test')
    if opt.train_mta:
        seqs_str = '''cam_0
                      cam_1
                      cam_2
                      cam_3
                      cam_4
                      cam_5'''
        data_root = os.path.join(opt.data_dir, 'mta_data/images/train')
    if opt.test_mta_partial:
        seqs_str = '''cam_0_t
                      cam_1_t
                      cam_2_t
                      cam_3_t
                      cam_4_t
                      cam_5_t'''
        data_root = os.path.join(opt.data_dir, 'mta_data/images/test')
    if opt.val_mta_long:
        seqs_str = '''cam_0_t
                      cam_1_t
                      cam_2_t
                      cam_3_t
                      cam_4_t
                      cam_5_t'''
        data_root = os.path.join(opt.data_dir, 'mta_data/images/train')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=opt.exp_id,
         show_image=False,
         save_images=False,
         save_videos=False)
