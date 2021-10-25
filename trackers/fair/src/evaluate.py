from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import motmetrics as mm

from tracking_utils.log import logger
from tracking_utils.evaluation import Evaluator

from opts import opts


def main(data_root, seqs):
    logger.setLevel(logging.INFO)
    # result_root = os.path.join(data_root, '..', 'results', exp_name)
    # result_root = "/Users/nolanzhang/Projects/mtmct/trackers/fair/data/MTA_short/mta_data/images/results/dla34_coco_wda_short_test/"
    result_root = "/Users/nolanzhang/Projects/mtmct/work_dirs/tracker/config_runs/new_test_20e_train/tracker_results_wda_to_fair/"
    data_type = 'mot'

    # run tracking
    accs = []
    for seq in seqs:
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))

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
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary.xlsx'))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    print(opt)

    if opt.test_mta:
        seqs_str = '''cam_0
                      cam_1
                      cam_2
                      cam_3
                      cam_4
                      cam_5'''
        # seqs_str = "cam_0"
        # data_root = os.path.join(opt.data_dir, 'mta_data/images/test')
        data_root = "/Users/nolanzhang/Projects/mtmct/trackers/fair/data/MTA_short/mta_data/images/test"
    if opt.train_mta:
        seqs_str = '''cam_0
                      cam_1
                      cam_2
                      cam_3
                      cam_4
                      cam_5'''
        # data_root = os.path.join(opt.data_dir, 'mta_data/images/train')
        data_root = "/Users/nolanzhang/Projects/mtmct/trackers/fair/data/MTA_short/mta_data/images/train"


    seqs = [seq.strip() for seq in seqs_str.split()]

    main(data_root=data_root,
         seqs=seqs)
