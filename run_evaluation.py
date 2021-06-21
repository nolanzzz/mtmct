

import sys

def append_to_pythonpath(paths):
    for path in paths:
        sys.path.append(path)


append_to_pythonpath(['/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/clustering',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/detectors/mmdetection',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/trackers/iou_tracker',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/evaluation/py_motmetrics',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/feature_extractors/reid-strong-baseline',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/clustering',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/detectors/mmdetection',
                      '/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/feature_extractors/ABD_Net'])

import argparse
import mmcv
import logging
from utils.logger import setup_logger
import os
import json
from evaluation.multicam_evaluation import splitted_multi_cam_evaluation
from evaluation.motmetrics_evaluation import splitted_single_cam_evaluation

class Run_clustering:
    def __init__(self,args):
        self.cfg = mmcv.Config.fromfile(args.config).root

        self.cfg.config_basename = os.path.basename(args.config).replace(".py", "")

        config_run_path = os.path.join(self.cfg.config_run_path, self.cfg.config_basename)
        setattr(self.cfg, "config_run_path", config_run_path)

        os.makedirs(config_run_path,exist_ok=True)

        self.logger = setup_logger("clustering_logger", self.cfg.config_run_path, 0)

        self.logger.info(json.dumps(self.cfg, sort_keys=True, indent=4))






    def run(self):

        if self.cfg.evaluate_multi_cam:

            multi_cam_eval_results_folder = os.path.join(self.cfg.config_run_path,"multi_cam_evaluation")

            self.logger.info("Started multi cam evaluation")
            splitted_multi_cam_evaluation(dataset_folder=self.cfg.dataset_folder
                                          ,
                                          track_results_folder=self.cfg.track_results_folder
                                          , results_output_folder=multi_cam_eval_results_folder
                                          , cam_ids=self.cfg.cam_ids
                                          , working_dir=self.cfg.working_dir
                                          , n_parts=self.cfg.n_parts
                                          )

            self.logger.info("Finished multi cam evaluation")



        if self.cfg.evaluate_single_cam:

            single_cam_eval_results_folder = os.path.join(self.cfg.config_run_path,"single_cam_evaluation")

            self.logger.info("Started single cam evaluation")
            splitted_single_cam_evaluation(dataset_folder=self.cfg.dataset_folder
                                          ,
                                          track_results_folder=self.cfg.track_results_folder
                                          , results_output_folder=single_cam_eval_results_folder
                                          , cam_ids=self.cfg.cam_ids
                                          , working_dir=self.cfg.working_dir
                                          , n_parts=self.cfg.n_parts
                                          )
            self.logger.info("Finished single cam evaluation")











def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)

    return parser.parse_args()



if __name__=="__main__":


    args = parse_args()

    run_clustering = Run_clustering(args)


    run_clustering.run()