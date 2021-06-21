
from detectors.mmdetection.mmdet.apis.inference import init_detector, inference_detector
from detectors.base_detector import Base_detector
from utilities.preprocessing import non_max_suppression
from utilities.helper import many_xyxy2xywh
import numpy as np
from utilities.helper import bboxes_round_int

import os
import mmcv

class Mmdetection_detector(Base_detector):


    def __init__(self,cfg):


        #Initialize the detector


        Base_detector.__init__(self, cfg)
        print("Mmdetection_detector init called")

        mmdetection_checkpoint_file = os.path.join(self.cfg.general.repository_root,self.cfg.detector.mmdetection_checkpoint_file)
        mmdetection_config = os.path.join(self.cfg.general.repository_root,self.cfg.detector.mmdetection_config)



        self.detector_model = init_detector(mmdetection_config
                                            , mmdetection_checkpoint_file
                                            , device=self.cfg.detector.device)

        pass


    def detect(self,img):


        # Run person detector
        result = inference_detector(self.detector_model, img)

        bboxes_with_score = result[0]


        # Only take those detections with a higher confidence than the min confidence
        bboxes_with_score = np.array(
            [bbox_with_s for bbox_with_s in bboxes_with_score if bbox_with_s[4] >= self.cfg.detector.min_confidence])

        if len(bboxes_with_score) == 0:
            return (np.array([]),np.array([]))

        #columns 0 to 3 are bbox
        bboxes = bboxes_with_score[:, 0:4]

        bboxes = many_xyxy2xywh(bboxes)

        # Run non max suppression
        scores = bboxes_with_score[:, 4]
        indices = non_max_suppression(
            bboxes, max_bbox_overlap=0.8, scores=scores)
        bboxes = [bboxes[i, :] for i in indices]

        bboxes = bboxes_round_int(bboxes)


        return bboxes, scores


