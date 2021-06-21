from modeling import build_model
import torch
from config import cfg

import torchvision.transforms as transforms

import cv2
import numpy as np
import os

class Reid_strong_extractor:

    def __init__(self,mc_cfg):
        '''
        Because the config from reid strong baseline is called cfg this name cannot be used.
        '''
        self.mc_cfg = mc_cfg
        os.environ['CUDA_VISIBLE_DEVICES'] = self.mc_cfg.reid_strong_extractor.visible_device

        cfg.merge_from_file(mc_cfg.reid_strong_extractor.reid_strong_baseline_config)
        cfg.TEST.WEIGHT = mc_cfg.reid_strong_extractor.checkpoint_file
        cfg.MODEL.PRETRAIN_CHOICE = ('self')




        cfg.freeze()
        self.model = build_model(cfg, 1361)
        self.model.load_param(cfg.TEST.WEIGHT)
        self.model.to(mc_cfg.reid_strong_extractor.device)

        #height,width
        self.size = (128, 256)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])



    def _preprocess(self, im_crops):
        """
        Copied :/ from the feature_extractor.py from deep_defaul (deep_sort_pytorch)
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            resized_im = cv2.resize(im.astype(np.float32)/255., size)
            return resized_im

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def extract(self, img_crops):
        im_batch = self._preprocess(img_crops)
        self.model.eval()
        with torch.no_grad():

            data = im_batch.to(self.mc_cfg.reid_strong_extractor.device) if torch.cuda.device_count() >= 1 else im_batch
            features = self.model(data)
            return features.cpu().numpy()



