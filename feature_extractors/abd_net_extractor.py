
import sys
import torch
import torch.nn as nn
from torchreid import models
from args import argument_parser, image_dataset_kwargs, optimizer_kwargs
from torchreid.transforms import build_transforms
from argparse import Namespace
import numpy as np
import PIL

import cv2
import os

from PIL import Image

class Abd_net_extractor:

    def __init__(self,params):

        self.params = params
        self.use_gpu = True
        self.load_args()

        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu_devices
        self.load_model_weights()
        height = self.args.height
        width = self.args.width
        self.transform_test = build_transforms(height, width, is_train=False, data_augment="none")


    def load_args(self):

        if isinstance(self.params,list):
            temp_argv = sys.argv
            sys.argv[1:] = self.params

            # global variables
            parser = argument_parser()

            self.args = parser.parse_args()
            sys.argv = temp_argv
        else:
            self.args = Namespace(**self.params)



        print(self.args)





    def load_model_weights(self):


        model = models.init_model(name=self.args.arch, num_classes=1500, loss={'xent'}, use_gpu=self.use_gpu,
                                  args=vars(self.args))

        # load pretrained weights but ignore layers that don't match in size
        try:
            checkpoint = torch.load(self.args.load_weights)
        except Exception as e:
            print(e)
            checkpoint = torch.load(self.args.load_weights, map_location={'cuda:0': 'cpu'})

        pretrain_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(self.args.load_weights))

        if self.use_gpu:
            model = nn.DataParallel(model).cuda()

        self.model = model


    def preprocess(self,img_crops):
        result = []
        for img in img_crops:
            pil_img = convert_cv2_img_to_pil_img(img)
            transformed_img = self.transform_test(pil_img)
            result.append(transformed_img)
        return result


    def extract(self,img_crops):
        result = []
        print(img_crops)
        self.model.eval()

        img_crops = self.preprocess(img_crops)

        with torch.no_grad():



            image_batch = torch.stack(img_crops)
            #img = img.unsqueeze(0)

            if self.use_gpu:
                image_batch = image_batch.cuda()

            features = self.model(image_batch)[0]

            features = features.cpu().numpy()
            # print("Features shape: ", features.shape)

            for i in range(features.shape[0]):
                feature = features[i,:]
                # print("Feature shape: ", feature.shape)
                # print("feature_content: ", feature)
                # feature = np.reshape(feature,(-1,))
                result.append(feature)
        return result

def convert_cv2_img_to_pil_img(cv2_img):
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = PIL.Image.fromarray(img)

    return pil_img

def test_show_image(path):


    np_img = cv2.imread(path)
    pil_img = convert_cv2_img_to_pil_img(np_img)
    pil_img.show()


if __name__ == "__main__":



    params_dict = dict(abd_dan=['cam', 'pam'], abd_dan_no_head=False, abd_dim=1024, abd_np=2, adam_beta1=0.9,
                  adam_beta2=0.999, arch='resnet50', branches=['global', 'abd'], compatibility=False, criterion='htri',
                  cuhk03_classic_split=False, cuhk03_labeled=False, dan_dan=[], dan_dan_no_head=False, dan_dim=1024,
                  data_augment=['crop,random-erase'], day_only=False, dropout=0.5, eval_freq=5, evaluate=False,
                  fixbase=False, fixbase_epoch=10, flip_eval=False, gamma=0.1, global_dim=1024,
                  global_max_pooling=False, gpu_devices='0', height=384, htri_only=False, label_smooth=True,
                  lambda_htri=0.1, lambda_xent=1, lr=0.0003, margin=1.2, max_epoch=80, min_height=-1,
                  momentum=0.9, night_only=False, np_dim=1024, np_max_pooling=False, np_np=2, np_with_global=False,
                  num_instances=4, of_beta=1e-06, of_position=['before', 'after', 'cam', 'pam', 'intermediate'],
                  of_start_epoch=23, open_layers=['classifier'], optim='adam', ow_beta=0.001,
                  pool_tracklet_features='avg', print_freq=10, resume='', rmsprop_alpha=0.99
                  , load_weights='/media/philipp/philippkoehl_ssd/work_dirs/feature_extractor/abd-net/checkpoint_ep30_non_clean.pth.tar'
                  , root='/media/philipp/philippkoehl_ssd/work_dirs/datasets'
                       , sample_method='evenly'
                       , save_dir='/media/philipp/philippkoehl_ssd/work_dirs/feature_extractor/abd-net/log/eval-resnet50'
                       , seed=1, seq_len=15,
                  sgd_dampening=0, sgd_nesterov=False, shallow_cam=True, source_names=['mta_ext'], split_id=0,
                  start_epoch=0, start_eval=0, stepsize=[20, 40], target_names=['market1501'],
                  test_batch_size=100, train_batch_size=64, train_sampler='', use_avai_gpus=False, use_cpu=False,
                  use_metric_cuhk03=False, use_of=True, use_ow=True, visualize_ranks=False, weight_decay=0.0005,
                  width=128, workers=4)

    abd_ext = Abd_net_extractor(params=params_dict)


    test_show_image("/media/philipp/philippkoehl_ssd/Dokumente/masterarbeit/JTA-MTMCT-Mod/deep_sort_mc/images/2.jpg")

    image = cv2.imread("/media/philipp/philippkoehl_ssd/Dokumente/masterarbeit/JTA-MTMCT-Mod/deep_sort_mc/images/2.jpg")

    features = abd_ext.extract(img_crops=[image])

    print("feature mean: {}".format(np.mean(features)))
    #output dimension 3072
    print(features)
