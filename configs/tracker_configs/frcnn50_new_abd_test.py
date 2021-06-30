root = {

    "general" : {

        "display_viewer" : False,
        #The visible GPUS will be restricted to the numbers listed here. The pytorch (cuda:0) numeration will start at 0
        #This is a trick to get everything onto the wanted gpus because just setting cuda:4 in the function calls will
        #not work for mmdetection. There will still be things on gpu cuda:0.
        "cuda_visible_devices" : "1, 2",
        "save_track_results" : True

    },

    "data" : {
        # To increase the speed while developing an specific interval of all frames can be set.
        "selection_interval" : [0,10000],

        "source" : {
            "base_folder" : "/u40/zhanr110/MTA_ext_short/test",
            "cam_ids" : [0,1,2,3,4,5]
        }


    },


    "detector" : {

        "mmdetection_config" : "detectors/mmdetection/configs/faster_rcnn_r50_fpn_1x_gta.py",
        "mmdetection_checkpoint_file" : "work_dirs/detector/faster_rcnn_gta22.07_epoch_5.pth",
        "device" : "cuda:0",
        #Remove all detections with a confidence less than min_confidence
        "min_confidence" : 0.8,

    },


    "feature_extractor" : {

        "feature_extractor_name" : "abd_net_extractor"

            ,"reid_strong_extractor": {
                "reid_strong_baseline_config": "feature_extractors/reid_strong_baseline/configs/softmax_triplet.yml",
                "checkpoint_file": "work_dirs/feature_extractor/strong_reid_baseline/resnet50_model_reid_GTA_softmax_triplet.pth",
                "device": "cuda:1,2"
                ,"visible_device" : "1,2"}

            ,"abd_net_extractor" : dict(abd_dan=['cam', 'pam'], abd_dan_no_head=False, abd_dim=1024, abd_np=2, adam_beta1=0.9,
                  adam_beta2=0.999, arch='resnet50', branches=['global', 'abd'], compatibility=False, criterion='htri',
                  cuhk03_classic_split=False, cuhk03_labeled=False, dan_dan=[], dan_dan_no_head=False, dan_dim=1024,
                  data_augment=['crop,random-erase'], day_only=False, dropout=0.5, eval_freq=5, evaluate=False,
                  fixbase=False, fixbase_epoch=10, flip_eval=False, gamma=0.1, global_dim=1024,
                  global_max_pooling=False, gpu_devices='1,2', height=384, htri_only=False, label_smooth=True,
                  lambda_htri=0.1, lambda_xent=1, lr=0.0003, margin=1.2, max_epoch=80, min_height=-1,
                  momentum=0.9, night_only=False, np_dim=1024, np_max_pooling=False, np_np=2, np_with_global=False,
                  num_instances=4, of_beta=1e-06, of_position=['before', 'after', 'cam', 'pam', 'intermediate'],
                  of_start_epoch=23, open_layers=['classifier'], optim='adam', ow_beta=0.001,
                  pool_tracklet_features='avg', print_freq=10, resume='', rmsprop_alpha=0.99
                  , load_weights='work_dirs/feature_extractor/abd-net/checkpoint_ep30_non_clean.pth.tar'
                  , root='work_dirs/datasets'
                       , sample_method='evenly'
                       , save_dir='work_dirs/feature_extractor/abd-net/log/eval-resnet50'
                       , seed=1, seq_len=15,
                  sgd_dampening=0, sgd_nesterov=False, shallow_cam=True, source_names=['mta_ext'], split_id=0,
                  start_epoch=0, start_eval=0, stepsize=[20, 40], target_names=['market1501'],
                  test_batch_size=100, train_batch_size=64, train_sampler='', use_avai_gpus=False, use_cpu=False,
                  use_metric_cuhk03=False, use_of=True, use_ow=True, visualize_ranks=False, weight_decay=0.0005,
                  width=128, workers=4)


    },

    "tracker" : {
        "type" : "DeepSort",
        "nn_budget" : 100

    }

}

