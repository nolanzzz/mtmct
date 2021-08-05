

root = {

        "work_dirs" : "/u40/zhanr110/mtmct/work_dirs"
        # "work_dirs" : "/Users/nolanzhang/Projects/mtmct/work_dirs"
        ,"train_track_results_folder" : "/u40/zhanr110/mtmct/work_dirs/tracker/config_runs/fair/tracker_results"
        # ,"train_track_results_folder" : "/Users/nolanzhang/Projects/mtmct/work_dirs/tracker/config_runs/fair/tracker_results"
        ,"test_track_results_folder" : "/u40/zhanr110/mtmct/work_dirs/tracker/config_runs/fair/tracker_results"
        # ,"test_track_results_folder" : "/Users/nolanzhang/Projects/mtmct/work_dirs/tracker/config_runs/fair/tracker_results"
        ,"train_dataset_folder" : "/u40/zhanr110/MTA_ext_short/train"
        # ,"train_dataset_folder" : "/Users/nolanzhang/Projects/mtmct/data/MTA_ext_short/train"
        ,"test_dataset_folder" : "/u40/zhanr110/MTA_ext_short/test"
        # ,"test_dataset_folder" : "/Users/nolanzhang/Projects/mtmct/data/MTA_ext_short/test"
        ,"cam_count" : 6

        ,"feature_extractor": {

               "feature_extractor_name": "abd_net_extractor"

               , "reid_strong_extractor": {
                   "reid_strong_baseline_config": "feature_extractors/reid_strong_baseline/configs/softmax_triplet.yml",
                   "checkpoint_file": "work_dirs/feature_extractor/strong_reid_baseline/resnet50_model_reid_GTA_softmax_triplet.pth",
                   "device": "cuda:0,1"
                   , "visible_device": "0,1"}

               , "abd_net_extractor": dict(abd_dan=['cam', 'pam'], abd_dan_no_head=False, abd_dim=1024, abd_np=2,
                                           adam_beta1=0.9,
                                           adam_beta2=0.999, arch='resnet50', branches=['global', 'abd'],
                                           compatibility=False, criterion='htri',
                                           cuhk03_classic_split=False, cuhk03_labeled=False, dan_dan=[],
                                           dan_dan_no_head=False, dan_dim=1024,
                                           data_augment=['crop,random-erase'], day_only=False, dropout=0.5, eval_freq=5,
                                           evaluate=False,
                                           fixbase=False, fixbase_epoch=10, flip_eval=False, gamma=0.1, global_dim=1024,
                                           global_max_pooling=False, gpu_devices='1,2', height=384, htri_only=False,
                                           label_smooth=True,
                                           lambda_htri=0.1, lambda_xent=1, lr=0.0003, margin=1.2, max_epoch=80,
                                           min_height=-1,
                                           momentum=0.9, night_only=False, np_dim=1024, np_max_pooling=False, np_np=2,
                                           np_with_global=False,
                                           num_instances=4, of_beta=1e-06,
                                           of_position=['before', 'after', 'cam', 'pam', 'intermediate'],
                                           of_start_epoch=23, open_layers=['classifier'], optim='adam', ow_beta=0.001,
                                           pool_tracklet_features='avg', print_freq=10, resume='', rmsprop_alpha=0.99
                                           ,
                                           load_weights='work_dirs/feature_extractor/abd-net/checkpoint_ep30_non_clean.pth.tar'
                                           , root='work_dirs/datasets'
                                           , sample_method='evenly'
                                           , save_dir='work_dirs/feature_extractor/abd-net/log/eval-resnet50'
                                           , seed=1, seq_len=15,
                                           sgd_dampening=0, sgd_nesterov=False, shallow_cam=True,
                                           source_names=['mta_ext'], split_id=0,
                                           start_epoch=0, start_eval=0, stepsize=[20, 40], target_names=['market1501'],
                                           test_batch_size=100, train_batch_size=64, train_sampler='',
                                           use_avai_gpus=False, use_cpu=False,
                                           use_metric_cuhk03=False, use_of=True, use_ow=True, visualize_ranks=False,
                                           weight_decay=0.0005,
                                           width=128, workers=4)


        }
        ,"cluster_from_weights" : {

            "split_count": 1

            ,"best_weights_path" : "no_path"
            , "default_weights" : {

                "are_tracks_cam_id_frame_disjunct": 1
                ,"are_tracks_frame_overlap_disjunct": 1
                ,"overlapping_match_score": 1.8
                ,"feature_mean_distance":2.5
                ,"track_pred_pos_start_distance" : 0.325

                }

            ,"dataset_type" : "train"
            ,"run" : True
        },


        "find_weights" : {

                #For fast testing a number of frames can be provided. To take all frames None can be passed.
                "take_frames_per_cam" : None

                #Default weights which will be partly overwritten during the search
                ,"dist_name_to_distance_weights" : {
                    "are_tracks_cam_id_frame_disjunct": 1
                    ,"are_tracks_frame_overlap_disjunct": 1
                    ,"overlapping_match_score": 0
                    ,"feature_mean_distance": 0
                    , "track_pred_pos_start_distance" : 0
            },

            "weight_search_configs" : [
                                        { "dist_name" : "feature_mean_distance"
                                                    ,"start_value" : 4.25
                                                    ,"stop_value" : 5
                                                    ,"steps" : 3},

                                        {"dist_name": "overlapping_match_score"
                                                    ,"start_value": 1.8
                                                    ,"stop_value": 2.3
                                                    ,"steps": 3},

                                        {"dist_name": "track_pred_pos_start_distance"
                                                    ,"start_value": 0.1
                                                    ,"stop_value": 1
                                                    ,"steps": 5}
                                                ]


        ,"run" : False

        }

}