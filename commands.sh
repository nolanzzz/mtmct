# Fair
CUDA_VISIBLE_DEVICES="0, 3" python track.py mot --exp_id fair_features_long_short_test --test_mta True --load_model '../exp/mot/mta_long_dla34/model_last.pth' --data_dir '../data/MTA_short/' --conf_thres 0.4 --gpus 0,3
CUDA_VISIBLE_DEVICES="0, 3" python train.py mot --exp_id mta_4_10000 --resume --num_epochs 35 --load_model '../exp/mot/mta_4_10000/model_last.pth' --data_cfg '../src/lib/cfg/mta.json' --gpus 0,3
CUDA_VISIBLE_DEVICES="1, 2" python demo.py mot --load_model '../exp/mot/mta_long_dla34/model_last.pth' --input-video '../data/MTA/mta_data/images/test/cam_5/cam_5.mp4' --output-root ../demos/mta_mot_short_newmodel_cam_5 --conf_thres 0.4 --gpus 1,2
CUDA_VISIBLE_DEVICES="1, 2" python visualize.py mot --exp_id mta_wda_vis_train_short --train_mta True --data_dir '../data/MTA/'

# wda_tracker
CUDA_VISIBLE_DEVICES="0, 3" python run_tracker.py --config configs/tracker_configs/frcnn50_new_abd_test.py
CUDA_VISIBLE_DEVICES="0, 3" python run_multi_cam_clustering.py --config configs/clustering_configs/mta_es_abd_non_clean.py