# Fair
CUDA_VISIBLE_DEVICES="0, 3" python track.py mot --exp_id fair_short_features_train --train_mta True --load_model '../exp/mot/mta_short_dla34/model_last.pth' --data_dir '../data/MTA_short/' --conf_thres 0.4 --gpus 0,3
CUDA_VISIBLE_DEVICES="0, 3" python train.py mot --exp_id mta_4_10000 --resume --num_epochs 35 --load_model '../exp/mot/mta_4_10000/model_last.pth' --data_cfg '../src/lib/cfg/mta.json' --gpus 0,3
CUDA_VISIBLE_DEVICES="0,2" python train.py mot --exp_id mta_full_5_epochs --num_epochs 5 --load_model '../exp/mot/mot17_dla34/model_last.pth' --data_cfg '../src/lib/cfg/mta.json' --gpus 0,2
CUDA_VISIBLE_DEVICES="1, 2" python demo.py mot --load_model '../exp/mot/mta_long_dla34/model_last.pth' --input-video '../data/MTA/mta_data/images/test/cam_5/cam_5.mp4' --output-root ../demos/mta_mot_short_newmodel_cam_5 --conf_thres 0.4 --gpus 1,2
CUDA_VISIBLE_DEVICES="1, 2" python visualize.py mot --exp_id mta_wda_vis_train_short --train_mta True --data_dir '../data/MTA/'

# wda_tracker
CUDA_VISIBLE_DEVICES="1,2,3" python run_tracker.py --config configs/tracker_configs/frcnn50_new_abd_test.py
CUDA_VISIBLE_DEVICES="0, 3" python run_multi_cam_clustering.py --config configs/clustering_configs/mta_es_abd_non_clean.py

# mmdetection
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py configs/mta/faster_rcnn_r50_mta.py --gpus 4
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/mta/faster_rcnn_r50_mta.py 4