# Fair
CUDA_VISIBLE_DEVICES="0, 3" python train.py mot --exp_id mta_4_10000 --resume --num_epochs 35 --load_model '../exp/mot/mta_4_10000/model_last.pth' --data_cfg '../src/lib/cfg/mta.json' --gpus 0,3
python train.py mot --exp_id fair_mot17_high_20e --num_epochs 20 --load_model '../exp/mot/mot17_dla34/model_last.pth' --data_cfg '../src/lib/cfg/mta.json' --gpus -2,3
python track.py mot --test_mta True --exp_id dla34_coco_wda_short_new_cam_1_full --load_model '../exp/mot/dla34_coco_wda_short/model_last.pth' --data_dir '../data/MTA_short/' --conf_thres 0.4
python evaluate.py mot --test_mta True --exp_id dla34_coco_wda_short_new_cam_1 --data_dir '../data/MTA_short/'
python demo.py mot --load_model '../exp/mot/mta_long_dla34/model_last.pth' --input-video '../data/MTA/mta_data/images/test/cam_5/cam_5.mp4' --output-root ../demos/mta_mot_short_newmodel_cam_5 --conf_thres 0.4 --gpus 1,2
python visualize.py mot --exp_id fair_high_20e --test_mta True --data_dir '../data/MTA_short/'
python visualize_fair.py mot --exp_id new_test_20e --test_mta True --data_dir '../data/MTA_short/'

# wda_tracker
python run_tracker.py --config configs/tracker_configs/fair_test_path.py
CUDA_VISIBLE_DEVICES="2" python run_multi_cam_clustering.py --config configs/clustering_configs/high_wda.py

# mmdetection
CUDA_VISIBLE_DEVICES="1,2,3" python tools/train.py configs/mta/mta_high_30.py --gpus 3
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/mta/faster_rcnn_r50_mta.py 4
python tools/test.py configs/mta/test_cam_0.py work_dirs/GtaDataset_30e/epoch_20.pth --eval bbox
python tools/test.py configs/mta/test_detector.py work_dirs/GtaDataset_30e/epoch_20.pth --out result.mp4