# Fair
CUDA_VISIBLE_DEVICES="0, 3" python track.py mot --exp_id fair_short_features_train --train_mta True --load_model '../exp/mot/mta_short_dla34/model_last.pth' --data_dir '../data/MTA_short/' --conf_thres 0.4 --gpus 0,3
CUDA_VISIBLE_DEVICES="0, 3" python train.py mot --exp_id mta_4_10000 --resume --num_epochs 35 --load_model '../exp/mot/mta_4_10000/model_last.pth' --data_cfg '../src/lib/cfg/mta.json' --gpus 0,3
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py mot --exp_id dla34_imagenet --num_epochs 20 --load_model '../models/dla34-imagenet.pth' --data_cfg '../src/lib/cfg/mta.json' --gpus 0,1,2,3


CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py mot --exp_id dla34-ba --num_epochs 20 --load_model '../models/dla34-ba.pth' --data_cfg '../src/lib/cfg/mta.json' --gpus 0,1,2,3
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py mot --exp_id fairmot_dla34_test --num_epochs 20 --load_model '../models/fairmot_dla34.pth' --data_cfg '../src/lib/cfg/mta.json' --gpus 0,1,2,3
CUDA_VISIBLE_DEVICES="2,3" python train.py mot --exp_id hrnetv2_w18_imagenet_pretrained.pth --arch 'hrnet_18' --num_epochs 20 --load_model '../models/hrnetv2_w18_imagenet_pretrained.pth' --data_cfg '../src/lib/cfg/mta.json' --gpus 2,3
CUDA_VISIBLE_DEVICES="0,1" python train.py mot --exp_id dla34_coco_wda_short --num_epochs 20 --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/mta.json' --gpus 0,1

python track.py mot --test_mta True --exp_id dla34_coco_wda_short_new_cam_1_full --load_model '../exp/mot/dla34_coco_wda_short/model_last.pth' --data_dir '../data/MTA_short/' --conf_thres 0.4
python track.py mot --test_mta True --exp_id dla34_coco_wda_short_new_cam_1_full --load_model '../exp/mot/dla34_coco_wda_short/model_last.pth' --data_dir '../data/MTA_short/' --conf_thres 0.4
python evaluate.py mot --test_mta True --exp_id dla34_coco_wda_short_new_cam_1 --data_dir '../data/MTA_short/'


CUDA_VISIBLE_DEVICES="0,2" python train.py mot --exp_id mta_full_5_epochs --num_epochs 5 --load_model '../exp/mot/mot17_dla34/model_last.pth' --data_cfg '../src/lib/cfg/mta.json' --gpus 0,2
CUDA_VISIBLE_DEVICES="1, 2" python demo.py mot --load_model '../exp/mot/mta_long_dla34/model_last.pth' --input-video '../data/MTA/mta_data/images/test/cam_5/cam_5.mp4' --output-root ../demos/mta_mot_short_newmodel_cam_5 --conf_thres 0.4 --gpus 1,2
python visualize.py mot --exp_id new_test_20e --test_mta True --data_dir '../data/MTA_short/'
python visualize_fair.py mot --exp_id new_test_20e --test_mta True --data_dir '../data/MTA_short/'


# wda_tracker
python run_tracker.py --config configs/tracker_configs/new_test_20e_cam_1_new_short_high_full.py
CUDA_VISIBLE_DEVICES="2" python run_multi_cam_clustering.py --config configs/clustering_configs/new_test_20e_train.py


# mmdetection
CUDA_VISIBLE_DEVICES="1,2" python tools/train.py configs/mta/faster_rcnn_r50_mta.py --gpus 2
CUDA_VISIBLE_DEVICES="2,3" python tools/train.py configs/mta/mta_full_test_data.py --gpus 2
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/mta/faster_rcnn_r50_mta.py 4

python tools/test.py configs/mta/test_cam_0.py work_dirs/GtaDataset_30e/epoch_20.pth --eval bbox
python tools/test.py configs/mta/test_cam_0.py /u40/zhanr110/mtmct/work_dirs/detector/faster_rcnn_gta22.07_epoch_5.pth --eval bbox
python tools/test.py configs/mta/test_detector.py work_dirs/GtaDataset_30e/epoch_20.pth --out result.mp4