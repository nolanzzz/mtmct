cd src
#CUDA_VISIBLE_DEVICES="1,2" python train.py mot --exp_id mta_4_10000 --resume --load_model '../exp/mot/mta_long_dla34/model_last.pth' --data_cfg '../src/lib/cfg/mta.json' --gpus 1,2
CUDA_VISIBLE_DEVICES="1, 2" python train.py mot --exp_id mta_4_10000 --resume --num_epochs 40 --load_model '../exp/mot/mta_4_10000/model_last.pth' --data_cfg '../src/lib/cfg/mta.json' --gpus 1,2
cd ..

CUDA_VISIBLE_DEVICES="1, 2" python track.py mot --exp_id mta_mot17_short_oldmodel --test_mta True --load_model '../exp/mot/mta_dla34/model_last.pth' --data_dir '../data/MTA/' --conf_thres 0.4 --gpus 1,2
CUDA_VISIBLE_DEVICES="1, 2" python demo.py mot --load_model '../exp/mot/mta_long_dla34/model_last.pth' --input-video '../data/MTA/mta_data/images/test/cam_5/cam_5.mp4' --output-root ../demos/mta_mot_short_newmodel_cam_5 --conf_thres 0.4 --gpus 1,2
CUDA_VISIBLE_DEVICES="0, 1" python track.py mot --exp_id mta_wda_short_test --test_mta True --load_model '../exp/mot/mta_short_dla34/model_last.pth' --data_dir '../data/MTA_short/' --conf_thres 0.4 --gpus 0,1
CUDA_VISIBLE_DEVICES="0, 1" python run_multi_cam_clustering.py --config configs/clustering_configs/mta_es_abd_non_clean.py
CUDA_VISIBLE_DEVICES="1, 2" python visualize.py mot --exp_id mta_wda_vis_gt --test_mta True --data_dir '../data/MTA/'

CUDA_VISIBLE_DEVICES="1" python run_tracker.py --config configs/tracker_configs/frcnn50_new_abd_test.py