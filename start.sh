# one-shot script to run single then multi-cam tracking
cd trackers/fair/src
CUDA_VISIBLE_DEVICES="0" python track.py mot --test_mta True --exp_id "$1" --load_model '../exp/mot/fair_high_30e/model_last.pth' --data_dir '../data/MTA_short/' --conf_thres 0.4 --gpus 0
cd ../../..
python run_tracker.py --config configs/tracker_configs/"$1".py
python run_multi_cam_clustering.py --config configs/clustering_configs/"$1".py
