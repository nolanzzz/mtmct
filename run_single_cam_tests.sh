# one-shot script to run all three inference using two baselines and our solution

# single-cam
cd trackers/fair/src
# 1 - Original FairMOT (20e model)
CUDA_VISIBLE_DEVICES="2" python track.py mot --test_mta True --use_original True --exp_id fair_high_20e_original --load_model '../exp/mot/high_20_complete/model_last.pth' --data_dir '../data/MTA_short/' --conf_thres 0.4 --gpus 2
# 2 - Ours - Fair-WDA (20e model)
CUDA_VISIBLE_DEVICES="2" python track.py mot --test_mta True --exp_id fair_high_20e --load_model '../exp/mot/high_20_complete/model_last.pth' --data_dir '../data/MTA_short/' --conf_thres 0.4 --gpus 2
cd ../../..
python run_tracker.py --config configs/tracker_configs/fair_high_20e.py
# 3 - Original WDA_Tracker (20e model)
python run_tracker.py --config configs/tracker_configs/high_wda_20e.py
