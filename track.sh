# run tracker on test dataset (cam 4,5 left) with 15e model
python run_tracker.py --config configs/tracker_configs/new_test_15e.py
# run tracker on train dataset with 15e model
CUDA_VISIBLE_DEVICES="1" python run_tracker.py --config configs/tracker_configs/new_test_30e.py