# run tracker on train dataset first (cam 4,5 left)
python run_tracker.py --config configs/tracker_configs/frcnn50_new_abd_train.py
# then run tracker on test dataset (cam 3,4,5 left)
python run_tracker.py --config configs/tracker_configs/frcnn50_new_abd_test.py