# run tracker on test dataset (cam 4,5 left) with 15e model
python run_tracker.py --config configs/tracker_configs/new_test_15e.py
# run tracker on train dataset with 15e model
CUDA_VISIBLE_DEVICES="1" python run_tracker.py --config configs/tracker_configs/new_test_30e.py


cd MTA_ext_short/train/cam_1
ffmpeg -i cam_1.mp4 -start_number 0 -r 41 img1/image_%d_1.png
cd ../cam_2
ffmpeg -i cam_2.mp4 -start_number 0 -r 41 img1/image_%d_2.png
cd ../cam_3
ffmpeg -i cam_3.mp4 -start_number 0 -r 41 img1/image_%d_3.png
cd ../cam_4
ffmpeg -i cam_4.mp4 -start_number 0 -r 41 img1/image_%d_4.png
cd ../cam_5
ffmpeg -i cam_5.mp4 -start_number 0 -r 41 img1/image_%d_5.png