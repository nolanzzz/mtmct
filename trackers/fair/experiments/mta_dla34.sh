cd src
CUDA_VISIBLE_DEVICES="1,2" python train.py mot --exp_id mta_long_dla34 --resume --load_model '../exp/mot/mta_long_dla34/model_last.pth' --data_cfg '../src/lib/cfg/mta.json' --gpus 1,2
cd ..

CUDA_VISIBLE_DEVICES="1, 2" python track.py mot --exp_id mta_mot17_short_oldmodel --test_mta True --load_model '../exp/mot/mta_dla34/model_last.pth' --data_dir '../data/MTA/' --conf_thres 0.4 --gpus 1,2
CUDA_VISIBLE_DEVICES="1, 2" python demo.py mot --load_model '../exp/mot/mta_long_dla34/model_last.pth' --input-video '../data/MTA/mta_data/images/test/cam_5/cam_5.mp4' --output-root ../demos/mta_mot_short_newmodel_cam_5 --conf_thres 0.4 --gpus 1,2
python track.py mot --exp_id mta_mot17 --test_mta True --load_model '../models/fairmot_dla34.pth' --data_dir '../data/MTA/' --conf_thres 0.4