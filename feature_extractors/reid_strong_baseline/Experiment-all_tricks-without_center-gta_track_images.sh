# Experiment all tricks without center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# without center loss
python3 tools/train.py \
--config_file=configs/softmax_triplet.yml \
MODEL.PRETRAIN_PATH "('/media/philipp/philippkoehl_ssd/work_dirs/feature_extractor/pretrained/resnet50-19c8e357.pth')" \
MODEL.DEVICE_ID "('0')" \
DATASETS.NAMES "('gta_track_images')" \
DATASETS.ROOT_DIR "('/media/philipp/philippkoehl_ssd/work_dirs/datasets')" \
OUTPUT_DIR "('/media/philipp/philippkoehl_ssd/work_dirs/track_images/logs/gta2207/Experiment-all-tricks-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-clean')"