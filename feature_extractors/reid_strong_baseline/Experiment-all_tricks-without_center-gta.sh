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
python3 tools/train.py --config_file='/home/koehlp/Dokumente/JTA-MTMCT-Mod/deep_sort_mc/feature_extractors/reid-strong-baseline/configs/softmax_triplet.yml' MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('gta2207')" DATASETS.ROOT_DIR "('/home/koehlp/Downloads/work_dirs/datasets/')" OUTPUT_DIR "('/home/koehlp/Downloads/work_dirs/feature_extractor/strong_reid_baseline/logs/gta2207/Experiment-all-tricks-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on')"