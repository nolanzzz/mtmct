# multi-cam clustering

# clear cache before start
rm -rf work_dirs/cache

# 1 - Ours - Cluster with our single-camera track results and features
python run_multi_cam_clustering.py --config configs/clustering_configs/fair_high_20e.py

# clear cache before next run
rm -rf work_dirs/cache

# 2 - WDA_Tracker - Cluster with WDA track results and features
CUDA_VISIBLE_DEVICES="2" python run_multi_cam_clustering.py --config configs/clustering_configs/high_wda_20e.py
