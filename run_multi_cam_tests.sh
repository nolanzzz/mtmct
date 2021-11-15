# multi-cam clustering

# 1 - WDA_Tracker - Cluster with WDA track results and features
CUDA_VISIBLE_DEVICES="3" python run_multi_cam_clustering.py --config configs/clustering_configs/high_wda_20e.py

# 2 - Ours - Cluster with our single-camera track results and features
CUDA_VISIBLE_DEVICES="3" python run_multi_cam_clustering.py --config configs/clustering_configs/fair_high_20e.py