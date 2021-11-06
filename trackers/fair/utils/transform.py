import os.path as osp
import os
import sys
import numpy as np

data_type = sys.argv[1]
folder_no = sys.argv[2]
# path = '/u40/zhanr110/mtmct/trackers/fair/data/MTA_short/mta_data/images/' + data_type + '/cam_' + folder_no
path = '../data/MTA_short/mta_data/images/' + data_type + '/cam_' + folder_no
filename = path + '/coords_fib_cam_' + folder_no + '.csv'

tid_curr = 0
tid_last = -1
gt = np.loadtxt(filename, dtype=np.float64, delimiter=',')
gt_fpath = path + '/gt/unsorted.txt'
for fid, tid, x1, y1, x2, y2 in gt:
    fid = int(fid)
    tid = int(tid)
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    width = int(x2 - x1)
    height = int(y2 - y1)
    rest = 1
    gt_str = '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d}\n'.format(
        fid, tid, x1, y1, width, height, rest, rest, rest)
    with open(gt_fpath, 'a') as f:
        f.write(gt_str)