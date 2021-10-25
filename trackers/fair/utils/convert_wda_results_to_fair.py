import os.path as osp
import os
import sys
import numpy as np


def main():
    seqs = [0,1,2,3,4,5]
    path = '/Users/nolanzhang/Projects/mtmct/work_dirs/tracker/config_runs/new_test_20e_train/tracker_results_wda_ready/'
    output_path = '/Users/nolanzhang/Projects/mtmct/work_dirs/tracker/config_runs/new_test_20e_train/tracker_results_wda_to_fair/'
    if not osp.exists(output_path):
        os.makedirs(output_path)

    for seq in seqs:
        filename = path + 'track_results_' + str(seq) + '.txt'
        output_filename = output_path + 'cam_' + str(seq) + '.txt'
        wda_file = np.loadtxt(filename, dtype=np.float64, delimiter=',')

        with open(output_filename, 'w') as f:
            for frame_id, cam_id, person_id, det_id, xtl, ytl, xbr, ybr in wda_file:
                save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
                w = xbr - xtl
                h = ybr - ytl
                line = save_format.format(frame=int(frame_id), id=int(person_id), x1=xtl, y1=ytl, x2=xbr, y2=ybr, w=w, h=h)
                f.write(line)


if __name__ == '__main__':
    main()