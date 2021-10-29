import os.path as osp
import os
import sys
import numpy as np

filename = './coords_fib_cam_1.csv'

file = np.loadtxt(filename, dtype=int, delimiter=',')
output = './result.csv'

for frame_no_cam, person_id, x_top_left_BB, y_top_left_BB, x_bottom_right_BB, y_bottom_right_BB in file:
    if 37843 <= frame_no_cam <= 42763:
        gt_str = '{:d},{:d},{:d},{:d},{:d},{:d}\n'.format(
            (frame_no_cam - 37843), person_id, x_top_left_BB, y_top_left_BB, x_bottom_right_BB, y_bottom_right_BB)
        with open(output, 'a') as f:
            f.write(gt_str)