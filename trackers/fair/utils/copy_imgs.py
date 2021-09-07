import shutil
import sys

use_steps = bool(sys.argv[1])
frames = int(sys.argv[2])
for i in range(6):
    if use_steps is True:
        j, period_i, curr_step = 1, 0, 0
        # train - 124230; test - 127880
        while j <= 124230:
            if curr_step == 4:
                break
            # enough frames for current step
            if period_i > frames:
                curr_step += 1
                period_i = 0
                j = j + 1000
                continue
            # not enough frames yet:
            j += 1
            period_i += 1
            img_title = str(j).zfill(6) + '.jpg'
            shutil.copy2('../data/MTA/mta_data/images/train/cam_' + str(i) + '/img1/' + img_title,
                         '../data/MTA/mta_data/images/train/cam_' + str(i) + '_t/img1/' + img_title)
    else:
        for j in range(1, frames):
            img_title = str(j).zfill(6) + '.jpg'
            shutil.copy2('../data/MTA/mta_data/images/test/cam_' + str(i) + '/img1/' + img_title,
                         '../data/MTA/mta_data/images/test/cam_' + str(i) + '_t/img1/')
