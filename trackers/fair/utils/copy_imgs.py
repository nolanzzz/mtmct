import shutil
import sys


def main():
    gen_type = sys.argv[1]
    cam_no = sys.argv[2]
    start_frame = int(sys.argv[3])
    total_num = int(sys.argv[4])

    for i in range(start_frame, start_frame + total_num):
        img_title = str(i).zfill(6) + '.jpg'
        shutil.copy2('../data/MTA/mta_data/images/' + gen_type + '/cam_' + str(cam_no) + '/img1/' + img_title,
                     '../data/MTA/mta_data/images/' + gen_type + '/cam_' + str(cam_no) + '/img1/')


def copy_steps():
    print("here")
    gen_type = sys.argv[1]
    use_steps = bool(sys.argv[2])
    expected_frames = int(sys.argv[3])
    num_steps = int(sys.argv[4])

    total_frames = 124230 if gen_type == 'train' else 127880
    period_frames, step_length = data_portion(total_frames, num_steps, expected_frames)

    for i in range(6):
        if use_steps is True:
            j, period_i, curr_step = 0, 0, 0
            # for i in range(len_all):
            while j < total_frames:
                # pass the last step, return here
                if curr_step == num_steps:
                    break
                # enough frames for current step
                if period_i > period_frames:
                    curr_step += 1
                    period_i = 0
                    j = j + step_length
                    continue
                # not enough frames yet:
                j += 1
                period_i += 1
                img_title = str(j).zfill(6) + '.jpg'
                shutil.copy2('../data/MTA/mta_data/images/' + gen_type + '/cam_' + str(i) + '/img1/' + img_title,
                             '../data/MTA/mta_data/images/' + gen_type + '/cam_' + str(i) + '_t/img1/')

            # j, period_i, curr_step = 1, 0, 0
            # # train - 124230; test - 127880
            # while j <= 124230:
            #     if curr_step == num_steps:
            #         break
            #     # enough frames for current step
            #     if period_i > frames:
            #         curr_step += 1
            #         period_i = 0
            #         j = j + 10
            #         continue
            #     # not enough frames yet:
            #     j += 1
            #     period_i += 1
            #     img_title = str(j).zfill(6) + '.jpg'
            #     shutil.copy2('../data/MTA/mta_data/images/train/cam_' + str(i) + '/img1/' + img_title,
            #                  '../data/MTA/mta_data/images/train/cam_' + str(i) + '_t/img1/')
        # else:
        #     for j in range(1, frames):
        #         img_title = str(j).zfill(6) + '.jpg'
        #         shutil.copy2('../data/MTA/mta_data/images/test/cam_' + str(i) + '/img1/' + img_title,
        #                      '../data/MTA/mta_data/images/test/cam_' + str(i) + '_t/img1/')


def data_portion(total_frames, num_portions, expected_frames):
    p_frames = expected_frames // num_portions
    step_len = total_frames // num_portions
    return p_frames, step_len


if __name__ == '__main__':
    main()
