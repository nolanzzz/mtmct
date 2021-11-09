import sys
import pandas as pd

data_type = sys.argv[1]
folder_no = sys.argv[2]

# filename = '../data/MTA/mta_data/images/' + data_type + '/cam_' + folder_no + '/coords_cam_' + folder_no + '.csv'
filename = '../data/MTA/MTA_videos_coords/' + data_type + '/cam_' + folder_no + '/coords_cam_' + folder_no + '.csv'
# filename = '../data/MTA/mta_data/images/' + data_type + '/cam_' + folder_no + '/coords_result.csv'

# file = np.loadtxt(filename, delimiter=',', skiprows=1)
# output = '../data/MTA/mta_data/images/' + data_type + '/cam_' + folder_no + '/coords_result.csv'
# output = '../data/MTA/mta_data/images/' + data_type + '/cam_' + folder_no + '/test.csv'
output = '../data/MTA/MTA_videos_coords/' + data_type + '/cam_' + folder_no + '/test_' + folder_no + '.csv'

# df = pd.read_csv(filename)
# res = df.query('frame_no_cam >= 37843 & frame_no_cam <= 42763')
# res = df.loc[(df['frame_no_cam'] >= 10) & (df['frame_no_cam'] <= 20)]

chunksize = 10 ** 6
chunk_no = 1
for chunk in pd.read_csv(filename, chunksize=chunksize):
    header = True if chunk_no == 1 else False
    df = chunk.query('frame_no_cam >= 37843 & frame_no_cam <= 42763')
    df['frame_no_cam'] = df['frame_no_cam'].apply(lambda x: x - 37843)
    df.to_csv(output, header=header, mode='a', index=False)
    chunk_no += 1

# res.to_csv(output)




# with open(output, 'a') as f:
#     header = 'Unnamed: 0,frame_no_gta,frame_no_cam,person_id,appearance_id,joint_type,x_2D_joint,y_2D_joint,x_3D_joint,y_3D_joint,z_3D_joint,joint_occluded,joint_self_occluded,x_3D_cam,y_3D_cam,z_3D_cam,x_rot_cam,y_rot_cam,z_rot_cam,fov,x_3D_person,y_3D_person,z_3D_person,x_2D_person,y_2D_person,ped_type,wears_glasses,yaw_person,hours_gta,minutes_gta,seconds_gta,x_top_left_BB,y_top_left_BB,x_bottom_right_BB,y_bottom_right_BB\n'
#     f.write(header)
#     for unnamed, frame_no_gta, frame_no_cam, person_id, appearance_id, joint_type, x_2D_joint, y_2D_joint, \
#             x_3D_joint, y_3D_joint, z_3D_joint, joint_occluded, joint_self_occluded, x_3D_cam, y_3D_cam, z_3D_cam, \
#             x_rot_cam, y_rot_cam, z_rot_cam, fov, x_3D_person, y_3D_person, z_3D_person, x_2D_person, y_2D_person, \
#             ped_type, wears_glasses, yaw_person, hours_gta, minutes_gta, seconds_gta, x_top_left_BB, y_top_left_BB, \
#             x_bottom_right_BB, y_bottom_right_BB in file:
#         if 37843 <= frame_no_cam <= 42763:
#             gt_str = '{:d},{:d},{:d},{:d},{:d},{:d},{:f},{:f},' \
#                      '{:f},{:f},{:f},{:d},{:d},{:f},{:f},{:f},' \
#                      '{:f},{:f},{:f},{:d},{:f},{:f},{:f},{:f},{:f},' \
#                      '{:d},{:d},{:f},{:d},{:d},{:d},{:f},{:f},' \
#                      '{:f},{:f}\n'.format(
#                         int(unnamed), int(frame_no_gta), int(frame_no_cam) - 37843, int(person_id), int(appearance_id), int(joint_type), x_2D_joint, y_2D_joint,
#                         x_3D_joint, y_3D_joint, z_3D_joint, int(joint_occluded), int(joint_self_occluded), x_3D_cam, y_3D_cam, z_3D_cam,
#                         x_rot_cam, y_rot_cam, z_rot_cam, int(fov), x_3D_person, y_3D_person, z_3D_person, x_2D_person, y_2D_person,
#                         int(ped_type), int(wears_glasses), yaw_person, int(hours_gta), int(minutes_gta), int(seconds_gta), x_top_left_BB, y_top_left_BB,
#                         x_bottom_right_BB, y_bottom_right_BB)
#             f.write(gt_str)
