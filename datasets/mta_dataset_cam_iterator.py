import mmcv
import os.path as osp
from utilities.helper import natural_keys
import os
import cv2

class Mta_dataset_image:
    def __init__(self,img,frame_no_cam,cam_id):

        self._img = img

        self._frame_no_cam = frame_no_cam
        self._cam_id = cam_id

        self._img_dims = (self._img.shape[:2][1],self._img.shape[:2][0])

    @property
    def cam_id(self):
        return self._cam_id

    @property
    def img(self):
        return self._img

    @property
    def frame_no_cam(self):
        return self._frame_no_cam

    @property
    def img_dims(self):
        return self._img_dims




def get_cam_iterators(cfg, cam_ids):

    iterators = []
    for cam_id in cam_ids:
        iterators.append(Mta_dataset_cam_iterator(cfg, cam_id))

    return iterators


class Mta_dataset_cam_iterator:

    def __init__(self, cfg, cam_id):
        super().__init__()


        self.cam_video_path = os.path.join(cfg.data.source.base_folder, "cam_{}".format(cam_id),"cam_{}.mp4".format(cam_id))

        self.video_capture = cv2.VideoCapture(self.cam_video_path)
        self.cfg = cfg

        self.cam_id = cam_id

        self.selection_interval = self.cfg.data.selection_interval
        assert self.selection_interval[0] < self.selection_interval[1]
        selection_start = self.selection_interval[0]


        self.video_capture .set(cv2.CAP_PROP_POS_FRAMES, selection_start)

        self.current_frame_no = selection_start


    def get_all_frame_nos(self):
        return list(range(self.selection_interval[0],self.selection_interval[1]+1))

    def __len__(self):
        real_selection_end = min(int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)),self.selection_interval[1])

        return real_selection_end - self.selection_interval[0] + 1



    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.video_capture.read()

        if not ret or self.selection_interval[1]  == self.current_frame_no - 1:
            raise StopIteration

        next_mta_image = Mta_dataset_image(img=frame, frame_no_cam=self.current_frame_no, cam_id=self.cam_id)
        self.current_frame_no += 1
        return next_mta_image


if __name__ == "__main__":
    cfg = mmcv.Config.fromfile("/configs/tracker_configs/frcnn50_new_abd_test.py").root
    mta_dataset = Mta_dataset_cam_iterator(cfg=cfg, cam_id=0)
    print("Dataset length {}".format(len(mta_dataset)))
    for dataset_image in mta_dataset:
        print("frame_no_cam: {} , cam_id: {} ".format(dataset_image.frame_no_cam,dataset_image.cam_id))
        cv2.imshow("",dataset_image.img)
        cv2.waitKey(1)

