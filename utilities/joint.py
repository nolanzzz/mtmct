# -*- coding: utf-8 -*-
# ---------------------

from typing import *
import pandas as pd
import cv2
import numpy as np


class Joint(object):
    """
    a Joint is a keypoint of the human body.
    """

    # list of joint names
    NAMES = [
        'head_top',
        'head_center',
        'neck',
        'right_clavicle',
        'right_shoulder',
        'right_elbow',
        'right_wrist',
        'left_clavicle',
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'spine0',
        'spine1',
        'spine2',
        'spine3',
        'spine4',
        'right_hip',
        'right_knee',
        'right_ankle',
        'left_hip',
        'left_knee',
        'left_ankle',
    ]

    def __init__(self, joint_row):
        """
        :param array: array version of the joint
        """

        if isinstance(joint_row,np.ndarray):
            array = joint_row
            self.frame = int(array[0])
            self.person_id = int(array[1])
            self.type = int(array[2])
            self.x2d = int(array[3])
            self.y2d = int(array[4])
            self.x3d = array[5]
            self.y3d = array[6]
            self.z3d = array[7]
            self.occ = bool(array[8])  # is this joint occluded?
            self.soc = bool(array[9])  # is this joint self-occluded?
            self.x_top_left_BB = array[10]
            self.y_top_left_BB = array[11]
            self.x_bottom_right_BB = array[12]
            self.y_bottom_right_BB = array[13]
            self.x_2D_person = array[14]
            self.y_2D_person = array[15]
            self.wears_glasses = array[16]
            self.ped_type = array[17]

        if isinstance(joint_row, pd.Series):
            joint_row = joint_row.astype(int)
            self.frame = joint_row["frame_no_cam"]
            self.person_id = joint_row["person_id"]
            self.type = joint_row["joint_type"]
            self.x2d = joint_row["x_2D_joint"]
            self.y2d = joint_row["y_2D_joint"]
            self.x3d = joint_row["x_3D_joint"]
            self.y3d = joint_row["y_3D_joint"]
            self.z3d = joint_row["z_3D_joint"]
            self.occ = bool(joint_row["joint_occluded"])  # is this joint occluded?
            self.soc = bool(joint_row["joint_self_occluded"])  # is this joint self-occluded?
            self.x_top_left_BB = joint_row["x_top_left_BB"]
            self.y_top_left_BB = joint_row["y_top_left_BB"]
            self.x_bottom_right_BB = joint_row["x_bottom_right_BB"]
            self.y_bottom_right_BB = joint_row["y_bottom_right_BB"]
            self.x_2D_person = joint_row["x_2D_person"]
            self.y_2D_person = joint_row["y_2D_person"]
            self.wears_glasses = joint_row["wears_glasses"]
            self.ped_type = joint_row["ped_type"]


    def get_bounding_box_height(self):
        return self.y_bottom_right_BB - self.y_top_left_BB

    @property
    def cam_distance(self):
        # type: () -> float
        """
        :return: distance of the joint from the camera
        """
        # NOTE: camera coords = (0, 0, 0)
        return np.sqrt(self.x3d ** 2 + self.y3d ** 2 + self.z3d ** 2)

    @property
    def is_on_screen(self):
        # type: () -> bool
        """
        :return: True if the joint is on screen, False otherwise
        """
        return (0 <= self.x2d <= 1920) and (0 <= self.y2d <= 1080)

    @property
    def visible(self):
        # type: () -> bool
        """
        :return: True if the joint is visible, False otherwise
        """
        return not (self.occ or self.soc)

    @property
    def personPosition(self):
        return int(self.x_2D_person),int(self.y_2D_person)

    @property
    def pos2d(self):
        # type: () -> Tuple[int, int]
        """
        :return: 2D coordinates of the joints [px]
        """
        return (self.x2d, self.y2d)

    @property
    def pos3d(self):
        # type: () -> Tuple[float, float, float]
        """
        :return: 3D coordinates of the joints [m]
        """
        return (self.x3d, self.y3d, self.z3d)

    @property
    def color(self):
        # type: () -> Tuple[int, int, int]
        """
        :return: the color with which to draw the joint;
        this color is chosen based on the visibility of the joint:
        (1) occluded joint --> RED
        (2) self-occluded joint --> ORANGE
        (2) visible joint --> GREEN
        """
        if self.occ:
            return (255, 0, 42)  # red
        elif self.soc:
            return (255, 128, 42)  # orange
        else:
            return (0, 255, 42)  # green

    @property
    def radius(self):
        # type: () -> int
        """
        :return: appropriate radius [px] for the circle that represents the joint;
        this radius is a function of the distance of the joint from the camera
        """
        #radius = int(round(100.0*np.power(10, 1 - (self.cam_distance / 20.0))))
        bbox_height = self.get_bounding_box_height()
        radius = int(round(bbox_height / 70))
        return radius if (radius >= 1) else 1


    @property
    def name(self):
        # type: () -> str
        """
        :return: name of the joint (eg: 'neck', 'left_elbow', ...)
        """
        return Joint.NAMES[self.type]

    def draw(self, image):
        # type: (np.ndarray) -> np.ndarray
        """
        :param image: image on which to draw the joint
        :return: image with the joint
        """
        image = cv2.circle(
            image, thickness=-1,
            center=self.pos2d,
            radius=self.radius,
            color=self.color,
        )
        return image

    def __str__(self):
        visibility = 'visible' if self.visible else 'occluded'
        return f'{self.name}|2D:({self.x2d},{self.y2d})|3D:({self.x3d},{self.y3d},{self.z3d})|{visibility}'

    __repr__ = __str__
