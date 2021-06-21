# -*- coding: utf-8 -*-
# ---------------------

from typing import *

import cv2
import numpy as np
import pandas as pd
from .joint import Joint

from .BoundingBoxTracked import BoundingBoxTracked

class Pose(list):
    """
    a Pose is a list of Joint(s) belonging to the same person.
    """

    LIMBS = [
        (0, 1),  # head_top -> head_center
        (1, 2),  # head_center -> neck
        (2, 3),  # neck -> right_clavicle
        (3, 4),  # right_clavicle -> right_shoulder
        (4, 5),  # right_shoulder -> right_elbow
        (5, 6),  # right_elbow -> right_wrist
        (2, 7),  # neck -> left_clavicle
        (7, 8),  # left_clavicle -> left_shoulder
        (8, 9),  # left_shoulder -> left_elbow
        (9, 10),  # left_elbow -> left_wrist
        (2, 11),  # neck -> spine0
        (11, 12),  # spine0 -> spine1
        (12, 13),  # spine1 -> spine2
        (13, 14),  # spine2 -> spine3
        (14, 15),  # spine3 -> spine4
        (15, 16),  # spine4 -> right_hip
        (16, 17),  # right_hip -> right_knee
        (17, 18),  # right_knee -> right_ankle
        (15, 19),  # spine4 -> left_hip
        (19, 20),  # left_hip -> left_knee
        (20, 21)  # left_knee -> left_ankle
    ]

    SKELETON = [[l[0] + 1, l[1] + 1] for l in LIMBS]

    def __init__(self, joints):
        if isinstance(joints,list):
            super().__init__(joints)


        if isinstance(joints, pd.DataFrame):
            initialized_joints = []
            for index,joint_row in joints.iterrows():
                initialized_joints.append(Joint(joint_row))
            super().__init__(initialized_joints)

    @property
    def invisible(self):
        # type: () -> bool
        """
        :return: True if all the joints of the pose are occluded, False otherwise
        """
        for j in self:
            if not j.occ:
                return False
        return True

    @property
    def bbox_2d(self):
        # type: () -> List[int]
        """
        :return: bounding box around the pose in format [x_min, y_min, width, height]
            - x_min = x of the top left corner of the bounding box
            - y_min = y of the top left corner of the bounding box
        """
        x_min = int(np.min([j.x2d for j in self]))
        y_min = int(np.min([j.y2d for j in self]))
        x_max = int(np.max([j.x2d for j in self]))
        y_max = int(np.max([j.y2d for j in self]))
        width = x_max - x_min
        height = y_max - y_min
        return [x_min, y_min, width, height]

    @property
    def coco_annotation(self):
        # type: () -> Dict
        """
        :return: COCO annotation dictionary of the pose
        ==========================================================
        NOTE#1: in COCO, each keypoint is represented by its (x,y)
        2D location and a visibility flag `v` defined as:
            - `v=0` ==> not labeled (in which case x=y=0)
            - `v=1` ==> labeled but not visible
            - `v=2` ==> labeled and visible
        ==========================================================
        NOTE#2: in COCO, a keypoint is considered visible if it
        falls inside the object segment. In JTA there are no
        object segments and every keypoint is labelled, so we
        v=2 for each keypoint.
        ==========================================================
        """
        keypoints = []
        for j in self:
            keypoints += [j.x2d, j.y2d, 2]
        annotation = {
            'keypoints': keypoints,
            'num_keypoints': len(self),
            'bbox': self.bbox_2d
        }
        return annotation


    @property
    def getPersonPosition(self):
        firstJoint: Joint = self[0]
        return None

    @property
    def getBoundingBoxPoints(self):
        """

        :returns: All four bounding box points starting at top left -> top right -> bottom right -> bottom left
        """
        firstJoint: Joint = self[0]

        resultBox = BoundingBoxTracked()
        resultBox.topLeft = (    int(firstJoint.x_top_left_BB)   ,   int(firstJoint.y_top_left_BB)   )

        resultBox.topRight = (   int(firstJoint.x_bottom_right_BB)  ,  int(firstJoint.y_top_left_BB)   )

        resultBox.bottomRight = (   int(firstJoint.x_bottom_right_BB)  ,  int(firstJoint.y_bottom_right_BB)   )

        resultBox.bottomLeft = (   int(firstJoint.x_top_left_BB)   , int(firstJoint.y_bottom_right_BB)    )

        return resultBox


    def drawTypeAndGlasses(self, image):

        firstJoint: Joint = self[0]

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = self.getBoundingBoxPoints.topLeft
        fontScale = 0.3
        fontColor = (0, 255, 0)
        lineType = 1

        cv2.putText(image, "pt: " + str(firstJoint.ped_type),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.putText(image, "wg: " + str(firstJoint.wears_glasses),
                    (bottomLeftCornerOfText[0],bottomLeftCornerOfText[1] - 9),
                    font,
                    fontScale,
                    fontColor,
                    lineType)

    def drawPersonId(self, image):

        firstJoint: Joint = self[0]

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = self.getBoundingBoxPoints.topLeft
        fontScale = 0.3
        fontColor = (0, 255, 0)
        lineType = 1

        cv2.putText(image, str(firstJoint.person_id),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)



    def drawPersonPosition(self, img, color):
        firstJoint: Joint = self[0]

        cv2.circle(
            img, thickness=2,
            center=firstJoint.personPosition,
            radius=firstJoint.radius,
            color=color,
        )


    def get_bbox_line_thickness(self):
        firstJoint: Joint = self[0]
        bbox_height = firstJoint.get_bounding_box_height()

        thickness = int(round(bbox_height / 100))
        return thickness if thickness > 0 else 1


    def drawBoundingBox(self, image, color):

        bbox_line_thickness = self.get_bbox_line_thickness()
        cv2.line(image, self.getBoundingBoxPoints.topLeft , self.getBoundingBoxPoints.bottomLeft, color=color, thickness=bbox_line_thickness)
        cv2.line(image, self.getBoundingBoxPoints.topLeft, self.getBoundingBoxPoints.topRight, color=color, thickness=bbox_line_thickness)
        cv2.line(image, self.getBoundingBoxPoints.topRight, self.getBoundingBoxPoints.bottomRight, color=color, thickness=bbox_line_thickness)
        cv2.line(image, self.getBoundingBoxPoints.bottomLeft, self.getBoundingBoxPoints.bottomRight, color=color, thickness=bbox_line_thickness)

    def drawPose(self, image, color):

        """
        :param image: image on which to draw the pose
        :param color: color of the limbs make up the pose
        :return: image with the pose
        """
        t = self.get_bbox_line_thickness()

        # draw limb(s) segments
        for (j_id_a, j_id_b) in Pose.LIMBS:
            joint_a = self[j_id_a]  # type: Joint
            joint_b = self[j_id_b]  # type: Joint

            if joint_a.is_on_screen and joint_b.is_on_screen:
                cv2.line(image, joint_a.pos2d, joint_b.pos2d, color=color, thickness=t)

        # draw joint(s) circles
        for joint in self:
            image = joint.draw(image)

        return image

    def __iter__(self):
        # type: () -> Iterator[Joint]
        return super().__iter__()
