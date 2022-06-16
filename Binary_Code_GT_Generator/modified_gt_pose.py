import numpy as np


def modified_gt_for_bowl(rot_pose, tra_pose):
    R11 = rot_pose[0, 0]
    R12 = rot_pose[0, 1]
    R21 = rot_pose[1, 0]
    R22 = rot_pose[1, 1]

    theta = np.arctan((R12 - R21) / (R11 + R22))
    if not np.sin(theta) * (R21 - R12) < np.cos(theta) * (R11 + R22):
        theta = theta + np.pi
    S = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    rot_pose = rot_pose.dot(S)
    tra_pose = rot_pose.dot(np.array([[0.],[0.],[0.]])) + tra_pose

    return rot_pose, tra_pose
