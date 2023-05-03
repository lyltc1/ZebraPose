""" modify gt to minimize |RS-I| """

import numpy as np


def modified_gt_for_symmetry(rot_pose, tra_pose, model_info):
    if 'symmetries_continuous' in model_info and 'symmetries_discrete' in model_info:
        ''' TODO this condition hasn't been tested yet '''
        #### step1: verify the case can be solved by the code
        assert len(model_info['symmetries_continuous']) == 1
        assert model_info['symmetries_continuous'][0]['axis'] == [0, 0, 1]
        assert model_info['symmetries_continuous'][0]['offset'] == [0, 0, 0]
        #### step2: store the symmetries_descrete in trans_discs
        trans_discs = [{'R': np.eye(3), 't': np.array([[0, 0, 0]]).T}]  # Identity.
        for sym in model_info['symmetries_discrete']:
            sym_4x4 = np.reshape(sym, (4, 4))
            R = sym_4x4[:3, :3]
            t = sym_4x4[:3, 3].reshape((3, 1))
            trans_discs.append({'R': R, 't': t})
        ### step3: store the best possible poses for every descrete symmetries counterpart in pose_tmps
        ### (considering descrete symmetries and continuous symmetries)
        pose_tmps = []
        for trans_disc in trans_discs:
            tra_pose_tmp = rot_pose.dot(trans_disc['t']) + tra_pose
            rot_pose_tmp = rot_pose.dot(trans_disc['R'])
            R11 = rot_pose_tmp[0, 0]
            R12 = rot_pose_tmp[0, 1]
            R21 = rot_pose_tmp[1, 0]
            R22 = rot_pose_tmp[1, 1]
            theta = np.arctan((R12 - R21) / (R11 + R22))
            if not np.sin(theta) * (R21 - R12) < np.cos(theta) * (R11 + R22):
                theta = theta + np.pi
            S = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
            tra_pose_tmp = rot_pose_tmp.dot(np.array([[0.], [0.], [0.]])) + tra_pose_tmp
            rot_pose_tmp = rot_pose_tmp.dot(S)
            pose_tmps.append({'R': rot_pose_tmp, 't': tra_pose_tmp})
        ### step4: choose best from pose_tmps
        best_pose = None
        froebenius_norm = 1e8
        for pose_tmp in pose_tmps:
            tmp_froebenius_norm = np.linalg.norm(pose_tmp['R'] - np.eye(3))
            if tmp_froebenius_norm < froebenius_norm:
                froebenius_norm = tmp_froebenius_norm
                best_pose = pose_tmp
        tra_pose = best_pose['t']
        rot_pose = best_pose['R']

    elif 'symmetries_continuous' in model_info:
        # currently not support for both discrete and continuous symmetry
        assert 'symmetries_discrete' not in model_info
        # currently not support for multi symmetries continuous
        assert len(model_info['symmetries_continuous']) == 1
        sym = model_info['symmetries_continuous'][0]
        if sym['axis'] == [0, 0, 1] and sym['offset'] == [0, 0, 0]:
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
            tra_pose = rot_pose.dot(np.array([[0.], [0.], [0.]])) + tra_pose
            rot_pose = rot_pose.dot(S)
        elif sym['axis'] == [0, 1, 0] and sym['offset'] == [0, 0, 0]:
            R11 = rot_pose[0, 0]
            R13 = rot_pose[0, 2]
            R31 = rot_pose[2, 0]
            R33 = rot_pose[2, 2]
            theta = np.arctan((R31 - R13) / (R11 + R33))
            if not np.sin(theta) * (R13 - R31) < np.cos(theta) * (R11 + R33):
                theta = theta + np.pi
            S = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])
            tra_pose = rot_pose.dot(np.array([[0.], [0.], [0.]])) + tra_pose
            rot_pose = rot_pose.dot(S)
        elif sym['axis'] == [1, 0, 0] and sym['offset'] == [0, 0, 0]:
            R22 = rot_pose[1, 1]
            R23 = rot_pose[1, 2]
            R32 = rot_pose[2, 1]
            R33 = rot_pose[2, 2]
            theta = np.arctan((R32 - R23) / (R22 + R33))
            if not (R22 + R33) * np.cos(theta) + (R32 - R23) * np.sin(theta) > 0:
                theta = theta + np.pi
            S = np.array([[1, 0, 0],
                          [0, np.cos(theta), np.sin(theta)],
                          [0, -np.sin(theta), np.cos(theta)]])
            tra_pose = rot_pose.dot(np.array([[0.], [0.], [0.]])) + tra_pose
            rot_pose = rot_pose.dot(S)
        else:
            raise NotImplementedError
    elif 'symmetries_discrete' in model_info:
        # currently not support for both discrete and continuous symmetry
        assert 'symmetries_continuous' not in model_info
        trans_disc = [{'R': np.eye(3), 't': np.array([[0, 0, 0]]).T}]  # Identity.
        for sym in model_info['symmetries_discrete']:
            sym_4x4 = np.reshape(sym, (4, 4))
            R = sym_4x4[:3, :3]
            t = sym_4x4[:3, 3].reshape((3, 1))
            trans_disc.append({'R': R, 't': t})
        best_R = None
        best_t = None
        froebenius_norm = 1e8
        for sym in trans_disc:
            R = sym['R']
            t = sym['t']
            tmp_froebenius_norm = np.linalg.norm(rot_pose.dot(R)-np.eye(3))
            if tmp_froebenius_norm < froebenius_norm:
                froebenius_norm = tmp_froebenius_norm
                best_R = R
                best_t = t
        tra_pose = rot_pose.dot(best_t) + tra_pose
        rot_pose = rot_pose.dot(best_R)
    elif (not 'symmetries_discrete' in model_info) and (not 'symmetries_continuous' in model_info):
        return rot_pose, tra_pose
    else:
        raise NotImplementedError
    return rot_pose, tra_pose
