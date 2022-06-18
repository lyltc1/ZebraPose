import os
import sys
import argparse
from bop_toolkit_lib import misc
from bop_toolkit_lib import transform as tr
sys.path.append("../zebrapose/tools_for_BOP")
import bop_io

import cv2
import Render
import numpy as np
from tqdm import tqdm

from modified_gt_pose import modified_gt_for_bowl


def generate_GT_images(bop_path, dataset_name, force_rewrite, is_training_data, data_folder, start_obj_id, end_obj_id):
    vis_path = os.path.join('/home/lyltc/git/BOP_DATASET', 'test_object_symmetries')
    vis_rgb_tpath = os.path.join('{vis_path}', '{dataset}', '{obj_id:06d}', '{view_id:06d}_{pose_id:06d}.jpg')
    views=[{'R': tr.rotation_matrix(0.5 * np.pi, [1, 0, 0]).dot(
                 tr.rotation_matrix(-0.5 * np.pi, [0, 0, 1])).dot(
                 tr.rotation_matrix(0.1 * np.pi, [0, 1, 0]))[:3, :3],
            't': np.array([[0, 0, 500]]).T },
           {'R': tr.rotation_matrix(0.2 * np.pi, [1, 0, 0]).dot(
               tr.rotation_matrix(-0.7 * np.pi, [0, 0, 1])).dot(
               tr.rotation_matrix(0.2 * np.pi, [0, 1, 0]))[:3, :3],
            't': np.array([[0, 0, 500]]).T},
           {'R': tr.rotation_matrix(0 * np.pi, [1, 0, 0]).dot(
               tr.rotation_matrix(0 * np.pi, [0, 0, 1])).dot(
               tr.rotation_matrix(0 * np.pi, [0, 1, 0]))[:3, :3],
            't': np.array([[0, 0, 500]]).T},
           {'R': tr.rotation_matrix(0.5 * np.pi, [1, 0, 0]).dot(
               tr.rotation_matrix(0 * np.pi, [0, 0, 1])).dot(
               tr.rotation_matrix(0 * np.pi, [0, 1, 0]))[:3, :3],
            't': np.array([[0, 0, 500]]).T},
           {'R': tr.rotation_matrix(0 * np.pi, [1, 0, 0]).dot(
               tr.rotation_matrix(0.5 * np.pi, [0, 0, 1])).dot(
               tr.rotation_matrix(0 * np.pi, [0, 1, 0]))[:3, :3],
            't': np.array([[0, 0, 500]]).T},
           {'R': tr.rotation_matrix(0 * np.pi, [1, 0, 0]).dot(
               tr.rotation_matrix(0 * np.pi, [0, 0, 1])).dot(
               tr.rotation_matrix(0.5 * np.pi, [0, 1, 0]))[:3, :3],
            't': np.array([[0, 0, 500]]).T}
           ]
    dataset_dir, source_dir, model_plys, model_info, model_ids, rgb_files, depth_files, mask_files, mask_visib_files, gts, gt_infos, cam_param_global, scene_cam = bop_io.get_dataset(
        bop_path, dataset_name, train=is_training_data, incl_param=True, data_folder=data_folder)

    target_dir = os.path.join(dataset_dir, data_folder + '_GT')

    im_width, im_height = cam_param_global['im_size']
    if dataset_name == 'tless':
        im_width = 720
        im_height = 540
        if data_folder == 'train_primesense':
            im_width = 400
            im_height = 400

    cam_K = cam_param_global['K']
    camera_parameters = np.array([im_width, im_height, cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]])
    print(camera_parameters)
    Render.init(camera_parameters, 1)
    model_scale = 0.1

    for model_to_render in range(start_obj_id, end_obj_id + 1):
        # only bind 1 model each time
        ply_fn = dataset_dir + "/models_GT_color/obj_{:06d}.ply".format(int(model_ids[model_to_render]))
        print("bind ", ply_fn, " to the render buffer position", 0)
        Render.bind_3D_model(ply_fn, 0, model_scale)

        poses = misc.get_symmetry_transformations(model_info[str(model_ids[model_to_render])], 0.04)
        for pose_id, pose in enumerate(poses):
            for view_id, view in enumerate(views):
                rot_pose = view['R'].dot(pose['R'])
                tra_pose = (view['R'].dot(pose['t']) + view['t']) * model_scale

                if dataset_name == 'ycbv' and model_to_render == 12:
                    rot_pose, tra_pose = modified_gt_for_bowl(rot_pose, tra_pose)
                rot_pose = rot_pose.flatten()

                vis_rgb_path = vis_rgb_tpath.format(
                    vis_path=vis_path, dataset=dataset_name, obj_id=model_to_render,
                    view_id=view_id, pose_id=pose_id)
                if not(os.path.exists(os.path.dirname(vis_rgb_path))):
                    os.makedirs(os.path.dirname(vis_rgb_path))
                Render.render_GT_visible_side(tra_pose, rot_pose, 0, vis_rgb_path)
    vis_path = os.path.join('/home/lyltc/git/BOP_DATASET', 'ori_object_symmetries')
    for model_to_render in range(start_obj_id, end_obj_id + 1):
        # only bind 1 model each time
        ply_fn = dataset_dir + "/models_GT_color/obj_{:06d}.ply".format(int(model_ids[model_to_render]))
        print("bind ", ply_fn, " to the render buffer position", 0)
        Render.bind_3D_model(ply_fn, 0, model_scale)

        poses = misc.get_symmetry_transformations(model_info[str(model_ids[model_to_render])], 0.04)
        for pose_id, pose in enumerate(poses):
            for view_id, view in enumerate(views):
                rot_pose = view['R'].dot(pose['R'])
                tra_pose = (view['R'].dot(pose['t']) + view['t']) * model_scale

                rot_pose = rot_pose.flatten()

                vis_rgb_path = vis_rgb_tpath.format(
                    vis_path=vis_path, dataset=dataset_name, obj_id=model_to_render,
                    view_id=view_id, pose_id=pose_id)
                if not(os.path.exists(os.path.dirname(vis_rgb_path))):
                    os.makedirs(os.path.dirname(vis_rgb_path))
                Render.render_GT_visible_side(tra_pose, rot_pose, 0, vis_rgb_path)
    Render.delete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate image labels for bop dataset')
    parser.add_argument('--bop_path', help='path to the bop folder', required=True)
    parser.add_argument('--dataset_name', help='the folder name of the dataset in the bop folder', required=True)
    parser.add_argument('--force_rewrite', choices=['True', 'False'], default='False', help='if rewrite the exist data',
                        required=True)
    parser.add_argument('--is_training_data', choices=['True', 'False'], default='True',
                        help='if is applied to training data ', required=True)
    parser.add_argument('--data_folder', help='which training data')
    parser.add_argument('--start_obj_id', help='start_obj_id')
    parser.add_argument('--end_obj_id', help='which training data')

    args = parser.parse_args()

    bop_path = args.bop_path
    dataset_name = args.dataset_name
    force_rewrite = args.force_rewrite == 'True'
    is_training_data = args.is_training_data == 'True'
    data_folder = args.data_folder
    start_obj_id = int(args.start_obj_id)
    end_obj_id = int(args.end_obj_id)

    generate_GT_images(bop_path, dataset_name, force_rewrite, is_training_data, data_folder, start_obj_id, end_obj_id)