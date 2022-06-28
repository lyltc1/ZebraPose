""" usage:
python test.py --cfg config/config_paper/ycbv/exp_ycbv_paper.txt --obj_name large_marker --ckpt_file /media/lyltc/mnt2/dataset/zebrapose/zebra_ckpts/paper/ycbv/large_marker
python test.py --cfg config/config_paper/ycbv/exp_ycbv_paper.txt --obj_name wood_block --ckpt_file /media/lyltc/mnt2/dataset/zebrapose/zebra_ckpts/paper/ycbv/wood_block --debug
"""

import os
import sys
import time

sys.path.insert(0, os.getcwd())

from config_parser import parse_cfg
import argparse
import cv2
from tqdm import tqdm


from tools_for_BOP import bop_io
from bop_dataset_pytorch import bop_dataset_single_obj_pytorch, get_roi

import torch
import numpy as np

from binary_code_helper.CNN_output_to_pose import load_dict_class_id_3D_points, CNN_outputs_to_object_pose
sys.path.append("../bop_toolkit")
from bop_toolkit_lib import inout

from model.BinaryCodeNet import BinaryCodeNet_Deeplab

from metric import Calculate_ADD_Error_BOP, Calculate_ADI_Error_BOP

from get_detection_results import get_detection_results, ycbv_select_keyframe, get_detection_scores
from common_ops import from_output_to_class_mask, from_output_to_class_binary_code
from tools_for_BOP.common_dataset_info import get_obj_info

from binary_code_helper.generate_new_dict import generate_new_corres_dict

from tools_for_BOP import write_to_cvs 


cur_dir = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(cur_dir, "..")


def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap


def compute_auc_posecnn(errors):
    # NOTE: Adapted from https://github.com/yuxng/YCB_Video_toolbox/blob/master/evaluate_poses_keyframe.m
    errors = errors.copy()
    d = np.sort(errors)
    d[d > 0.1] = np.inf
    accuracy = np.cumsum(np.ones(d.shape[0])) / d.shape[0]
    ids = np.isfinite(d)
    d = d[ids]
    accuracy = accuracy[ids]
    if len(ids) == 0 or ids.sum() == 0:
        return np.nan
    rec = d
    prec = accuracy
    mrec = np.concatenate(([0], rec, [0.1]))
    mpre = np.concatenate(([0], prec, [prec[-1]]))
    for i in np.arange(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.arange(1, len(mpre))
    ids = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = ((mrec[ids] - mrec[ids-1]) * mpre[ids]).sum() * 10
    return ap


def main(configs):
    #### training dataset
    bop_challange = configs['bop_challange']
    bop_path = configs['bop_path']
    obj_name = configs['obj_name']
    dataset_name = configs['dataset_name']
    training_data_folder=configs['training_data_folder']
    training_data_folder_2=configs['training_data_folder_2']
    test_folder=configs['test_folder']                                  # usually is 'test'
    second_dataset_ratio = configs['second_dataset_ratio']              # the percentage of second dataset in the batch
    num_workers = configs['num_workers']
    train_obj_visible_theshold = configs['train_obj_visible_theshold']  # for test is always 0.1, for training we can set different values, usually 0.2
    #### network settings
    BoundingBox_CropSize_image = configs['BoundingBox_CropSize_image']  # input image size
    BoundingBox_CropSize_GT = configs['BoundingBox_CropSize_GT']        # network output size
    BinaryCode_Loss_Type = configs['BinaryCode_Loss_Type']              # now only support "L1" or "BCE"
    output_kernel_size = configs['output_kernel_size']                  # last layer kernel size
    resnet_layer = configs['resnet_layer']                              # usually resnet 34
    concat=configs['concat_encoder_decoder']                   

    #### augmentations
    Detection_reaults=configs['Detection_reaults']                       # for the test, the detected bounding box provided by GDR Net
    padding_ratio=configs['padding_ratio']                               # pad the bounding box for training and test
    resize_method = configs['resize_method']                             # how to resize the roi images to 256*256
    use_peper_salt= configs['use_peper_salt']                            # if add additional peper_salt in the augmentation
    use_motion_blur= configs['use_motion_blur']                          # if add additional motion_blur in the augmentation
    # pixel code settings
    divide_number_each_itration = configs['divide_number_each_itration']
    number_of_itration = configs['number_of_itration']

    torch.manual_seed(0)      # the both are only good for ablation study
    np.random.seed(0)         # if can be removed in the final experiments

    calc_add_and_adi=False

    # get dataset informations
    dataset_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,cam_param_global, cam_params = bop_io.get_dataset(bop_path,dataset_name, train=True, data_folder=training_data_folder, data_per_obj=True, incl_param=True, train_obj_visible_theshold=train_obj_visible_theshold)
    obj_name_obj_id, symmetry_obj = get_obj_info(dataset_name)
    obj_id = int(obj_name_obj_id[obj_name] - 1) # now the obj_id started from 0
    if obj_name in symmetry_obj:
        Calculate_Pose_Error_Main = Calculate_ADI_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADD_Error_BOP
        main_metric_name = 'ADI'
        supp_metric_name = 'ADD'
    else:
        Calculate_Pose_Error_Main = Calculate_ADD_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADI_Error_BOP
        main_metric_name = 'ADD'
        supp_metric_name = 'ADI'
    
    mesh_path = model_plys[obj_id+1] # mesh_path is a dict, the obj_id should start from 1
    obj_diameter = model_info[str(obj_id+1)]['diameter']
    print("obj_diameter", obj_diameter)
    path_dict = os.path.join(dataset_dir, "models_GT_color", "Class_CorresPoint{:06d}.txt".format(obj_id+1))
    total_numer_class, _, _, dict_class_id_3D_points = load_dict_class_id_3D_points(path_dict)
    divide_number_each_itration = int(divide_number_each_itration)
    total_numer_class = int(total_numer_class)
    number_of_itration = int(number_of_itration)
    if divide_number_each_itration ** number_of_itration != total_numer_class:
        raise AssertionError("the combination is not valid")
    GT_code_infos = [divide_number_each_itration, number_of_itration, total_numer_class]

    if divide_number_each_itration != 2 and (BinaryCode_Loss_Type=='BCE' or BinaryCode_Loss_Type=='L1'):
        raise AssertionError("for non-binary case, use CE as loss function")
    if divide_number_each_itration == 2 and BinaryCode_Loss_Type=='CE':
        raise AssertionError("not support for now")

    vertices = inout.load_ply(mesh_path)["pts"]

    # define test data loader
    if not bop_challange:
        dataset_dir_test,_,_,_,_,test_rgb_files,_,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_, camera_params_test = bop_io.get_dataset(bop_path, dataset_name,train=False, data_folder=test_folder, data_per_obj=True, incl_param=True, train_obj_visible_theshold=train_obj_visible_theshold)
        if dataset_name == 'ycbv' and Detection_reaults != 'none':
            print("select key frames from ycbv test images")
            key_frame_index = ycbv_select_keyframe(Detection_reaults, test_rgb_files[obj_id])
            test_rgb_files_keyframe = [test_rgb_files[obj_id][i] for i in key_frame_index]
            test_mask_files_keyframe = [test_mask_files[obj_id][i] for i in key_frame_index]
            test_mask_visib_files_keyframe = [test_mask_visib_files[obj_id][i] for i in key_frame_index]
            test_gts_keyframe = [test_gts[obj_id][i] for i in key_frame_index]
            test_gt_infos_keyframe = [test_gt_infos[obj_id][i] for i in key_frame_index]
            camera_params_test_keyframe = [camera_params_test[obj_id][i] for i in key_frame_index]
            test_rgb_files[obj_id] = test_rgb_files_keyframe
            test_mask_files[obj_id] = test_mask_files_keyframe
            test_mask_visib_files[obj_id] = test_mask_visib_files_keyframe
            test_gts[obj_id] = test_gts_keyframe
            test_gt_infos[obj_id] = test_gt_infos_keyframe
            camera_params_test[obj_id] = camera_params_test_keyframe
    else:
        print("use BOP test images")
        dataset_dir_test,_,_,_,_,test_rgb_files,_,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_, camera_params_test = bop_io.get_bop_challange_test_data(bop_path, dataset_name, target_obj_id=obj_id+1, data_folder=test_folder)

    if Detection_reaults != 'none':
        Det_Bbox = get_detection_results(Detection_reaults, test_rgb_files[obj_id], obj_id+1, 0)
        scores = get_detection_scores(Detection_reaults, test_rgb_files[obj_id], obj_id+1, 0)
    else:
        Det_Bbox = None

    test_dataset = bop_dataset_single_obj_pytorch(
                                            dataset_dir_test, test_folder, test_rgb_files[obj_id], test_mask_files[obj_id], test_mask_visib_files[obj_id], 
                                            test_gts[obj_id], test_gt_infos[obj_id], camera_params_test[obj_id], False, 
                                            BoundingBox_CropSize_image, BoundingBox_CropSize_GT, GT_code_infos, 
                                            padding_ratio=padding_ratio, resize_method=resize_method, Detect_Bbox=Det_Bbox,
                                            use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur
                                        )
    print("test image example:", test_rgb_files[obj_id][0], flush=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    binary_code_length = number_of_itration
    print("predicted binary_code_length", binary_code_length)
    configs['binary_code_length'] = binary_code_length
 
    net = BinaryCodeNet_Deeplab(
                num_resnet_layers=resnet_layer, 
                concat=concat, 
                binary_code_length=binary_code_length, 
                divided_number_each_iteration = divide_number_each_itration, 
                output_kernel_size = output_kernel_size
            )

    if torch.cuda.is_available():
        net=net.cuda()

    checkpoint = torch.load( configs['checkpoint_file'] )
    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()

    #test with test data
    debug_image_root = os.path.abspath(
        os.path.join(PROJ_ROOT, ".cache/{}_{}_{}/refine_by_{}_{}".
                     format(time.strftime('%Y-%m-%d %H:%M:%S'), test_folder, obj_name,
                     configs.get('refine_entire_mask_type'), configs.get('refine_mask_type'))))

    ADX_passed=np.zeros(len(test_loader.dataset))
    ADX_passed_5 = np.zeros(len(test_loader.dataset))
    ADX_passed_2 = np.zeros(len(test_loader.dataset))
    ADX_error=np.zeros(len(test_loader.dataset))
    AUC_ADX_error=np.zeros(len(test_loader.dataset))
    if calc_add_and_adi:
        ADY_passed=np.zeros(len(test_loader.dataset))
        ADY_error=np.zeros(len(test_loader.dataset))
        AUC_ADY_error=np.zeros(len(test_loader.dataset))

    print("test dataset")
    print(len(test_loader.dataset))

    ignore_bit = configs['ignore_bit']
    if ignore_bit!=0:
        new_dict_class_id_3D_points = generate_new_corres_dict(dict_class_id_3D_points, 16, 16-ignore_bit)
    
    img_ids = []
    scene_ids = []
    estimated_Rs = []
    estimated_Ts = []
    for rgb_fn in test_rgb_files[obj_id]:
        rgb_fn = rgb_fn.split("/")
        scene_id = rgb_fn[-3]
        img_id = rgb_fn[-1].split(".")[0]
        img_ids.append(img_id)
        scene_ids.append(scene_id)

    for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, class_code_images, cam_Ks) in enumerate(tqdm(test_loader)):
        if torch.cuda.is_available():
            data=data.cuda()
            masks = masks.cuda()
            class_code_images = class_code_images.cuda()

        pred_mask_prob, pred_code_prob = net(data)

        pred_masks = from_output_to_class_mask(pred_mask_prob)
        pred_code_images = from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type, divided_num_each_interation=divide_number_each_itration, binary_code_length=binary_code_length)
       
        # from binary code to pose
        pred_code_images = pred_code_images.transpose(0, 2, 3, 1)

        pred_masks = pred_masks.transpose(0, 2, 3, 1)
        pred_masks = pred_masks.squeeze(axis=-1).astype('uint8')

        Rs = Rs.detach().cpu().numpy()
        ts = ts.detach().cpu().numpy()
        Bboxes = Bboxes.detach().cpu().numpy()

        class_code_images = class_code_images.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(axis=0).astype('uint8')
        
        for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
            if ignore_bit!=0:
                R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_code_images[counter][:,:,:-ignore_bit],
                                                                            Bbox, BoundingBox_CropSize_GT, divide_number_each_itration, new_dict_class_id_3D_points, 
                                                                            intrinsic_matrix=cam_K)
            else:
                R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_code_images[counter], 
                                                                            Bbox, BoundingBox_CropSize_GT, divide_number_each_itration, dict_class_id_3D_points, 
                                                                            intrinsic_matrix=cam_K)

            if success:     
                if configs.get('refine', False):
                    ### refine code
                    gt_entire_mask = entire_masks[counter].cpu().numpy().astype('uint8')
                    gt_mask = masks[counter].cpu().numpy().astype('uint8')
                    pre_mask = pred_masks[counter]
                    entire_mask = gt_entire_mask
                    pre_entire_mask = gt_entire_mask
                    mask = None
                    if configs.get('refine_mask_type') == 'pre_mask':
                        mask = pre_mask
                    elif configs.get('refine_mask_type') == 'gt_mask':
                        mask = gt_mask

                    #################################
                    ##### visualize for debugging
                    #################################
                    if debug:
                        debug_image_dir = os.path.join(debug_image_root,
                                                       "image_{}_{}".format(scene_ids[batch_idx], img_ids[batch_idx]))
                        print("visualize for debugging_{}_{}".format(scene_ids[batch_idx],img_ids[batch_idx]))
                        if not os.path.exists(debug_image_dir):
                            os.makedirs(debug_image_dir)
                    contours, _ = cv2.findContours(entire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    visible_contour = np.empty(shape=(0, 2))
                    for contour in contours:
                        for i in range(len(contour)):
                            x, y = contour[i, 0]  # [x,y]
                            if np.any(mask[y - 1:y + 1, x - 1:x + 1]) and x > 0 and y > 0 and x < entire_mask.shape[1] and y < entire_mask.shape[0]:
                                visible_contour = np.append(visible_contour, contour[i], axis=0)
                    visible_contour = mapping_pixel_position_to_original_position(visible_contour, Bbox, BoundingBox_CropSize_GT)
                    add_err_before_refine = Calculate_Pose_Error_Main(r_GT, t_GT, R_predict, t_predict, vertices)
                    R_predict_refine, t_predict_refine = py_edge_refine(R_predict, t_predict / 1000., visible_contour, mesh_path,
                                                          debug_image_dir)
                    t_predict_refine = t_predict_refine.reshape(3, 1) * 1000.

                    if debug:
                        renderer = Renderer(model_plys.values(),
                                            vertex_tmp_store_folder=os.path.join(PROJ_ROOT, ".cache"),
                                            vertex_scale=0.001)
                        predict_img, predict_depth = renderer.render(obj_id, 640, 480, cam_K, R_predict, t_predict / 1000., 0.25, 6.0)
                        predict_img = get_roi(predict_img, Bbox, BoundingBox_CropSize_GT, interpolation=cv2.INTER_LINEAR, resize_method=resize_method)
                        predict_depth = get_roi(predict_depth, Bbox, BoundingBox_CropSize_GT, interpolation=cv2.INTER_LINEAR, resize_method=resize_method)
                        predict_depth = (predict_depth > 0).astype("uint8")

                        predict_img_refine, predict_depth_refine = renderer.render(obj_id, 640, 480, cam_K, R_predict_refine, t_predict_refine / 1000., 0.25, 6.0)
                        predict_img_refine = get_roi(predict_img_refine, Bbox, BoundingBox_CropSize_GT, interpolation=cv2.INTER_LINEAR,resize_method=resize_method)
                        predict_depth_refine = get_roi(predict_depth_refine, Bbox, BoundingBox_CropSize_GT, interpolation=cv2.INTER_LINEAR,resize_method=resize_method)
                        predict_depth_refine = (predict_depth_refine > 0).astype("uint8")

                        mean = torch.as_tensor((0.485, 0.456, 0.406), dtype=data.dtype, device=data.device)
                        std = torch.as_tensor((0.229, 0.224, 0.225), dtype=data.dtype, device=data.device)
                        if mean.ndim == 1:
                            mean = mean.view(-1, 1, 1)
                        if std.ndim == 1:
                            std = std.view(-1, 1, 1)
                        x = data.mul(std).add(mean)
                        x = x.detach().cpu().numpy()
                        x = (x.transpose(0, 2, 3, 1)[counter]*255.).astype('uint8')
                        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
                        x = cv2.resize(x, (128, 128))

                        pre_mask_contour, _ = cv2.findContours(pre_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        x_pre_mask_contour = x.copy()
                        cv2.drawContours(x_pre_mask_contour, pre_mask_contour, -1, (0, 0, 255), 1)

                        pre_entire_mask_contour, _ = cv2.findContours(pre_entire_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        x_pre_entire_mask_contour = x.copy()
                        cv2.drawContours(x_pre_entire_mask_contour, pre_entire_mask_contour, -1, (0, 0, 255), 1)

                        predict_depth_contour, _ = cv2.findContours(predict_depth, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        predict_depth_refine_contour, _ = cv2.findContours(predict_depth_refine, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        x_predict_depth_before_and_after_contour = x.copy()
                        cv2.drawContours(x_predict_depth_before_and_after_contour, predict_depth_contour, -1, (0, 0, 255), 1)
                        cv2.drawContours(x_predict_depth_before_and_after_contour, predict_depth_refine_contour, -1, (255, 0, 0), 1)


                        gt_mask_contour, _ = cv2.findContours(gt_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        x_gt_mask_contour = x.copy()
                        cv2.drawContours(x_gt_mask_contour, gt_mask_contour, -1, (0, 0, 255), 1)

                        gt_entire_mask_contour, _ = cv2.findContours(gt_entire_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        x_gt_entire_mask_contour = x.copy()
                        cv2.drawContours(x_gt_entire_mask_contour, gt_entire_mask_contour, -1, (255, 0, 0), 1)

                        show_ims = [x,
                                    x_pre_mask_contour,
                                    x_pre_entire_mask_contour,
                                    x_predict_depth_before_and_after_contour,
                                    predict_img[:, :, [2, 1, 0]],
                                    predict_img_refine[:, :, [2, 1, 0]],
                                    x_gt_mask_contour,
                                    x_gt_entire_mask_contour]
                        show_titles = ["image",
                                       "predict_visible_mask",
                                       "predict_entire_mask",
                                       "predict_pose_mask_before(b)_and_after(r)_refine",
                                       "predict_pose_before_refine",
                                       "predict_pose_refine",
                                       "groundTruth_visible_mask",
                                       "groundTruth_entire_mask"]
                        grid_show(show_ims, show_titles, row=2, col=4, save_path=os.path.join(debug_image_dir, "debug_img.jpg"))

                        right_bit_code_images = np.zeros((class_code_images.shape[0], class_code_images.shape[1]))
                        for i in range(8):
                            right_bit_code_images = right_bit_code_images + (class_code_images[:, :, i] == pred_code_images[counter][:, :, i]) * (
                                        2 ** (8 - 1 - i))

                        show_ims = [x,
                                    class_code_images[:, :, 0: 3]*255,
                                    pred_code_images[counter][:, :, 0: 3]*255,
                                    class_code_images[:, :, 1: 4] * 255,
                                    pred_code_images[counter][:, :, 1: 4] * 255,
                                    class_code_images[:, :, 3: 6] * 255,
                                    pred_code_images[counter][:, :, 3: 6] * 255,
                                    right_bit_code_images
                                    ]
                        show_titles = ["image",
                                       "groundTruth_code_images[0:3]",
                                       "pred_code_images[0:3]",
                                       "groundTruth_code_images[1:4]",
                                       "pred_code_images[1:4]",
                                       "groundTruth_code_images[3:6]",
                                       "pred_code_images[3:6]",
                                       "right_bit_code_images"
                                       ]
                        grid_show(show_ims, show_titles, row=2, col=4, save_path=os.path.join(debug_image_dir, "code_img.jpg"))

                        right_bit_code_images = (class_code_images == pred_code_images[counter])
                        for i in range(1,16):
                            right_bit_code_images[:,:,i] = np.logical_and(right_bit_code_images[:,:,i-1], right_bit_code_images[:,:,i])
                        show_ims = []
                        show_titles = []
                        for i in range(16):
                            show_ims.append(right_bit_code_images[:, :, i]*255)
                            show_titles.append("right_bit_code_images" + str(i))
                        grid_show(show_ims, show_titles, row=4, col=4, save_path=os.path.join(debug_image_dir, "per_code_img.jpg"))

                    R_predict = R_predict_refine
                    t_predict = t_predict_refine

                    add_err_after_refine = Calculate_Pose_Error_Main(r_GT, t_GT, R_predict, t_predict, vertices)
                    if debug:
                        with open(os.path.join(debug_image_dir,"add_err.txt"), "w") as f:
                            f.write("add_err_before_refine\n")
                            f.write(str(add_err_before_refine))
                            f.write("\nadd_err_after_refine\n")
                            f.write(str(add_err_after_refine))
                            f.write("\nobject diameter\n")
                            f.write(str(obj_diameter))
                estimated_Rs.append(R_predict)
                estimated_Ts.append(t_predict)
            else:
                R_ = np.zeros((3,3))
                R_[0,0] = 1
                R_[1,1] = 1
                R_[2,2] = 1
                estimated_Rs.append(R_)
                estimated_Ts.append(np.zeros((3,1)))

            adx_error = 10000
            if success:
                adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, R_predict, t_predict, vertices)
                if np.isnan(adx_error):
                    adx_error = 10000
                    
            if adx_error < obj_diameter*0.1:
                ADX_passed[batch_idx] = 1
            if adx_error < obj_diameter*0.05:
                ADX_passed_5[batch_idx] = 1
            if adx_error < obj_diameter*0.02:
                ADX_passed_2[batch_idx] = 1
            ADX_error[batch_idx] = adx_error
            th = np.linspace(10, 100, num=10)
            sum_correct = 0
            for t in th:
                if adx_error < t:
                    sum_correct = sum_correct + 1
            AUC_ADX_error[batch_idx] = sum_correct/10
           
            if calc_add_and_adi:
                ady_error = 10000
                if success:
                    ady_error = Calculate_Pose_Error_Supp(r_GT, t_GT, R_predict, t_predict, vertices)
                    if np.isnan(ady_error):
                        ady_error = 10000
                if ady_error < obj_diameter*0.1:
                    ADY_passed[batch_idx] = 1
                ADY_error[batch_idx] = ady_error
               
                th = np.linspace(10, 100, num=10)
                sum_correct = 0
                for t in th:
                    if ady_error < t:
                        sum_correct = sum_correct + 1
                AUC_ADY_error[batch_idx] = sum_correct/10

    scores = [1 for x in range(len(estimated_Rs))]
    cvs_path = os.path.join(eval_output_path, 'pose_result_bop/')
    if not os.path.exists(cvs_path):
        os.makedirs(cvs_path)

    write_to_cvs.write_cvs(cvs_path, "{}_{}".format(dataset_name, obj_name), obj_id+1, scene_ids, img_ids, estimated_Rs, estimated_Ts, scores)

    ADX_passed = np.mean(ADX_passed)
    ADX_passed_5 = np.mean(ADX_passed_5)
    ADX_passed_2 = np.mean(ADX_passed_2)
    ADX_error_mean= np.mean(ADX_error)
    AUC_ADX_error = np.mean(AUC_ADX_error)
    print('{}/{}'.format(main_metric_name,main_metric_name), ADX_passed)
    print('AUC_{}/{}'.format(main_metric_name,main_metric_name), AUC_ADX_error)
    AUC_ADX_error_posecnn = compute_auc_posecnn(ADX_error/1000.)
    print('AUC_posecnn_{}/{}'.format(main_metric_name,main_metric_name), AUC_ADX_error_posecnn)

    if calc_add_and_adi:
        ADY_passed = np.mean(ADY_passed)
        ADY_error_mean= np.mean(ADY_error)
        AUC_ADY_error = np.mean(AUC_ADY_error)
        print('{}/{}'.format(supp_metric_name,supp_metric_name), ADY_passed)
        print('AUC_{}/{}'.format(supp_metric_name,supp_metric_name), AUC_ADY_error)
        AUC_ADY_error_posecnn = compute_auc_posecnn(ADY_error/1000.)
        print('AUC_posecnn_{}/{}'.format(supp_metric_name,supp_metric_name), AUC_ADY_error_posecnn)

    ####save results to file
    path = os.path.join(eval_output_path, "ADD_result/")
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + "{}_{}".format(dataset_name, obj_name) + ".txt" 
    #path = path + dataset_name + obj_name  + "ignorebit_" + str(configs['ignore_bit']) + ".txt"
    #path = path + dataset_name + obj_name + "radix" + "_" + str(divide_number_each_itration)+"_"+str(number_of_itration) + ".txt"
    print('save ADD results to', path)
    print(path)
    f = open(path, "w")
    f.write('{}_10/{} '.format(main_metric_name,main_metric_name))
    f.write(str(ADX_passed.item()))
    f.write('\n')
    f.write('{}_5/{} '.format(main_metric_name,main_metric_name))
    f.write(str(ADX_passed_5.item()))
    f.write('\n')
    f.write('{}_2/{} '.format(main_metric_name,main_metric_name))
    f.write(str(ADX_passed_2.item()))
    f.write('\n')

    f.write('AUC_{}/{} '.format(main_metric_name,main_metric_name))
    f.write(str(AUC_ADX_error.item()))
    f.write('\n')
    f.write('AUC_posecnn_{}/{} '.format(main_metric_name,main_metric_name))
    f.write(str(AUC_ADX_error_posecnn.item()))
    f.write('\n')

    f.close()
    ####

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    parser.add_argument('--cfg', type=str) # config file
    parser.add_argument('--obj_name', type=str)
    parser.add_argument('--ckpt_file', type=str)
    parser.add_argument('--ignore_bit', default='0', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    config_file = args.cfg
    checkpoint_file = args.ckpt_file
    obj_name = args.obj_name
    debug = args.debug
    configs = parse_cfg(config_file)

    configs['obj_name'] = obj_name
    if configs['test_folder'] != 'test':
        configs['Detection_reaults'] = 'none'
    if configs['Detection_reaults'] != 'none':
        Detection_reaults = configs['Detection_reaults']
        dirname = os.path.dirname(__file__)
        Detection_reaults = os.path.join(dirname, Detection_reaults)
        configs['Detection_reaults'] = Detection_reaults

    configs['checkpoint_file'] = checkpoint_file
    eval_output_path = os.path.join(configs['eval_output_path'], time.strftime('%Y-%m-%d %H:%M:%S'))
    configs['eval_output_path'] = eval_output_path
    configs['ignore_bit'] = int(args.ignore_bit)
    if not os.path.exists(eval_output_path):
        os.makedirs(eval_output_path)
    with open(os.path.join(eval_output_path, 'config.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in configs.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    f = open(os.path.join(eval_output_path, 'log.txt'), 'a')
    sys.stdout = f
    sys.stderr = f  # redirect std err, if necessary

    if configs.get('refine', False):
        from binary_code_helper.CNN_output_to_pose import mapping_pixel_position_to_original_position

    if debug:
        from edge_refine.build.examples.edge_refine_debug import py_edge_refine
        from vis_util.image import grid_show
        from lib.meshrenderer.meshrenderer_phong import Renderer
    else:
        from edge_refine.build.examples.edge_refine import py_edge_refine
    main(configs)