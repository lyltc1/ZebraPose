""" train_v5 with GT_v1 and net_v3 and distributed training """
""" CUDA_VISIBLE_DEVICES=2 python -u train_v5.py --cfg config/config_BOP/tless/exp_tless_BOP.txt --obj_name obj03 > tmp_danka.txt """
""" CUDA_VISIBLE_DEVICES=0,1,2 python -u train_v5.py --cfg config/config_BOP/tless/exp_tless_BOP.txt --obj_name obj03 --multiprocessing-distributed --world-size 1 --rank 0 > tmp_sanka.txt """

import os
import sys
import time

sys.path.insert(0, os.getcwd())

from config_parser import parse_cfg
import argparse

from tools_for_BOP import bop_io
from tools_for_BOP.common_dataset_info import get_obj_info
from bop_dataset_pytorch import bop_dataset_single_obj_pytorch

import torch
from torch import optim
import numpy as np

from binary_code_helper.CNN_output_to_pose import load_dict_class_id_3D_points, CNN_outputs_to_object_pose

sys.path.append("../bop_toolkit")
from bop_toolkit_lib import inout
from model.BinaryCodeNet_v3 import BinaryCodeNet_Deeplab_v3
from model.BinaryCodeNet import MaskLoss, BinaryCodeLoss

from torch.utils.tensorboard import SummaryWriter

from utils_v2 import save_checkpoint, get_checkpoint, save_best_checkpoint
from metric import Calculate_ADD_Error_BOP, Calculate_ADI_Error_BOP

from get_detection_results import get_detection_results, ycbv_select_keyframe

from common_ops import from_output_to_class_mask, from_output_to_class_binary_code, get_batch_size

from test_network_with_test_data_v2 import test_network_with_single_obj

# dist-related 
import torch.multiprocessing as mp
import torch.distributed as dist

def main(gpu, configs, args):
    # dist-related
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    config_file_name = configs['config_file_name']
    #### training dataset
    bop_challange = configs['bop_challange']
    bop_path = configs['bop_path']
    obj_name = configs['obj_name']
    dataset_name = configs['dataset_name']
    training_data_folder=configs['training_data_folder']
    training_data_folder_2=configs['training_data_folder_2']
    val_folder=configs['val_folder']                                  # usually is 'test'
    second_dataset_ratio = configs['second_dataset_ratio']              # the percentage of second dataset in the batch
    num_workers = configs['num_workers']                                # for data loader
    train_obj_visible_theshold = configs['train_obj_visible_theshold']  # for test is always 0.1, for training we can set different values
    #### network settings
    BoundingBox_CropSize_image = configs['BoundingBox_CropSize_image']  # input image size
    BoundingBox_CropSize_GT = configs['BoundingBox_CropSize_GT']        # network output size
    BinaryCode_Loss_Type = configs['BinaryCode_Loss_Type']              # now only support "L1" or "BCE"
    mask_binary_code_loss=configs['mask_binary_code_loss']          # if binary code loss only applied for object mask region
    use_histgramm_weighted_binary_loss = configs['use_histgramm_weighted_binary_loss']
    output_kernel_size = configs['output_kernel_size']                  # last layer kernel size
    resnet_layer = configs['resnet_layer']                              # usually resnet 34
    concat=configs['concat_encoder_decoder']  
    predict_entire_mask=configs['predict_entire_mask']                  # if predict the entire object part rather than the visible one
    #### check points
    load_checkpoint = configs['load_checkpoint']
    tensorboard_path = configs['tensorboard_path']
    check_point_path = configs['check_point_path']
    total_iteration = configs['total_iteration']                         # train how many steps
    if args.distributed:
        total_iteration = total_iteration // args.world_size
        configs['total_iteration'] = total_iteration
    #### optimizer
    optimizer_type = configs['optimizer_type']                           # Adam is the best sofar
    batch_size=configs['batch_size']                                     # 32 is the best so far, set to 16 for debug in local machine
    learning_rate = configs['learning_rate']                             # 0.002 or 0.003 is the best so far
    if args.distributed:
        learning_rate = learning_rate * args.world_size
        configs['learning_rate'] = learning_rate
    binary_loss_weight = configs['binary_loss_weight']                     # 3 is the best so far
    #### augmentations
    Detection_reaults=configs['Detection_reaults']                       # for the test, the detected bounding box provided by GDR Net
    padding_ratio=configs['padding_ratio']                               # pad the bounding box for training and test
    resize_method = configs['resize_method']                             # how to resize the roi images to 256*256
    use_peper_salt= configs['use_peper_salt']                            # if add additional peper_salt in the augmentation
    use_motion_blur= configs['use_motion_blur']                          # if add additional motion_blur in the augmentation
    # vertex code settings
    divide_number_each_itration = configs['divide_number_each_itration']
    number_of_itration = configs['number_of_itration']
    #print the configurations
    if args.rank==0 or args.rank==-1:
        for key in configs:
            print("configs[", key, "]: ", configs[key], flush=True)
        print(args)


    # get dataset informations
    dataset_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,cam_param_global, cam_params = bop_io.get_dataset(bop_path,dataset_name, train=True, data_folder=training_data_folder, data_per_obj=True, incl_param=True, train_obj_visible_theshold=train_obj_visible_theshold)
    obj_name_obj_id, symmetry_obj = get_obj_info(dataset_name)
    obj_id = int(obj_name_obj_id[obj_name] - 1)    # now the obj_id started from 0
    if obj_name in symmetry_obj:
        Calculate_Pose_Error = Calculate_ADI_Error_BOP
    else:
        Calculate_Pose_Error = Calculate_ADD_Error_BOP
    mesh_path = model_plys[obj_id+1]         # mesh_path is a dict, the obj_id should start from 1
    if args.rank==0 or args.rank==-1:
        print(mesh_path, flush=True)
    obj_diameter = model_info[str(obj_id+1)]['diameter']
    if args.rank==0 or args.rank==-1:
        print("obj_diameter", obj_diameter, flush=True)
    path_dict = os.path.join(dataset_dir, "models_GT_color", "Class_CorresPoint{:06d}.txt".format(obj_id+1))
    total_numer_class, _, _, dict_class_id_3D_points = load_dict_class_id_3D_points(path_dict)
    divide_number_each_itration = int(divide_number_each_itration)
    total_numer_class = int(total_numer_class)
    number_of_itration = int(number_of_itration)
    if divide_number_each_itration ** number_of_itration != total_numer_class:
        raise AssertionError("the combination is not valid")
    if BoundingBox_CropSize_image / BoundingBox_CropSize_GT != 2:
        raise AssertionError("currnet endoder-decoder only support input_size/output_size = 2")
    GT_code_infos = [divide_number_each_itration, number_of_itration, total_numer_class]

    if divide_number_each_itration != 2 and (BinaryCode_Loss_Type=='BCE' or BinaryCode_Loss_Type=='L1'):
        raise AssertionError("for non-binary case, use CE as loss function")
    if divide_number_each_itration == 2 and BinaryCode_Loss_Type=='CE':
        raise AssertionError("not support for now")

    vertices = inout.load_ply(mesh_path)["pts"]

    ########################## define data loader
    batch_size_1_dataset, batch_size_2_dataset = get_batch_size(second_dataset_ratio, batch_size)

    train_dataset = bop_dataset_single_obj_pytorch(
                                                    dataset_dir, training_data_folder, rgb_files[obj_id], mask_files[obj_id], mask_visib_files[obj_id], 
                                                    gts[obj_id], gt_infos[obj_id], cam_params[obj_id], True, BoundingBox_CropSize_image, 
                                                    BoundingBox_CropSize_GT, GT_code_infos,  padding_ratio=padding_ratio, resize_method=resize_method, 
                                                    use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur
                                                    )
    if args.rank==0 or args.rank==-1:
        print("training_data_folder image example:", rgb_files[obj_id][0], flush=True)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if training_data_folder_2 != 'none':
        dataset_dir_pbr,_,_,_,_,rgb_files_pbr,_,mask_files_pbr,mask_visib_files_pbr,gts_pbr,gt_infos_pbr,_, camera_params_pbr = bop_io.get_dataset(bop_path, dataset_name, train=True, data_folder=training_data_folder_2, data_per_obj=True, incl_param=True, train_obj_visible_theshold=train_obj_visible_theshold)
        train_dataset_2 = bop_dataset_single_obj_pytorch(
                                                        dataset_dir_pbr, training_data_folder_2, rgb_files_pbr[obj_id], mask_files_pbr[obj_id], mask_visib_files_pbr[obj_id], 
                                                        gts_pbr[obj_id], gt_infos_pbr[obj_id], camera_params_pbr[obj_id], True, 
                                                        BoundingBox_CropSize_image, BoundingBox_CropSize_GT, GT_code_infos, 
                                                        padding_ratio=padding_ratio, resize_method=resize_method,
                                                        use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur
                                                    )
        if args.rank==0 or args.rank==-1:
            print("training_data_folder_2 image example:", rgb_files_pbr[obj_id][0], flush=True)
        if args.distributed:
            train_sampler_2 = torch.utils.data.distributed.DistributedSampler(train_dataset_2)
        else:
            train_sampler_2 = None
        train_loader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=batch_size_2_dataset, num_workers=num_workers, drop_last=True, sampler=train_sampler_2)                     
        train_loader_2_iter = iter(train_loader_2)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_1_dataset, num_workers=num_workers, drop_last=True, sampler=train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, sampler=train_sampler)

    # define test data loader
    if not bop_challange:
        dataset_dir_test,_,_,_,_,test_rgb_files,_,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_, camera_params_test = bop_io.get_dataset(bop_path, dataset_name,train=False, data_folder=val_folder, data_per_obj=True, incl_param=True, train_obj_visible_theshold=train_obj_visible_theshold)
        if dataset_name == 'ycbv':
            if args.rank==0 or args.rank==-1:
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
        dataset_dir_test,_,_,_,_,test_rgb_files,_,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_, camera_params_test = bop_io.get_bop_challange_test_data(bop_path, dataset_name, target_obj_id=obj_id+1, data_folder=val_folder)
    if args.rank==0 or args.rank==-1:
        print('len(test_rgb_files)', len(test_rgb_files[obj_id]))
        print('test_rgb_file exsample', test_rgb_files[obj_id][0])

    if Detection_reaults != 'none':
        Det_Bbox = get_detection_results(Detection_reaults, test_rgb_files[obj_id], obj_id+1, 0)
    else:
        Det_Bbox = None

    test_dataset = bop_dataset_single_obj_pytorch(
                                            dataset_dir_test, val_folder, test_rgb_files[obj_id], test_mask_files[obj_id], test_mask_visib_files[obj_id], 
                                            test_gts[obj_id], test_gt_infos[obj_id], camera_params_test[obj_id], False, 
                                            BoundingBox_CropSize_image, BoundingBox_CropSize_GT, GT_code_infos, 
                                            padding_ratio=padding_ratio, resize_method=resize_method, Detect_Bbox=Det_Bbox,
                                            use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur
                                        )
    if args.rank==0 or args.rank==-1:
        print("number of test images: ", len(test_dataset), flush=True)
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=False)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, num_workers=num_workers, sampler=test_sampler)

    #############build the network 
    binary_code_length = number_of_itration
    if args.rank==0 or args.rank==-1:
        print("binary_code_length: ", binary_code_length)
    configs['binary_code_length'] = binary_code_length
    
    net = BinaryCodeNet_Deeplab_v3(
                num_resnet_layers=resnet_layer, 
                concat=concat, 
                binary_code_length=binary_code_length, 
                divided_number_each_iteration = divide_number_each_itration, 
                output_kernel_size = output_kernel_size
            )
    maskLoss = MaskLoss()
    binarycode_loss = BinaryCodeLoss(BinaryCode_Loss_Type, mask_binary_code_loss, divide_number_each_itration, use_histgramm_weighted_binary_loss=use_histgramm_weighted_binary_loss)
    
    #visulize input image, ground truth code, ground truth mask
    writer = None
    if args.rank==0 or args.rank==-1:
        writer = SummaryWriter(tensorboard_path)

    #visulize_input_data_and_network(writer, train_loader, net)
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        print('using DDP.')
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            net.cuda(args.gpu)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
        else:
            raise NotImplementedError("--multiprocessing_distributed is required when running DDP.")
    else:
        print('using single GPU')
        net.cuda()
        
        

    optimizer = None
    lr_scheduler = None
    if optimizer_type == 'SGD':
        optimizer=optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == 'Adam':
        optimizer=optim.Adam(net.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    else:
        raise NotImplementedError(f"unknown optimizer type: {optimizer_type}")

    best_score_path = os.path.join(check_point_path, 'best_score')
    if args.rank==0 or args.rank==-1:
        if not os.path.isdir(best_score_path):
            os.makedirs(best_score_path)
    best_score = 0
    iteration_step = 0
    if not (load_checkpoint is None or load_checkpoint != 'none'):
        checkpoint = torch.load( get_checkpoint(load_checkpoint) )
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_score = checkpoint['best_score']
        iteration_step = checkpoint['iteration_step']
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    # train the network
    while True:
        end_training = False
        if args.distributed:
            train_sampler.set_epoch(iteration_step) # change ddp sampler seed
        for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, class_code_images, cam_Ks) in enumerate(train_loader):
            # if multiple training sets, get data from the second set
            if training_data_folder_2 != 'none':
                try:
                    data_2, entire_masks_2, masks_2, Rs_2, ts_2, Bboxes_2, class_code_images_2, cam_Ks_2 = next(train_loader_2_iter)
                except StopIteration:
                    train_loader_2_iter = iter(train_loader_2)
                    if args.distributed:
                        train_sampler_2.set_epoch(iteration_step) # change ddp sampler seed
                    data_2, entire_masks_2, masks_2, Rs_2, ts_2, Bboxes_2, class_code_images_2, cam_Ks_2 = next(train_loader_2_iter)

                data = torch.cat((data, data_2), 0)
                entire_masks = torch.cat((entire_masks, entire_masks_2), 0)
                masks = torch.cat((masks, masks_2), 0)
                class_code_images = torch.cat((class_code_images, class_code_images_2), 0)
                Rs = torch.cat((Rs, Rs_2), 0)
                ts = torch.cat((ts, ts_2), 0)
                cam_Ks = torch.cat((cam_Ks, cam_Ks_2), 0)
                Bboxes = torch.cat((Bboxes, Bboxes_2), 0)
            # data to GPU
            if torch.cuda.is_available():
                data=data.cuda()
                entire_masks=entire_masks.cuda()
                masks = masks.cuda()
                class_code_images = class_code_images.cuda()

            optimizer.zero_grad()
            if data.shape[0]!= batch_size:
                raise ValueError(f"batch size wrong")
            pred_mask_prob, pred_entire_mask_prob, pred_code_prob = net(data)

            # loss for predicted binary coding
            pred_mask_for_loss = from_output_to_class_mask(pred_mask_prob)
            pred_mask_for_loss = torch.tensor(pred_mask_for_loss).cuda()
            loss_b = binarycode_loss(pred_code_prob, pred_mask_for_loss, class_code_images)

            # loss for predicted mask
            loss_mask = maskLoss(pred_mask_prob, masks)
            loss_entire_mask = maskLoss(pred_entire_mask_prob, entire_masks)

            loss = binary_loss_weight*loss_b + loss_mask + loss_entire_mask

            loss.backward()
            optimizer.step()
        
            if args.rank==0 or args.rank==-1:
                print(config_file_name, " iteration_step:", iteration_step, 
                "loss_b:", loss_b.item(),  
                "loss_m:", loss_mask.item(),
                "loss_em:", loss_entire_mask.item(),
                "loss:", loss.item(),
                flush=True
                )
            if args.rank == 0 or args.rank==-1:
                writer.add_scalar('Loss/training loss total', loss, iteration_step)
                writer.add_scalar('Loss/training loss mask', loss_mask, iteration_step)
                writer.add_scalar('Loss/training loss entire mask', loss_entire_mask, iteration_step)
                writer.add_scalar('Loss/training loss binary code', loss_b, iteration_step)

            # test the trained CNN
            log_freq = 1000

            if (iteration_step+1) % log_freq == 0:
                
                if binarycode_loss.histogram is not None:
                    np.set_printoptions(formatter={'float': lambda x: "{0:.2f}".format(x)})
                    if args.rank==0 or args.rank==-1:
                        print('Train err:{}'.format(binarycode_loss.histogram.detach().cpu().numpy()))
               
                pred_masks = from_output_to_class_mask(pred_mask_prob) 
                pred_codes = from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type, divided_num_each_interation=divide_number_each_itration, binary_code_length=binary_code_length)

                if args.rank == 0 or args.rank==-1:    
                    save_checkpoint(check_point_path, net, iteration_step, best_score, optimizer, lr_scheduler, 3)

                pred_codes = pred_codes.transpose(0, 2, 3, 1)

                pred_masks = pred_masks.transpose(0, 2, 3, 1)
                pred_masks = pred_masks.squeeze(axis=-1)

                Rs = Rs.detach().cpu().numpy()
                ts = ts.detach().cpu().numpy()
                Bboxes = Bboxes.detach().cpu().numpy()
                cam_Ks = cam_Ks.detach().cpu().numpy()

                ADD_passed=np.zeros(batch_size)
                ADD_error=np.zeros(batch_size) 

                for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
                    R_predict, t_predict, success = CNN_outputs_to_object_pose( pred_masks[counter], 
                                                                                pred_codes[counter], 
                                                                                Bbox, 
                                                                                BoundingBox_CropSize_GT, 
                                                                                divide_number_each_itration, 
                                                                                dict_class_id_3D_points, 
                                                                                intrinsic_matrix=cam_K)    

                    add_error = 10000
                    if success:
                        add_error = Calculate_Pose_Error(r_GT, t_GT, R_predict, t_predict, vertices)
                        if np.isnan(add_error):
                           add_error = 10000
                    if add_error < obj_diameter*0.1:
                        ADD_passed[counter] = 1
                    ADD_error[counter] = add_error
                    

                ADD_passed = np.mean(ADD_passed)
                ADD_error= np.mean(ADD_error)

                lr_scheduler.step()
                # dist-related
                if args.distributed:
                    tmp_ADD_passed = torch.tensor([ADD_passed, 1]).cuda(args.rank)
                    tmp_ADD_error = torch.tensor([ADD_error, 1]).cuda(args.rank)
                    dist.all_reduce(tmp_ADD_passed, op=dist.ReduceOp.SUM, async_op=False)
                    ADD_passed = np.array((tmp_ADD_passed[0]/tmp_ADD_passed[1]).cpu())
                    dist.all_reduce(tmp_ADD_error, op=dist.ReduceOp.SUM, async_op=False)
                    ADD_error = np.array((tmp_ADD_error[0]/tmp_ADD_error[1]).cpu())
                print(f'rank: {args.rank}: ADD_passed: {ADD_passed}, ADD_error: {ADD_error}')
                if args.rank == 0 or args.rank == -1:
                    writer.add_scalar('TRAIN_ADD/ADD_Train', ADD_passed, iteration_step)
                    writer.add_scalar('TRAIN_ADD/ADD_Error_Train', ADD_error, iteration_step)
                    writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], iteration_step)
                ADD_passed = test_network_with_single_obj(net, test_loader, obj_diameter, writer, dict_class_id_3D_points, vertices, iteration_step, configs, 0,calc_add_and_adi=False,args=args)
                print("ADD_passed", ADD_passed, "args.rank", args.rank)
                if args.rank == 0 or args.rank == -1:
                    if ADD_passed >= best_score:
                        best_score = ADD_passed
                        print("best_score", best_score)
                        save_best_checkpoint(best_score_path, net, optimizer, lr_scheduler, best_score, iteration_step)

            iteration_step = iteration_step + 1
            if iteration_step >=total_iteration:
                end_training = True
                break
        if end_training == True:
            print('end the training in iteration_step:', iteration_step)
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet_train')
    parser.add_argument('--cfg', type=str) # config file
    parser.add_argument('--obj_name', type=str)

    # dist-related
    parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    args = parser.parse_args()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    config_file = args.cfg
    configs = parse_cfg(config_file)
    configs['obj_name'] = args.obj_name

    check_point_path = configs['check_point_path']
    tensorboard_path= configs['tensorboard_path']

    config_file_name = os.path.basename(config_file)
    config_file_name = os.path.splitext(config_file_name)[0]
    time_suffix = time.strftime('%Y%m%d_%H%M%S')
    check_point_path = check_point_path + config_file_name + args.obj_name + '_v5_' + time_suffix + '/checkpoints'
    tensorboard_path = tensorboard_path + config_file_name + args.obj_name + '_v5_' + time_suffix + '/tensorboard_logs/runs'
    configs['check_point_path'] = check_point_path
    configs['tensorboard_path'] = tensorboard_path

    configs['config_file_name'] = config_file_name

    if configs['Detection_reaults'] != 'none':
        Detection_reaults = configs['Detection_reaults']
        dirname = os.path.dirname(__file__)
        Detection_reaults = os.path.join(dirname, Detection_reaults)
        configs['Detection_reaults'] = Detection_reaults
    
    # dist-related
    args.ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main, nprocs=args.ngpus_per_node, args=(configs, args))
    else:
        main(args.gpu, configs, args)
