import os

""" foam_brick """
""" v2_no_refine """
# TODO, exp_ycbv_paper.txt ==> refine = False
os.system(
    'python test_v2.py --cfg config/config_paper/ycbv/exp_ycbv_paper.txt --obj_name foam_brick --ckpt_file /home/lyltc/git/ZebraPose/results/checkpoints/exp_ycbv_paperfoam_brick_v2/best_score/0_9097step376000 ')

""" v2_refine_pre """
# TODO, exp_ycbv_paper.txt ==> refine = True ;
# TODO, exp_ycbv_paper.txt ==> refine_entire_mask_type = pre_entire_mask
# TODO, exp_ycbv_paper.txt ==> refine_mask_type = pre_mask
os.system(
    'python test_v2.py --cfg config/config_paper/ycbv/exp_ycbv_paper.txt --obj_name foam_brick --ckpt_file /home/lyltc/git/ZebraPose/results/checkpoints/exp_ycbv_paperfoam_brick_v2/best_score/0_9097step376000 --debug')

""" v2_refine_gt """
# # TODO, exp_ycbv_paper.txt ==> refine = True ;
# # TODO, exp_ycbv_paper.txt ==> refine_entire_mask_type = gt_entire_mask
# # TODO, exp_ycbv_paper.txt ==> refine_mask_type = gt_mask
os.system('python test_v2.py --cfg config/config_paper/ycbv/exp_ycbv_paper.txt --obj_name foam_brick '
          '--ckpt_file /home/lyltc/git/ZebraPose/results/checkpoints/exp_ycbv_paperfoam_brick_v2/best_score/0_9097step376000 '
          '--debug')

""" large_clamp """
""" v2_no_refine """
# TODO, exp_ycbv_paper.txt ==> refine = False
os.system(
    'python test_v2.py --cfg config/config_paper/ycbv/exp_ycbv_paper.txt --obj_name large_clamp --ckpt_file /home/lyltc/git/ZebraPose/results/checkpoints/exp_ycbv_paperlarge_clamp_v2/best_score/0_9017step249000 ')

""" v2_refine_pre """
# TODO, exp_ycbv_paper.txt ==> refine = True ;
# TODO, exp_ycbv_paper.txt ==> refine_entire_mask_type = pre_entire_mask
# TODO, exp_ycbv_paper.txt ==> refine_mask_type = pre_mask
os.system(
    'python test_v2.py --cfg config/config_paper/ycbv/exp_ycbv_paper.txt --obj_name large_clamp --ckpt_file /home/lyltc/git/ZebraPose/results/checkpoints/exp_ycbv_paperlarge_clamp_v2/best_score/0_9017step249000 --debug')

""" v2_refine_gt """
# # TODO, exp_ycbv_paper.txt ==> refine = True ;
# # TODO, exp_ycbv_paper.txt ==> refine_entire_mask_type = gt_entire_mask
# # TODO, exp_ycbv_paper.txt ==> refine_mask_type = gt_mask
os.system(
    'python test_v2.py --cfg config/config_paper/ycbv/exp_ycbv_paper.txt --obj_name large_clamp --ckpt_file /home/lyltc/git/ZebraPose/results/checkpoints/exp_ycbv_paperlarge_clamp_v2/best_score/0_9017step249000 --debug')

""" run 2022.7.3 """
os.system(
    'CUDA_VISIBLE_DEVICES=0 python train.py --cfg config/config_BOP/tless/exp_tless_BOP.txt --obj_name obj04 > log_obj04.txt')
os.system(
    'CUDA_VISIBLE_DEVICES=1 python train_v4.py --cfg config/config_BOP/tless/exp_tless_BOP.txt --obj_name obj04 > log_obj04_v4.txt')
os.system(
    'CUDA_VISIBLE_DEVICES=2 python train_v4.py --cfg config/config_paper/ycbv/exp_ycbv_paper.txt --obj_name large_marker > log_large_marker.txt')