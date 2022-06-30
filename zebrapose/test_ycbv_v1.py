import os

command = 'python test.py --cfg config/config_paper/ycbv/exp_ycbv_paper.txt --obj_name {} --ckpt_file ' \
          '/home/lyl/git/ZebraPose/results/zebra_ckpts/paper/ycbv/{} --eval_output_path ' \
          '/home/lyl/git/ZebraPose/results/evaluate_report'
er
filename_list = ['banana', 'bleach_cleanser', 'bowl', 'cracker_box', 'extra_large_clamp', 'foam_brick', 'gelatin_box',
                 'large_clamp', 'large_marker', 'master_chef_can', 'mug', 'mustard_bottle', 'pitcher_base',
                 'potted_meat_can', 'power_drill', 'pudding_box', 'scissors', 'sugar_box', 'tomato_soup_can',
                 'tuna_fish_can', 'wood_block']

for filename in filename_list:
    os.system(command.format(filename, filename))
    print("finished {}".format(filename))
