import os

from zebrapose.tools_for_BOP.common_dataset_info import get_obj_info
obj_name_obj_id, _ = get_obj_info("tless")
for obj_name in obj_name_obj_id.keys():
    print(obj_name)
    os.system("python test.py --cfg config/config_BOP/tless/exp_tless_BOP.txt --obj_name {} --ckpt_file /media/lyltc/mnt2/dataset/zebrapose/zebra_ckpts/bop/tless/{}".format(obj_name, obj_name))