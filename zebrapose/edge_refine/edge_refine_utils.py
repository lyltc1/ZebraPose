import numpy as np
from core.csrc.edge_refine.build.examples.edge_refine import py_edge_refine

if __name__=="__main__":
    import os.path as osp
    cur_dir = osp.dirname(osp.abspath(__file__))
    PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../.."))
    rot_est = np.array([[ 0.12275933, 0.99077314, 0.05743508],
                        [0.5620873, -0.02171489, -0.82679284],
                       [-0.817917, 0.13378006, -0.5595666 ]])
    trans_est = np.array([-0.10697047, -0.11897808, 1.0259193])
    contour = np.array([[222,135], [222,136], [222,137], [222,138], [222,139], [222,140],
                        [222,141], [222,142], [222,143], [222,144], [222,145],[222,146],
                        [222,147], [222,148], [222,149], [222,150], [222,151], [222,152],
                        [222,153], [222,154], [222,155], [222,156], [222,157], [222,158],
                        [222,159], [222,160], [222,161], [222,162], [222,163], [222,164],
                        [222,165], [222,166], [222,167], [222,168], [222,169], [222,170]])
    model_path = '/home/lyl/git/GDR-Net/datasets/BOP_DATASETS/lm/models/obj_000001.ply'
    rot_refined, trans_refined = py_edge_refine(rot_est, trans_est, contour,
                                                model_path,
                                                osp.abspath(osp.join(PROJ_ROOT, ".cache")))
    print(rot_refined)
    print(trans_refined)

# output:
# [[ 0.41262424 -0.05337209 -0.9093361 ]
#  [ 0.18694368 -0.97207075  0.14188246]
#  [-0.89151204 -0.22853889 -0.3911224 ]]
# [-0.10696586 -0.11893506  1.0259362 ]
