## Build Docker
```
docker build -t lyltc1/zebrapose:latest .
```
## Usage
```
docker run -it --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all \
--gpus all --shm-size 12G --device=/dev/dri --group-add video \
--volume=/tmp/.X11-unix:/tmp/.X11-unix --env="DISPLAY=$DISPLAY" \
--env="QT_X11_NO_MITSHM=1" --name zebrapose \
-v path/to/dataset/:/home/dataset:ro \
-v path/to/output/:/home/HiPose/output:rw \
lyltc1/zebrapose:latest /bin/bash
```
i.e.
```
docker run -it --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all \
--gpus all --shm-size 12G --device=/dev/dri --group-add video \
--volume=/tmp/.X11-unix:/tmp/.X11-unix --env="DISPLAY=$DISPLAY" \
--env="QT_X11_NO_MITSHM=1" --name zebrapose \
-v /mnt/2T/Data/BOP:/home/dataset:ro \
-v /home/lyl/git/ZebraPose/data/:/home/data:rw \
-v /home/lyl/git/ZebraPose/results/:/home/ZebraPose/results:rw \
lyltc1/zebrapose:latest /bin/bash
```

## Prepare soft link

```
mkdir /home/ZebraPose/datasets/
mkdir /home/ZebraPose/datasets/tless/
ln -sf /home/dataset/tless/* /home/ZebraPose/datasets/tless/
ln -sf /home/dataset/tless/models_cad /home/ZebraPose/datasets/tless/models

ln -sf /home/data/GT/tless/test_primesense_bop_GT/ /home/ZebraPose/datasets/tless/
ln -sf /home/data/GT/tless/models_GT_color/ /home/ZebraPose/datasets/tless/
ln -sf /home/data/ckpt/tless /home/ZebraPose/data/tless

```

## Run for visualization
1. To compare zebrapose-tless-real and zebrapose-tless-pbr visualization
```
cd /home/ZebraPose/zebrapose
python test_for_visulize.py --cfg config/config_BOP/tless/exp_tless_BOP.txt --obj_name obj27 --ckpt_file /home/data/ckpt/tless/pbr_obj27
```