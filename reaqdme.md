# README

## Download
1. Download opencv = 3.4.12 from https://github.com/opencv/opencv/archive/refs/tags/3.4.12.zip
2. Download opencv_contrib = 3.4.12 from https://github.com/opencv/opencv_contrib/archive/refs/tags/3.4.12.zip


## Build 
```bash
docker build -t z3d:0.0.1 .
(or without cache) docker build --no-cache -t z3d:0.0.1 .
```
## Usage
```
docker run -it --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all --gpus all -p 8024:22 --shm-size 12G --device=/dev/dri --group-add video --volume=/tmp/.X11-unix:/tmp/.X11-unix --env="DISPLAY=$DISPLAY" --env="QT_X11_NO_MITSHM=1" --name zebrapose -v /home/lyl/dataset/:/home/dataset:ro -v /home/lyl/git/ZebraPose/:/home/ZebraPose:rw z3d:0.0.1 /bin/bash
```

## prepare soft link
```
ln -sf /home/dataset/pbr/lmo/* /home/ZebraPose/data/lmo/
ln -sf /home/dataset/zebrapose/data/lmo/* /home/z3d/data/lmo/

ln -sf /home/dataset/pbr/ycbv/* /home/ZebraPose/data/ycbv/
ln -sf /home/dataset/zebrapose/data/ycbv/* /home/ZebraPose/data/ycbv/

ln -sf /home/dataset/zebrapose/ckpt/bop/ycbv/bowl ZebraPose/results/zebra_ckpts/paper/ycbv/
```


## evaluate
```
cd /home/zebrapose/zebrapose
python test.py --cfg config/config_zebra3d/lmo_zebra3D_32_no_hier_lmo_bop_gdrnpp_.txt --obj_name ape --ckpt_file /home/dataset/z3d/lmo_zebra3D_32_no_hier_lmo_bop_ape/0_7668step37000 --ignore_bit 0 --eval_output_path /home/z3d/output/
```

## Docker Usage
```
docker stop \z3d
docker rm \z3d
```
