ARG PYTORCH="1.13.1"
ARG CUDA="11.6"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

RUN apt-get update
RUN apt-get install -y git

WORKDIR /home
RUN git clone --recurse-submodules https://github.com/lyltc1/ZebraPose

# Install other dependency
RUN apt-get install -y libgl1
RUN apt-get install -y libglib2.0-0  # for opencv
RUN pip install numpy==1.24
RUN pip install tqdm tensorboard
RUN pip install imgaug pypng opencv-python-headless
RUN pip install pandas scikit-learn open3d-cpu
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y python3-tk

WORKDIR /home/ZebraPose/bop_toolkit
RUN pip install -e .

WORKDIR /home/ZebraPose

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
RUN git pull

# docker build -t lyltc1/zebrapose:latest .
