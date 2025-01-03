FROM nvidia/cudagl:11.4.2-runtime-ubuntu20.04

# Dependencies for PCL
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    linux-libc-dev \
    cmake \
    cmake-gui \
    libusb-1.0-0-dev \
    libusb-dev \
    libudev-dev \
    mpi-default-dev \
    openmpi-bin \
    openmpi-common \
    libflann1.9 \
    libflann-dev \
    libeigen3-dev \
    libboost-all-dev \
    libqhull* \
    libgtest-dev \
    freeglut3-dev \
    pkg-config \
    libxmu-dev \
    libxi-dev \
    mono-complete \
    libopenni-dev \
    libopenni2-dev \
    openjdk-8-jdk \
    openjdk-8-jre

# Install PCL
RUN git clone https://github.com/PointCloudLibrary/pcl.git /opt/pcl
WORKDIR /opt/pcl
RUN mkdir release && cd release && \
    cmake -DCMAKE_BUILD_TYPE=None -DCMAKE_INSTALL_PREFIX=/usr \
    -DBUILD_GPU=ON -DBUILD_apps=ON -DBUILD_examples=ON .. && \
    make -j$(nproc) && make install

# Install OpenCV 3.4.5
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" && \
    apt-get update && \
    apt-get install -y libjasper1 libjasper-dev
RUN apt-get install -y \
    build-essential \
    wget \
    unzip \
    cmake \
    libgtk2.0-dev \
    libtiff-dev \
    libtiff5-dev \
    libjasper-dev \
    libavformat-dev \
    libswscale-dev \
    libavcodec-dev \
    libjpeg-turbo8-dev \
    pkg-config \
    ffmpeg

RUN wget https://github.com/opencv/opencv/archive/3.4.5.zip -O /tmp/opencv-3.4.5.zip && \
    unzip /tmp/opencv-3.4.5.zip -d /opt && \
    rm /tmp/opencv-3.4.5.zip && mkdir /opt/opencv-3.4.5/build
WORKDIR /opt/opencv-3.4.5/build
RUN cmake .. && make -j$(nproc) && make install

# Clone ZebraPose repository
RUN git clone https://github.com/lyltc1/ZebraPose.git /home/ZebraPose
WORKDIR /home/ZebraPose
RUN git pull
WORKDIR /home/ZebraPose/Binary_Code_GT_Generator/Generate_Mesh_with_GT_Color
RUN mkdir build && cd build && cmake .. && make

RUN ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen && ln -sf /usr/include/eigen3/unsupported /usr/include/unsupported
RUN apt-get install -y libglfw3-dev libglfw3 libassimp-dev assimp-utils
WORKDIR /home/ZebraPose/Binary_Code_GT_Generator/Render_GT_Color_Mesh_to_GT_Img/
RUN mkdir build && cd build && cmake .. && make

RUN apt-get install -y python3-pip && pip3 install numpy==1.24.4
WORKDIR /home/ZebraPose/Binary_Code_GT_Generator/Render_GT_Color_Mesh_to_GT_Img/Render_Python_API
RUN mkdir build && python3 setup.py build
RUN export PYTHONPATH=$PYTHONPATH:/home/ZebraPose/Binary_Code_GT_Generator/Render_GT_Color_Mesh_to_GT_Img/Render_Python_API/build/lib.linux-x86_64-3.8/ && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ZebraPose/Binary_Code_GT_Generator/Render_GT_Color_Mesh_to_GT_Img/build

WORKDIR /home/ZebraPose
RUN git clone https://github.com/lyltc1/bop_toolkit.git
WORKDIR /home/ZebraPose/bop_toolkit
RUN pip install -e .
# docker build -t lyltc1/zebrapose_generator:latest -f docker/Generator.Dockerfile .
# docker run -it --gpus all lyltc1/zebrapose_generator:latest