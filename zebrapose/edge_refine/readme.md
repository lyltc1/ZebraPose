1.
because of using the features in C++17, we can not use original GNU7.5.0 which is not supported
do:
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt install gcc-10
sudo apt install g++-10
2.
specified the opencv path in /path/to/edge_refine/CMakeLists.txt like:
set(CMAKE_PREFIX_PATH /home/lyltc/git/opencv-master/install)
find_package(OpenCV 4.0.0 REQUIRED)
3.
specified the python path in /path/to/edge_refine/CMakeLists.txt like:
set(PYTHON_EXECUTABLE "/home/lyltc/miniconda3/envs/pose/bin/python")
set(PYTHON_INCLUDE_DIR "/home/lyltc/miniconda3/envs/pose/include/python3.8")
set(PYTHON_LIBRARY "/home/lyltc/miniconda3/envs/pose/lib")
4. ? needn't
install sophus
https://blog.csdn.net/qq_36955294/article/details/109498531

cmake ..
-- The C compiler identification is GNU 10.3.0
-- The CXX compiler identification is GNU 10.3.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/g++
-- Check for working CXX compiler: /usr/bin/g++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found OpenGL: /usr/lib/x86_64-linux-gnu/libOpenGL.so   
-- Found GLEW: /usr/include  
-- Found OpenCV: /home/lyltc/git/opencv-master/install (found suitable version "4.6.0", minimum required is "4.0.0") 
-- Found OpenMP_C: -fopenmp (found version "4.5") 
-- Found OpenMP_CXX: -fopenmp (found version "4.5") 
-- Found OpenMP: TRUE (found version "4.5")  
-- Performing Test HAS_MARCH
-- Performing Test HAS_MARCH - Failed
-- Performing Test HAS_MTUNE
-- Performing Test HAS_MTUNE - Failed
-- Performing Test HAS_GGDB
-- Performing Test HAS_GGDB - Success
-- Performing Test HAS_Z7
-- Performing Test HAS_Z7 - Failed
-- Performing Test HAS_FTRAPV
-- Performing Test HAS_FTRAPV - Success
-- Performing Test HAS_OD
-- Performing Test HAS_OD - Failed
-- Performing Test HAS_OB3
-- Performing Test HAS_OB3 - Failed
-- Performing Test HAS_O3
-- Performing Test HAS_O3 - Success
-- Performing Test HAS_OB2
-- Performing Test HAS_OB2 - Failed
-- Performing Test HAS_O2
-- Performing Test HAS_O2 - Success
-- pybind11 v2.10.0 dev1
-- Found PythonInterp: /home/lyltc/miniconda3/envs/pose/bin/python (found suitable version "3.8.13", minimum required is "3.6") 
-- Found PythonLibs: /home/lyltc/miniconda3/envs/pose/lib
-- Performing Test HAS_FLTO
-- Performing Test HAS_FLTO - Success
-- Configuring done
-- Generating done
-- Build files have been written to: /home/lyltc/git/ZebraPose/zebrapose/edge_refine/build
