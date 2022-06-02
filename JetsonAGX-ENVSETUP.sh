#!/bin/bash
set -e

# remove something and maxpower
cd ~
sudo /usr/sbin/nvpmodel -m 0
sudo apt -y remove libreoffice* firefox*
sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y

#install Jetpack
sudo apt install -y nvidia-jetpack
cd ~

#install nomachine
cd Downloads
wget https://download.nomachine.com/download/7.9/Arm/nomachine_7.9.2_1_arm64.deb
sudo apt install ./nomachine_7.9.2_1_arm64.deb
rm -rf ./nomachine_7.9.2_1_arm64.deb
cd ~

#Camera Setup
cd ~
wget https://raw.githubusercontent.com/ZacharyZhang-NY/AGXSETUP/main/camera_overrides.isp
sudo mv ./camera_overrides.isp /var/nvidia/nvcam/settings
sudo chmod 664 /var/nvidia/nvcam/settings/camera_overrides.isp
sudo chown root:root /var/nvidia/nvcam/settings/camera_overrides.isp
cd ~

#Env Setup
sudo apt install -y libopenblas-base libgeos-dev libopenblas-dev python3-pip python3-venv python3-dev libpython3-dev python3-testresources gfortran libopenmpi-dev liblapack-dev libatlas-base-dev libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y build-essential cmake git unzip pkg-config zlib1g-dev libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev libpng-dev libtiff-dev libglew-dev libavcodec-dev libavformat-dev libswscale-dev libgtk2.0-dev libgtk-3-dev libcanberra-gtk* libavcodec-dev libavformat-dev libswscale-dev libgtk2.0-dev libgtk-3-dev libcanberra-gtk* libxvidcore-dev libx264-dev libgtk-3-dev libtbb2 libtbb-dev libdc1394-22-dev libxine2-dev gstreamer1.0-tools libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev libv4l-dev v4l-utils v4l2ucp qv4l2 libtesseract-dev libxine2-dev libpostproc-dev libavresample-dev libvorbis-dev libfaac-dev libmp3lame-dev libtheora-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenblas-dev libatlas-base-dev libblas-dev liblapack-dev liblapacke-dev libeigen3-dev gfortran libhdf5-dev libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev

cd ~
TMPDIR=$(mktemp -d)
echo "moving to $TMPDIR"
cd $TMPDIR

export PATH=/home/zacharyzhang/.local/bin:$PATH

pip3 install -U pip

# avoids warnings caused by invoking the pip3 wrapper script
PIP_INSTALL="python3 -m pip install --user"
export OPENBLAS_CORETYPE=ARMV8

${PIP_INSTALL} -U Cython
${PIP_INSTALL} -U protobuf
${PIP_INSTALL} -U numpy pandas 
${PIP_INSTALL} -U "matplotlib==3.3.4"
${PIP_INSTALL} -U scipy sklearn scikit-image requests

# install dependencies from source and PyTorch from wheel provided by NVIDIA
${PIP_INSTALL} -U future psutil dataclasses typing-extensions pyyaml tqdm seaborn
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
${PIP_INSTALL} torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# check whether everything's good so far
python3 -c 'import torch; assert torch.cuda.is_available()' || { echo "something went wrong when importing torch and checking for cuda"; exit 1; }

# install torchvision from source
${PIP_INSTALL} -U pillow
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python3 setup.py install --user
cd ~

# reveal the CUDA location
cd ~
sudo sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf"
sudo ldconfig

# install the dependencies
sudo apt-get install -y build-essential cmake git unzip pkg-config zlib1g-dev
sudo apt-get install -y libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libglew-dev
sudo apt-get install -y libgtk2.0-dev libgtk-3-dev libcanberra-gtk*
sudo apt-get install -y python-dev python-numpy python-pip
sudo apt-get install -y python3-dev python3-numpy python3-pip
sudo apt-get install -y libxvidcore-dev libx264-dev libgtk-3-dev
sudo apt-get install -y libtbb2 libtbb-dev libdc1394-22-dev libxine2-dev
sudo apt-get install -y gstreamer1.0-tools libv4l-dev v4l-utils v4l2ucp  qv4l2 
sudo apt-get install -y libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev
sudo apt-get install -y libavresample-dev libvorbis-dev libxine2-dev libtesseract-dev
sudo apt-get install -y libfaac-dev libmp3lame-dev libtheora-dev libpostproc-dev
sudo apt-get install -y libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get install -y libopenblas-dev libatlas-base-dev libblas-dev
sudo apt-get install -y liblapack-dev liblapacke-dev libeigen3-dev gfortran
sudo apt-get install -y libhdf5-dev protobuf-compiler
sudo apt-get install -y libprotobuf-dev libgoogle-glog-dev libgflags-dev

# remove old versions or previous builds
cd ~ 
sudo rm -rf opencv*
# download the latest version
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.5.zip 
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.5.zip 
# unpack
unzip opencv.zip 
unzip opencv_contrib.zip 
# some administration to make live easier later on
mv opencv-4.5.5 opencv
mv opencv_contrib-4.5.5 opencv_contrib
# clean up the zip files
rm opencv.zip
rm opencv_contrib.zip

# set install dir
cd ~/opencv
mkdir build
cd build

# run cmake
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
-D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
-D WITH_OPENCL=OFF \
-D WITH_CUDA=ON \
-D CUDA_ARCH_BIN=5.3 \
-D CUDA_ARCH_PTX="" \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=ON \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_NEON=ON \
-D WITH_QT=OFF \
-D WITH_OPENMP=ON \
-D BUILD_TIFF=ON \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D WITH_TBB=ON \
-D BUILD_TBB=ON \
-D BUILD_TESTS=OFF \
-D WITH_EIGEN=ON \
-D WITH_V4L=ON \
-D WITH_LIBV4L=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D PYTHON3_PACKAGES_PATH=/home/zacharyzhang/.local/lib/python3.6/site-packages \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_EXAMPLES=OFF \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D BUILD_opencv_python3=ON \
-D HAVE_opencv_python3=ON \
-D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 ..

# run make
FREE_MEM="$(free -m | awk '/^Swap/ {print $2}')"
# Use "-j 4" only swap space is larger than 5.5GB
if [[ "FREE_MEM" -gt "5500" ]]; then
  NO_JOB=8
else
  echo "Due to limited swap, make only uses 1 core"
  NO_JOB=8
fi
make -j ${NO_JOB} 

sudo rm -r /usr/include/opencv4/opencv2
sudo make install
sudo ldconfig

# cleaning (frees 300 MB)
make clean
sudo apt-get update

# clone yolov5 dependencies
cd ~
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5/utils
rm -rf dataloaders.py
wget https://raw.githubusercontent.com/ZacharyZhang-NY/AGXSETUP/main/dataloaders.py
cd ~

#setup env
cd ~
echo "export PATH=/home/zacharyzhang/.local/bin:$PATH" >> ~/.bashrc
echo "export OPENBLAS_CORETYPE=ARMV8" >> ~/.bashrc
source ~/.bashrc


# boot from ssd
git clone https://github.com/jetsonhacks/rootOnNVMe.git # clone
cd rootOnNVMe
./copy-rootfs-ssd.sh
./setup-service.sh
sudo reboot
