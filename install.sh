#!/bin/bash

echo 'Cloning ultralytics yolov5'
rm -rf ./yolov5_ultralytics
git clone -b v5.0 https://github.com/ultralytics/yolov5.git yolov5_ultralytics

echo 'Cloning tensorrtx'
rm -rf ./tensorrtx
git clone -b yolov5-v5.0 https://github.com/wang-xinyu/tensorrtx.git

echo 'Cloning scripts and guides'
rm -rf ./scripts-and-guides
git clone https://github.com/embedded-machine-learning/scripts-and-guides.git scripts-and-guides

echo 'Creating venv'
deactivate
rm -rf ./venv
python3 -m venv ./venv/torch
source ./venv/torch/bin/activate

echo 'Installing pip packages'
python -m pip install --upgrade pip --no-cache
pip install --upgrade setuptools wheel numpy==1.19.4
pip install -r requirements.txt

echo 'Installing torch and torchvision'
wget https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
pip install ./torch-1.7.0-cp36-cp36m-linux_aarch64.whl
rm ./torch-1.7.0-cp36-cp36m-linux_aarch64.whl

sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.8.1 https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
cd torchvision
export BUILD_VERSION=0.8.1  # where 0.x.0 is the torchvision version  
python setup.py install
cd ../  # attempting to load torchvision from build dir will result in import error
rm -rf torchvision

echo 'Done!'
