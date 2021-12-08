#!/bin/bash

get_model_name()
{
  MYFILENAME=`basename "$0"`
  MODELNAME=`echo $MYFILENAME | sed 's/convert_yolov5_to_trt_//' | sed 's/.sh//'`
  echo Selected model: $MODELNAME
}

get_yolo_arch()
{
  filter_arch="${MODELNAME##*_yolov5}"
  ARCH="${filter_arch:0:1}"
  echo Selected Architecture: $ARCH
}

set_yolo_parameters()
{
  MODEL_SIZE="${MODELNAME##*_yolov5[a-z]_}"
  MODEL_SIZE="${MODEL_SIZE%x*}"
  sed -i -e s/"INPUT_H = .*;"/"INPUT_H = $MODEL_SIZE;"/ ../../tensorrtx/yolov5/yololayer.h
  sed -i -e s/"INPUT_W = .*;"/"INPUT_W = $MODEL_SIZE;"/ ../../tensorrtx/yolov5/yololayer.h
  echo Model Inputsize: $MODEL_SIZE

  sed -i -e s/"#define USE_FP16"/""/ ../../tensorrtx/yolov5/yolov5.cpp
  echo Set default Optimization to None

  sed -i -e s/"\.\/coco_calib"/"\.\.\/calib"/ ../../tensorrtx/yolov5/yolov5.cpp
  echo Set default calibration datset to tensorrtx/yolov5/calib

  sed -i -e s/"CLASS_NUM = .*;"/"CLASS_NUM = $CLASS_NUM;"/ ../../tensorrtx/yolov5/yololayer.h
  echo Set number of classes

  rm ../../tensorrtx/yolov5/calib
  ln -s $CALIBRATION ../../tensorrtx/yolov5/calib
  rm images
  ln -s $VALIDATION images
  echo Set symbolic links
}

setup_env()
{
  # echo "Activate environment"
  source ../../init_env_torch.sh
  
  # echo "Setup task spooler socket."
  source ../../init_ts.sh
}

echo ==============================================
echo  CDLEML Process YOLOv5 Object Detection Conversion
echo ==============================================

CLASS_NUM=1
CALIBRATION=/media/cdleml/128GB/datasets/pedestrian_detection_graz_val_only_ss10/images/calib
VALIDATION=/media/cdleml/128GB/datasets/pedestrian_detection_graz_val_only_ss10/images/val

MODELSOURCE=exported-models
MODELEXPORT=exported-models-trt

#Extract model name from this filename
get_model_name
get_yolo_arch
set_yolo_parameters
#Setup environment
setup_env

echo ==============================================
echo  Extracting weights from $MODELNAME
echo ==============================================
cd ../..
cp ./tensorrtx/yolov5/gen_wts.py ./yolov5_ultralytics/gen_wts.py
cp ./demonstration_projects/inference-demo/$MODELSOURCE/$MODELNAME.pt ./yolov5_ultralytics/model.pt
python ./yolov5_ultralytics/gen_wts.py -w ./yolov5_ultralytics/model.pt -o ./yolov5_ultralytics/model.wts
mkdir ./demonstration_projects/inference-demo/$MODELEXPORT

echo ==============================================
echo  Building the TRT engines for $MODELNAME
echo ==============================================
# Iterate the string array using for loop
declare -a StringArray=(INT8 FP16 FP32)

for val in ${StringArray[@]}; do
  echo "===" $val "==="
  
  rm -r -f ./tensorrtx/yolov5/build
  mkdir ./tensorrtx/yolov5/build
  cp ./yolov5_ultralytics/model.wts ./tensorrtx/yolov5/build/model.wts
  
  cd ./tensorrtx/yolov5/build/
  {
    cmake .. 
    make CXX_FLAGS=-DUSE_$val
    sudo ./yolov5 -s model.wts yolov5s.engine $ARCH
  } && {
    mkdir ../../../demonstration_projects/inference-demo/$MODELEXPORT/${MODELNAME}_TRT${val} 2>/dev/null || :
    cp ../../../tensorrtx/yolov5/build/yolov5s.engine \
      ../../../demonstration_projects/inference-demo/$MODELEXPORT/${MODELNAME}_TRT${val}/yolov5s.engine 2>/dev/null || :
    cp ../../../tensorrtx/yolov5/build/libmyplugins.so \
      ../../../demonstration_projects/inference-demo/$MODELEXPORT/${MODELNAME}_TRT${val}/libmyplugins.so 2>/dev/null || :
  }
  cd ../../..
done