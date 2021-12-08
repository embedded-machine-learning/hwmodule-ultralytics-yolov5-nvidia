#!/bin/bash

# Functions
setup_env()
{
  # echo "Activate environment"
  source ../../init_env_torch.sh
  
  # echo "Setup task spooler socket."
  source ../../init_ts.sh
}

add_job()
{
  echo "Generate Training Script for $MODELNAME"

  # echo "Copy convert_yolov5_to_trt_TEMPLATE.sh to convert_yolov5_to_trt_$MODELNAME.sh"
  # cp convert_yolov5_to_trt_TEMPLATE.sh convert_yolov5_to_trt_$MODELNAME.sh
  
  echo "Copy yolov5_inf_eval_saved_model_TEMPLATE.sh to yolov5_inf_eval_saved_model_$MODELNAME.sh"
  cp yolov5_inf_eval_saved_model_TEMPLATE.sh yolov5_inf_eval_saved_model_$MODELNAME.sh
  
  echo "Copy yolov5_inf_eval_saved_model_trt_TEMPLATE.sh to yolov5_inf_eval_saved_model_trt_$MODELNAME\_TRTFP32.sh"
  cp yolov5_inf_eval_saved_model_trt_TEMPLATE.sh yolov5_inf_eval_saved_model_trt_$MODELNAME\_TRTFP32.sh
  
  echo "Copy yolov5_inf_eval_saved_model_trt_TEMPLATE.sh to yolov5_inf_eval_saved_model_trt_$MODELNAME\_TRTFP16.sh"
  cp yolov5_inf_eval_saved_model_trt_TEMPLATE.sh yolov5_inf_eval_saved_model_trt_$MODELNAME\_TRTFP16.sh
  
  echo "Copy yolov5_inf_eval_saved_model_trt_TEMPLATE.sh to yolov5_inf_eval_saved_model_trt_$MODELNAME\_TRTINT8.sh"
  cp yolov5_inf_eval_saved_model_trt_TEMPLATE.sh yolov5_inf_eval_saved_model_trt_$MODELNAME\_TRTINT8.sh
  
  echo "Add task spooler jobs for $MODELNAME to the task spooler"

  # echo "Add shell script convert_yolov5_to_trt_$MODELNAME.sh"
  # tsp -L AW_conv_$MODELNAME $CURRENTFOLDER/convert_yolov5_to_trt_$MODELNAME.sh

  echo "Add shell script yolov5_inf_eval_saved_model_$MODELNAME.sh"
  tsp -L AW_inf_orig_$MODELNAME $CURRENTFOLDER/yolov5_inf_eval_saved_model_$MODELNAME.sh

  echo "Add shell script yolov5_inf_eval_saved_model_trt_$MODELNAME\_TRTFP32.sh"
  tsp -L AW_inf_FP32_$MODELNAME $CURRENTFOLDER/yolov5_inf_eval_saved_model_trt_$MODELNAME\_TRTFP32.sh

  echo "Add shell script yolov5_inf_eval_saved_model_trt_$MODELNAME\_TRTFP16.sh"
  tsp -L AW_inf_FP16_$MODELNAME $CURRENTFOLDER/yolov5_inf_eval_saved_model_trt_$MODELNAME\_TRTFP16.sh

  echo "Add shell script yolov5_inf_eval_saved_model_trt_$MODELNAME\_TRTINT8.sh"
  tsp -L AW_inf_INT8_$MODELNAME $CURRENTFOLDER/yolov5_inf_eval_saved_model_trt_$MODELNAME\_TRTINT8.sh

}

# Constant Definition
BASEPATH=.
CURRENTFOLDER=`pwd`
MODELSOURCE=exported-models/*

#Setup environment
setup_env

echo "This file converts a model from the ultralytics repository in the .pt format from exported-models into a TRT models for FP32, FP16 and INT8. Then it executes inference on all 4 models and saved the models in results."

for f in $MODELSOURCE
do
  #echo "$f"
  MODELNAME=`basename ${f%%.*}`
  echo $MODELNAME
  add_job
  
  # take action on each file. $f store current file name
  #cat $f
done
