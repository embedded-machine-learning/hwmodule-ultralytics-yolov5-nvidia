#!/bin/bash

setup_env()
{
  # echo "Activate environment"
  source ../../init_env_torch.sh
  
  # echo "Setup task spooler socket."
  source ../../init_ts.sh
  
  export PYTHONPATH=/usr/lib/python3.6/dist-packages:$PYTHONPATH
}

get_model_name()
{
  MYFILENAME=`basename "$0"`
  MODELNAME=`echo $MYFILENAME | sed 's/yolov5_inf_eval_saved_model_//' | sed 's/.sh//'`
  echo Selected model: $MODELNAME
}

echo ==============================================#
echo  CDLEML Process YOLOv5 Object Detection
echo ==============================================#

HARDWARENAME="Xavier"
GROUNDTRUTH="/media/cdleml/128GB/datasets/pedestrian_detection_graz_val_only_ss10/annotations/coco_val_annotations_zero_start.json"
DATASETCONF="/media/cdleml/128GB/datasets/pedestrian_detection_graz_val_only_ss10/dataset.yaml"

# Constant Definition
PYTHONENV=torch
BASEPATH=`pwd`
SCRIPTPREFIX=../../scripts-and-guides/scripts
MODELSOURCE=exported-models

#Extract model name from this filename
get_model_name
#Setup environment
setup_env

#echo "Start training of $MODELNAME on EDA02" | mail -s "Start training of $MODELNAME" $USEREMAIL

echo ====================================
echo  Infer Images from $MODELNAME
echo ====================================
echo Inference from model 
python3 ./yolov5_pt.py \
--data=$DATASETCONF \
--weights="exported-models/$MODELNAME.pt" \
--batch-size=1 \
--task="val" \
--save-json \
--latency_runs=100 \
--device_name=$HARDWARENAME \
--model_name=$MODELNAME \
--device=0 \
--output="./results" \
--index_save_file="./tmp/index.txt"

echo ====================================
echo  Evaluate with Coco Metrics
echo ====================================
# echo coco evaluation
python3 $SCRIPTPREFIX/inference_evaluation/objdet_pycoco_evaluation.py \
--groundtruth_file=$GROUNDTRUTH \
--detection_file="results/$MODELNAME/$HARDWARENAME/coco_predictions.json" \
--output_file="results/performance_$HARDWARENAME.csv" \
--model_name=$MODELNAME \
--hardware_name=$HARDWARENAME \
--index_save_file="./tmp/index.txt"

echo ====================================
echo  Merge results to one result table
echo ====================================
# echo merge latency and evaluation metrics
python3 $SCRIPTPREFIX/inference_evaluation/merge_results.py \
--latency_file="results/latency_$HARDWARENAME.csv" \
--coco_eval_file="results/performance_$HARDWARENAME.csv" \
--output_file="results/combined_results_$HARDWARENAME.csv"
