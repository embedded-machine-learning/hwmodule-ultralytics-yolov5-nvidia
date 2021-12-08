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
  MODELNAME=`echo $MYFILENAME | sed 's/yolov5_inf_eval_saved_model_trt_//' | sed 's/.sh//'`
  echo Selected model: $MODELNAME
}

echo ==============================================#
echo  CDLEML Process YOLOv5 Object Detection
echo ==============================================#

HARDWARENAME="Xavier"
GROUNDTRUTH="/media/cdleml/128GB/datasets/pedestrian_detection_graz_val_only_ss10/annotations/coco_val_annotations_zero_start.json"


# Constant Definition
PYTHONENV=torch
BASEPATH=`pwd`
SCRIPTPREFIX=../../scripts-and-guides/scripts
MODELSOURCE=exported-models-trt

#Extract model name from this filename
get_model_name
#Setup environment
setup_env

#echo "Start training of $MODELNAME on EDA02" | mail -s "Start training of $MODELNAME" $USEREMAIL

echo ====================================
echo  Infer Images from $MODELNAME
echo ====================================
echo Inference from model 
python3 yolov5_trt.py \
--lib="$MODELSOURCE/$MODELNAME/libmyplugins.so" \
--engine="$MODELSOURCE/$MODELNAME/yolov5s.engine" \
--output="results" \
--images="images" \
--latency_runs=100 \
--device_name=$HARDWARENAME \
--index_save_file="./tmp/index.txt"

#--model_short_name=%MODELNAMESHORT% unused because the name is created in the csv file

# echo ====================================#
# echo  Convert Detections to Pascal VOC Format
# echo ====================================#
#echo Convert TF CSV Format similar to voc to Pascal VOC XML
#python3 $SCRIPTPREFIX/conversion/convert_tfcsv_to_voc.py \
#--annotation_file="results/$MODELNAME/$HARDWARENAME/detections.csv" \
#--output_dir="results/$MODELNAME/$HARDWARENAME/det_xmls" \
#--labelmap_file="annotations/$LABELMAP"

echo ====================================#
echo  Convert to Pycoco Tools JSON Format
echo ====================================#
# echo Convert TF CSV to Pycoco Tools csv
python3 $SCRIPTPREFIX/conversion/convert_tfcsv_to_pycocodetections.py \
--annotation_file="results/$MODELNAME/$HARDWARENAME/detections.csv" \
--output_file="results/$MODELNAME/$HARDWARENAME/coco_detections.json"

echo ====================================#
echo  Evaluate with Coco Metrics
echo ====================================#
# echo coco evaluation
python3 $SCRIPTPREFIX/inference_evaluation/objdet_pycoco_evaluation.py \
--groundtruth_file=$GROUNDTRUTH \
--detection_file="results/$MODELNAME/$HARDWARENAME/coco_detections.json" \
--output_file="results/performance_$HARDWARENAME.csv" \
--model_name=$MODELNAME \
--hardware_name=$HARDWARENAME \
--index_save_file="./tmp/index.txt"

echo ====================================#
echo  Merge results to one result table
echo ====================================#
# echo merge latency and evaluation metrics
python3 $SCRIPTPREFIX/inference_evaluation/merge_results.py \
--latency_file="results/latency_$HARDWARENAME.csv" \
--coco_eval_file="results/performance_$HARDWARENAME.csv" \
--output_file="results/combined_results_$HARDWARENAME.csv"
