# hwmodule-ultralytics-yolov5-nvidia
This repository contains a demo project to run inference of Ultralytics YoloV5 on NVIDIA devices Jetson Nano, TX2 and Xavier.

# Setup

To setup the project, you first have to run the install script

``
./install.sh
``

This creates a virtual environment, installs the needed packages and clones needed 3rd party repositories.

## Script variables
After the install-script has finished you need to change some variables in the following files:

### convert_yolov5_to_trt_TEMPLATE.sh

Enter the number of classes:  
``CLASS_NUM=xx``

(for INT8 optimization)  
Enter the absolute path to the images-folder of your calibration dataset.  
``CALIBRATION=xx``

Enter the absolute path to the images-folder of your validation dataset.  
``VALIDATION=xx``

### yolov5_inf_eval_saved_model_TEMPLATE.sh
Enter the name of your device:  
``HARDWARENAME=xx``

Enter the path to your validation dataset groundtruth coco file:  
``GROUNDTRUTH=xx``

Enter the path to your dataset config .yaml file:  
``DATASETCONF=xx``

### yolov5_inf_eval_saved_model_trt_TEMPLATE.sh
Enter the name of your device:  
``HARDWARENAME=xx``

Enter the path to your validation dataset groundtruth coco file:  
``GROUNDTRUTH=xx``

### init_env_torch.sh
Enter the absolute path to the project folder:  
``PROJECTPATH=xx``

### init_ts.sh
If your task-spooler socket is located in another directory, you also need to change the path in this file.

## System changes
You need to allow passwordless sudo execution for a specific command. This allows the scripts to run without asking for your sudo password.

Type:  
``sudo visudo``

Add the following line at the end, add the absolute path to your yolov5 directory:  
``cdleml  ALL=NOPASSWD:<ABSOLUTE_PATH>/yolov5_jetson/tensorrtx/yolov5/build/yolov5``


# Usage

## Model preperation

In the demonstration project you need to place your yolov5-Pytorch models in the 'exported-models' directory.

The name of your models must follow a specific format to set all parameters properly:

``
<framework>_yolov5<s|m|l|x>_<width>x<height>_<dataset_name>  
``

Example: pt_yolov5s_640x360_peddet

## Convert and run inference

To optimize all models in the 'exported-models' folder to the TensorRT FP32, FP16, INT8 format and benchmark the resulting models (including the original .pt model) run

``
./add_folder_conv_and_inf_all_jobs
``

This script will create all necessary scripts and put them into the task-spooler queue. The results will be saved in the 'results' folder.

## Run inference only

When you have already converted your models you can run the inference part only with

``
./add_folder_inference_jobs.sh
``
