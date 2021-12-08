#!/bin/bash

PROJECTPATH=xx

echo "Activate Torch Python Environment"
# use absolute path here
source $PROJECTPATH/venv/torch/bin/activate
echo "Done"

echo "Executing jetson_clocks"
sudo jetson_clocks --fan
echo "Done"
