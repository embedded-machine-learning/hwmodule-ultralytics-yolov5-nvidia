"""
An example that uses TensorRT's Python api to make inferences.
Original script (c) https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5
"""
import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision
import argparse
import concurrent.futures
from tqdm import tqdm
import pandas as pd

PATH_TO_SCRIPTS = "../../scripts-and-guides/scripts/inference_evaluation/inference_utils.py"
CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4

sys.path.append(os.path.dirname(PATH_TO_SCRIPTS))
from inference_utils import save_latencies_to_csv, convert_reduced_detections_tf2_to_df, generate_measurement_index

def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, raw_image_generator):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        for i, image_raw in enumerate(raw_image_generator):
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.post_process(
                output[i * 6001: (i + 1) * 6001], batch_origin_h[i], batch_origin_w[i]
            )
            # Draw rectangles and labels on the original image
            for j in range(len(result_boxes)):
                box = result_boxes[j]
                plot_one_box(
                    box,
                    batch_image_raw[i],
                    label="{}:{:.2f}".format(str(result_classid[j]), result_scores[j]),
                )
        return batch_image_raw, end - start, result_boxes, result_scores, result_classid

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        
    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)
        
    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # to a torch Tensor
        pred = torch.Tensor(pred).cuda()
        # Get the boxes
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4]
        # Get the classid
        classid = pred[:, 5]
        # Choose those boxes that score > CONF_THRESH
        si = scores > CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        # Do nms
        indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_classid = classid[indices].cpu()
        return result_boxes, result_scores, result_classid

class InferThread(threading.Thread):
    def __init__(self, yolov5_wrapper, image_path_batch, output_dir, detection_file, save_images):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
        self.image_path_batch = image_path_batch
        self.output_dir = output_dir
        self.detections_file = os.path.join(output_dir, detection_file)
        self.save_images = save_images

    def run(self):
        batch_image_raw, use_time, result_boxes, result_scores, result_classid = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image(self.image_path_batch))
        for i, img_path in enumerate(self.image_path_batch):
            parent, filename = os.path.split(img_path)
            # Save image
            if self.save_images:
                os.makedirs(os.path.join(self.output_dir, 'detections'), exist_ok=True) 
                save_name = os.path.join(self.output_dir, 'detections', filename)
                cv2.imwrite(save_name, batch_image_raw[i])

            box_swapped = torch.zeros(size=(result_boxes.shape))
            
            for box, swap_box in zip(result_boxes, box_swapped):
                box[0] = box[0] / batch_image_raw[i].shape[1]
                box[1] = box[1] / batch_image_raw[i].shape[0]
                box[2] = box[2] / batch_image_raw[i].shape[1]
                box[3] = box[3] / batch_image_raw[i].shape[0]
                swap_box[0] = box[1]
                swap_box[1] = box[0]
                swap_box[3] = box[2]
                swap_box[2] = box[3]

            df_xml = convert_reduced_detections_tf2_to_df(filename, batch_image_raw[i], box_swapped, result_classid.numpy().astype(np.int), result_scores.numpy(), 0.5)
            if os.path.isfile(self.detections_file):
                df_xml.to_csv(self.detections_file, index=False, mode='a', header=False, sep=';')
            else:
                df_xml.to_csv(self.detections_file, index=False, mode='w', header=True, sep=';')

def warmupFunc(yolov5_wrapper):
    batch_image_raw, use_time, result_boxes, result_scores, result_classid = yolov5_wrapper.infer(yolov5_wrapper.get_raw_image_zeros())
    return use_time

# argparse check dir function
def dir_path(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path

#argparse valid file
def file_valid(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser(description='Script for executing a yolov5 TRT model.')
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('-l', '--lib', type=lambda x: file_valid(parser, x), help="Path to the Plugin Library.")
    required.add_argument('-e', '--engine', type=lambda x: file_valid(parser, x), help="Path to the TRT engine file.")
    optional.add_argument('-o', '--output', default="output/", type=dir_path, help="Output directory.")
    optional.add_argument('-i', '--images', default="images/", type=dir_path, help="Image directory.")
    optional.add_argument('--latency_runs', default=100, type=int, help="Number of runs for latency measurements.")
    optional.add_argument('-d', '--device_name', default="jetson", type=str, help="Name of the device.")
    optional.add_argument('--save_images', default=False, help="If the detected images should be saved.")
    optional.add_argument('--index_save_file', default="./tmp/index.txt", help="index file.")
    args = parser.parse_args()
    # load custom plugins

    PLUGIN_LIBRARY = str(args.lib)
    engine_file_path = str(args.engine)

    ctypes.CDLL(PLUGIN_LIBRARY)

    # if os.path.exists(args.output):
    #     shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)
    model_dir_name = os.path.basename(os.path.dirname(engine_file_path))
    detection_path = os.path.join(args.output, model_dir_name, args.device_name)
    shutil.rmtree(detection_path, ignore_errors=True)
    os.makedirs(detection_path)
    
    # a YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    try:
        print('batch size is', yolov5_wrapper.batch_size)
        
        image_dir = args.images
        image_path_batches = get_img_path_batches(yolov5_wrapper.batch_size, image_dir)
        
        detection_file = os.path.join("detections.csv")
        
        latency_file = os.path.join("latency_" + args.device_name + ".csv")
        latency_file_path = os.path.join(args.output, latency_file)

        if os.path.isfile(os.path.join(args.output, detection_file)):
            os.remove(os.path.join(args.output, detection_file))

        ########################################################################

        print("Starting warm up runs...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(10):
                future = executor.submit(warmupFunc, yolov5_wrapper)
                future.result()

        ########################################################################
        
        print("Starting latency runs...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            latencies = []
            for i in range(args.latency_runs):
                future = executor.submit(warmupFunc, yolov5_wrapper)
                latencies.append(future.result()*1000.0)#to milliseconds

            index = generate_measurement_index(model_dir_name)
            save_latencies_to_csv(latencies,
                                    yolov5_wrapper.batch_size,
                                    args.latency_runs,
                                    args.device_name,
                                    model_dir_name,
                                    model_dir_name,
                                    latency_file_path, 
                                    index=index)
            #Save index to a file
            os.makedirs(os.path.dirname(args.index_save_file), exist_ok=True)
            file1 = open(args.index_save_file, 'w')
            file1.write(index)
            print("Index {} used for latency measurement".format(index))

        ########################################################################

        print("Starting inference runs...")
        for batch in image_path_batches:
            thread1 = InferThread(yolov5_wrapper, 
                                    batch, 
                                    detection_path, 
                                    detection_file,
                                    args.save_images)
            thread1.start()
            thread1.join()

    finally:
        # destroy the instance
        yolov5_wrapper.destroy()
