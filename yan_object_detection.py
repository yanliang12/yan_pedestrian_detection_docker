###############yan_object_detection.py#################
from __future__ import division

from models import *
from models.darknet import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torchvision.transforms.functional as TF

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


img_size = 416

#######

def load_yolo_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(
        "config/yolov3.cfg", 
        img_size=img_size).to(device)
    model.load_darknet_weights("yolov3.weights")
    model.eval()  # Set in evaluation mode
    classes = load_classes("data/coco.names")  # Extracts class labels from file
    return model, classes

model, classes = load_yolo_model()

#######

def object_detection_from_image(
    input_image_path, 
    ):
    ouput = []
    image_original = Image.open(input_image_path)
    image = image_original.resize((img_size,img_size), Image.ANTIALIAS)
    image = np.array(image)
    image = TF.to_tensor(image)
    ###
    input_imgs = Variable(image.type(Tensor))
    input_imgs = torch.unsqueeze(input_imgs, 0)
    ###
    detections = model(input_imgs)
    detections = non_max_suppression(detections, 0.8, 0.4)[0]
    detections1 = rescale_boxes(detections, img_size, image_original.size)
    for detection in detections1:
        x1, y1, x2, y2, conf, cls_conf, cls_pred = detection
        print(detection)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        label = classes[int(cls_pred)]
        score = cls_conf.item()
        ouput.append({
            "x1":x1, "x2":x2, "y1": y1, "y2":y2, "label":label, "score":score
            })
    return ouput

###############yan_object_detection.py#################
