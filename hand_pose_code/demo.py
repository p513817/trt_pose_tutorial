import torch
import torchvision.transforms as transforms

import cv2
import PIL.Image
import json
import time

import math
import os
import numpy as np
import traitlets

from torch2trt import TRTModule
import trt_pose
import trt_pose.coco
import trt_pose.models
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

#############################################################

print_div = lambda x: print(f"\n{x}\n")

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

############################################################

print_div("INIT")

with open('preprocess/hand_pose.json', 'r') as f:
    hand_pose = json.load(f)

# 改變標籤結構 : 增加頸部keypoint以及 paf的維度空間
topology = trt_pose.coco.coco_category_to_topology(hand_pose)
# 用於解析預測後的 cmap與paf
parse_objects = ParseObjects(topology, cmap_threshold=0.30, link_threshold=0.30)
# 用於將keypoint繪製到圖片上
draw_objects = DrawObjects(topology)

############################################################

print_div("LOAD TENSORRT ENGINE")

OPTIMIZED_MODEL = 'model/hand_pose_resnet18_baseline_att_224x224.trt'

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

###########################################################

print_div("START STREAM")

cap = cv2.VideoCapture(0)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
crop_size = (w-h)//2
# print(w, h, crop_size)

while(True):
    
    t_start = time.time()
    
    ret, frame = cap.read()
    
    if not ret:
        continue

    frame = frame[:, crop_size:(w-crop_size)]
    image = cv2.resize(frame, (224,224))
    data = preprocess(image)

    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)

    t_end = time.time()
    cv2.putText(image, f"FPS:{int(1/(t_end-t_start))}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1,  cv2.LINE_AA)
    cv2.imshow('pose esimation', image)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#############################################################
