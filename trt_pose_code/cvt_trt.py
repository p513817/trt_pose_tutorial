import json

import torch2trt
import torch

import trt_pose.coco
import trt_pose.models

def print_div(txt):    
    print(f"\n{txt}\n")


print_div("LOAD MODEL")

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

# 取得 keypoint 數量
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

# 修改輸出層
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

# 載入權重
MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))

print_div("COVERTING")

WIDTH, HEIGHT = 224, 224
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

print_div("SAVING TENSORRT")

OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A.trt'
torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

print_div("FINISH")