# trt_pose_tutorial
簡化 trt_pose、trt_pose_hand 的 Jupyter 程式，讓一般 Python 使用者能夠快速上手。

## 人體辨識DEMO
![image](./figures/image003_pose_estimation.gif)

## 手部辨識DEMO
![image](./figures/image004_trt_pose_hand.gif)

# 程式
| 名稱  | 描述  |
| ---   | ---   |
| cvt_trt.py  |  透過torch2trt轉換成TensorRT引擎可用之模型  
| demo.py     |  即時影像推論之程式

# 使用方法 ( 教學文章 )

* [使用Jetson Nano 進行姿態辨識與手部辨識](https://chiachun0818.medium.com/%E4%BD%BF%E7%94%A8jetson-nano-%E9%80%B2%E8%A1%8C%E5%A7%BF%E6%85%8B%E8%BE%A8%E8%AD%98%E8%88%87%E6%89%8B%E9%83%A8%E8%BE%A8%E8%AD%98-851d172cb273)
* [Jetson Nano 運行手部辨識範例TRT_POSE_HAND 與其它應用](https://chiachun0818.medium.com/jetson-nano-%E9%81%8B%E8%A1%8C%E6%89%8B%E9%83%A8%E8%BE%A8%E8%AD%98%E7%AF%84%E4%BE%8Btrt-pose-hand-%E8%88%87%E5%85%B6%E5%AE%83%E6%87%89%E7%94%A8-db71d3c73680)

# 參考
* [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose)
* [trt_pose_hand](https://github.com/NVIDIA-AI-IOT/trt_pose_hand)
