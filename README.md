# TensorRTx

Fork来源: https://github.com/wang-xinyu/tensorrtx

新增了Nanodet

新增了Yolov5-face

新增了stnet (权重来自paddle)

新增了arcface-r100-glink360, pth权重从官方网盘下载，将gen_wts.py放到insightface/recognition/arcface_torch https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch 名称是backbone.pth

新增了trt_pose, 官方项目https://github.com/NVIDIA-AI-IOT/trt_pose， 这里使用PaddlePddle的复现权重，项目地址: https://aistudio.baidu.com/aistudio/projectdetail/3516206

新增了yolov5n-v6-prune, 支持yolov5n-v6模型，以及通道数量自适应的裁剪模型, pytorch和paddle都可以转换权重，Paddle项目地址:https://aistudio.baidu.com/aistudio/projectdetail/3452279 
