# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Shuo Wang
# --------------------------------------------------------
import os
import sys
import cv2
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'rfcn'))

import Inference_Shuo_AICity

if __name__ == "__main__":
    print("Extracting frames...")
    vidcap = cv2.VideoCapture('/mxnet/Deformable-ConvNets/input_video/video.mp4')
    success,image = vidcap.read()
    count = 0
    with open('file.txt', 'a') as file:
        while success:
            cv2.imwrite('/mxnet/Deformable-ConvNets/data/data_Shuo/VOC1080/JPEGImages/' + "frame%d.jpeg" % count, image)     # save frame as JPEG file
            file.write("frame%d.jpg" % count)
            success,image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1
        print("Detection started...")
    Inference_Shuo_AICity.main()

