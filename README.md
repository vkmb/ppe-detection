# Personal Protection Equipment Detection based on Deep Learning

Real time Personal Protection Equipment(PPE) detection running on NVIDIA Jetson TX2 and Ubuntu 16.04

  - Person, HardHat and Vest detection
  - Input from Video file or USB Camera
  - A backend service which can push message to "console" or "Cisco® Webex Teams space" when an abnormal event is detected.

![PPE Image](data/ppe.jpg)

# Requirements
  - NVIDIA Jetson TX2 or Ubuntu 16.04
  - NVIDIA GPU on Ubuntu 16.04 is optional
  - Python3

# How to run

## Video Inference Service

```sh
$ git clone https://github.com/vkmb/ppe-detection
$ cd ~/ppe-detection
$ pip3 install -r requirements.txt
$ python3 predict.py --video_file_name=xxx
```
* video_file_name: input video file name or usb camera device name, you can get camera device name on ubuntu or NVIDIA Jeston by running
```sh
$ ls /dev/video* 
```
* show_video_window: the flag to show video window, the options are {0, 1}
* camera_id: It is just convenient for humans to distinguish between different cameras, and you can assign any value, such as camera001


By default, it will use the console notification, this just print the notification to stdout.

# Training Program
Based on TensorFlow Object Detection API, using pretrained ssd_mobilenet_v1 on COCO dataset to initialize weights.

# Reference work
* TensorFlow Object Detection: https://github.com/tensorflow/models/tree/master/research/object_detection
