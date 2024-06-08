# Dog Detection using YOLOv5

## Description
This repository is part of a project that I worked on with https://github.com/Better1337 as part of the Artificial Intelligence Techniques(Metody Sztucznej Inteligencji) course. We used https://github.com/ultralytics/yolov5 and the dataset from http://vision.stanford.edu/aditya86/ImageNetDogs/

## Requirements
- Python 3.x
- PyTorch
- OpenCV
- Pillow
- NumPy

## Data
The training model utilizes a set of labeled images, available from the following  dataset: [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). Ensure your dataset is configured correctly for YOLOv5. The parameters used to train the model(best.pt) can be found at the begginning of the CamDetection.py

- **Dataset Setup**: Annotations must adhere to YOLOv5's requirements, formatted as .txt files containing `class_index x_center y_center width height`. For comprehensive guidance on preparing your data, visit YOLOv5's guide to [Training on Custom Data](https://docs.ultralytics.com/tutorials/train-custom-data).
