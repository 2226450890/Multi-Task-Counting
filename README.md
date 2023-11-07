# Multi-Task-Counting
![image](https://github.com/2226450890/Multi-Task-Counting/master/图片9.jpg)
![image](https://github.com/2226450890/Multi-Task-Counting/master/fig16.jpg)
Chinese soft-shelled Turtle Counting under Irregular Densities Using Multi-task Strategy.

## Prerequisites
We strongly recommend Anaconda as the environment.

Python: 3.8

PyTorch: 1.11

CUDA: 11.3

## Data Setup
Download Turtle Dataset from
baidu drive: [link](https://pan.baidu.com/s/1waS1ir8chkn0bTln_jlJjA?pwd=58i5) 

## Evaluation
&emsp;1. We are providing our pretrained model, and the evaluation code can be used without the training. Download pretrained model from baidu drive: [link](https://pan.baidu.com/s/11hh0OlH3dAIKs5GejnEJVA?pwd=trzc).  
&emsp;2. To run code quickly, We describe the main documents.
    
```
Multi-Task-Counting                              # Project folder. Typically we run our code from this folder.
│───density.py                                   # Counting with Density Maps.
│───detect.py                                    # Counting with YOLOv5n.
└───predict.py                                   # Density classification and selection of counting modules.
```
&emsp;3. Evaluate the model
```
python predict.py
```  

