# Multi-Task-Counting
![image](https://github.com/2226450890/Multi-Task-Counting/blob/master/图片9.jpg)
![image](https://github.com/2226450890/Multi-Task-Counting/blob/master/fig16.jpg)
Chinese soft-shelled Turtle Counting under Irregular Densities Using Multi-task Strategy.

## Prerequisites
We strongly recommend Anaconda as the environment.

Python: 3.8

PyTorch: 1.11

CUDA: 11.3

## Data Setup
Download Turtle Dataset from
OneDrive: [link](https://stuscaueducn-my.sharepoint.com/:u:/g/personal/3170062_stu_scau_edu_cn/EY_Tj-2shqxLuhjlP1BO86YBmVsAM7Ih0D-DtGwb72TBjw) 

## Evaluation
&emsp;1. We are providing our pretrained model, and the evaluation code can be used without the training. Download pretrained model from OneDrive: [link](https://stuscaueducn-my.sharepoint.com/:u:/g/personal/3170062_stu_scau_edu_cn/EbDcoJGDzXBOnNsiX0u62w8BO8Z2PJD6fdzuLH57bL0lyQ).  
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

