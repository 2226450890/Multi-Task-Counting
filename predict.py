from __future__ import print_function, division
from sklearn.metrics import classification_report, confusion_matrix
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from tqdm import tqdm
from density import density_map, density_front
from detect import object_detect, detect_front

cla_dict = {
    '0':"High_density",
    '1':"Low_density",
    '2':"Middle_density"
}


def predict(model, imagepath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(299),
         transforms.CenterCrop(299),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = imagepath
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # prediction
    model.eval()
    print(type(cla_dict))

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device)))
        predict = torch.softmax(output, dim=0).cpu()
        predict_cla = torch.argmax(predict).numpy()
        print(cla_dict[str(predict_cla)])

    print_res = "class: {}   prob: {:.3}".format(cla_dict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    classify = []
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(cla_dict[str(i)], predict[i].numpy()))
        classify.append(predict[i].numpy())
    print("The most likely category is ：",np.argmax(classify))
    return np.argmax(classify)

view_img, save_txt, imgsz, model, save_dir, save_img, stride, device, half = detect_front()
resnet50_conv, regressor = density_front()
model_conv = torch.load("densenet_169.pt")
data_path = './turtle/'
data_split_file = data_path + 'extra_all.json'
im_dir = data_path + 'new_extra'
gt_path = "turtle/labels/extra/"
split = "extra_all"

with open(data_split_file) as f:
    data_split = json.load(f)
im_ids = data_split[split]
pbar = tqdm(im_ids)

cnt = 0
SAE = 0
SSE = 0
t_all = []



for im_id in pbar:
    t1 = time.time()
    image = im_dir+"/"+im_id
    classify = predict(model_conv, image)
    if classify == 1:
        pred_cnt = object_detect(view_img, save_txt, imgsz, model, save_dir, save_img, stride, device, half, image)
    else:
        pred_cnt = density_map(im_id,image, resnet50_conv, regressor)
    t2 = time.time()
    print(pred_cnt)
    t_all.append(t2 - t1)
    l_path = gt_path + im_id
    l_path = l_path[:-4]
    gt_cnt = len(open(l_path + '.txt', 'rU').readlines())
    cnt += 1
    err = abs(gt_cnt - pred_cnt)
    SAE += err
    SSE += err ** 2

    pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'. \
                         format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE / cnt, (SSE / cnt) ** 0.5))

print('average time:', np.mean(t_all) / 1)
print('average fps:', 1 / np.mean(t_all))
print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(split, SAE/cnt, (SSE/cnt)**0.5))
