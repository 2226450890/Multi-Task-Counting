import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from model import CBAM

criterion = nn.MSELoss().cuda()
MAPS = ['map3', 'map4']
MIN_HW = 384
MAX_HW = 600
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


def ExtraLoss(output, gt, boxes):
    Loss = 0
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            X = output[:, :, y1:y2, x1:x2].sum()
            Y = gt[:, :, y1:y2, x1:x2].sum()
            Loss += F.mse_loss(X, Y)  
            break
    return Loss

def extract_features0(feature_model, image, boxes, feat_map_keys=['map3', 'map4']):
    N = image.shape[0]
     # [1,3,960,1280]
    """
    Getting features for the image N * C * H * W
    """
    Image_features = feature_model(image)
    map3_conv = torch.tensor(np.load("turtle/conv/map3_conv.npy")).cuda()
    map4_conv = torch.tensor(np.load("turtle/conv/map4_conv.npy")).cuda()
    """
    Getting features for the examples (N*M) * C * h * w
    """
    for ix in range(0, N):

        cnter = 0
        for keys in feat_map_keys:
            image_features = Image_features[keys][ix].unsqueeze(0)

            """
            Convolving example features over image features
            """
            if keys == 'map3':
                h, w = map3_conv.shape[2], map3_conv.shape[3]
                conv = map3_conv
            if keys == 'map4':
                h, w = map4_conv.shape[2], map4_conv.shape[3]
                conv = map4_conv
            #ROI pool
            features = F.conv2d(
                    F.pad(image_features, ((int(w/2)), int((w-1)/2), int(h/2), int((h-1)/2))),
                    conv
                )  # [1,100,120,160] [1,100,60,80]
            combined = features.permute([1, 0, 2, 3])
            if cnter == 0:
                Combined = 1.0 * combined
            else:
                if Combined.shape[2] != combined.shape[2] or Combined.shape[3] != combined.shape[3]:
                    combined = F.interpolate(combined, size=(Combined.shape[2], Combined.shape[3]), mode='bilinear')
                Combined = torch.cat((Combined, combined), dim=1)  # [100,2,120,160]
            cnter += 1
        if ix == 0:
            All_feat = 1.0 * Combined.unsqueeze(0)
        else:
            All_feat = torch.cat((All_feat, Combined.unsqueeze(0)), dim=0)
    return All_feat 

def extract_features1(feature_model, image, boxes, feat_map_keys=['map5']):
    N = image.shape[0]
     # [1,3,960,1280]
    """
    Getting features for the image N * C * H * W
    """
    Image_features = feature_model(image)
    # map4_conv = torch.tensor(np.load("turtle/conv/map4_conv_1.npy")).cuda()
    map5_conv = torch.tensor(np.load("turtle/conv/map5_conv_1.npy")).cuda()
    """
    Getting features for the examples (N*M) * C * h * w
    """
    for ix in range(0, N):

        cnter = 0
        for keys in feat_map_keys:
            image_features = Image_features[keys][ix].unsqueeze(0)
            # if keys == 'map4':
            #     h, w = map4_conv.shape[2], map4_conv.shape[3]
            #     conv = map4_conv
            if keys == 'map5':
                h, w = map5_conv.shape[2], map5_conv.shape[3]
                conv = map5_conv
            #ROI pool
            features = F.conv2d(
                    F.pad(image_features, ((int(w/2)), int((w-1)/2), int(h/2), int((h-1)/2))),
                    conv
                ) 
            combined = features.permute([1, 0, 2, 3])

#             if keys == 'map4':
#                 conv =  torch.tensor(np.ones((10, 256, 1, 1))).cuda().float()
#                 h,w = 1,1
#                 # net = CBAMLayer(512).cuda()
#                 # image_features = net.forward(image_features)
#             if keys == 'map5':
#                 conv =  torch.tensor(np.ones((1, 512, 1, 1))).cuda().float()
#                 h,w = 1,1
#                 # net = CBAMLayer(1024).cuda()
#                 # image_features = net.forward(image_features)
#             # #ROI pool
#             features = F.conv2d(
#                     F.pad(image_features, ((int(w/2)), int((w-1)/2), int(h/2), int((h-1)/2))),
#                     conv
#                 )  # [1,100,120,160] [1,100,60,80]
          
#             # features = F.conv2d(
#             #         F.pad(image_features, ((int(w/2)), int((w-1)/2), int(h/2), int((h-1)/2))),
#             #         conv
#             #     )  # [1,100,120,160] [1,100,60,80]
#             combined = features.permute([1, 0, 2, 3])
            if cnter == 0:
                Combined = 1.0 * combined
                Combined = F.interpolate(Combined, size=(Combined.shape[2]*2, Combined.shape[3]*2), mode='bilinear')
            else:
                if Combined.shape[2] != combined.shape[2] or Combined.shape[3] != combined.shape[3]:
                    combined = F.interpolate(combined, size=(Combined.shape[2], Combined.shape[3]), mode='bilinear')
                Combined = torch.cat((Combined, combined), dim=1)  # [100,2,120,160]
            cnter += 1
        if ix == 0:
            All_feat = 1.0 * Combined.unsqueeze(0)
        else:
            All_feat = torch.cat((All_feat, Combined.unsqueeze(0)), dim=0)
            
    return All_feat 

def extract_features(feature_model, image, feat_map_keys=['map3', 'map4']):
    N = image.shape[0]
     # [1,3,960,1280]
    """
    Getting features convolution kernels
    """
    Image_features = feature_model(image)
    map3_conv = torch.tensor(np.load("chicken/conv/map3_conv.npy")).cuda()
    map4_conv = torch.tensor(np.load("chicken/conv/map4_conv.npy")).cuda()
   
    """
    Getting Correlation map
    """
    for ix in range(0, N):
        cnter = 0
        for keys in feat_map_keys:
            image_features = Image_features[keys][ix].unsqueeze(0)
            """
            features convolution kernels
            """
            if keys == 'map3':
                h1, w1 = map3_conv.shape[2], map3_conv.shape[3]
                conv1 = map3_conv
               
            if keys == 'map4':
                h1, w1 = map4_conv.shape[2], map4_conv.shape[3]
                conv1 = map4_conv
                
            """
            convolution operation
            """
            features1 = F.conv2d(
                    F.pad(image_features, ((int(w1/2)), int((w1-1)/2), int(h1/2), int((h1-1)/2))),
                    conv1
                )  # [1,100,120,160] [1,100,60,80]
           
            combined1 = features1.permute([1, 0, 2, 3])
            
            """
            Correlation map concatenation
            """
            if cnter == 0:
                Combined = 1.0 * combined1
            else:
                combined1 = F.interpolate(combined1, size=(Combined.shape[2], Combined.shape[3]), mode='bilinear')
                Combined = torch.cat((Combined, combined1), dim=1)
               
            cnter += 1
        if ix == 0:
            Correlation_map = 1.0 * Combined.unsqueeze(0)
        else:
            Correlation_map = torch.cat((Correlation_map, Combined.unsqueeze(0)), dim=0)
    return Correlation_map  # [1,100,2,120,160]


class resizeImage0(object):

    def __init__(self, MAX_HW=600):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image = sample['image']

        W, H = image.size

        scale_factor = 600.0/1920
        new_H = 8 * int(H * scale_factor / 8)
        new_W = 8 * int(W * scale_factor / 8)
        resized_image = transforms.Resize((new_H, new_W))(image)

        resized_image = Normalize(resized_image)
        sample = {'image': resized_image}
        return sample

class resizeImage1(object):

    def __init__(self, MAX_HW=600):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image = sample['image']
        W, H = image.size
        scale_factor = 600.0/1920
        new_H = 8 * int(H * scale_factor / 8)
        new_W = 8 * int(W * scale_factor / 8)
        resized_image = transforms.Resize((new_H, new_W))(image)

        resized_image = Normalize(resized_image)
        sample = {'image': resized_image}
        return sample
    
class resizeImage(object):
    
    def __init__(self, MAX_HW=600):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image = sample['image']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw) / max(H, W)
            # new_H = 352
            # new_W = 608
            new_H = 8 * int(H * scale_factor / 8)
            new_W = 8 * int(W * scale_factor / 8)
            resized_image = transforms.Resize((new_H, new_W))(image)
        else:
            resized_image = image

        resized_image = Normalize(resized_image)
        sample = {'image': resized_image}
        return sample

class resizeImage3(object):
    
    def __init__(self, MAX_HW=600):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image = sample['image']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw) / max(H, W)
            new_H = 352
            new_W = 608
            # new_H = 8 * int(H * scale_factor / 8)
            # new_W = 8 * int(W * scale_factor / 8)
            resized_image = transforms.Resize((new_H, new_W))(image)
        else:
            resized_image = image

        resized_image = Normalize(resized_image)
        sample = {'image': resized_image}
        return sample
    
class resizeImageWithGT(object):

    def __init__(self, MAX_HW = 600):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image, lines_boxes, density = sample['image'], sample['boxes'], sample['gt_density']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8 * int(H * scale_factor / 8)
            new_W = 8 * int(W * scale_factor / 8)
            resized_image = transforms.Resize((new_H, new_W))(image)
            resized_density = cv2.resize(density, (new_W, new_H))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)

            if new_count > 0:
                resized_density = resized_density * (orig_count / new_count)
            
        else:
            scale_factor = 1
            resized_image = image
            resized_density = density
        boxes = list()
        for box in lines_boxes:
            box2 = [int(int(k)*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1, x1, y2, x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image': resized_image, 'boxes': boxes, 'gt_density': resized_density}
        return sample


Normalize = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])

Transform0 = transforms.Compose([resizeImage0(MAX_HW)])
Transform = transforms.Compose([resizeImage(MAX_HW)])
Transform3 = transforms.Compose([resizeImage3(MAX_HW)])
Transform1 = transforms.Compose([resizeImage1(MAX_HW)])
TransformTrain = transforms.Compose([resizeImageWithGT(MAX_HW)])


def denormalize(tensor, means = IM_NORM_MEAN, stds = IM_NORM_STD):
    denormalized = tensor.clone()
    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)
    return denormalized


def scale_and_clip(val, scale_factor, min_val, max_val):
    new_val = int(round(val*scale_factor))
    new_val = max(new_val, min_val)
    new_val = min(new_val, max_val)
    return new_val


def visualize(input_, output, save_path, figsize=(20, 12)):

    # get the total count
    pred_cnt = output.sum().item()
    img1 = format_for_plotting(denormalize(input_))
    output = format_for_plotting(output)
    fig = plt.figure(figsize=figsize)

    # display the input image
    # scale_factor = 608.0/1920
    ax = fig.add_subplot(2, 2, 1)
    ax.set_axis_off()
    # ax.set_title("Input image, gt count: {}".format(dots.shape[0]))
    # ax.scatter(dots[:, 0]*scale_factor, dots[:, 1]*scale_factor, c='red', edgecolors='blue')
    ax.imshow(img1, cmap='gray')
        
    ax = fig.add_subplot(2, 2, 2)
    ax.set_axis_off()
    ax.set_title("Overlaid result, predicted count: {:.2f}".format(pred_cnt))

    img2 = 0.2989*img1[:, :, 0] + 0.5870*img1[:, :, 1] + 0.1140*img1[:, :, 2]   # Convert to grayscale
    ax.imshow(img2, cmap='gray')
    ax.imshow(output, cmap=plt.cm.viridis, alpha=0.5)

    # display the density map
    ax = fig.add_subplot(2, 2, 3)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ax.imshow(output, cmap=plt.cm.jet)

    ax = fig.add_subplot(2, 2, 4)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ret_fig = ax.imshow(output)

    fig.colorbar(ret_fig, ax=ax)

    print(save_path)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close()


def format_for_plotting(tensor):

    has_batch_dimension = len(tensor.shape) == 4
    formatted = tensor.clone()

    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()




