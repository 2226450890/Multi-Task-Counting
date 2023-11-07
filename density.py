from model import DensityRegressor, Resnet18FPN
from util import MAPS, Transform, extract_features1, visualize, Transform3
import os
import torch
from PIL import Image


def density_front():
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    resnet50_conv = Resnet18FPN()
    resnet50_conv = torch.load("./out33/backbone.pth")
    resnet50_conv.cuda()
    resnet50_conv.eval()

    regressor = DensityRegressor(1, pool='mean')
    regressor.load_state_dict(torch.load("./out33/generate_density.pth"))
    regressor.cuda()
    regressor.eval()

    return resnet50_conv,regressor


def density_map(im_id,image, resnet50_conv, regressor):

    image = Image.open(image)
    image.load()
    sample = {'image': image}
    sample = Transform(sample)
    image = sample['image']
    image = image.cuda()

    with torch.no_grad():
        output = regressor(extract_features1(resnet50_conv, image.unsqueeze(0), MAPS))

    pred_cnt = output.sum().item()
    rslt_file = "{}/{}".format("showing", im_id)
    """
    Save the visualization
    """
    #visualize(image3.detach().cpu(), output.detach().cpu(), rslt_file)

    return pred_cnt
