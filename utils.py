import torch
import numpy as np
from PIL import Image
from prompt_generator import PromptGenerator
import monai.losses.dice
import argparse

# 各种loss函数
mask_loss_dict = {
    "DL": monai.losses.DiceLoss(sigmoid=True, squared_pred=True),
    "DCL": monai.losses.DiceCELoss(sigmoid=True, squared_pred=True),
    "DFL": monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, lambda_focal=20/21, lambda_dice=1/21),
    "GDL": monai.losses.GeneralizedDiceLoss(sigmoid=True),
    "GDFL": monai.losses.GeneralizedDiceFocalLoss(sigmoid=True)
}

# 训练数据和测试数据的文件名
data_dict = {
    "train":
        ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0021", "0022",
            "0023", "0024", "0025", "0026", "0027", "0028", "0029", "0030", "0031", "0032", "0033", "0034"],
    "test":
        ["0035", "0036", "0037", "0038", "0039", "0040"],
    "all":
        ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0021", "0022",
            "0023", "0024", "0025", "0026", "0027", "0028", "0029", "0030", "0031", "0032", "0033", "0034",
            "0035", "0036", "0037", "0038", "0039", "0040"]
}


range_dict = {
    "0": [0],
    "1": [1],
    "2": [2],
    "3": [3],
    "4": [4],
    "5": [5],
    "6": [6],
    "7": [7],
    "8": [8],
    "9": [9],
    "10": [10],
    "11": [11],
    "12": [12],
    "13": [13],
    "organ": range(1, 14),
    "all": range(0, 14),
}

# 标签与名称的映射
labels_dict = [
    "background",
    "spleen",
    "rkid",
    "lkid",
    "gall",
    "eso",
    "liver",
    "sto",
    "aorta",
    "IVC",
    "veins",
    "pancreas",
    "rad",
    "lad"
]

# 提示器，分别是单点，多点，边界框，类别
prompter = {
    "single": PromptGenerator('point', 1),
    "multi": PromptGenerator('point', 4),
    "box": PromptGenerator('box', (0.05, 0.1)),
    "class": PromptGenerator('class')
}


def get_args():
    r"""取参数函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/data/whma/Abdomen/RawData/Training")
    parser.add_argument('--model_type', type=str, default='vit_h')
    parser.add_argument('--checkpoint', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--device', type=str, default='cuda:9')

    parser.add_argument('--loss_fn', type=str, default='DCL')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parser.add_argument('--save_dir', type=str, default='/data/whma/sam_model')

    parser.add_argument('--prompter', type=str, default="single")
    parser.add_argument('--train_class', action="store_true")
    parser.add_argument('--train_prompt', action="store_true")
    parser.add_argument('--grid', action="store_true")

    args = parser.parse_args()
    return args


def dice_fun(output, target):
    r"""2d dice"""
    if output.shape != target.shape:
        raise AssertionError("AssertionError")
    intersection = torch.logical_and(output, target)
    if len(output.shape) == 3:
        X = torch.sum(output, dim=(1, 2))
        Y = torch.sum(target, dim=(1, 2))
        Z = torch.sum(intersection, dim=(1, 2))
        return (2 * Z / (X + Y))
    elif len(output.shape) == 4:
        X = torch.sum(output, dim=(2, 3))
        Y = torch.sum(target, dim=(2, 3))
        Z = torch.sum(intersection, dim=(2, 3))
        return (2 * Z / (X + Y))[:, 0]


def dice_fun_3D(output, target):
    r"""3d dice"""
    if output.shape != target.shape:
        raise AssertionError("AssertionError")
    intersection = torch.logical_and(output, target)
    X = torch.sum(output, dim=(0, 1, 2))
    Y = torch.sum(target, dim=(0, 1, 2))
    Z = torch.sum(intersection, dim=(0, 1, 2))
    return (2 * Z / (X + Y))


def save_png_box(low_mask, res_mask, label, box, name):
    r"""用于debug"""
    with torch.no_grad():
        low_mask_ = torch.clip(low_mask + 128, 0, 255).type(torch.uint8)
        res_mask_ = (res_mask * 128).type(torch.uint8)
        label_ = (label * 128).type(torch.uint8)
        pic = torch.stack((low_mask_, res_mask_, label_), dim=2)
        box_ = (box / 2).type(torch.int)[0]
        pic[box_[1]:box_[3], box_[0]:box_[2], :] += 32
        Image.fromarray(np.array(pic.cpu())).save(name)


def save_png_point(low_mask, res_mask, label, box, name):
    r"""用于debug"""
    with torch.no_grad():
        low_mask_ = torch.clip(low_mask + 128, 0, 255).type(torch.uint8)
        res_mask_ = (res_mask * 128).type(torch.uint8)
        label_ = (label * 128).type(torch.uint8)
        # print(label.sum())
        # print(res_mask.sum())
        pic = torch.stack((low_mask_, res_mask_, label_), dim=2)
        box_ = (box / 2).type(torch.int)[0]
        pic[box_[1] - 5:box_[1] + 5, box_[0] - 5:box_[0] + 5, :] += 32
        # Image.fromarray(np.array(pic.cpu())).save(name)
    # time.sleep(0.25)
