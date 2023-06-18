'''
    预处理数据集,将图像转化成feature保存起来,以加速训练步骤
    参数(参见args.py):
    args.data_dir: 从args.data_dir/img/文件夹下加载3D图像,转化成feature保存到args.data_dir/feature/下
    args.device: 使用设备
    args.checkpoint: 模型路径
    args.model_type: 模型种类
'''

import nibabel as nib
import torch
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from utils import data_dict, get_args
import os


def open_picture(file_name: str):
    nifti_img = nib.load(file_name)
    raw_data = nifti_img.get_fdata()
    layercnt = len(raw_data[0][0])
    images = []
    for layer in range(layercnt):
        img = torch.tensor((raw_data[:, :, layer] + 512) / 6)
        img = torch.clip(img, 0, 255).type(torch.uint8)
        img = torch.repeat_interleave(img.unsqueeze(2), 3, dim=2)
        images.append(img)
    return images


def feature(file_name, predictor):
    images = open_picture(file_name)
    features = []
    print(file_name, ':')
    for image in tqdm(images):
        predictor.set_image(np.array(image))
        features.append(predictor.get_image_embedding().cpu())
    return features


def main():
    args = get_args()
    img_dir = args.data_dir + '/img/'
    os.makedirs(args.data_dir + '/feature', exist_ok=True)
    feature_dir = args.data_dir + '/feature/'
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)
    for index in data_dict["all"]:
        file_name = img_dir + "img" + index + '.nii.gz'
        features = feature(file_name, predictor)
        torch.save(features, feature_dir + "feature" + index + ".pth")


if __name__ == "__main__":
    main()
