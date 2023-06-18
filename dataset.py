'''
    数据集OurDataset类
'''
import torch
from torch.utils.data import Dataset
import nibabel as nib
from utils import data_dict
from prompt_generator import PromptGenerator


class OurDataset(Dataset):
    r"""
    数据集,每次取数据返回image_feature, label, clas, prompt

    image_feature: 图像特征向量。

    label: groundtruth标签。

    clas: 类别。

    prompt: 提示。

    初始化参数：

    train: 训练集or测试集。

    data_dir: 数据路径。

    prompt_gen: 提示生成器,参见prompt_generator.py中的PromptGenerator类和utils.py中的prompter字典。

    """

    def __init__(self, train: bool, data_dir: str, prompt_gen: PromptGenerator, organs_kept: list[int] = range(1, 14)):

        if (train):
            index_list = data_dict["train"]
        else:
            index_list = data_dict["test"]
        label_dir = data_dir + '/label/'
        feature_dir = data_dir + '/feature/'
        self.features = []
        self.indices = []
        self.labels = []
        self.prompt_gen = prompt_gen
        ind = 0
        for index in index_list:
            # data
            feature = torch.load(feature_dir + "feature" + index + '.pth')
            # self.len = self.len + len(feature) * 13
            self.features.extend(feature)
            # label
            label_3d = nib.load(label_dir + "label" + index + '.nii.gz').get_fdata()
            layer_cnt = label_3d.shape[2]
            if (layer_cnt != len(feature)):
                raise AssertionError("Label length doesn't match data length! >_< ")
            for layer in range(layer_cnt):
                img = label_3d[:, :, layer]
                for i in organs_kept:
                    label = (img == i)
                    if (label.sum() >= 100):
                        self.indices.append((ind, i))
                        self.labels.append(label)
                ind = ind + 1
        self.len = len(self.labels)
        if (self.len == 0):
            raise AssertionError("dataset length = 0: Failed to load! >_< ")

    def __len__(self):
        return self.len

    def __getitem__(self, index: int):
        if (index >= self.len):
            raise IndexError("Index too large! >_< ")
        findex, clas = self.indices[index]
        image_feature = self.features[findex][0]
        label = self.labels[index]
        if self.prompt_gen is None:
            AssertionError("self.prompt_gen is None >_< ")
        prompt, clas = self.prompt_gen(label, clas)
        return image_feature, label[None, :, :], clas, prompt * 2

    def set_prompt_gen(self, prompt_gen):
        self.prompt_gen = prompt_gen
