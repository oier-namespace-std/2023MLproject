"""
    提示生成器
"""

import numpy as np


class PromptGenerator():
    r"""
    提示生成器

    初始化参数：

    str: 提示类型

    para: 类型的参数

    检查__init__函数获取更多信息
    """

    def __init__(self, str=None, para=None) -> None:
        if (str == 'box'):
            self.handler = self.get_box
            (self.a, self.b) = para
        elif (str == 'point'):
            self.handler = self.get_point
            self.N = para
        elif (str == 'class'):
            self.handler = self.class_feature_point
        else:
            raise AssertionError("Error str")

    def __call__(self, label, clas=None):
        return self.handler(label, clas)

    def get_box(self, label, clas=None):
        indices = np.argwhere(label)
        if len(indices) == 0:
            return np.zeros((1, 4), dtype=int), -1
        x_min, y_min = indices.min(axis=0)
        x_max, y_max = indices.max(axis=0)
        x = x_max - x_min
        y = y_max - y_min
        ax = np.array((y_min, x_min, y_max, x_max))
        r = np.random.uniform(self.a, self.b, size=4)
        sz = np.array((-y, -x, y, x))
        res = ax + r * sz
        res = np.clip(res, 0, 512).astype(int)
        return np.array(res)[None, :], clas

    def get_point(self, label, clas=None):
        indices = np.argwhere(label)
        choice = np.random.choice(indices.shape[0], self.N)
        return np.array(indices[choice][:, ::-1]), clas

    def class_feature_point(self, label, clas=None):
        x = clas // 4
        y = clas % 4
        coord = np.array([(x, y)]) * 128 + 64
        return coord, clas

    def random_point(self, label, clas=None):
        return np.random.randint(0, 512, (1, 2)), np.random.randint(1, 14)
