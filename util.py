import numpy as np
import torch
import os
import os.path as osp


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        MIoU = np.nanmean(MIoU)
        return MIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
        # count = np.bincount(label, minlength=self.num_class**2)
        # count = np.bincount(label, minlength=self.num_class**2)[: self.num_class**2]
        count = np.bincount(label.astype("int64"), minlength=self.num_class**2)[
            : self.num_class**2
        ]

        # ↑これがおかしい
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, labels, n=1):
        if type(labels).__module__ == "tensor":
            labels = labels.item()
        self.labels = labels
        # self.sum += labels
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename, dir_data_name):
    file_path = osp.join(dir_data_name, filename)
    if not os.path.exists(dir_data_name):
        os.makedirs(dir_data_name)
    torch.save(state.state_dict(), file_path)
