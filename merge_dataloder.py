from torch import nn
import torchvision
from torch.utils.data import Dataset
from torch.utils import data
from PIL import Image
import os
import scipy.io as scio
import numpy as np
from torchvision.transforms import transforms
import torch

imgspath = '/data/03-scanpath/datasets_new/MIT_SALICON_OSIE/images/'
salspath = '/data/03-scanpath/datasets_new/MIT_SALICON_OSIE/saliency_maps_SalGAN/'
gtspath = '/data/03-scanpath/datasets_new/MIT_SALICON_OSIE/gt_fixations/'

transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
])


class MergeDataset(Dataset):
    def __init__(self, choose, cfg):
        self.choose = choose
        self.imgspathdir = os.listdir(imgspath)
        self.imgspath = imgspath
        self.imgspathdir.sort()  # 排序
        self.pool2d = nn.AvgPool2d(cfg.patch_size)
        self.cfg = cfg

    def __getitem__(self, index):
        image_name = self.imgspathdir[index]
        image_path = os.path.join(self.imgspath, image_name)
        img = Image.open(image_path).convert('RGB')  # （Image类型）
        h, w = img.height, img.width
        img = img.resize((self.cfg.image_w, self.cfg.image_h))
        img = transform(img)

        sal_name = image_name
        sal_path = os.path.join(salspath, sal_name)
        sal = Image.open(sal_path)
        sal = sal.resize((self.cfg.image_w, self.cfg.image_h))
        sal = transform(sal).squeeze()

        # 获取图片真实预测点
        gt_name = image_name[:-4] + '.mat'
        gt_path = os.path.join(gtspath, gt_name)
        gt_fixations = scio.loadmat(gt_path)
        gt_fixations = gt_fixations['gt_fixations']
        valid_len = len(gt_fixations)

        gt_fixations = gt_fixations.astype(np.float)
        gt_fixations[:, 0] /= h
        gt_fixations[:, 1] /= w
        if len(gt_fixations) >= self.cfg.seq_len:
            gt = gt_fixations[:self.cfg.seq_len]
            valid_len = self.cfg.seq_len
        else:
            gt = np.vstack((gt_fixations, [[0., 0.]] * (self.cfg.seq_len - len(gt_fixations))))

        dec_input = gt[:-1]
        if self.cfg.query:
            dec_input = np.vstack(([0.5, 0.5], dec_input))

        dec_mask = torch.zeros(self.cfg.seq_len)
        dec_mask[valid_len:] = 1
        durations_gt = 0

        return img, sal, gt, valid_len, dec_input, dec_mask, gt_name, durations_gt

    def __len__(self):
        return len(self.imgspathdir)


class Merge_iter:
    def __init__(self, cfg):
        self.merge_data = MergeDataset(choose='all', cfg=cfg)
        self.merge_data_iter = data.DataLoader(self.merge_data, batch_size=cfg.train_batch_size, shuffle=True,
                                          num_workers=16, drop_last=True, pin_memory=True, worker_init_fn=self._init_fn)
        self.cfg = cfg

    def _init_fn(self, worker_id):
        np.random.seed(int(self.cfg.seed) + worker_id)


