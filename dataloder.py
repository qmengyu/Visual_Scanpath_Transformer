from torch import nn
import torchvision
from torch.utils.data import Dataset
from torch.utils import data
import scipy.io as scio
from torchvision.transforms import transforms
from utils import *

Merge_imgspath = '/data/03-scanpath/datasets_new/MIT_SALICON_OSIE/images/'
Merge_salspath = '/data/03-scanpath/datasets_new/MIT_SALICON_OSIE/saliency_maps_SalGAN/'
Merge_gtspath = '/data/03-scanpath/datasets_new/MIT_SALICON_OSIE/gt_fixations/'

mit_imgspath = '/data/03-scanpath/datasets/MIT/images/'
mit_train_imgspath = '/data/03-scanpath/datasets_new/MIT/images/train/'
mit_val_imgspath = '/data/03-scanpath/datasets_new/MIT/images/val/'
mit_salspath = '/data/03-scanpath/datasets_new/MIT/saliency_maps/SalGAN/'
mit_gtspath = '/data/03-scanpath/datasets_new/MIT/gt_fixations/'

salicon_imgspath = '/data/03-scanpath/datasets_new/SALICON/images/all/'
salicon_train_imgspath = '/data/03-scanpath/datasets/SALICON/SALICON/images/train'
salicon_val_imgspath = '/data/03-scanpath/datasets/SALICON/SALICON/images/val'
salicon_salspath = '/data/03-scanpath/datasets_new/SALICON/saliency_maps/SalGAN/'
salicon_gtspath = '/data/03-scanpath/datasets_new/SALICON/gt_fixations_best'

osie_imgspath = '/data/03-scanpath/datasets/OSIE/images/'
osie_train_imgspath = '/data/03-scanpath/datasets_new/OSIE/images/train_new/'
osie_val_imgspath = '/data/03-scanpath/datasets_new/OSIE/images/val_new/'
osie_salspath = '/data/03-scanpath/datasets_new/OSIE/saliency_maps/SalGAN/'
osie_gtspath = '/data/03-scanpath/datasets_new/OSIE/gt_fixations_test/'

isun_imgspath = '/data/03-scanpath/datasets_new/iSUN/images_padding_600_800/'
isun_salspath = '/data/03-scanpath/datasets_new/iSUN/saliency_maps_padding_600_800/SalGAN'
isun_gtspath = '/data/03-scanpath/datasets_new/iSUN/gt_fixations_padding_600_800/'


class ScanPath_Dataset(Dataset):
    def __init__(self, choose, dataset, cfg):
        self.choose = choose
        self.dataset = dataset
        self.cfg = cfg
        self.pool2d = nn.AvgPool2d(cfg.patch_size)
        self.transform = torchvision.transforms.Compose([
            transforms.ToTensor(),
        ])
        if self.dataset == "salicon":
            self.salspath = salicon_salspath
            self.gtspath = salicon_gtspath
            if choose == 'train':
                self.imgspathdir = os.listdir(salicon_train_imgspath)
                self.imgspath = salicon_train_imgspath
            elif choose == 'val':
                self.imgspathdir = os.listdir(salicon_val_imgspath)
                self.imgspath = salicon_val_imgspath
            elif choose == 'all':
                self.imgspathdir = os.listdir(salicon_imgspath)
                self.imgspath = salicon_imgspath
        elif self.dataset == "osie":
            self.salspath = osie_salspath
            self.gtspath = osie_gtspath
            if choose == 'train':
                self.imgspathdir = os.listdir(osie_train_imgspath)
                self.imgspath = osie_train_imgspath
            elif choose == 'val':
                self.imgspathdir = os.listdir(osie_val_imgspath)
                self.imgspath = osie_val_imgspath
            elif choose == 'all':
                self.imgspathdir = os.listdir(osie_imgspath)
                self.imgspath = osie_imgspath
        elif self.dataset == "mit":
            self.salspath = mit_salspath
            self.gtspath = mit_gtspath
            if choose == 'train':
                self.imgspathdir = os.listdir(mit_train_imgspath)
                self.imgspath = mit_train_imgspath
            elif choose == 'val':
                self.imgspathdir = os.listdir(mit_val_imgspath)
                self.imgspath = mit_val_imgspath
            elif choose == 'all':
                self.imgspathdir = os.listdir(mit_imgspath)
                self.imgspath = mit_imgspath
        elif self.dataset == "isun":
            self.salspath = isun_salspath
            self.gtspath = isun_gtspath
            if choose == 'all':
                self.imgspathdir = os.listdir(isun_imgspath)
                self.imgspath = isun_imgspath

        self.imgspathdir.sort()  # 排序

    def __getitem__(self, index):
        image_name = self.imgspathdir[index]
        image_path = os.path.join(self.imgspath, image_name)
        img = Image.open(image_path).convert('RGB')  # （Image类型）
        img_size = torch.Tensor([img.height, img.width])
        img, pad_size = pad_img(img, target_w=self.cfg.image_w, target_h=self.cfg.image_h)
        img = img.resize((self.cfg.image_w, self.cfg.image_h))
        img = self.transform(img)

        sal_name = image_name
        sal_path = os.path.join(self.salspath, sal_name)
        sal = Image.open(sal_path)
        sal, _ = pad_sal(sal, target_w=self.cfg.image_w, target_h=self.cfg.image_h)
        sal = sal.resize((self.cfg.image_w, self.cfg.image_h))
        sal = self.transform(sal).squeeze()

        # 获取图片真实预测点
        gt_name = image_name[:-4] + '.mat'
        gt_path = os.path.join(self.gtspath, gt_name)
        gt_fixations = scio.loadmat(gt_path)
        index = random.randint(0, 14)
        if self.dataset == "osie":
            max_index = gt_fixations['max_index'][0][0]
            durations = gt_fixations['durations'][0]
            if self.choose == 'train':
                gt_fixations = gt_fixations['gt_fixations'][0][max_index]
                durations = durations[max_index][0]
            else:
                gt_fixations = gt_fixations['gt_fixations'][0][index]
                durations = durations[index][0]
        elif self.dataset == 'mit':
            gt_fixations = gt_fixations['gt_fixations'][0][index]
            durations = 0
        elif self.dataset == 'isun':
            max_index = gt_fixations['max_index'][0][0]
            gt_fixations = gt_fixations['gt_fixations'][0][max_index]
            durations = 0
        else:
            gt_fixations = gt_fixations['gt_fixations']
            durations = 0
        valid_len = len(gt_fixations)
        gt_fixations = gt_fixations.astype(np.float)
        gt_fixations[:, 0] /= img_size[0]
        gt_fixations[:, 1] /= img_size[1]
        if len(gt_fixations) >= self.cfg.seq_len:
            gt = gt_fixations[:self.cfg.seq_len]
            valid_len = self.cfg.seq_len
        else:
            gt = np.vstack((gt_fixations, [[0., 0.]] * (self.cfg.seq_len - len(gt_fixations))))
        enc_input = sal.flatten().unsqueeze(-1)
        dec_input = gt[:-1]
        dec_mask = torch.zeros(self.cfg.seq_len)
        dec_mask[valid_len:] = 1
        durations_gt = 0

        return img, sal, gt, valid_len, dec_input, dec_mask, gt_name, \
               img_size.unsqueeze(0).repeat(self.cfg.seq_len, 1), \
               torch.Tensor(pad_size).repeat(self.cfg.seq_len, 1), \
               durations_gt

    def __len__(self):
        return len(self.imgspathdir)


class Dataiter:
    def __init__(self, cfg):
        self.cfg = cfg

        self.mit_train_data = ScanPath_Dataset(choose='train', dataset='mit', cfg=cfg)
        self.mit_val_data = ScanPath_Dataset(choose='val', dataset='mit', cfg=cfg)
        self.mit_data = ScanPath_Dataset(choose='all', dataset='mit', cfg=cfg)
        self.mit_train_data_iter = data.DataLoader(self.mit_train_data, batch_size=cfg.train_batch_size, shuffle=True,
                                                   num_workers=16, drop_last=True, pin_memory=True,
                                                   worker_init_fn=self._init_fn)
        self.mit_val_data_iter = data.DataLoader(self.mit_val_data, batch_size=cfg.val_batch_size, shuffle=True,
                                                 num_workers=16, drop_last=False, pin_memory=True,
                                                 worker_init_fn=self._init_fn)
        self.mit_data_iter = data.DataLoader(self.mit_data, batch_size=cfg.val_batch_size, shuffle=True,
                                             num_workers=16, drop_last=False, pin_memory=True,
                                             worker_init_fn=self._init_fn)

        self.osie_train_data = ScanPath_Dataset(choose='train', dataset='osie', cfg=cfg)
        self.osie_val_data = ScanPath_Dataset(choose='val', dataset='osie', cfg=cfg)
        self.osie_data = ScanPath_Dataset(choose='all', dataset='osie', cfg=cfg)
        self.osie_train_data_iter = data.DataLoader(self.osie_train_data, batch_size=cfg.train_batch_size, shuffle=True,
                                                    num_workers=16, drop_last=True, pin_memory=True,
                                                    worker_init_fn=self._init_fn)
        self.osie_val_data_iter = data.DataLoader(self.osie_val_data, batch_size=cfg.val_batch_size, shuffle=True,
                                                  num_workers=16, drop_last=False, pin_memory=True,
                                                  worker_init_fn=self._init_fn)
        self.osie_data_iter = data.DataLoader(self.osie_data, batch_size=cfg.val_batch_size, shuffle=True,
                                              num_workers=16, drop_last=False, pin_memory=True,
                                              worker_init_fn=self._init_fn)

        self.salicon_train_data = ScanPath_Dataset(choose='train', dataset='salicon', cfg=cfg)
        self.salicon_val_data = ScanPath_Dataset(choose='val', dataset='salicon', cfg=cfg)
        self.salicon_data = ScanPath_Dataset(choose='all', dataset='salicon', cfg=cfg)
        self.salicon_train_data_iter = data.DataLoader(self.salicon_train_data, batch_size=cfg.train_batch_size,
                                                       shuffle=True,
                                                       num_workers=16, drop_last=True, pin_memory=True,
                                                       worker_init_fn=self._init_fn)
        self.salicon_val_data_iter = data.DataLoader(self.salicon_val_data, batch_size=cfg.val_batch_size, shuffle=True,
                                                     num_workers=16, drop_last=False, pin_memory=True,
                                                     worker_init_fn=self._init_fn)
        self.salicon_data_iter = data.DataLoader(self.salicon_data, batch_size=cfg.val_batch_size, shuffle=True,
                                                 num_workers=16, drop_last=False, pin_memory=True,
                                                 worker_init_fn=self._init_fn)

        self.isun_data = ScanPath_Dataset(choose='all', dataset='isun', cfg=cfg)
        self.isun_data_iter = data.DataLoader(self.isun_data, batch_size=cfg.val_batch_size, shuffle=True,
                                              num_workers=16, drop_last=False, pin_memory=True,
                                              worker_init_fn=self._init_fn)

    def _init_fn(self, worker_id):
        np.random.seed(int(self.cfg.seed) + worker_id)
