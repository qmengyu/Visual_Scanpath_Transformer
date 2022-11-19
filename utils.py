import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import os
from PIL import Image
import torch

from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, \
    ExponentialLR, CosineAnnealingWarmRestarts
from warmup_scheduler import GradualWarmupScheduler
from metrics.ScanMatch import ScanMatch
from metrics.utils import ScanMatchInfo_salicon


def pairs_eval(seq_preds, gts, valid_lens):
    batch_size = seq_preds.size(0)
    num_sample = seq_preds.size(1)
    seq_preds[:, :, :, 0] *= 480
    seq_preds[:, :, :, 1] *= 640
    gts[:, :, 0] *= 480
    gts[:, :, 1] *= 640
    metrics_reward = torch.zeros(batch_size, num_sample)
    for index in range(batch_size):
        for num in range(num_sample):
            gt = gts[index][:int(valid_lens[index])].cpu().numpy()
            seq_pred = seq_preds[index][num][:9].cpu().numpy()
            score = ScanMatch(seq_pred.astype(np.int), gt.astype(np.int), ScanMatchInfo_salicon)
            metrics_reward[index, num] = score
    print(metrics_reward.mean())
    return metrics_reward


# 按比例填充图片
def pad_sal(sal, target_w, target_h, pad_value=0):
    w, h = sal.size
    if w / h == target_w / target_h:
        return sal, np.array([0, 0])
    if w < h or w * (target_h / target_w) < h:
        new_w = int(h * (target_w / target_h))
        new_img = Image.new('L', (new_w, h), color=pad_value)
        new_img.paste(sal, (int((new_w - w) // 2), 0))
        return new_img, np.array([0, new_w - w])
    else:  #
        new_h = int(w * (target_h / target_w))
        new_img = Image.new('L', (w, new_h), color=pad_value)
        new_img.paste(sal, (0, int((new_h - h) // 2)))
        return new_img, np.array([new_h - h, 0])


# 按比例填充图片
def pad_img(img, target_w, target_h, pad_value=(124, 116, 104)):
    w, h = img.size
    if w / h == target_w / target_h:
        return img, np.array([0, 0])

    if w < h or w * (target_h / target_w) < h:
        new_w = int(h * (target_w / target_h))
        new_img = Image.new('RGB', (new_w, h), color=pad_value)
        new_img.paste(img, (int((new_w - w) // 2), 0))
        return new_img, np.array([0, new_w - w])
    else:  #
        new_h = int(w * (target_h / target_w))
        new_img = Image.new('RGB', (w, new_h), color=pad_value)
        new_img.paste(img, (0, int((new_h - h) // 2)))
        return new_img, np.array([new_h - h, 0])


def show_tensor_heatmap(img, annot=None, fmt=".1f", save_path=None):
    plt.figure(figsize=(10, 20))  # 画布大小
    sns.set()
    ax = sns.heatmap(img, cmap="rainbow", annot=annot, fmt=fmt)  # cmap是热力图颜色的参数

    # plt.rcParams["font.sans-serif"] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


class Accumulator:  # @save
    """在`n`个变量上累加。"""

    def __init__(self, n):
        self.data = [0.0] * n
        self.num = 0

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        self.num += 1

    def divided(self, arg):
        self.data = [a / float(arg) for a in self.data]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def save_str_file(save_path, str0):
    filename = open(save_path, 'w')
    filename.write(str0)
    filename.close()


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)


def save_checkpoint(epoch_num, model, optimizer, cfg):
    checkpointName = 'ep{}.pth.tar'.format(epoch_num)
    checkpointpath = f'{cfg.work_dir}/checkpoint/'
    if not os.path.exists(checkpointpath):
        os.makedirs(checkpointpath)
    checkpoint = {
        'epoch': epoch_num,
        'model': model.state_dict(),
        'lr': optimizer.param_groups[0]['lr']
    }
    torch.save(checkpoint, os.path.join(checkpointpath, checkpointName))


def loadCheckpoint(epoch, model, optimizer, cfg):
    model_dir_name = f'{cfg.work_dir}/checkpoint/'
    if not os.path.exists(model_dir_name):
        os.mkdir(model_dir_name)

    model_dir = os.listdir(model_dir_name)  # 列出文件夹下文件名
    model_dir.sort(key=lambda x: int(x[2:-8]))  # 文件名按数字排序
    if len(model_dir) == 0:
        return 0, model, optimizer
    if epoch == -1:
        checkpointName = model_dir[epoch]  # 获取文件 , epoch = -1 获取最后一个文件
    else:
        checkpointName = f'ep{epoch}.pth.tar'
    checkpointName = os.path.join(model_dir_name, checkpointName)

    if os.path.isfile(checkpointName):
        print(f"Loading {checkpointName}...")
        checkpoint = torch.load(checkpointName, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.param_groups[0]['lr'] = checkpoint['lr']
        print('Checkpoint loaded')
    else:
        raise OSError('Checkpoint not found')

    return checkpoint['epoch'], model, optimizer


def build_scheduler(cfg, optimizer):
    name_scheduler = cfg.lr_scheduler.type
    scheduler = None

    if name_scheduler == 'StepLR':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step()
        scheduler = StepLR(optimizer=optimizer, step_size=cfg.lr_scheduler.step_size, gamma=cfg.lr_scheduler.gamma)
    elif name_scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=cfg.lr_scheduler.T_max)
    elif name_scheduler == 'ReduceLROnPlateau':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step(val_loss)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode=cfg.lr_scheduler.mode)
    elif name_scheduler == 'LambdaLR':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step()
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=cfg.lr_scheduler.lr_lambda)
    elif name_scheduler == 'MultiStepLR':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step()
        scheduler = MultiStepLR(optimizer=optimizer, milestones=cfg.lr_scheduler.milestones, gamma=cfg.lr_scheduler.gamma)
    elif name_scheduler == 'CyclicLR':
        # >>> for epoch in range(10):
        # >>>   for batch in data_loader:
        # >>>       train_batch(...)
        # >>>       scheduler.step()
        scheduler = CyclicLR(optimizer=optimizer, base_lr=cfg.lr_scheduler.base_lr, max_lr=cfg.lr_scheduler.max_lr)
    elif name_scheduler == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer=optimizer, gamma=cfg.lr_scheduler.gamma)
    elif name_scheduler == 'CosineAnnealingWarmRestarts':
        # >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
        # >>> for epoch in range(20):
        #     >>> scheduler.step()
        # >>> scheduler.step(26)
        # >>> scheduler.step()  # scheduler.step(27), instead of scheduler(20)
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=cfg.lr_scheduler.T_0,
                                                T_mult=cfg.lr_scheduler.T_mult)

    if cfg.warmup_epochs != 0:
        scheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=1, total_epoch=cfg.warmup_epochs, after_scheduler=scheduler)

    if scheduler is None:
        raise Exception('scheduler is wrong')
    return scheduler


