import argparse
from torch import optim
from dataloder.merge_dataloder import Merge_iter
from eval import Eval
from generator import Feature_Extrator
from torch.utils.tensorboard import SummaryWriter
from model import Transformer
from utils import *
from mmcv import Config, DictAction


parser = argparse.ArgumentParser(description='Train a models')

parser.add_argument('--config', default='./config.py', help='config.py path')
parser.add_argument('--work_dir', help='path to save logs and weights')
parser.add_argument('--device', help='cuda:n')
parser.add_argument('--options', nargs='+', action=DictAction, help='arguments in dict')

args = parser.parse_args()
cfg = Config.fromfile(args.config)
cfg.merge_from_dict(vars(args))
if args.options is not None:
    cfg.merge_from_dict(args.options)
cfg.work_dir = os.path.join('/data/qmengyu/logs/', cfg.work_dir)

writer = SummaryWriter(log_dir=cfg.work_dir)
cfg.dump(os.path.join(cfg.work_dir, 'config.py'))
save_str_file(f'{cfg.work_dir}/config.py', cfg.pretty_text)

setup_seed(cfg.seed)
model = Transformer(cfg).to(cfg.device)
feature_extrator = Feature_Extrator(cfg).get_features_seqs


epoch_start = 0
optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

if cfg.reload:
    epoch_start, model, optimizer = loadCheckpoint(-1, model, optimizer, cfg)

scheduler = build_scheduler(cfg=cfg, optimizer=optimizer)

criterion = torch.nn.CrossEntropyLoss()
criterion_duration = torch.nn.MSELoss()

merge_data_iter = Merge_iter(cfg).merge_data_iter
validation = Eval(cfg).validation

for epoch in range(epoch_start, cfg.epoch_num):
    if epoch % cfg.val_step == 0:
        validation(model, epoch, dataset_name='osie', writer=writer, cfg=cfg, feature_extrator=feature_extrator)
        save_checkpoint(epoch, model, optimizer, cfg)
    model.train()
    train_perfomance = Accumulator(1)
    for imgs, sals, gts, valid_lens, dec_inputs, dec_masks, names, durations_gt in merge_data_iter:
        optimizer.zero_grad()
        enc_inputs = feature_extrator(imgs.to(cfg.device), sals.to(cfg.device))
        pis, mus, sigmas, rhos, eos, durations = model(enc_inputs.to(cfg.device), dec_inputs.float().to(cfg.device),
                                                       dec_masks.to(cfg.device))

        probs = model.mdn.mixture_probability(pis, mus, sigmas, rhos, gts.unsqueeze(-1).to(cfg.device)).squeeze()
        probs_mask = torch.arange(cfg.seq_len).expand(cfg.train_batch_size, cfg.seq_len). \
            lt(valid_lens.unsqueeze(-1).expand(cfg.train_batch_size, cfg.seq_len)).to(cfg.device)
        probs = torch.masked_select(probs, probs_mask)
        loss_fixation = torch.mean(-torch.log(probs))
        loss = loss_fixation

        loss_item = loss.detach().cpu().item()
        train_perfomance.add(loss_item)
        print('Epoch:', '%04d' % epoch, loss_item)
        loss.backward()
        optimizer.step()

    train_perfomance.divided(train_perfomance.num)
    writer.add_scalar('AA_Scalar/train_loss', train_perfomance[0], epoch)

    learning_rate = float(optimizer.param_groups[0]['lr'])
    writer.add_scalar('BB_Scalar/train_lr', learning_rate, epoch)
    scheduler.step()
