from scipy import io
from torch import nn
from MDN import MDN
from dataloder.dataloder import Dataiter
from metrics.utils import get_score_salicon, get_score_filename_osie, get_score_filename_mit, get_score_filename_isun
from utils import *
import torch
criterion = nn.CrossEntropyLoss()


class Eval:
    def __init__(self, cfg):
        super(Eval, self).__init__()
        self.cfg = cfg
        self.dataiter = Dataiter(cfg)

    def validation(self, model, epoch, dataset_name, cfg, feature_extrator, save=False, writer=None, ):
        model.eval()
        save_dir_path = os.path.join(cfg.work_dir, f'{epoch}_results', dataset_name)
        if save and not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        with torch.no_grad():
            val_perfomance = Accumulator(16)
            if dataset_name == 'salicon':
                val_data_iter = self.dataiter.salicon_val_data_iter
            elif dataset_name == 'osie':
                val_data_iter = self.dataiter.osie_val_data_iter
            elif dataset_name == 'isun':
                val_data_iter = self.dataiter.isun_data_iter
            elif dataset_name == 'mit':
                val_data_iter = self.dataiter.mit_data_iter
            for imgs, sals, gts, valid_lens, _, _, names, img_size, pad_size, durations_gt in val_data_iter:
                val_batch_size = sals.size(0)
                enc_inputs = feature_extrator(imgs.to(cfg.device), sals.to(cfg.device))
                if self.cfg.query:
                    dec_inputs = torch.ones(val_batch_size, 1, 2).to(cfg.device) * 0.5
                else:
                    dec_inputs = torch.zeros(val_batch_size, 0, 2).to(cfg.device)
                enc_outputs, enc_self_attns = model.encoder(enc_inputs.to(cfg.device))

                for n in range(cfg.seq_len):
                    dec_outputs, dec_self_attns, dec_enc_attns = model.decoder(enc_outputs, dec_inputs,
                                                                               dec_masks=torch.zeros(val_batch_size,
                                                                                                     n + 1).to(cfg.device))

                    pis, mus, sigmas, rhos, eos, durtions = model.mdn(dec_outputs)
                    pred_roi_maps = model.mdn.mixture_probability_map(pis, mus, sigmas, rhos).reshape(
                        (-1, n + 1, cfg.action_map_h, cfg.action_map_w)).flatten(2)
                    _, indexs = pred_roi_maps.max(-1)
                    indexs_w = ((indexs % cfg.action_map_w) / cfg.action_map_w).unsqueeze(-1)
                    indexs_h = (torch.div(indexs, cfg.action_map_w, rounding_mode='trunc') / cfg.action_map_h).unsqueeze(-1)
                    outputs = torch.cat((indexs_h, indexs_w), -1)
                    outputs = outputs.clamp(0, 0.99)
                    # dec_inputs = outputs
                    dec_inputs = torch.cat((dec_inputs, outputs[:, -1].unsqueeze(1)), dim=1)

                len_error = 0
                duration_error = 0
                probs = model.mdn.mixture_probability(pis, mus, sigmas, rhos, gts.unsqueeze(-1).to(cfg.device)).squeeze()
                probs_mask = torch.arange(cfg.seq_len).expand(val_batch_size, cfg.seq_len). \
                    lt(valid_lens.unsqueeze(-1).expand(val_batch_size, cfg.seq_len)).to(cfg.device)
                probs = torch.masked_select(probs, probs_mask)
                loss_fixation = torch.mean(-torch.log(probs))

                loss = loss_fixation
                loss_item = loss.detach().cpu().item()

                pred_roi_maps = model.mdn.mixture_probability_map(pis, mus, sigmas, rhos).reshape(
                    (-1, cfg.seq_len, cfg.action_map_h, cfg.action_map_w))
                _, indexs = pred_roi_maps.flatten(2).max(-1)
                indexs_w = ((indexs % cfg.action_map_w) / cfg.action_map_w).unsqueeze(-1)
                indexs_h = (torch.div(indexs, cfg.action_map_w, rounding_mode='trunc') / cfg.action_map_h).unsqueeze(-1)
                outputs = torch.cat((indexs_h, indexs_w), -1).clamp(0, 0.99).detach().cpu()
                gts = gts.cpu()

                outputs = ((outputs * (img_size + pad_size)) - pad_size/2).clamp(min=0).numpy()
                gts = (gts * img_size).numpy()

                if dataset_name == 'osie' or dataset_name == 'salicon':
                    clamp_len = 9
                elif dataset_name == 'mit':
                    clamp_len = 8
                elif dataset_name == 'isun':
                    clamp_len = 6
                for batch_index in range(len(gts)):
                    output = outputs[batch_index][:clamp_len]
                    if save:
                        save_path = os.path.join(save_dir_path, names[batch_index])
                        io.savemat(save_path,
                                   {'fixations': output})
                    gt = gts[batch_index][:valid_lens[batch_index]]
                    if len(output) < 3 or len(gt) < 3:
                        continue
                    if dataset_name == 'salicon':
                        scanmatch_score, H_scores, MM_scores, mutimatch_scores = get_score_salicon(output, gt)
                    elif dataset_name == 'osie':
                        scanmatch_score, H_scores, MM_scores, mutimatch_scores = get_score_filename_osie(output, names[batch_index])
                    elif dataset_name == 'isun':
                        scanmatch_score, H_scores, MM_scores, mutimatch_scores = get_score_filename_isun(output, names[batch_index])
                    elif dataset_name == 'mit':
                        scanmatch_score, H_scores, MM_scores, mutimatch_scores = get_score_filename_mit(output, names[batch_index])

                    print('Epoch:', '%04d' % epoch, 'val_loss =', f'{loss_item:.6f}'
                                                                  f'val_score {scanmatch_score}')

                    val_perfomance.add(loss_item, scanmatch_score,
                                       H_scores[0], H_scores[1], H_scores[2], H_scores[3],
                                       MM_scores[0], MM_scores[1], MM_scores[2], MM_scores[3],
                                       mutimatch_scores[0], mutimatch_scores[1], mutimatch_scores[2], mutimatch_scores[3],
                                       len_error, duration_error)
            val_perfomance.divided(val_perfomance.num)
            print('epoch', epoch, f'{dataset_name}_val_loss', val_perfomance[0], f'{dataset_name}_val_score', val_perfomance[1])
            if not save:
                writer.add_scalar(f'BB_Scalar/val_{dataset_name}_loss', val_perfomance[0], epoch)
                writer.add_scalar(f'AA_Scalar/val_{dataset_name}_Score', val_perfomance[1], epoch)

                writer.add_scalar(f'val_{dataset_name}_H_TDE/score1', val_perfomance[2], epoch)
                writer.add_scalar(f'val_{dataset_name}_H_TDE/score2', val_perfomance[3], epoch)
                writer.add_scalar(f'val_{dataset_name}_H_TDE/score3', val_perfomance[4], epoch)
                writer.add_scalar(f'val_{dataset_name}_H_TDE/score4', val_perfomance[5], epoch)

                writer.add_scalar(f'val_{dataset_name}_MM_TDE/score1', val_perfomance[6], epoch)
                writer.add_scalar(f'val_{dataset_name}_MM_TDE/score2', val_perfomance[7], epoch)
                writer.add_scalar(f'val_{dataset_name}_MM_TDE/score3', val_perfomance[8], epoch)
                writer.add_scalar(f'val_{dataset_name}_MM_TDE/score4', val_perfomance[9], epoch)

                writer.add_scalar(f'val_{dataset_name}_mutimatch_scores/vector', val_perfomance[10], epoch)
                writer.add_scalar(f'val_{dataset_name}_mutimatch_scores/dirction', val_perfomance[11], epoch)
                writer.add_scalar(f'val_{dataset_name}_mutimatch_scores/length', val_perfomance[12], epoch)
                writer.add_scalar(f'val_{dataset_name}_mutimatch_scores/position', val_perfomance[13], epoch)
