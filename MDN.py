import math
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import torch


class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians, cfg):
        super(MDN, self).__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians     # 混合网络个数

        self.h, self.w = np.mgrid[0:self.cfg.action_map_h, 0:self.cfg.action_map_w]
        self.w_t = torch.from_numpy(self.w / float(self.cfg.action_map_w)).float().reshape(1, 1, -1)
        self.h_t = torch.from_numpy(self.h / float(self.cfg.action_map_h)).float().reshape(1, 1, -1)
        self.hw_t = torch.cat([self.h_t, self.w_t], dim=1).to(self.cfg.device)

        self.pi = nn.Sequential(
            nn.Linear(self.input_dim, cfg.MDN_hidden_num),
            nn.ReLU(),
            nn.Linear(cfg.MDN_hidden_num, self.num_gaussians),
            nn.Softmax(dim=-1)
        )
        self.mu = nn.Sequential(
            nn.Linear(self.input_dim, cfg.MDN_hidden_num),
            nn.ReLU(),
            nn.Linear(cfg.MDN_hidden_num, self.output_dim * self.num_gaussians)
        )
        self.std = nn.Sequential(
            nn.Linear(self.input_dim, cfg.MDN_hidden_num),
            nn.ReLU(),
            nn.Linear(cfg.MDN_hidden_num, self.output_dim * self.num_gaussians)
        )
        self.rho = nn.Sequential(
            nn.Linear(self.input_dim, cfg.MDN_hidden_num),
            nn.ReLU(),
            nn.Linear(cfg.MDN_hidden_num, self.num_gaussians)
        )
        self.mu[-1].bias.data.copy_(torch.rand_like(self.mu[-1].bias))

    def forward(self, x):
        pi = self.pi(x)
        mu = self.mu(x)
        sigma = torch.exp(self.std(x))
        # print(sigma.mean(0))
        # sigma = torch.clamp(sigma, 0.06, 10)
        rho = torch.tanh((self.rho(x)))

        eos = 0
        duration = 0
        # rho = torch.clamp(self.rho(x), -0.25, 0.25)
        mu = mu.reshape(-1, mu.size(1), self.num_gaussians, self.output_dim)
        sigma = sigma.reshape(-1, sigma.size(1), self.num_gaussians, self.output_dim)
        rho = rho.reshape(-1, rho.size(1), self.num_gaussians, 1)

        return pi, mu, sigma, rho, eos, duration

    def gaussian_probability(self, mu, sigma, rho, data):
        mean_y, mean_x = torch.chunk(mu, 2, dim=-1)
        std_y, std_x = torch.chunk(sigma, 2, dim=-1)
        y, x = torch.chunk(data, 2, dim=2)
        dx = x - mean_x
        dy = y - mean_y
        std_xy = std_x * std_y
        z = (dx * dx) / (std_x * std_x) + (dy * dy) / (std_y * std_y) - (2 * rho * dx * dy) / std_xy
        training_stablizer = 2
        norm = 1 / (training_stablizer * math.pi * std_x * std_y * torch.sqrt(1 - rho * rho))
        p = norm * torch.exp(-z / (1 - rho * rho) * 0.5)
        return p

    def mixture_probability(self, pi, mu, sigma, rho, data):
        pi = pi.unsqueeze(-1)
        prob = pi * self.gaussian_probability(mu, sigma, rho, data)
        prob = torch.sum(prob, dim=2)
        return prob

    def mixture_probability_map(self, pi, mu, sigma, rho):
        pi = pi.unsqueeze(-1)
        prob = pi * self.gaussian_probability(mu, sigma, rho, self.hw_t.unsqueeze(0))
        prob = torch.sum(prob, dim=2)
        return prob

    def sample_mdn_simp(self, pi, mu, sigma, rho, sample_num=1):
        pi, mu, sigma, rho = pi.float().cpu(), mu.float().cpu(), sigma.float().cpu(), rho.float().cpu()
        batch_size = pi.size(0)
        seq_len = pi.size(1)
        cat = Categorical(pi)
        pis = cat.sample()
        samples_seq = torch.zeros(batch_size, sample_num, seq_len, 2)
        samples_neg_log_probs = torch.zeros(batch_size, sample_num, seq_len)
        for index in range(batch_size):
            for num in range(sample_num):
                for t in range(seq_len):
                    idx = pis[index, t]
                    loc = mu[index, t, idx]
                    std = sigma[index, t, idx]
                    std_y, std_x = std[0].item(), std[1].item()
                    r = rho[index, t, idx].item()
                    cov_mat = torch.tensor([[std_y * std_y, std_y * std_x * r], [std_y * std_x * r, std_x * std_x]])

                    MN = MultivariateNormal(loc, covariance_matrix=cov_mat)
                    fixation = MN.sample()
                    neg_log_probs = -MN.log_prob(fixation)  # ?
                    samples_neg_log_probs[index, num, t] = neg_log_probs

                    samples_seq[index, num, t, 0] = fixation[0]
                    samples_seq[index, num, t, 1] = fixation[1]

        return samples_seq, samples_neg_log_probs

    def sample_mdn(self, pi, mu, sigma, rho, sample_num=10):
        seq_len = pi.size(1)
        pi, mu, sigma, rho = pi.float(), mu.float(), sigma.float(), rho.float()
        cat = Categorical(pi)
        pis = cat.sample()
        samples = list()
        samples_neg_log_probs = list()
        for num in range(sample_num):
            mask_index = torch.arange(5).unsqueeze(0).unsqueeze(0).expand(self.cfg.train_batch_size, seq_len, self.cfg.num_gauss).to(self.cfg.device) \
                         == pis.unsqueeze(-1).expand(self.cfg.train_batch_size, seq_len, self.cfg.num_gauss)

            mu_sample = mu[mask_index].reshape(self.cfg.train_batch_size, seq_len, 2)  # ??
            sigma_sample = sigma[mask_index].reshape(self.cfg.train_batch_size, seq_len, 2)
            rho_sample = rho[mask_index].reshape(self.cfg.train_batch_size, seq_len)

            loc = mu_sample
            std_y = sigma_sample[:, :, 0]
            std_x = sigma_sample[:, :, 1]
            r = rho_sample
            cov_mat = torch.zeros(self.cfg.train_batch_size, seq_len, 2, 2).to(self.cfg.device)
            cov_mat[:, :, 0, 0] = std_y * std_y
            cov_mat[:, :, 0, 1] = std_y * std_x * r
            cov_mat[:, :, 1, 0] = std_y * std_x * r
            cov_mat[:, :, 1, 1] = std_x * std_x

            MN = MultivariateNormal(loc, covariance_matrix=cov_mat)
            fixations = MN.sample()
            neg_log_probs = -MN.log_prob(fixations)

            samples.append(fixations.unsqueeze(0))
            samples_neg_log_probs.append(neg_log_probs.unsqueeze(0))
        samples = torch.cat(samples, dim=0).transpose(0, 1)
        samples_neg_log_probs = torch.cat(samples_neg_log_probs, dim=0).transpose(0, 1)
        return samples, samples_neg_log_probs

