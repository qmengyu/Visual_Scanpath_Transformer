import math
import numpy as np
import torch.nn as nn
from MDN import MDN
from einops import rearrange, repeat
import torch


class FixedAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size, position_embedding_type):
        super().__init__()

        self.position_embedding_type = position_embedding_type
        self.is_absolute = True

        inv_freq = 1. / (10000 ** (torch.arange(0, hidden_size, 2, dtype=torch.float) / hidden_size))
        position = torch.arange(max_position_embeddings, dtype=torch.float)
        sinusoid_inp = torch.einsum('i,j -> ij', position, inv_freq)
        embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('embeddings', embeddings)

    def forward_fixed(self, x):
        """
        return (b l d)
        """
        return x + self.embeddings[None, :x.size(1), :]

    def forward_rope(self, x):
        """
        return (b l d)
        """
        embeddings = self.embeddings[None, :x.size(1), :] # b l d
        embeddings = rearrange(embeddings, 'b l (j d) -> b l j d', j=2)
        sin, cos = embeddings.unbind(dim=-2) # b l d//2
        sin, cos = map(lambda t: repeat(t, '... d -> ... (d 2)'), (sin, cos)) # b l d
        return x * cos + self.rotate_every_two(x) * sin

    @staticmethod
    def rotate_every_two(x):
        x = rearrange(x, '... (d j) -> ... d j', j=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d j -> ... (d j)')

    def _forward(self, x):
        if self.position_embedding_type == 'fixed':
            return self.forward_fixed(x)

        elif self.position_embedding_type == 'rope':
            return self.forward_rope(x)

    def forward(self, x):
        if x.dim() == 3:
            return self._forward(x)

        elif x.dim() == 4:
            h = x.size(1)
            x = rearrange(x, 'b h l d -> (b h) l d')
            x = self._forward(x)
            x = rearrange(x, '(b h) l d -> b h l d', h=h)
            return x


class LearnableAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.is_absolute = True
        self.embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.register_buffer('position_ids', torch.arange(max_position_embeddings))

    def forward(self, x):
        """
        return (b l d) / (b h l d)
        """
        position_ids = self.position_ids[:x.size(-2)]

        if x.dim() == 3:
            return x + self.embeddings(position_ids)[None, :, :]
        elif x.dim() == 4:
            # x: [batch_size x n_heads x len_q x d_k]
            h = x.size(1)
            x = rearrange(x, 'b h l d -> b l (h d)')
            x = x + self.embeddings(position_ids)[None, :, :]
            x = rearrange(x, 'b l (h d) -> b h l d', h=h)
            return x


def PositionEmbeddingSine(cfg):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    num_pos_feats = cfg.d_model / 2
    temperature = 10000
    scale = 2 * math.pi
    normalize = False
    mask = torch.zeros(1, cfg.num_patch_h, cfg.num_patch_w).bool().to(cfg.device)
    assert mask is not None
    not_mask = ~mask
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=cfg.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
    # print(pos.shape)
    return pos


def get_attn_pad_mask(seq_q_mask, seq_k_mask):
    batch_size, len_q = seq_q_mask.size()
    batch_size, len_k = seq_k_mask.size()
    pad_attn_mask = seq_k_mask.data.unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k).bool()  # batch_size x len_q x len_k


def get_attn_subsequent_mask(seq, cfg):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask.to(cfg.device)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, cfg):
        super(ScaledDotProductAttention, self).__init__()
        self.cfg = cfg

    def forward(self, Q, K, V, attn_mask):
        # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.cfg.d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)

        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(cfg.d_model, cfg.d_k * cfg.n_heads)
        self.W_K = nn.Linear(cfg.d_model, cfg.d_k * cfg.n_heads)
        self.W_V = nn.Linear(cfg.d_model, cfg.d_v * cfg.n_heads)
        self.linear = nn.Linear(cfg.n_heads * cfg.d_v, cfg.d_model)
        self.layer_norm = nn.LayerNorm(cfg.d_model)

        self.cfg = cfg

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.cfg.n_heads, self.cfg.d_k).transpose(1,
                                                                       2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.cfg.n_heads, self.cfg.d_k).transpose(1,
                                                                       2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.cfg.n_heads, self.cfg.d_v).transpose(1,
                                                                       2)  # v_s: [batch_size x n_heads x len_k x d_v]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.cfg.n_heads, 1,
                                                      1)  # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention(self.cfg)(q_s, k_s, v_s, attn_mask)
        # context: [batch_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.cfg.n_heads * self.cfg.d_v)

        output = self.linear(context)
        return self.layer_norm(output + residual), attn  # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, cfg):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=cfg.d_model, out_channels=cfg.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=cfg.d_ff, out_channels=cfg.d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.dropout(output)
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(cfg)
        self.pos_ffn = PoswiseFeedForwardNet(cfg)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(cfg)
        self.dec_enc_attn = MultiHeadAttention(cfg)
        self.pos_ffn = PoswiseFeedForwardNet(cfg)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs = dec_inputs
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_outputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, None, dec_enc_attn


# class Len_DecoderLayer(nn.Module):
#     def __init__(self):
#         super(Len_DecoderLayer, self).__init__()
#         self.dec_enc_attn = MultiHeadAttention()
#         self.pos_ffn = PoswiseFeedForwardNet()
#
#     def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
#         dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_inputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
#         dec_outputs = self.pos_ffn(dec_outputs)
#         return dec_outputs, None, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.src_emb = nn.Linear(cfg.feature_dim, cfg.d_model)
        self.pos_emb = PositionEmbeddingSine(cfg)  # Transformer中位置编码时固定的，不需要学习
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.enc_n_layers)])

    def forward(self, enc_inputs):  # enc_inputs : [batch_size x source_len]
        enc_outputs = self.src_emb(enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs = enc_outputs + self.pos_emb.repeat(enc_inputs.size(0), 1, 1)
            enc_outputs, enc_self_attn = layer(enc_outputs, None)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Linear(2, cfg.d_model)
        if cfg.postion_method == 'fixed':
            self.pos_embed = FixedAbsolutePositionEmbedding(cfg.seq_len, cfg.d_model, position_embedding_type='fixed')  # Fix 位置编码
        elif cfg. postion_method == 'learning':
            self.pos_embed = LearnableAbsolutePositionEmbedding(cfg.seq_len, cfg.d_k * cfg.n_heads)  # learning 位置编码

        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.dec_n_layers)])
        if not cfg.query:
            self.start = torch.nn.Parameter(torch.randn(1, 1, cfg.d_model)).to(cfg.device)
        self.cfg = cfg

    def forward(self, enc_outputs, dec_inputs, dec_masks):
        if self.cfg.query:
            dec_outputs = self.tgt_emb(dec_inputs)
        else:
            dec_outputs = torch.cat((self.start.repeat(enc_outputs.size(0), 1, 1), self.tgt_emb(dec_inputs)), dim=1)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_masks, dec_masks)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_masks, self.cfg)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_masks,
                                              torch.zeros(enc_outputs.size(0), enc_outputs.size(1)).to(self.cfg.device))

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs = self.pos_embed(dec_outputs)
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


# class Len_Decoder(nn.Module):
#     def __init__(self):
#         super(Len_Decoder, self).__init__()
#         self.layers = nn.ModuleList([Len_DecoderLayer() for _ in range(cfg.dec_n_layers)])
#         self.len_query = torch.nn.Parameter(torch.randn(1, 1, cfg.d_model)).to(cfg.device)
#
#     def forward(self, enc_outputs):
#         len_dec_outputs = self.len_query.repeat(enc_outputs.size(0), 1, 1)
#         dec_self_attns, dec_enc_attns = [], []
#         for layer in self.layers:
#             len_dec_outputs, dec_self_attn, dec_enc_attn = layer(len_dec_outputs, enc_outputs, None, None)
#             dec_self_attns.append(dec_self_attn)
#             dec_enc_attns.append(dec_enc_attn)
#         return len_dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.cfg = cfg
        self.mdn = MDN(cfg.d_model, 2, cfg.num_gauss, cfg=cfg).to(cfg.device)

    def forward(self, enc_inputs, dec_inputs, dec_masks):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(enc_outputs, dec_inputs, dec_masks)
        pi, mu, sigma, rho, eos, duration = self.mdn(dec_outputs)

        return pi, mu, sigma, rho, eos, duration

