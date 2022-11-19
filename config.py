work_dir = 'first_query'
image_h = 480
image_w = 640
action_map_h = 480
action_map_w = 640

patch_size = (16, 16)
num_patchs = 1200
num_patch_h = image_h // patch_size[0]
num_patch_w = image_w // patch_size[1]
feature_dim = 64
seq_len = 18
clamp_len = 9


# 模型参数
d_model = 128  # Embedding Size
d_ff = 128  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
enc_n_layers = 2  # number of Encoder of Decoder Layer
dec_n_layers = 4  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
dropout = 0
num_gauss = 5
MDN_hidden_num = 16
postion_method = 'fixed'


# train setting
lr = 1e-3
weight_decay = 0
warmup_epochs = 20
lr_scheduler = dict(type='MultiStepLR', milestones=[50, 100, 150, 200, 250], gamma=0.5)

train_batch_size = 40
val_batch_size = 12
epoch_num = 300
val_step = 5
seed = 1218

reload = True
device = "cuda:1"

# test
query = True