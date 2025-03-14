from torch.utils.data import DataLoader
from function.losses import *
from function.datasets.vaihingen_dataset import *
from function.model.GDGNet_model import GDGNet
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 200
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name ="GDGNet_200"
weights_path = "model_weights/vaihingen/{}".format(weights_name)
test_weights_name = "last_weight"
log_name = 'vaihingen/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1

pretrained_ckpt_path = None
gpus = 'auto'
resume_ckpt_path = None
net = GDGNet(num_classes=num_classes)
loss = GDGNetLoss(ignore_index=ignore_index)


train_dataset = VaihingenDataset(data_root='data/vaihingen/train', mode='train', img_dir='images_1024', mask_dir='masks_1024',
                                 mosaic_ratio=0.25, transform=train_aug)
val_dataset = VaihingenDataset(data_root='data/vaihingen/test', mode='val', img_dir='images_1024', mask_dir='masks_1024_eroded',
                                transform=val_aug)
test_dataset = VaihingenDataset(data_root='data/vaihingen/test', mode='val', img_dir='images_1024', mask_dir='masks_1024_eroded',
                                transform=val_aug)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

